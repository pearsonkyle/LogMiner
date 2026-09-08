"""Quality scoring for agent-SFT conversations.

Produces a single ``score`` in ``[0, 1]`` from a weighted sum of saturating
signals (length, tool engagement, tool diversity, edit chains, tool-call
pairing) minus ratio penalties (error ratio, truly-empty-turn ratio). Trivial
sessions hit a hard floor and score ``0.0``; pure-chatter sessions are capped
below the default ``--min-score 0.5`` threshold.

The scoring shape is designed to *reward* the long, tool-diverse, edit-chained
trajectories that make good agent SFT data — a previous version inverted this
by penalizing normal Claude "silent tool call" turns (assistant content="" with
tool_calls set) and by using absolute (rather than ratio) error counts that
inevitably fired on long sessions.
"""

import json
import math
import re
from pathlib import Path
from typing import Any


def estimate_token_count(text: str) -> int:
    """Estimate token count from text.

    Uses a rough heuristic: word_count + len/4. This approximates ~4 chars/token
    on average and is not specific to any LLM tokenizer.
    """
    if not text:
        return 0
    return max(1, len(text.split()) + len(text) // 4)


def extract_thinking_tags(content: str) -> list[str]:
    return re.findall(r"<think>(.*?)</think>", content, re.DOTALL)


def _tool_pair_counts(messages: list[dict[str, Any]]) -> tuple[int, int]:
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                call_ids.add(tc.get("id", ""))
        elif msg.get("role") == "tool":
            result_ids.add(msg.get("tool_call_id", ""))
    return len(call_ids & result_ids), len(call_ids - result_ids)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# Saturation constants — tuned for agent-SFT trajectories. Past these points
# additional length/calls/diversity stop paying, which lets the score still
# discriminate among the "very good" tail by other axes (edit chains, etc).
_LENGTH_SATURATION_TOKENS = 8000
_CALLS_SATURATION = 20
_DIVERSITY_SATURATION = 5  # 5+ distinct tools = full diversity credit

_ERROR_RATIO_FLOOR = 0.30  # penalty kicks in past 30% errors
_ERROR_RATIO_SPAN = 0.30  # ... and saturates 30 percentage points later
_EMPTY_RATIO_FLOOR = 0.10
_EMPTY_RATIO_SPAN = 0.30

_READ_TOOLS = frozenset({"read_file", "view", "cat", "read"})
_WRITE_TOOLS = frozenset(
    {
        "delete_file",
        "write_file",
        "write_to_file",
        "edit",
        "replace_in_file",
        "create_file",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "write",
    }
)


def evaluate_conversation(
    conversation: dict[str, Any],
    min_turns: int = 2,
    min_token_count: int = 1000,
) -> dict[str, Any]:
    """Score a conversation for agent-SFT quality.

    Returns the original conversation dict with ``score``, ``label``,
    ``reasons``, ``metrics``, ``tools_used``, ``tool_count`` merged in.

    ``score`` is in ``[0, 1]``. ``label`` is ``"good"`` (>=0.7), ``"maybe"``
    (>=0.5), or ``"bad"``.
    """
    messages = conversation.get("messages", [])

    total_tokens = 0
    user_turns = 0
    assistant_turns = 0
    tool_calls_count = 0
    tool_results_count = 0
    tool_error_count = 0
    truly_empty_assistant = 0
    thinking_count = 0
    tool_names: set[str] = set()
    read_ops = 0
    write_ops = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if not isinstance(content, str):
            # Some providers leave content as a list of blocks; coerce for
            # token estimation only.
            content = json.dumps(content)

        if role == "user":
            user_turns += 1
            total_tokens += estimate_token_count(content)
        elif role == "assistant":
            assistant_turns += 1
            total_tokens += estimate_token_count(content)
            thinking_count += len(extract_thinking_tags(content))
            tool_calls = msg.get("tool_calls") or []
            if not content.strip() and not tool_calls:
                truly_empty_assistant += 1
            for tc in tool_calls:
                tool_calls_count += 1
                name = (tc.get("function", {}) or {}).get("name", "").lower()
                if name:
                    tool_names.add(name)
                if name in _READ_TOOLS:
                    read_ops += 1
                if name in _WRITE_TOOLS:
                    write_ops += 1
        elif role == "tool":
            tool_results_count += 1
            total_tokens += estimate_token_count(content)
            if "Error:" in content or msg.get("is_error", False):
                tool_error_count += 1

    complete_pairs, incomplete_calls = _tool_pair_counts(messages)
    tool_diversity = len(tool_names)
    has_edit_chain = read_ops > 0 and write_ops > 0
    err_ratio = (tool_error_count / tool_results_count) if tool_results_count else 0.0
    empty_ratio = (truly_empty_assistant / assistant_turns) if assistant_turns else 0.0

    # --- Saturating positive signals (each in [0, 1]) ---
    s_length = _clamp(math.log1p(total_tokens) / math.log1p(_LENGTH_SATURATION_TOKENS))
    s_calls = _clamp(math.log1p(tool_calls_count) / math.log1p(_CALLS_SATURATION))
    s_diversity = (
        _clamp((tool_diversity - 1) / (_DIVERSITY_SATURATION - 1)) if tool_diversity else 0.0
    )
    s_editchain = 1.0 if has_edit_chain else 0.0
    s_pairing = (complete_pairs / tool_calls_count) if tool_calls_count else 0.0
    s_thinking = 1.0 if thinking_count > 0 else 0.0
    s_user = 1.0 if user_turns >= 1 else 0.0

    # --- Ratio penalties (each in [0, 1]) ---
    p_err = _clamp((err_ratio - _ERROR_RATIO_FLOOR) / _ERROR_RATIO_SPAN)
    p_empty = _clamp((empty_ratio - _EMPTY_RATIO_FLOOR) / _EMPTY_RATIO_SPAN)

    score = (
        (
            0.35 * s_length
            + 0.20 * s_calls
            + 0.15 * s_diversity
            + 0.10 * s_editchain
            + 0.10 * s_pairing
            + 0.05 * s_thinking
            + 0.05 * s_user
        )
        - 0.30 * p_err
        - 0.20 * p_empty
    )

    reasons: list[str] = [
        f"length:{s_length:.2f}",
        f"calls:{s_calls:.2f}",
        f"diversity:{s_diversity:.2f}",
        f"editchain:{s_editchain:.2f}",
        f"pairing:{s_pairing:.2f}",
    ]
    if s_thinking:
        reasons.append("thinking")
    if not s_user:
        reasons.append("no_user_input")
    if p_err > 0:
        reasons.append(f"-error_ratio:{err_ratio:.2f}")
    if p_empty > 0:
        reasons.append(f"-empty_ratio:{empty_ratio:.2f}")

    # --- Hard floors override the additive score ---
    if len(messages) < min_turns or total_tokens < min_token_count:
        score = 0.0
        reasons.append("hard_floor:trivial")
    elif tool_calls_count == 0:
        # Chatter caps below the default --min-score 0.5 threshold so the
        # filter drops it by default but you can opt back in with a lower bar.
        score = min(score, 0.4)
        reasons.append("cap:no_tool_calls")

    score = _clamp(score)

    if score >= 0.7:
        label = "good"
    elif score >= 0.5:
        label = "maybe"
    else:
        label = "bad"

    evaluation = {
        "session_id": conversation.get("id", ""),
        "score": round(score, 3),
        "label": label,
        "reasons": reasons,
        "metrics": {
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "tool_calls": tool_calls_count,
            "tool_results": tool_results_count,
            "incomplete_tool_chains": incomplete_calls,
            "thinking_tags": thinking_count,
            "tool_error_count": tool_error_count,
            "error_ratio": round(err_ratio, 3),
            "truly_empty_assistant": truly_empty_assistant,
        },
        "tools_used": sorted(tool_names),
        "tool_count": tool_diversity,
    }
    return {**conversation, **evaluation}


def evaluate_file(
    input_path: Path,
    output_path: Path,
    min_turns: int = 2,
    min_token_count: int = 1000,
) -> None:
    conversations: list[dict[str, Any]] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                conv["id"] = conv.get("id", str(len(conversations)))
                conversations.append(conv)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(conversations)} conversations")
    results = [
        evaluate_conversation(
            c,
            min_turns=min_turns,
            min_token_count=min_token_count,
        )
        for c in conversations
    ]

    good = sum(1 for r in results if r["label"] == "good")
    maybe = sum(1 for r in results if r["label"] == "maybe")
    bad = sum(1 for r in results if r["label"] == "bad")
    print(f"Good: {good}  Maybe: {maybe}  Bad: {bad}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Output written to: {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate conversation quality for training data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--min-token-count", type=int, default=1000)
    args = parser.parse_args()

    evaluate_file(
        Path(args.input),
        Path(args.output),
        min_turns=args.min_turns,
        min_token_count=args.min_token_count,
    )


if __name__ == "__main__":
    main()
