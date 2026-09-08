import contextlib
import json
import re
from pathlib import Path
from typing import Any

# Narrow set of unambiguous shell-failure signatures. The earlier version had
# regexes like `error:`, `failed`, `not found:`, `permission denied`, which
# matched on:
#   - source code being read (e.g. `sys.exit("ERROR: ...")`)
#   - OpenAPI enum values literally named "Failed"
#   - Claude's plan-approval system text
#   - the agent's own narrative output leaking via injection
# Empirically on real Claude logs those produced ~45% false positives and
# destroyed exactly the failed-then-recovered trajectories that are the most
# valuable agent-SFT signal.
#
# Keep only signatures that are unambiguous evidence of a *command* (not a
# file read or an arbitrary string) failing.
_NO_SUCH_FILE_RX = re.compile(r"no such file or directory", re.IGNORECASE)
_COMMAND_NOT_FOUND_RX = re.compile(r"command not found", re.IGNORECASE)
_EXIT_CODE_RX = re.compile(r"Exit Code:\s*(\d+)", re.IGNORECASE)

# Tools whose nonzero exit codes are normal/expected.
_NONZERO_OK_TOKENS = ("grep", "ps aux |", "lsof -i :", "sleep ", "jobs -l", "curl")


def has_failed_command(content: str) -> bool:
    """True iff the tool result is unambiguously a failed shell command.

    Only fires on signatures that cannot plausibly appear in successful
    output: ``Exit Code: <nonzero>`` (with known-noisy commands excepted),
    ``command not found``, and ``no such file or directory``. Broader
    "error"/"failed" matching was removed — see module-level comment above.
    """
    if not isinstance(content, str):
        return False

    exit_match = _EXIT_CODE_RX.search(content)
    if exit_match and int(exit_match.group(1)) != 0:
        cl = content.lower()
        if not any(token in cl for token in _NONZERO_OK_TOKENS):
            return True

    if _COMMAND_NOT_FOUND_RX.search(content):
        return True
    return bool(_NO_SUCH_FILE_RX.search(content))


def remove_orphaned_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip tool_calls entries that have no matching role:tool result."""
    result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            result_ids.add(msg.get("tool_call_id", ""))

    cleaned = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            kept_calls = [tc for tc in msg["tool_calls"] if tc.get("id", "") in result_ids]
            if not kept_calls and not msg.get("content", "").strip():
                continue
            msg = (
                {**msg, "tool_calls": kept_calls}
                if kept_calls
                else {k: v for k, v in msg.items() if k != "tool_calls"}
            )
        cleaned.append(msg)
    return cleaned


def clean_conversation(
    conversation: dict[str, Any],
    remove_orphaned: bool = True,
) -> tuple[dict[str, Any], int]:
    """Clean a conversation by removing structurally-broken messages.

    Drops:
    - Tool results that are unambiguous shell-command failures
      (``has_failed_command``: nonzero exit code, ``command not found``,
      ``no such file or directory``).
    - Empty assistant turns (no content, no tool_calls).
    - Orphaned assistant ``tool_calls`` whose IDs have no matching
      ``role: "tool"`` result (when ``remove_orphaned=True``).

    Does NOT drop tool results just because they have ``is_error=True`` or
    because their content contains the words "error" or "failed". The
    earlier broad-regex filter destroyed ~45% of legitimate file-read and
    traceback results — exactly the failure-recovery trajectories that are
    the most valuable signal for agent SFT. Quality scoring of
    error-heavy sessions is the scorer's job, not the cleaner's.
    """
    messages = conversation.get("messages", [])
    removed_count = 0

    cleaned_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")

        if role in ("system", "user"):
            cleaned_messages.append(msg)
            continue

        if role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if content.strip() or tool_calls:
                cleaned_messages.append(msg)
            continue

        if role == "tool":
            content = msg.get("content", "")
            if has_failed_command(content):
                removed_count += 1
                continue
            cleaned_messages.append(msg)

    if remove_orphaned:
        cleaned_messages = remove_orphaned_tool_calls(cleaned_messages)

    return (
        {
            **conversation,
            "messages": cleaned_messages,
            "_cleaned_stats": (
                {"removed_tool_results": removed_count} if removed_count > 0 else None
            ),
        },
        removed_count,
    )


DEFAULT_MAX_TOKENS = 131_072

# A line must appear in at least this share of a source's system prompts to
# count as boilerplate rather than session context.
DEFAULT_BOILERPLATE_SHARE = 0.9
# Below this many sessions there is no meaningful "corpus-wide" frequency,
# so nothing is condensed.
_MIN_GROUP_FOR_BOILERPLATE = 5

BOILERPLATE_MARKER = "[... standard agent instructions elided ...]"


def condense_system_prompts(
    records: list[dict[str, Any]],
    min_share: float = DEFAULT_BOILERPLATE_SHARE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collapse the boilerplate that every system prompt in a source shares.

    Agent CLIs ship a large fixed system prompt — measured here at 12.6k
    characters for Claude Code and 24.3k for Qwen Code, together 8.2% of the
    corpus. Because it is byte-identical across sessions it carries no
    discriminative signal: a model cannot learn "when the system prompt says
    X, do Y" from a corpus where it always says X. It only consumes context
    that the actual trajectory could use.

    What *does* vary is the session context the CLI injects into the same
    message — the environment block, the project's CLAUDE.md, git status.
    That conditions behaviour and is kept verbatim.

    So: group by ``source``, find the lines present in ``min_share`` of that
    group's system prompts, and replace each contiguous run of them with a
    single marker. The first line is always preserved — it names the agent,
    which is the one piece of boilerplate worth conditioning on.

    Returns ``(records, stats)``. Records are not mutated in place.
    """
    groups: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        messages = rec.get("messages") or []
        if messages and messages[0].get("role") == "system":
            groups.setdefault(rec.get("source", ""), []).append(i)

    static_by_source: dict[str, set[str]] = {}
    for source, idxs in groups.items():
        if len(idxs) < _MIN_GROUP_FOR_BOILERPLATE:
            continue
        freq: dict[str, int] = {}
        for i in idxs:
            content = records[i]["messages"][0].get("content") or ""
            for line in set(content.split("\n")):
                freq[line] = freq.get(line, 0) + 1
        threshold = min_share * len(idxs)
        static_by_source[source] = {ln for ln, c in freq.items() if c >= threshold and ln.strip()}

    out = list(records)
    stats = {"records_condensed": 0, "chars_removed": 0}
    for source, idxs in groups.items():
        static = static_by_source.get(source)
        if not static:
            continue
        for i in idxs:
            rec = out[i]
            messages = rec["messages"]
            original = messages[0].get("content") or ""
            condensed = _condense_lines(original, static)
            if len(condensed) >= len(original):
                continue
            stats["records_condensed"] += 1
            stats["chars_removed"] += len(original) - len(condensed)
            new_first = {**messages[0], "content": condensed}
            out[i] = {**rec, "messages": [new_first, *messages[1:]]}
    return out, stats


def _condense_lines(content: str, static: set[str]) -> str:
    """Replace each contiguous run of ``static`` lines with one marker."""
    lines = content.split("\n")
    result: list[str] = []
    in_run = False
    for idx, line in enumerate(lines):
        # Always keep the first line: it identifies the agent.
        if idx == 0:
            result.append(line)
            continue
        if line in static:
            if not in_run:
                result.append(BOILERPLATE_MARKER)
                in_run = True
            continue
        # Blank lines are never "static" (they appear everywhere), so without
        # this they would break one boilerplate block into a run of markers
        # separated by blanks. Absorb them while inside a run, and re-emit a
        # single separator when real content resumes.
        if in_run and not line.strip():
            continue
        if in_run:
            result.append("")
            in_run = False
        result.append(line)
    return "\n".join(result)


def _message_token_cost(msg: dict[str, Any]) -> int:
    """Estimated tokens for one message, including its tool-call payload."""
    from logminer.pipeline.evaluate import estimate_token_count

    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content)
    total = estimate_token_count(content)
    if msg.get("tool_calls"):
        total += estimate_token_count(json.dumps(msg["tool_calls"]))
    return total


def cap_conversation_length(
    conversation: dict[str, Any],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[dict[str, Any], bool]:
    """Truncate a conversation to a token budget at a safe boundary.

    Measured on a 522-session corpus, only 37% of records fit a 32k
    context and the largest was ~3.8M estimated tokens, so an uncapped
    dataset silently relies on whatever the trainer does when a sample
    overflows.

    Truncation keeps a *prefix* of the conversation rather than dropping
    the record: a prefix of an agent trajectory is itself a valid
    trajectory, whereas dropping loses the long multi-step sessions that
    are the most valuable training signal. The system message is always
    retained (it carries the embedded tool schemas), and
    :func:`remove_orphaned_tool_calls` runs afterwards so the cut can
    never leave an assistant ``tool_calls`` entry without its matching
    result.

    Token counts use the same ``estimate_token_count`` heuristic as the
    scorer, which over-estimates relative to a real BPE tokenizer. The
    cap is therefore conservative — output fits the budget with room to
    spare rather than landing just over it.

    Returns ``(conversation, was_truncated)``.
    """
    messages = conversation.get("messages", [])
    if not messages:
        return conversation, False

    kept: list[dict[str, Any]] = []
    used = 0
    for i, msg in enumerate(messages):
        cost = _message_token_cost(msg)
        # Always keep the first message even if it alone blows the budget;
        # a record with no system turn is worse than one that is oversized.
        if i > 0 and used + cost > max_tokens:
            break
        kept.append(msg)
        used += cost

    if len(kept) == len(messages):
        return conversation, False

    # Ending on a tool result gives the model nothing to predict, but dropping
    # that turn can orphan the assistant call before it — and removing *that*
    # can expose another trailing tool result. Alternate until stable.
    while kept:
        while kept and kept[-1].get("role") == "tool":
            kept.pop()
        pruned = remove_orphaned_tool_calls(kept)
        if len(pruned) == len(kept):
            kept = pruned
            break
        kept = pruned

    return {**conversation, "messages": kept}, True


def format_for_training(conversation: dict[str, Any]) -> dict[str, Any]:
    """Prepare a conversation record for ``trainer.py`` consumption.

    The trainer loads JSONL via ``load_dataset("json", …)`` and then calls
    ``ds.select_columns(["messages"])``, so any fields outside ``messages``
    are dropped.  To ensure the tool schemas and tool-call arguments survive:

    1. **Embed tools in first message** – moves the top-level ``tools`` list
       into ``messages[0]["tools"]`` where ``_prepare_messages_and_tools``
       will find it.
    2. **Parse ``arguments`` to dicts** – the parsers store ``arguments`` as
       JSON strings (OpenAI API format), but ``apply_chat_template`` expects
       dicts so it can serialise them itself.
    3. **Strip internal fields** – removes ``_cleaned_stats`` and other
       pipeline-only metadata that the trainer doesn't need.
    """
    conv = dict(conversation)
    messages = [dict(m) for m in conv.get("messages", [])]

    # 1. Embed top-level tools into the first message
    tools = conv.get("tools")
    if tools and messages:
        messages[0] = {**messages[0], "tools": tools}

    # 2. Convert tool_call arguments from JSON strings → dicts
    for msg in messages:
        if msg.get("tool_calls"):
            new_calls = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                func = tc.get("function")
                if isinstance(func, dict):
                    func = dict(func)
                    args = func.get("arguments")
                    if isinstance(args, str):
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            func["arguments"] = json.loads(args)
                    tc["function"] = func
                new_calls.append(tc)
            msg["tool_calls"] = new_calls

    conv["messages"] = messages

    # 3. Strip internal pipeline fields
    for key in ("_cleaned_stats",):
        conv.pop(key, None)

    return conv


def clean_file(
    input_path: Path,
    output_path: Path,
    remove_orphaned: bool = True,
) -> None:
    """Clean all conversations in a JSONL file."""
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
    total_removed = 0
    cleaned_conversations = []

    for conv in conversations:
        cleaned, removed = clean_conversation(conv, remove_orphaned=remove_orphaned)
        if removed > 0:
            print(f"Cleaned {conv.get('id', '')}: removed {removed} tool results")
        total_removed += removed
        cleaned_conversations.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for conv in cleaned_conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"Total tool results removed: {total_removed}")
    print(f"Output written to: {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up training data by removing problematic tool calls/results"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--no-remove-orphaned",
        action="store_true",
        help="Keep orphaned tool calls",
    )
    args = parser.parse_args()

    clean_file(
        Path(args.input),
        Path(args.output),
        remove_orphaned=not args.no_remove_orphaned,
    )


if __name__ == "__main__":
    main()
