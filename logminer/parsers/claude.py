import json
from pathlib import Path
from typing import Any

from logminer.parsers.base import BaseParser, build_tool_schema, normalize_tool_arguments

_BUNDLED_SYSTEM_PROMPT = Path(__file__).parent / "claude_system_prompt.md"
_USER_SYSTEM_PROMPT = Path.home() / ".claude" / "system.md"


def _load_system_prompt() -> str | None:
    for candidate in (_USER_SYSTEM_PROMPT, _BUNDLED_SYSTEM_PROMPT):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def _stitch_resumed_sessions(parsed: list[dict]) -> list[dict]:
    """Merge sessions linked by parentUuid/leafUuid across files.

    Claude Code splits one logical conversation across multiple JSONL files
    when `/resume` runs or auto-compaction fires. The continuation's first
    parented entry has a parentUuid pointing into the predecessor's uuids,
    and/or the continuation contains a `summary` entry whose leafUuid lives
    in the predecessor. Subagent sessions are left alone — they're children
    of a main session, not continuations of it.
    """
    main = [s for s in parsed if not s.get("_link", {}).get("is_subagent")]
    subs = [s for s in parsed if s.get("_link", {}).get("is_subagent")]

    by_uuid: dict[str, dict] = {}
    for s in main:
        for u in s["_link"]["own_uuids"]:
            by_uuid[u] = s

    # successor-id -> predecessor; predecessor-id -> successor
    pred_ids: set[int] = set()
    succ: dict[int, dict] = {}
    for s in main:
        link = s["_link"]
        anchor = link["first_parent"] or link["summary_leaf"]
        if anchor and anchor in by_uuid:
            p = by_uuid[anchor]
            if p is not s and id(p) not in succ:
                succ[id(p)] = s
                pred_ids.add(id(s))

    roots = [s for s in main if id(s) not in pred_ids]
    roots.sort(key=lambda x: x["_link"].get("start_time") or "")

    merged: list[dict] = []
    for root in roots:
        chain = [root]
        cur = root
        while id(cur) in succ:
            cur = succ[id(cur)]
            if cur in chain:
                break
            chain.append(cur)

        if len(chain) == 1:
            merged.append(_strip_link(root))
            continue

        head = chain[0]
        out = {
            "id": head["id"],
            "source": head["source"],
            "metadata": {
                **head["metadata"],
                "resumed_from": [c["id"] for c in chain[1:]],
                "end_time": chain[-1]["metadata"].get("end_time"),
            },
            "tools": list(
                {t["function"]["name"]: t for c in chain for t in c.get("tools", [])}.values()
            ),
            "messages": [m for c in chain for m in c["messages"]],
        }
        merged.append(out)

    merged.extend(_strip_link(s) for s in subs)
    return merged


def _strip_link(s: dict) -> dict:
    return {k: v for k, v in s.items() if k != "_link"}


def _make_tool_result_msg(tid: str, name: str, result_block: dict[str, Any]) -> dict[str, Any]:
    rc = result_block.get("content", "")
    if isinstance(rc, list):
        rc_parts = []
        for rb in rc:
            if isinstance(rb, dict):
                rc_parts.append(rb.get("text", str(rb)))
            else:
                rc_parts.append(str(rb))
        rc = "\n".join(rc_parts)
    msg: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tid,
        "name": name,
        "content": str(rc) if rc is not None else "",
    }
    if result_block.get("is_error"):
        msg["is_error"] = True
    return msg


class ClaudeParser(BaseParser):
    SOURCE = "claude"
    DEFAULT_LOG_DIR = Path.home() / ".claude" / "projects"

    def discover_sessions(self, input_path: Path) -> list[Path]:
        return sorted(input_path.glob("**/*.jsonl"))

    def parse_all(self, input_path: Path, **kwargs: Any) -> list[dict]:
        sessions = self.discover_sessions(input_path)
        parsed: list[dict] = []
        for sp in sessions:
            try:
                r = self.parse_session(sp, **kwargs)
            except Exception as e:
                print(f"Error parsing {sp}: {e}")
                continue
            if r is not None:
                parsed.append(r)
        stitched = _stitch_resumed_sessions(parsed)
        system_prompt = _load_system_prompt()
        if system_prompt:
            for s in stitched:
                s["messages"].insert(0, {"role": "system", "content": system_prompt})
        return stitched

    def parse_session(
        self,
        session_path: Path,
        include_thinking: bool = True,
        min_turns: int = 2,
        **kwargs: Any,
    ) -> dict | None:
        entries: list[dict[str, Any]] = []
        with open(session_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not entries:
            return None

        is_subagent = "/subagents/" in str(session_path)

        # Capture all uuids in this file, and the first parented entry's
        # parentUuid — used later to stitch resumed/compacted sessions.
        own_uuids: set[str] = set()
        first_parent: str | None = None
        for e in entries:
            u = e.get("uuid")
            if u:
                own_uuids.add(u)
            if first_parent is None and e.get("parentUuid"):
                first_parent = e.get("parentUuid")
        # If first_parent points back into this same file, it isn't a
        # cross-file link.
        if first_parent in own_uuids:
            first_parent = None

        # Summary entries carry a leafUuid pointing into a prior chain
        # that this file's contents pre-compacted from.
        summary_leaf: str | None = None
        for e in entries:
            if e.get("type") == "summary" and e.get("leafUuid"):
                summary_leaf = e.get("leafUuid")
                break

        # Pre-pass: collect tool results keyed by tool_use_id
        tool_results: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if entry.get("type") != "user":
                continue
            msg = entry.get("message", {})
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tid = block.get("tool_use_id", "")
                        if tid:
                            tool_results[tid] = block

        messages: list[dict[str, Any]] = []
        seen_tools: dict[str, dict[str, Any]] = {}
        metadata: dict[str, Any] = {}

        for entry in entries:
            entry_type = entry.get("type")

            # Extract metadata from first entries
            if not metadata:
                metadata = {
                    "model": entry.get("message", {}).get("model"),
                    "git_branch": entry.get("gitBranch"),
                    "cwd": entry.get("cwd"),
                    "start_time": entry.get("timestamp"),
                    "end_time": None,
                    "version": entry.get("version"),
                    "session_id": entry.get("sessionId"),
                }
            if entry.get("timestamp"):
                metadata["end_time"] = entry.get("timestamp")

            # Synthetic context-rebuild turns after auto-compaction look
            # like user messages but aren't human input. Same for isMeta.
            if entry.get("isCompactSummary") or entry.get("isMeta"):
                continue

            if entry_type == "user":
                msg = entry.get("message", {})
                content = msg.get("content", "")
                text_parts: list[str] = []
                if isinstance(content, str):
                    if content.strip():
                        text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text":
                            t = block.get("text", "")
                            if t.strip():
                                text_parts.append(t)
                        # skip tool_result blocks
                user_text = "\n".join(text_parts).strip()
                if user_text:
                    messages.append({"role": "user", "content": user_text})

            elif entry_type == "assistant":
                msg = entry.get("message", {})
                content = msg.get("content", [])
                if not metadata.get("model") and msg.get("model"):
                    metadata["model"] = msg.get("model")

                text_parts = []
                tool_calls: list[dict[str, Any]] = []

                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            t = block.get("text", "")
                            if t.strip():
                                text_parts.append(t)
                        elif btype == "thinking" and include_thinking:
                            t = block.get("thinking", "")
                            if t.strip():
                                text_parts.insert(0, f"<think>{t}</think>")
                        elif btype == "tool_use":
                            tid = block.get("id", "")
                            name = block.get("name", "")
                            inp = normalize_tool_arguments(block.get("input"))
                            if name and name not in seen_tools:
                                seen_tools[name] = build_tool_schema(name, inp)
                            tool_call: dict[str, Any] = {
                                "id": tid,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(inp),
                                },
                            }
                            tool_calls.append(tool_call)

                assistant_msg: dict[str, Any] = {"role": "assistant"}
                combined_text = "\n".join(text_parts).strip()
                assistant_msg["content"] = combined_text
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                if "content" in assistant_msg or "tool_calls" in assistant_msg:
                    messages.append(assistant_msg)

                # Append tool result messages after the assistant message
                for tc in tool_calls:
                    tid = tc["id"]
                    result_block = tool_results.get(tid)
                    if result_block is not None:
                        messages.append(
                            _make_tool_result_msg(tid, tc["function"]["name"], result_block)
                        )

        # Filter out empty messages
        messages = [m for m in messages if m.get("content") or m.get("tool_calls")]

        if len(messages) < min_turns:
            return None

        session_id = metadata.get("session_id") or session_path.stem
        if is_subagent:
            metadata["is_subagent"] = True
            session_id = f"{session_id}::{session_path.stem}"

        return {
            "id": session_id,
            "source": self.SOURCE,
            "metadata": metadata,
            "tools": list(seen_tools.values()),
            "messages": messages,
            "_link": {
                "own_uuids": own_uuids,
                "first_parent": first_parent,
                "summary_leaf": summary_leaf,
                "is_subagent": is_subagent,
                "start_time": metadata.get("start_time"),
            },
        }
