import json
import sqlite3
from pathlib import Path
from typing import Any

from logminer.parsers.base import BaseParser, build_tool_schema, normalize_tool_arguments

_BUNDLED_PROMPT_DIR = Path(__file__).parent / "opencode_prompts"
_USER_SYSTEM_PROMPT = Path.home() / ".local" / "share" / "opencode" / "system.md"


def _select_variant(model_id: str | None) -> str:
    """Mirror OpenCode's runtime selection (SystemPrompt.provider in opencode bin)."""
    mid = (model_id or "").strip()
    mid_lc = mid.lower()
    if "gpt-4" in mid or "o1" in mid or "o3" in mid:
        return "beast"
    if "gpt" in mid:
        return "codex" if "codex" in mid else "gpt"
    if "gemini-" in mid:
        return "gemini"
    if "claude" in mid:
        return "anthropic"
    if "trinity" in mid_lc:
        return "trinity"
    if "kimi" in mid_lc:
        return "kimi"
    return "default"


def _load_system_prompt(model_id: str | None) -> str | None:
    if _USER_SYSTEM_PROMPT.is_file():
        text = _USER_SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
        if text:
            return text
    bundled = _BUNDLED_PROMPT_DIR / f"{_select_variant(model_id)}.md"
    if bundled.is_file():
        text = bundled.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


class OpenCodeParser(BaseParser):
    SOURCE = "opencode"
    DEFAULT_LOG_DIR = Path.home() / ".local" / "share" / "opencode"

    def discover_sessions(self, input_path: Path) -> list[Path]:
        db = input_path / "opencode.db" if input_path.is_dir() else input_path
        return [db] if db.exists() else []

    def parse_session(self, session_path: Path, **kwargs: Any) -> dict | None:
        # Not directly used — use parse_all / _parse_db for OpenCode
        return None

    def parse_all(self, input_path: Path, **kwargs: Any) -> list[dict]:
        results = []
        for db_path in self.discover_sessions(input_path):
            try:
                results.extend(self._parse_db(db_path, **kwargs))
            except Exception as e:
                print(f"Error parsing {db_path}: {e}")
        return results

    def _parse_db(
        self,
        db_path: Path,
        include_thinking: bool = True,
        min_turns: int = 2,
        **kwargs: Any,
    ) -> list[dict]:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        results = []
        try:
            sessions = conn.execute(
                "SELECT id, directory, time_created, time_updated FROM session"
            ).fetchall()
            for session_row in sessions:
                session_id = session_row["id"]
                try:
                    result = self._parse_session_from_db(
                        conn,
                        session_id,
                        session_row,
                        include_thinking=include_thinking,
                        min_turns=min_turns,
                    )
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error parsing session {session_id}: {e}")
        finally:
            conn.close()
        return results

    def _parse_session_from_db(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        session_row: sqlite3.Row,
        include_thinking: bool = True,
        min_turns: int = 2,
    ) -> dict | None:
        messages_rows = conn.execute(
            "SELECT id, data, time_created FROM message "
            "WHERE session_id = ? ORDER BY time_created ASC, rowid ASC",
            (session_id,),
        ).fetchall()

        conversation: list[dict[str, Any]] = []
        seen_tools: dict[str, dict[str, Any]] = {}

        for msg_row in messages_rows:
            msg_data: dict[str, Any] = json.loads(msg_row["data"])
            role = msg_data.get("role", "")
            msg_id = msg_row["id"]

            parts_rows = conn.execute(
                "SELECT data FROM part WHERE message_id = ? ORDER BY time_created ASC, rowid ASC",
                (msg_id,),
            ).fetchall()

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for part_row in parts_rows:
                part_data: dict[str, Any] = json.loads(part_row["data"])
                ptype = part_data.get("type", "")

                if ptype == "text":
                    t = part_data.get("text", "")
                    if t.strip():
                        text_parts.append(t)

                elif ptype == "reasoning" and include_thinking:
                    t = part_data.get("text", "")
                    if t.strip():
                        text_parts.insert(0, f"<think>{t}</think>")

                elif ptype == "tool":
                    tool_name = part_data.get("tool", "")
                    state = part_data.get("state", {})
                    inp = normalize_tool_arguments(state.get("input"))
                    status = state.get("status", "")
                    output = state.get("output", "")
                    tool_id = part_data.get("id", f"call_{len(tool_calls)}")

                    if role == "assistant":
                        if tool_name and tool_name not in seen_tools:
                            seen_tools[tool_name] = build_tool_schema(tool_name, inp)
                        tool_calls.append(
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(inp),
                                },
                            }
                        )
                        tr: dict[str, Any] = {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": str(output) if output is not None else "",
                        }
                        if status == "error":
                            tr["is_error"] = True
                        tool_results.append(tr)

            if role == "user":
                user_text = "\n".join(text_parts).strip()
                if user_text:
                    conversation.append({"role": "user", "content": user_text})

            elif role == "assistant":
                # Tool-only turns need an explicit empty content value for
                # consistent chat-template rendering across all parsers.
                combined = "\n".join(text_parts).strip()
                asst_msg: dict[str, Any] = {"role": "assistant", "content": combined}
                if tool_calls:
                    asst_msg["tool_calls"] = tool_calls
                if "content" in asst_msg or "tool_calls" in asst_msg:
                    conversation.append(asst_msg)
                    conversation.extend(tool_results)

        valid = [m for m in conversation if m.get("content") or m.get("tool_calls")]
        if len(valid) < min_turns:
            return None

        model_val = None
        for msg_row in messages_rows:
            d = json.loads(msg_row["data"])
            m = d.get("model")
            if isinstance(m, dict):
                m = m.get("modelID")
            if m:
                model_val = m
                break

        system_prompt = _load_system_prompt(model_val)
        if system_prompt:
            valid.insert(0, {"role": "system", "content": system_prompt})

        return {
            "id": session_id,
            "source": self.SOURCE,
            "metadata": {
                "model": model_val,
                "cwd": session_row["directory"],
                "start_time": session_row["time_created"],
                "end_time": session_row["time_updated"],
            },
            "tools": list(seen_tools.values()),
            "messages": valid,
        }
