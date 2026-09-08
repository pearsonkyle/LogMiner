"""Parser for Codex CLI rollout JSONL files.

Codex persists an ordered stream of Responses API items in
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``.  The stream is richer than
the Chat Completions message format used by the training pipeline: function
calls, their outputs, and reasoning are separate items.  This parser folds
those items into the project's normal assistant/tool turn representation
without losing call IDs or JSON arguments.
"""

import contextlib
import json
from pathlib import Path
from typing import Any

from logminer.parsers.base import BaseParser, build_tool_schema


def _content_text(content: Any) -> str:
    """Extract text blocks from a Responses API message item."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part for part in parts if part.strip()).strip()


def _reasoning_text(value: Any) -> str:
    """Flatten the public reasoning-summary shapes used by Codex logs."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(filter(None, (_reasoning_text(item) for item in value))).strip()
    if isinstance(value, dict):
        # Current Responses API summaries contain text blocks; accepting both
        # fields also makes the parser tolerant of older Codex builds.
        return _reasoning_text(value.get("text") or value.get("content") or value.get("summary"))
    return ""


class CodexParser(BaseParser):
    SOURCE = "codex"
    DEFAULT_LOG_DIR = Path.home() / ".codex" / "sessions"

    def discover_sessions(self, input_path: Path) -> list[Path]:
        if input_path.is_file():
            return [input_path] if input_path.suffix == ".jsonl" else []
        return sorted(input_path.rglob("rollout-*.jsonl"))

    def parse_session(
        self,
        session_path: Path,
        include_thinking: bool = True,
        min_turns: int = 2,
        **kwargs: Any,
    ) -> dict | None:
        entries: list[dict[str, Any]] = []
        with open(session_path, encoding="utf-8") as handle:
            for line in handle:
                with contextlib.suppress(json.JSONDecodeError):
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        entries.append(entry)

        if not entries:
            return None

        entries.sort(key=lambda entry: entry.get("timestamp", ""))
        metadata: dict[str, Any] = {
            "session_id": session_path.stem,
            "start_time": None,
            "end_time": None,
        }
        messages: list[dict[str, Any]] = []
        seen_tools: dict[str, dict[str, Any]] = {}
        pending_thinking: list[str] = []

        def add_thinking(message: dict[str, Any]) -> None:
            if not include_thinking or not pending_thinking:
                return
            thought = "\n".join(pending_thinking).strip()
            pending_thinking.clear()
            if thought:
                prefix = f"<think>\n{thought}\n</think>"
                message["content"] = f"{prefix}\n{message.get('content', '')}".strip()

        def assistant_for_call() -> dict[str, Any]:
            # A Responses output can contain an assistant message followed by
            # one or more calls.  Combine those ordered items into one chat
            # assistant turn, unless a tool result/user turn has intervened.
            if messages and messages[-1].get("role") == "assistant":
                return messages[-1]
            message: dict[str, Any] = {"role": "assistant", "content": ""}
            add_thinking(message)
            messages.append(message)
            return message

        for entry in entries:
            timestamp = entry.get("timestamp")
            if timestamp:
                metadata["start_time"] = metadata["start_time"] or timestamp
                metadata["end_time"] = timestamp

            entry_type = entry.get("type")
            payload = entry.get("payload") or {}
            if not isinstance(payload, dict):
                continue

            if entry_type == "session_meta":
                metadata.update(
                    {
                        "session_id": payload.get("id") or metadata["session_id"],
                        "cwd": payload.get("cwd"),
                        "cli_version": payload.get("cli_version"),
                        "model_provider": payload.get("model_provider"),
                    }
                )
                continue
            if entry_type == "turn_context":
                metadata["model"] = payload.get("model") or metadata.get("model")
                metadata["cwd"] = payload.get("cwd") or metadata.get("cwd")
                continue

            # Older Codex versions record raw reasoning separately from the
            # Responses item.  It immediately precedes the corresponding
            # assistant item, so queue it until that item is emitted.
            if entry_type == "event_msg" and payload.get("type") == "agent_reasoning_raw_content":
                text = _reasoning_text(payload.get("text"))
                if text:
                    pending_thinking.append(text)
                continue

            if entry_type != "response_item":
                continue

            item_type = payload.get("type")
            if item_type == "reasoning":
                text = _reasoning_text(payload.get("summary")) or _reasoning_text(
                    payload.get("content")
                )
                if text:
                    pending_thinking.append(text)
                continue

            if item_type == "message":
                role = payload.get("role")
                text = _content_text(payload.get("content"))
                if role in ("developer", "system") and text:
                    messages.append({"role": "system", "content": text})
                elif role == "user" and text:
                    messages.append({"role": "user", "content": text})
                elif role == "assistant":
                    message = {"role": "assistant", "content": text}
                    add_thinking(message)
                    if message.get("content"):
                        messages.append(message)
                continue

            if item_type in ("function_call", "custom_tool_call"):
                name = payload.get("name", "")
                call_id = payload.get("call_id", "")
                # Custom tools use ``input`` while function tools use the
                # Responses API's ``arguments`` field.
                raw_arguments = payload.get("arguments", payload.get("input", "{}"))
                if not isinstance(raw_arguments, str):
                    raw_arguments = json.dumps(raw_arguments)
                try:
                    args = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    args = raw_arguments
                    arguments_are_object = False
                else:
                    arguments_are_object = isinstance(args, dict)
                if not arguments_are_object:
                    args = {"input": args}
                # Chat-template tool calls require an object argument value.
                # Codex custom tools commonly persist a bare string in
                # ``input``; wrap it while preserving the original value.
                if not arguments_are_object:
                    raw_arguments = json.dumps(args)
                if name and name not in seen_tools:
                    seen_tools[name] = build_tool_schema(name, args)
                if name and call_id:
                    assistant = assistant_for_call()
                    assistant.setdefault("tool_calls", []).append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": raw_arguments},
                        }
                    )
                continue

            if item_type in ("function_call_output", "custom_tool_call_output"):
                call_id = payload.get("call_id", "")
                output = payload.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output)
                if call_id:
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

        # A response stream can end after a reasoning item (for example when
        # the client was interrupted).  Do not attach it to a later user turn.
        valid = [
            message for message in messages if message.get("content") or message.get("tool_calls")
        ]
        if len(valid) < min_turns:
            return None
        return {
            "id": metadata["session_id"],
            "source": self.SOURCE,
            "metadata": metadata,
            "tools": list(seen_tools.values()),
            "messages": valid,
        }
