"""Parser for Cline task transcripts.

Cline stores each task as an ``api_conversation_history.json`` array.  Its
model-facing protocol is XML-like: assistant text contains tool tags and the
following user turn contains ``[tool for '…'] Result:``.  This module turns
that protocol into the OpenAI chat/tool format used by the rest of LogMiner.
"""

import contextlib
import html
import json
import re
from pathlib import Path
from typing import Any

from logminer.parsers.base import BaseParser, build_tool_schema

# These are Cline's top-level XML tools.  Restricting extraction to this list
# avoids treating HTML, examples, and arbitrary XML in an answer as a call.
_TOOLS = frozenset(
    {
        "ask_followup_question", "attempt_completion", "browser_action",
        "delete_file", "execute_command", "list_code_definition_names",
        "list_files", "plan_mode_respond", "read_file", "replace_in_file",
        "search_files", "task_complete", "use_mcp_tool", "write_to_file",
    }
)
_COMPLETE_TOOLS = frozenset({"task_complete", "attempt_completion"})
_TOOL_TAG = re.compile(r"<(?P<name>[A-Za-z_][\w-]*)>(?P<body>.*?)</(?P=name)>", re.DOTALL)
_RESULT = re.compile(r"^\[(?P<name>[^\] ]+)(?:\s+for\s+[^\]]+)?\]\s*Result:\s*", re.DOTALL)
_ARGUMENT_TAG = re.compile(
    r"<(?P<name>[A-Za-z_][\w-]*)>(?P<body>.*?)</(?P=name)>", re.DOTALL
)


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(
        part.get("text", "") for part in value
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
    )


def _user_text_content(value: Any) -> str:
    """Remove Cline UI/context injections from model-facing user content.

    These are separate content blocks that Cline appends after real prompts
    and after tool results.  They are control-plane state, not a user request
    or tool observation, and include rapidly changing editor state and clocks.
    """
    if not isinstance(value, list):
        return _text_content(value)
    ignored_prefixes = (
        "<environment_details>",
        "# task_progress",
        "# TODO LIST UPDATE REQUIRED",
        "[TASK RESUMPTION]",
    )
    parts = []
    for part in value:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text", "")
        if isinstance(text, str) and not text.lstrip().startswith(ignored_prefixes):
            parts.append(text)
    return "\n".join(parts)


def _tool_arguments(body: str) -> dict[str, Any]:
    """Extract XML parameters without parsing literal code as XML.

    Cline uses XML delimiters but does not XML-escape a file's HTML, a shell
    heredoc, or a diff. ``ElementTree`` therefore rejects many perfectly
    valid Cline calls and previously collapsed them into a lossy ``input``.
    Matching only the immediate parameter delimiters retains those literals.
    """
    result: dict[str, Any] = {}
    for match in _ARGUMENT_TAG.finditer(body):
        name = match["name"]
        # This parameter is UI-only bookkeeping. Keeping it teaches the model
        # to emit Cline's progress widget rather than to perform the task.
        if name == "task_progress":
            continue
        value = html.unescape(match["body"]).strip()
        if name in result:
            result[name] = result[name] if isinstance(result[name], list) else [result[name]]
            result[name].append(value)
        else:
            result[name] = value
    if result:
        # MCP arguments are frequently encoded as JSON inside one XML tag.
        arguments = result.get("arguments")
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError):
                result["arguments"] = json.loads(arguments)
        return result
    text = re.sub(r"<[^>]+>", "", body).strip()
    return {"input": text} if text else {}


def _extract_assistant(text: str, include_thinking: bool) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def replace(match: re.Match[str]) -> str:
        name, body = match["name"], match["body"]
        if name in ("think", "thinking"):
            return f"<think>{body.strip()}</think>" if include_thinking and body.strip() else ""
        if name in _TOOLS:
            calls.append((name, _tool_arguments(body)))
            return ""
        return match.group(0)

    content = _TOOL_TAG.sub(replace, text)
    # Older Cline builds emitted an unclosed task_complete at stream end.
    for name in _TOOLS:
        marker = f"<{name}>"
        if marker in content:
            before, body = content.split(marker, 1)
            calls.append((name, _tool_arguments(body)))
            content = before
            break
    return content.strip(), calls


class ClineParser(BaseParser):
    SOURCE = "cline"
    DEFAULT_LOG_DIR = Path.home() / "Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/tasks"

    def discover_sessions(self, input_path: Path) -> list[Path]:
        if input_path.is_file():
            return [input_path] if input_path.name == "api_conversation_history.json" else []
        return sorted(input_path.rglob("api_conversation_history.json"))

    def parse_session(self, session_path: Path, include_thinking: bool = True, min_turns: int = 2, **kwargs: Any) -> dict | None:
        with contextlib.suppress(OSError, json.JSONDecodeError):
            entries = json.loads(session_path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                return None
        if not isinstance(locals().get("entries"), list):
            return None

        metadata: dict[str, Any] = {"session_id": session_path.parent.name}
        metadata_path = session_path.parent / "task_metadata.json"
        with contextlib.suppress(OSError, json.JSONDecodeError):
            task_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(task_metadata, dict):
                usage = task_metadata.get("model_usage") or []
                if usage and isinstance(usage[-1], dict):
                    metadata.update({"model": usage[-1].get("model_id"), "model_provider": usage[-1].get("model_provider_id")})

        messages: list[dict[str, Any]] = []
        tools: dict[str, dict[str, Any]] = {}
        pending: list[tuple[str, str]] = []
        complete_calls = 0
        stop_after_result = False

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            text = (
                _user_text_content(entry.get("content"))
                if role == "user"
                else _text_content(entry.get("content"))
            )
            if not text:
                continue
            if entry.get("ts"):
                metadata["start_time"] = metadata.get("start_time") or entry["ts"]
                metadata["end_time"] = entry["ts"]

            if role == "assistant":
                if stop_after_result:
                    break
                content, calls = _extract_assistant(text, include_thinking)
                assistant: dict[str, Any] = {"role": "assistant", "content": content}
                for index, (name, args) in enumerate(calls):
                    # A malformed/combined transcript can contain several
                    # completion tags in one turn; never admit a third one.
                    if name in _COMPLETE_TOOLS and complete_calls >= 2:
                        break
                    call_id = f"cline-{session_path.parent.name}-{len(messages)}-{index}"
                    assistant.setdefault("tool_calls", []).append({"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
                    pending.append((call_id, name))
                    tools.setdefault(name, build_tool_schema(name, args))
                    if name in _COMPLETE_TOOLS:
                        complete_calls += 1
                if assistant["content"] or assistant.get("tool_calls"):
                    messages.append(assistant)
                if complete_calls >= 2:
                    stop_after_result = True
                continue

            if role != "user":
                continue
            match = _RESULT.match(text)
            if match and pending:
                result_name = match["name"]
                index = next((i for i, (_, name) in enumerate(pending) if name == result_name), 0)
                call_id, name = pending.pop(index)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": text[match.end():]})
                if stop_after_result:
                    break
            else:
                messages.append({"role": "user", "content": text})

        valid = [message for message in messages if message.get("content") or message.get("tool_calls")]
        if len(valid) < min_turns:
            return None
        return {"id": session_path.parent.name, "source": self.SOURCE, "metadata": metadata, "tools": list(tools.values()), "messages": valid}
