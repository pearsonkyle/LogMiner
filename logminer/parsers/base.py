from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def normalize_tool_arguments(value: Any) -> dict[str, Any]:
    """Return an object suitable for OpenAI-style chat-template tool calls.

    Harnesses normally persist object arguments, but a few custom/MCP tools
    use scalar or list input.  ``apply_chat_template`` expects an object, so
    preserve those values under ``input`` rather than emitting an invalid
    tool-call payload.
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"input": value}


def build_tool_schema(func_name: str, args: dict) -> dict:
    """Build an OpenAI-style tool schema from a function name and sample args."""
    properties = {}
    required = []
    for key, value in args.items():
        if isinstance(value, str):
            prop_type = "string"
        elif isinstance(value, bool):
            prop_type = "boolean"
        elif isinstance(value, int | float):
            prop_type = "number"
        elif isinstance(value, list):
            prop_type = "array"
        elif isinstance(value, dict):
            prop_type = "object"
        else:
            prop_type = "string"
        properties[key] = {"type": prop_type}
        if not key.startswith("_") and value is not None:
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": f"Tool function: {func_name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class BaseParser(ABC):
    SOURCE: str
    DEFAULT_LOG_DIR: Path

    @abstractmethod
    def discover_sessions(self, input_path: Path) -> list[Path]:
        """Find all session files/dirs under input_path."""

    @abstractmethod
    def parse_session(self, session_path: Path, **kwargs: Any) -> dict | None:
        """Parse a single session into OpenAI-style training format."""

    def parse_all(self, input_path: Path, **kwargs: Any) -> list[dict]:
        """Discover + parse all sessions."""
        sessions = self.discover_sessions(input_path)
        results = []
        for session_path in sessions:
            try:
                result = self.parse_session(session_path, **kwargs)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error parsing {session_path}: {e}")
        return results
