from logminer.parsers.base import BaseParser
from logminer.parsers.claude import ClaudeParser
from logminer.parsers.cline import ClineParser
from logminer.parsers.codex import CodexParser
from logminer.parsers.opencode import OpenCodeParser
from logminer.parsers.qwen import QwenParser

REGISTRY: dict[str, type[BaseParser]] = {
    "claude": ClaudeParser,
    "cline": ClineParser,
    "codex": CodexParser,
    "opencode": OpenCodeParser,
    "qwen": QwenParser,
}


def get_parser(source: str) -> BaseParser:
    if source not in REGISTRY:
        raise ValueError(f"Unknown parser: {source}. Available: {list(REGISTRY)}")
    return REGISTRY[source]()
