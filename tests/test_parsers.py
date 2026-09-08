"""Tests for parser modules."""

import json
import sqlite3
from pathlib import Path

import pytest

from logminer.parsers import REGISTRY, get_parser
from logminer.parsers.claude import ClaudeParser
from logminer.parsers.cline import ClineParser
from logminer.parsers.codex import CodexParser
from logminer.parsers.opencode import OpenCodeParser
from logminer.parsers.qwen import QwenParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLAUDE_ENTRY_USER = {
    "type": "user",
    "sessionId": "ses-123",
    "timestamp": "2024-01-01T00:00:00Z",
    "gitBranch": "main",
    "cwd": "/projects/myapp",
    "version": "1.0",
    "message": {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please list files in the current directory."},
            {
                "type": "tool_result",
                "tool_use_id": "tool_1",
                "content": "file1.py\nfile2.py",
                "is_error": False,
            },
        ],
    },
}

CLAUDE_ENTRY_ASSISTANT = {
    "type": "assistant",
    "timestamp": "2024-01-01T00:01:00Z",
    "message": {
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [
            {"type": "text", "text": "I'll list the files for you."},
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "bash",
                "input": {"command": "ls -la"},
            },
        ],
    },
}

CLAUDE_ENTRY_USER2 = {
    "type": "user",
    "timestamp": "2024-01-01T00:02:00Z",
    "message": {
        "role": "user",
        "content": "Thanks!",
    },
}


def make_claude_jsonl(tmp_path: Path) -> Path:
    session_dir = tmp_path / "project1"
    session_dir.mkdir()
    session_file = session_dir / "session1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps(CLAUDE_ENTRY_USER) + "\n")
        f.write(json.dumps(CLAUDE_ENTRY_ASSISTANT) + "\n")
        f.write(json.dumps(CLAUDE_ENTRY_USER2) + "\n")
    return tmp_path


QWEN_MSG_USER = {
    "uuid": "u1",
    "type": "user",
    "timestamp": "2024-01-01T00:00:00Z",
    "message": {"parts": [{"text": "What files are here?"}]},
}

QWEN_MSG_ASSISTANT = {
    "uuid": "u2",
    "type": "assistant",
    "timestamp": "2024-01-01T00:01:00Z",
    "message": {
        "parts": [
            {"text": "Let me check."},
            {
                "functionCall": {
                    "name": "bash",
                    "args": {"command": "ls"},
                    "id": "call_1",
                }
            },
        ]
    },
}

QWEN_MSG_TOOL = {
    "uuid": "u3",
    "type": "tool_result",
    "timestamp": "2024-01-01T00:01:05Z",
    "toolCallResult": {"callId": "call_1"},
    "message": {
        "parts": [
            {
                "functionResponse": {
                    "name": "bash",
                    "response": {"output": "file1.py\nfile2.py", "error": ""},
                }
            }
        ]
    },
}

QWEN_MSG_USER2 = {
    "uuid": "u4",
    "type": "user",
    "timestamp": "2024-01-01T00:02:00Z",
    "message": {"parts": [{"text": "Thanks!"}]},
}


def make_qwen_sessions(tmp_path: Path) -> Path:
    proj_dir = tmp_path / "myproject"
    chats_dir = proj_dir / "chats"
    chats_dir.mkdir(parents=True)
    session_file = chats_dir / "sess1.jsonl"
    with open(session_file, "w") as f:
        for msg in [QWEN_MSG_USER, QWEN_MSG_ASSISTANT, QWEN_MSG_TOOL, QWEN_MSG_USER2]:
            f.write(json.dumps(msg) + "\n")
    return tmp_path


def make_opencode_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE session (id TEXT, directory TEXT, time_created TEXT, time_updated TEXT)"
    )
    conn.execute("CREATE TABLE message (id TEXT, session_id TEXT, data TEXT, time_created TEXT)")
    conn.execute("CREATE TABLE part (message_id TEXT, data TEXT, time_created TEXT)")
    conn.execute(
        "INSERT INTO session VALUES (?, ?, ?, ?)",
        ("sess-abc", "/projects/myapp", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"),
    )
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?)",
        (
            "msg-1",
            "sess-abc",
            json.dumps({"role": "user", "model": ""}),
            "2024-01-01T00:00:00Z",
        ),
    )
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?)",
        (
            "msg-1",
            json.dumps({"type": "text", "text": "List the files please"}),
            "2024-01-01T00:00:00Z",
        ),
    )
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?)",
        (
            "msg-2",
            "sess-abc",
            json.dumps({"role": "assistant", "model": "gpt-4o"}),
            "2024-01-01T00:01:00Z",
        ),
    )
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?)",
        (
            "msg-2",
            json.dumps({"type": "text", "text": "Sure, let me check."}),
            "2024-01-01T00:01:00Z",
        ),
    )
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?)",
        (
            "msg-2",
            json.dumps(
                {
                    "type": "tool",
                    "id": "tc1",
                    "tool": "bash",
                    "state": {
                        "input": {"command": "ls"},
                        "status": "completed",
                        "output": "file1.py\nfile2.py",
                    },
                }
            ),
            "2024-01-01T00:01:05Z",
        ),
    )
    conn.commit()
    conn.close()
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_registry_contains_all_parsers():
    assert "claude" in REGISTRY
    assert "opencode" in REGISTRY
    assert "qwen" in REGISTRY
    assert "codex" in REGISTRY
    assert "cline" in REGISTRY


def test_get_parser_returns_correct_type():
    assert isinstance(get_parser("claude"), ClaudeParser)
    assert isinstance(get_parser("opencode"), OpenCodeParser)
    assert isinstance(get_parser("qwen"), QwenParser)
    assert isinstance(get_parser("codex"), CodexParser)
    assert isinstance(get_parser("cline"), ClineParser)


# ---------------------------------------------------------------------------
# Cline parser tests
# ---------------------------------------------------------------------------


def test_cline_parser_extracts_reasoning_tools_and_caps_completions(tmp_path):
    """Cline XML calls pair with its following bracketed tool-result turns."""
    task = tmp_path / "123"
    task.mkdir()
    transcript = [
        {"role": "user", "content": [{"type": "text", "text": "Fix the bug."}], "ts": 1},
        {"role": "assistant", "content": [{"type": "text", "text": "<thinking>Inspect first.</thinking><read_file><path>a.py</path></read_file>"}], "ts": 2},
        {"role": "user", "content": [{"type": "text", "text": "[read_file for 'a.py'] Result:\nprint('x')"}], "ts": 3},
        {"role": "assistant", "content": [{"type": "text", "text": "<attempt_completion><result>first</result></attempt_completion>"}], "ts": 4},
        {"role": "user", "content": [{"type": "text", "text": "[attempt_completion] Result:\nok"}], "ts": 5},
        {"role": "assistant", "content": [{"type": "text", "text": "<task_complete><result>second</result></task_complete>"}], "ts": 6},
        {"role": "user", "content": [{"type": "text", "text": "[task_complete] Result:\ndone"}], "ts": 7},
        {"role": "user", "content": [{"type": "text", "text": "This must be excluded."}], "ts": 8},
    ]
    (task / "api_conversation_history.json").write_text(json.dumps(transcript))
    (task / "task_metadata.json").write_text(json.dumps({"model_usage": [{"model_id": "test-model"}]}))

    result = ClineParser().parse_session(task / "api_conversation_history.json")

    assert result is not None
    assert result["source"] == "cline"
    assert result["metadata"]["model"] == "test-model"
    read_turn = next(
        m
        for m in result["messages"]
        if any(tc["function"]["name"] == "read_file" for tc in m.get("tool_calls", []))
    )
    assert read_turn["content"] == "<think>Inspect first.</think>"
    assert read_turn["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}'
    assert any(m["role"] == "tool" and m["content"] == "print('x')" for m in result["messages"])
    assert [tc["function"]["name"] for m in result["messages"] for tc in m.get("tool_calls", [])].count("attempt_completion") == 1
    assert [tc["function"]["name"] for m in result["messages"] for tc in m.get("tool_calls", [])].count("task_complete") == 1
    assert not any("excluded" in m.get("content", "") for m in result["messages"])
    schema = next(t for t in result["tools"] if t["function"]["name"] == "read_file")
    assert schema["function"]["parameters"]["properties"] == {"path": {"type": "string"}}


def test_cline_parser_strips_ui_injections_and_preserves_literal_tool_content(tmp_path):
    task = tmp_path / "123"
    task.mkdir()
    transcript = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Implement the page."},
                {"type": "text", "text": "# task_progress RECOMMENDED\nUse a todo list."},
                {"type": "text", "text": "<environment_details>editor state</environment_details>"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "<write_to_file><path>page.html</path>"
                        "<content><div>literal HTML</div></content>"
                        "<task_progress>- [x] done</task_progress></write_to_file>"
                    ),
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "[write_to_file] Result:\nok"}]},
    ]
    (task / "api_conversation_history.json").write_text(json.dumps(transcript))

    result = ClineParser().parse_session(task / "api_conversation_history.json")

    assert result is not None
    assert result["messages"][0] == {"role": "user", "content": "Implement the page."}
    args = json.loads(result["messages"][1]["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "page.html", "content": "<div>literal HTML</div>"}


# ---------------------------------------------------------------------------
# Codex parser tests
# ---------------------------------------------------------------------------


def test_codex_parser_preserves_tool_pairing_and_reasoning(tmp_path):
    session = tmp_path / "2026" / "01" / "02" / "rollout-test.jsonl"
    session.parent.mkdir(parents=True)
    entries = [
        {
            "timestamp": "2026-01-02T00:00:00Z",
            "type": "session_meta",
            "payload": {"id": "codex-session", "cwd": "/projects/demo", "cli_version": "1"},
        },
        {
            "timestamp": "2026-01-02T00:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "List the files."}],
            },
        },
        {
            "timestamp": "2026-01-02T00:00:02Z",
            "type": "event_msg",
            "payload": {"type": "agent_reasoning_raw_content", "text": "Inspect the directory."},
        },
        {
            "timestamp": "2026-01-02T00:00:03Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell",
                "arguments": '{"command":"ls","all":true}',
                "call_id": "call_1",
            },
        },
        {
            "timestamp": "2026-01-02T00:00:04Z",
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "call_1", "output": "a.py"},
        },
        {
            "timestamp": "2026-01-02T00:00:05Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "There is one file."}],
            },
        },
    ]
    session.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")

    result = CodexParser().parse_session(session)

    assert result is not None
    assert result["id"] == "codex-session"
    assert result["messages"][1]["tool_calls"][0] == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "shell", "arguments": '{"command":"ls","all":true}'},
    }
    assert result["messages"][1]["content"] == "<think>\nInspect the directory.\n</think>"
    assert result["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "a.py"}
    properties = result["tools"][0]["function"]["parameters"]["properties"]
    assert properties == {"command": {"type": "string"}, "all": {"type": "boolean"}}


def test_get_parser_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown parser"):
        get_parser("nonexistent")


# ---------------------------------------------------------------------------
# Claude parser tests
# ---------------------------------------------------------------------------


def test_claude_discover_sessions(tmp_path):
    base = make_claude_jsonl(tmp_path)
    parser = ClaudeParser()
    sessions = parser.discover_sessions(base)
    assert len(sessions) == 1
    assert sessions[0].suffix == ".jsonl"


def test_claude_parse_session_returns_correct_format(tmp_path):
    base = make_claude_jsonl(tmp_path)
    parser = ClaudeParser()
    sessions = parser.discover_sessions(base)
    result = parser.parse_session(sessions[0])
    assert result is not None
    assert result["source"] == "claude"
    assert "id" in result
    assert "metadata" in result
    assert "tools" in result
    assert "messages" in result


def test_claude_parse_session_messages(tmp_path):
    base = make_claude_jsonl(tmp_path)
    parser = ClaudeParser()
    sessions = parser.discover_sessions(base)
    result = parser.parse_session(sessions[0])
    assert result is not None
    roles = [m["role"] for m in result["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_claude_parse_session_tool_result_attached(tmp_path):
    base = make_claude_jsonl(tmp_path)
    parser = ClaudeParser()
    sessions = parser.discover_sessions(base)
    result = parser.parse_session(sessions[0])
    assert result is not None
    tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) >= 1
    assert tool_msgs[0]["content"] == "file1.py\nfile2.py"


def test_claude_parse_session_tool_schema_built(tmp_path):
    base = make_claude_jsonl(tmp_path)
    parser = ClaudeParser()
    sessions = parser.discover_sessions(base)
    result = parser.parse_session(sessions[0])
    assert result is not None
    assert len(result["tools"]) >= 1
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert "name" in tool["function"]


def test_claude_resumed_sessions_stitched(tmp_path):
    """Two files where file2's first parented entry points into file1's uuids
    should be merged by parse_all into one logical session."""
    proj = tmp_path / "proj"
    proj.mkdir()

    f1 = proj / "ses-A.jsonl"
    user_a = {**CLAUDE_ENTRY_USER, "uuid": "uA", "sessionId": "ses-A"}
    asst_a = {**CLAUDE_ENTRY_ASSISTANT, "uuid": "uA2", "parentUuid": "uA"}
    f1.write_text(json.dumps(user_a) + "\n" + json.dumps(asst_a) + "\n")

    f2 = proj / "ses-B.jsonl"
    user_b = {
        **CLAUDE_ENTRY_USER2,
        "uuid": "uB",
        "parentUuid": "uA2",  # points into f1
        "sessionId": "ses-B",
        "timestamp": "2024-01-01T00:05:00Z",
        "message": {"role": "user", "content": "follow-up"},
    }
    asst_b = {**CLAUDE_ENTRY_ASSISTANT, "uuid": "uB2", "parentUuid": "uB"}
    f2.write_text(json.dumps(user_b) + "\n" + json.dumps(asst_b) + "\n")

    parser = ClaudeParser()
    results = parser.parse_all(tmp_path)
    assert len(results) == 1
    merged = results[0]
    assert "resumed_from" in merged["metadata"]
    contents = [m.get("content", "") for m in merged["messages"]]
    assert any("list files" in c for c in contents)
    assert any("follow-up" in c for c in contents)


def test_claude_compact_summary_filtered(tmp_path):
    """User entries with isCompactSummary or isMeta are synthetic, not human."""
    proj = tmp_path / "proj"
    proj.mkdir()
    f = proj / "s.jsonl"
    synthetic = {
        **CLAUDE_ENTRY_USER,
        "uuid": "uS",
        "isCompactSummary": True,
        "message": {"role": "user", "content": "AUTO-COMPACT RECAP"},
    }
    real = {
        **CLAUDE_ENTRY_USER,
        "uuid": "uR",
        "message": {"role": "user", "content": "real question"},
    }
    asst = {**CLAUDE_ENTRY_ASSISTANT, "uuid": "uA"}
    f.write_text("\n".join(json.dumps(e) for e in (synthetic, real, asst)) + "\n")

    parser = ClaudeParser()
    result = parser.parse_session(f)
    assert result is not None
    contents = [m.get("content", "") for m in result["messages"]]
    assert not any("AUTO-COMPACT" in c for c in contents)
    assert any("real question" in c for c in contents)


def test_claude_subagent_included(tmp_path):
    """Subagent jsonl files under /subagents/ should be discovered, and parsed
    sessions should be tagged with metadata.is_subagent=True."""
    proj = tmp_path / "project1"
    proj.mkdir()
    normal = proj / "session1.jsonl"
    normal.write_text(
        json.dumps(CLAUDE_ENTRY_USER) + "\n" + json.dumps(CLAUDE_ENTRY_ASSISTANT) + "\n"
    )

    sub_dir = proj / "abc123" / "subagents"
    sub_dir.mkdir(parents=True)
    subagent = sub_dir / "agent-def456.jsonl"
    subagent.write_text(
        json.dumps(CLAUDE_ENTRY_USER) + "\n" + json.dumps(CLAUDE_ENTRY_ASSISTANT) + "\n"
    )

    parser = ClaudeParser()
    sessions = parser.discover_sessions(tmp_path)
    assert len(sessions) == 2

    results = parser.parse_all(tmp_path)
    flags = [r["metadata"].get("is_subagent", False) for r in results]
    assert True in flags
    assert False in flags


def test_claude_tool_only_assistant_has_content(tmp_path):
    """Assistant messages with only tool_use (no text) should have content=''."""
    entry_asst_tool_only = {
        "type": "assistant",
        "timestamp": "2024-01-01T00:01:00Z",
        "message": {
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "Read",
                    "input": {"file_path": "/tmp/test.py"},
                },
            ],
        },
    }
    entry_user_result = {
        "type": "user",
        "timestamp": "2024-01-01T00:02:00Z",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_2",
                    "content": "print('hello')",
                },
            ],
        },
    }
    proj = tmp_path / "project1"
    proj.mkdir()
    session_file = proj / "session.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps(CLAUDE_ENTRY_USER) + "\n")
        f.write(json.dumps(entry_asst_tool_only) + "\n")
        f.write(json.dumps(entry_user_result) + "\n")
        f.write(json.dumps(CLAUDE_ENTRY_USER2) + "\n")

    parser = ClaudeParser()
    result = parser.parse_session(session_file)
    assert result is not None
    asst_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
    assert len(asst_msgs) >= 1
    # The tool-only assistant message should have content="" (not missing)
    tool_only = [m for m in asst_msgs if m.get("tool_calls")]
    assert len(tool_only) >= 1
    assert "content" in tool_only[0]
    assert tool_only[0]["content"] == ""


def test_claude_multi_tool_calls(tmp_path):
    """Multiple tool_use blocks in one assistant entry should all be captured."""
    entry_asst_multi = {
        "type": "assistant",
        "timestamp": "2024-01-01T00:01:00Z",
        "message": {
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [
                {"type": "text", "text": "Let me check both files."},
                {
                    "type": "tool_use",
                    "id": "tc_a",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a.py"},
                },
                {
                    "type": "tool_use",
                    "id": "tc_b",
                    "name": "Read",
                    "input": {"file_path": "/tmp/b.py"},
                },
            ],
        },
    }
    entry_user_results = {
        "type": "user",
        "timestamp": "2024-01-01T00:02:00Z",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tc_a", "content": "aaa"},
                {"type": "tool_result", "tool_use_id": "tc_b", "content": "bbb"},
            ],
        },
    }
    proj = tmp_path / "p"
    proj.mkdir()
    sf = proj / "s.jsonl"
    with open(sf, "w") as f:
        f.write(json.dumps(CLAUDE_ENTRY_USER) + "\n")
        f.write(json.dumps(entry_asst_multi) + "\n")
        f.write(json.dumps(entry_user_results) + "\n")

    parser = ClaudeParser()
    result = parser.parse_session(sf)
    assert result is not None
    asst = [m for m in result["messages"] if m.get("tool_calls")]
    assert len(asst) == 1
    assert len(asst[0]["tool_calls"]) == 2
    tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert {m["content"] for m in tool_msgs} == {"aaa", "bbb"}


def test_claude_thinking_with_tool_use(tmp_path):
    """Thinking blocks and tool_use in the same assistant entry."""
    entry = {
        "type": "assistant",
        "timestamp": "2024-01-01T00:01:00Z",
        "message": {
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [
                {"type": "thinking", "thinking": "I should read the file first."},
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tc_think",
                    "name": "Bash",
                    "input": {"command": "cat /tmp/x"},
                },
            ],
        },
    }
    entry_result = {
        "type": "user",
        "timestamp": "2024-01-01T00:02:00Z",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tc_think", "content": "data"},
            ],
        },
    }
    proj = tmp_path / "p"
    proj.mkdir()
    sf = proj / "s.jsonl"
    with open(sf, "w") as f:
        f.write(json.dumps(CLAUDE_ENTRY_USER) + "\n")
        f.write(json.dumps(entry) + "\n")
        f.write(json.dumps(entry_result) + "\n")

    parser = ClaudeParser()
    result = parser.parse_session(sf)
    assert result is not None
    asst = [m for m in result["messages"] if m.get("tool_calls")]
    assert len(asst) == 1
    # Should have thinking + text in content
    assert "<think>" in asst[0]["content"]
    assert "Let me check" in asst[0]["content"]
    assert len(asst[0]["tool_calls"]) == 1


def test_claude_format_for_training_embeds_tools(tmp_path):
    """End-to-end: parse → format_for_training → tools embedded in messages[0]."""
    from logminer.pipeline.cleanup import format_for_training

    base = make_claude_jsonl(tmp_path)
    parser = ClaudeParser()
    sessions = parser.discover_sessions(base)
    result = parser.parse_session(sessions[0])
    assert result is not None
    assert len(result["tools"]) >= 1

    formatted = format_for_training(result)
    messages = formatted["messages"]
    # Tools should be embedded in first message
    assert "tools" in messages[0]
    assert len(messages[0]["tools"]) >= 1
    # Tool call arguments should be dicts (not JSON strings)
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            args = tc.get("function", {}).get("arguments")
            if args is not None:
                assert isinstance(args, dict), f"arguments should be dict, got {type(args)}"


def test_claude_min_turns_filters(tmp_path):
    session_dir = tmp_path / "proj"
    session_dir.mkdir()
    session_file = session_dir / "short.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps(CLAUDE_ENTRY_USER) + "\n")
    parser = ClaudeParser()
    result = parser.parse_session(session_file, min_turns=5)
    assert result is None


# ---------------------------------------------------------------------------
# Qwen parser tests
# ---------------------------------------------------------------------------


def test_qwen_discover_sessions(tmp_path):
    base = make_qwen_sessions(tmp_path)
    parser = QwenParser()
    sessions = parser.discover_sessions(base)
    assert len(sessions) == 1


def test_qwen_parse_session_format(tmp_path):
    base = make_qwen_sessions(tmp_path)
    parser = QwenParser()
    sessions = parser.discover_sessions(base)
    result = parser.parse_session(sessions[0])
    assert result is not None
    assert result["source"] == "qwen"
    assert result["id"] == "sess1"
    assert "metadata" in result
    assert "tools" in result
    assert "messages" in result


def test_qwen_parse_session_messages(tmp_path):
    base = make_qwen_sessions(tmp_path)
    parser = QwenParser()
    sessions = parser.discover_sessions(base)
    result = parser.parse_session(sessions[0])
    assert result is not None
    roles = [m["role"] for m in result["messages"]]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


def test_qwen_parse_tool_call_format(tmp_path):
    base = make_qwen_sessions(tmp_path)
    parser = QwenParser()
    sessions = parser.discover_sessions(base)
    result = parser.parse_session(sessions[0])
    assert result is not None
    asst_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
    assert len(asst_msgs) >= 1
    assert "tool_calls" in asst_msgs[0]
    tc = asst_msgs[0]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "bash"


# ---------------------------------------------------------------------------
# OpenCode parser tests
# ---------------------------------------------------------------------------


def test_opencode_discover_sessions(tmp_path):
    base = make_opencode_db(tmp_path)
    parser = OpenCodeParser()
    sessions = parser.discover_sessions(base)
    assert len(sessions) == 1
    assert sessions[0].name == "opencode.db"


def test_opencode_parse_all(tmp_path):
    base = make_opencode_db(tmp_path)
    parser = OpenCodeParser()
    results = parser.parse_all(base)
    assert len(results) == 1
    result = results[0]
    assert result["source"] == "opencode"
    assert result["id"] == "sess-abc"
    assert "metadata" in result
    assert result["metadata"]["cwd"] == "/projects/myapp"
    assert result["metadata"]["model"] == "gpt-4o"
    assert "messages" in result
    assert "tools" in result


def test_opencode_parse_all_messages(tmp_path):
    base = make_opencode_db(tmp_path)
    parser = OpenCodeParser()
    results = parser.parse_all(base)
    assert results
    roles = [m["role"] for m in results[0]["messages"]]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


def test_opencode_empty_db(tmp_path):
    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE session (id TEXT, directory TEXT, time_created TEXT, time_updated TEXT)"
    )
    conn.execute("CREATE TABLE message (id TEXT, session_id TEXT, data TEXT, time_created TEXT)")
    conn.execute("CREATE TABLE part (message_id TEXT, data TEXT, time_created TEXT)")
    conn.commit()
    conn.close()
    parser = OpenCodeParser()
    results = parser.parse_all(tmp_path)
    assert results == []


def test_opencode_nonexistent_db(tmp_path):
    parser = OpenCodeParser()
    sessions = parser.discover_sessions(tmp_path)
    assert sessions == []
