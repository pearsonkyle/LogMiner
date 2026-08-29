"""Tests for pipeline modules (cleanup + evaluate)."""

import json

from logminer.pipeline.cleanup import (
    cap_conversation_length,
    clean_conversation,
    format_for_training,
    has_failed_command,
)
from logminer.pipeline.evaluate import evaluate_conversation, evaluate_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conversation(**kwargs):
    """Build a minimal conversation dict with sensible defaults."""
    base = {
        "id": "test-conv",
        "source": "claude",
        "metadata": {"model": "test-model"},
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "user", "content": "Hello, please help me with a task."},
            {"role": "assistant", "content": "Sure, I can help with that!"},
        ],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# evaluate_conversation preserves original fields
# ---------------------------------------------------------------------------


class TestEvaluatePreservesFields:
    def test_messages_preserved(self):
        conv = _make_conversation()
        result = evaluate_conversation(conv, min_turns=1, min_token_count=1)
        assert "messages" in result
        assert len(result["messages"]) == 2

    def test_source_preserved(self):
        conv = _make_conversation(source="opencode")
        result = evaluate_conversation(conv, min_turns=1, min_token_count=1)
        assert result["source"] == "opencode"

    def test_metadata_preserved(self):
        conv = _make_conversation()
        result = evaluate_conversation(conv, min_turns=1, min_token_count=1)
        assert result["metadata"] == {"model": "test-model"}

    def test_tools_preserved(self):
        conv = _make_conversation()
        result = evaluate_conversation(conv, min_turns=1, min_token_count=1)
        assert result["tools"] == conv["tools"]

    def test_score_and_label_present(self):
        conv = _make_conversation()
        result = evaluate_conversation(conv, min_turns=1, min_token_count=1)
        assert "score" in result
        assert "label" in result
        assert "metrics" in result


# ---------------------------------------------------------------------------
# evaluate_file preserves messages through file I/O
# ---------------------------------------------------------------------------


class TestEvaluateFilePreservesMessages:
    def test_messages_in_output_file(self, tmp_path):
        conv = _make_conversation()
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        input_path.write_text(json.dumps(conv) + "\n")

        evaluate_file(input_path, output_path, min_turns=1, min_token_count=1)

        with open(output_path) as f:
            result = json.loads(f.readline())
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert "score" in result


# ---------------------------------------------------------------------------
# clean_conversation preserves original fields
# ---------------------------------------------------------------------------


class TestCleanPreservesFields:
    def test_messages_and_metadata_preserved(self):
        conv = _make_conversation()
        cleaned, removed = clean_conversation(conv)
        assert "messages" in cleaned
        assert cleaned["source"] == "claude"
        assert cleaned["metadata"] == {"model": "test-model"}
        assert cleaned["tools"] == conv["tools"]


# ---------------------------------------------------------------------------
# End-to-end: clean → evaluate preserves messages
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    def test_messages_survive_clean_then_evaluate(self):
        conv = _make_conversation(
            messages=[
                {"role": "user", "content": "Fix the bug in main.py please."},
                {
                    "role": "assistant",
                    "content": "<think>I should read the file first.</think>",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "main.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "def main(): pass",
                },
                {
                    "role": "assistant",
                    "content": "The file looks fine, no bug found.",
                },
            ]
        )

        cleaned, _ = clean_conversation(conv)
        result = evaluate_conversation(cleaned, min_turns=2, min_token_count=1)

        assert "messages" in result
        assert len(result["messages"]) >= 3
        assert result["source"] == "claude"
        assert "score" in result
        assert result["label"] in ("good", "maybe", "bad")


# ---------------------------------------------------------------------------
# format_for_training
# ---------------------------------------------------------------------------


class TestFormatForTraining:
    def test_tools_embedded_in_first_message(self):
        """Top-level tools list must be copied into messages[0]['tools']."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ]
        conv = _make_conversation(tools=tools)
        result = format_for_training(conv)
        assert result["messages"][0].get("tools") == tools

    def test_tools_not_embedded_when_no_tools(self):
        """No crash when tools list is empty."""
        conv = _make_conversation(tools=[])
        result = format_for_training(conv)
        assert "tools" not in result["messages"][0]

    def test_arguments_converted_to_dicts(self):
        """tool_call arguments should be dicts, not JSON strings."""
        conv = _make_conversation(
            messages=[
                {"role": "user", "content": "help"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "main.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "def main(): pass",
                },
            ]
        )
        result = format_for_training(conv)
        tc = result["messages"][1]["tool_calls"][0]
        assert isinstance(tc["function"]["arguments"], dict)
        assert tc["function"]["arguments"] == {"path": "main.py"}

    def test_arguments_already_dict_unchanged(self):
        """If arguments is already a dict, leave it alone."""
        conv = _make_conversation(
            messages=[
                {"role": "user", "content": "help"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": {"cmd": "ls"},
                            },
                        }
                    ],
                },
            ]
        )
        result = format_for_training(conv)
        tc = result["messages"][1]["tool_calls"][0]
        assert tc["function"]["arguments"] == {"cmd": "ls"}

    def test_cleaned_stats_removed(self):
        """Internal _cleaned_stats field must be stripped."""
        conv = _make_conversation()
        conv["_cleaned_stats"] = {"removed_tool_results": 3}
        result = format_for_training(conv)
        assert "_cleaned_stats" not in result

    def test_original_record_not_mutated(self):
        """format_for_training must not modify the input dict."""
        conv = _make_conversation(
            tools=[{"type": "function", "function": {"name": "bash"}}],
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"cmd": "ls"}',
                            },
                        }
                    ],
                },
            ],
        )
        format_for_training(conv)
        # Original should still have no tools key in first message
        assert "tools" not in conv["messages"][0]
        # Original arguments should still be a string
        assert conv["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"cmd": "ls"}'

    def test_full_pipeline_produces_trainer_compatible_output(self):
        """End-to-end: clean → evaluate → format produces trainer-ready data."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Tool function: read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        conv = _make_conversation(
            tools=tools,
            messages=[
                {"role": "user", "content": "Fix the bug in main.py please."},
                {
                    "role": "assistant",
                    "content": "<think>I should read the file first.</think>",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "main.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "def main(): pass",
                },
                {
                    "role": "assistant",
                    "content": "The file looks fine.",
                },
            ],
        )

        cleaned, _ = clean_conversation(conv)
        scored = evaluate_conversation(cleaned, min_turns=2, min_token_count=1)
        result = format_for_training(scored)

        # Must have messages with tools embedded in first message
        assert "messages" in result
        assert result["messages"][0].get("tools") == tools
        # Tool call arguments should be dicts
        tc = result["messages"][1]["tool_calls"][0]
        assert isinstance(tc["function"]["arguments"], dict)
        # Should have score from evaluation
        assert "score" in result
        # Internal fields stripped
        assert "_cleaned_stats" not in result

    def test_real_qwen_output_schema_compatible(self):
        """Verify real Qwen pipeline output converts to trainer-compatible format.

        This mirrors the exact structure produced by `python -m logminer run --source qwen`.
        The trainer expects:
        - tools embedded in messages[0]["tools"]
        - tool_call arguments as dicts (not JSON strings)
        - tool response messages with tool_call_id and name
        """
        conv = {
            "id": "0d9d7f06-8894-429d-90d6-94f0ab217589",
            "source": "qwen",
            "metadata": {
                "model": None,
                "cwd": "/Users/user/projects/my-app",
                "start_time": "2026-03-20T03:25:42.050Z",
                "end_time": "2026-03-20T03:43:14.732Z",
            },
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Tool function: read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"absolute_path": {"type": "string"}},
                            "required": ["absolute_path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "description": "Tool function: edit",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                            },
                            "required": ["file_path", "old_string", "new_string"],
                        },
                    },
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": "Add error handling to the main function",
                },
                {
                    "role": "assistant",
                    "content": "<think>\nI need to read the file first.\n</think>\nLet me read the file.",  # noqa: E501
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"absolute_path": "/app/main.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "read_file",
                    "tool_call_id": "call_abc123",
                    "content": "def main():\n    data = load_data()\n    process(data)",
                },
                {
                    "role": "assistant",
                    "content": "I'll add try/except error handling.",
                    "tool_calls": [
                        {
                            "id": "call_def456",
                            "type": "function",
                            "function": {
                                "name": "edit",
                                "arguments": '{"file_path": "/app/main.py", "old_string": "def main():\\n    data = load_data()\\n    process(data)", "new_string": "def main():\\n    try:\\n        data = load_data()\\n        process(data)\\n    except Exception as e:\\n        print(f\\"Error: {e}\\")"}',  # noqa: E501
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "edit",
                    "tool_call_id": "call_def456",
                    "content": "File edited successfully.",
                },
                {
                    "role": "assistant",
                    "content": "Done! I've added error handling to the main function.",
                },
            ],
            "score": 0.85,
            "label": "good",
        }

        result = format_for_training(conv)

        # Tools embedded in first message for trainer discovery
        assert result["messages"][0]["tools"] == conv["tools"]
        assert len(result["messages"][0]["tools"]) == 2

        # All tool_call arguments are dicts, not strings
        for msg in result["messages"]:
            for tc in msg.get("tool_calls", []):
                args = tc["function"]["arguments"]
                assert isinstance(args, dict), f"arguments should be dict, got {type(args)}: {args}"

        # Tool response messages have required fields
        for msg in result["messages"]:
            if msg.get("role") == "tool":
                assert "tool_call_id" in msg
                assert "name" in msg
                assert "content" in msg

        # Scoring metadata still present
        assert result["score"] == 0.85
        assert result["label"] == "good"


# ---------------------------------------------------------------------------
# Score semantics — what the new scorer should and shouldn't reward
# ---------------------------------------------------------------------------


def _agentic_conversation(
    n_calls: int = 20,
    n_tools: int = 5,
    user_text_chars: int = 4000,
    include_edit_chain: bool = True,
    error_fraction: float = 0.0,
    silent_tool_calls: bool = False,
):
    """Build a synthetic agentic conversation for score-semantics tests.

    `n_tools` distinct tool names spread across `n_calls` calls. `read_file`
    and `write_file` are included when `include_edit_chain=True`. Each tool
    call has a matching tool result; a `error_fraction` of those results
    contain "Error:". When `silent_tool_calls=True`, assistant content is
    empty for tool-call turns (the Claude default).
    """
    base_pool = [
        "read_file",
        "write_file",
        "bash",
        "grep",
        "list_files",
        "edit",
        "search",
        "fetch_url",
        "create_file",
    ]
    pool: list[str] = []
    if include_edit_chain:
        pool.extend(["read_file", "write_file"])
    for name in base_pool:
        if name not in pool:
            pool.append(name)
        if len(pool) >= n_tools:
            break
    pool = pool[:n_tools]

    messages = [{"role": "user", "content": "u " * (user_text_chars // 2)}]
    n_errors = int(round(n_calls * error_fraction))
    for i in range(n_calls):
        name = pool[i % len(pool)]
        call_id = f"call_{i}"
        messages.append(
            {
                "role": "assistant",
                "content": "" if silent_tool_calls else f"Calling {name}.",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                ],
            }
        )
        result_text = "Error: boom" if i < n_errors else f"ok-{i}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_text,
            }
        )
    messages.append({"role": "assistant", "content": "Done."})
    return {"id": "synth", "source": "claude", "messages": messages}


class TestScoreSemantics:
    """Behavioral assertions for the agent-SFT scorer.

    These pin score *bands*, not exact magnitudes, so the weights inside
    evaluate_conversation can be retuned without churning the tests.
    """

    def test_trivial_session_scores_zero(self):
        """2-message hi/hello below min_token_count must score 0.0."""
        conv = {
            "id": "trivial",
            "source": "claude",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
        result = evaluate_conversation(conv)
        assert result["score"] == 0.0
        assert result["label"] == "bad"
        assert any("hard_floor" in r for r in result["reasons"])

    def test_chatter_capped_below_threshold(self):
        """Long no-tool conversation must cap below default 0.5 threshold."""
        long_text = "alpha beta " * 400  # ~2800 chars -> well above min_tokens
        conv = {
            "id": "chatter",
            "source": "claude",
            "messages": [
                {"role": "user", "content": long_text},
                {"role": "assistant", "content": long_text},
                {"role": "user", "content": long_text},
                {"role": "assistant", "content": long_text},
            ],
        }
        result = evaluate_conversation(conv)
        assert result["score"] <= 0.4
        assert any("no_tool_calls" in r for r in result["reasons"])

    def test_substantive_agentic_session_scores_high(self):
        """20 calls, 5 tools, read+write, plenty of tokens → score >= 0.7."""
        conv = _agentic_conversation(
            n_calls=20,
            n_tools=5,
            user_text_chars=4000,
            include_edit_chain=True,
            error_fraction=0.0,
        )
        result = evaluate_conversation(conv)
        assert result["score"] >= 0.7, (result["score"], result["reasons"])
        assert result["label"] == "good"

    def test_error_swamp_scores_low(self):
        """High error ratio must drive score below 0.5."""
        conv = _agentic_conversation(
            n_calls=10,
            n_tools=3,
            user_text_chars=4000,
            include_edit_chain=True,
            error_fraction=0.8,
        )
        result = evaluate_conversation(conv)
        assert result["score"] < 0.5, (result["score"], result["reasons"])
        assert any("error_ratio" in r for r in result["reasons"])

    def test_length_monotone_above_floor(self):
        """Above the hard floor, longer sessions should not score lower."""
        scores = []
        for chars in (2000, 8000, 16000, 32000):
            conv = _agentic_conversation(
                n_calls=10,
                n_tools=3,
                user_text_chars=chars,
                include_edit_chain=True,
                error_fraction=0.0,
            )
            scores.append(evaluate_conversation(conv)["score"])
        # Monotone non-decreasing
        assert all(scores[i] <= scores[i + 1] + 1e-9 for i in range(len(scores) - 1)), scores

    def test_trivial_token_count_hits_hard_floor(self):
        """Even with tool calls, below min_token_count → score 0.0."""
        conv = _agentic_conversation(
            n_calls=2,
            n_tools=1,
            user_text_chars=10,
            include_edit_chain=False,
            error_fraction=0.0,
        )
        result = evaluate_conversation(conv)  # default min_token_count=1000
        assert result["score"] == 0.0

    def test_diversity_monotone(self):
        """Holding tokens and calls fixed, more distinct tools → higher score."""
        scores = []
        for n_tools in (1, 3, 5):
            conv = _agentic_conversation(
                n_calls=15,
                n_tools=n_tools,
                user_text_chars=4000,
                include_edit_chain=(n_tools >= 2),
                error_fraction=0.0,
            )
            scores.append(evaluate_conversation(conv)["score"])
        assert scores[0] < scores[2], scores

    def test_silent_tool_calls_not_penalized(self):
        """Assistant content="" + tool_calls is the normal Claude shape.

        It must not be classified as 'empty' and must not push a long,
        tool-diverse session below the keep threshold.
        """
        conv = _agentic_conversation(
            n_calls=15,
            n_tools=4,
            user_text_chars=4000,
            include_edit_chain=True,
            error_fraction=0.0,
            silent_tool_calls=True,
        )
        result = evaluate_conversation(conv)
        assert result["score"] >= 0.5, (result["score"], result["reasons"])
        # The empty penalty should NOT appear in reasons.
        assert not any("empty_ratio" in r for r in result["reasons"])
        # And the metric should count 0 truly-empty assistants (because each
        # silent assistant has tool_calls attached).
        assert result["metrics"]["truly_empty_assistant"] == 0


# ---------------------------------------------------------------------------
# Cleaner boundaries — what the narrow has_failed_command should/shouldn't drop
# ---------------------------------------------------------------------------


class TestCleanerBoundaries:
    """Pin the narrow has_failed_command rules.

    The earlier broad regex (``error:``/``failed``/``permission denied``/
    ``not found:``) matched on file contents, OpenAPI enum values, and
    Claude's plan-approval system text — destroying ~45% of legitimate
    tool results on real Claude logs. The new function only fires on
    unambiguous *command*-failure signatures.
    """

    # --- True positives: real shell failures stay dropped ---

    def test_command_not_found_dropped(self):
        assert has_failed_command("bash: foobar: command not found") is True

    def test_no_such_file_dropped(self):
        assert has_failed_command("ls: /nope: No such file or directory") is True

    def test_nonzero_exit_code_dropped(self):
        assert has_failed_command("output\n\nExit Code: 127") is True

    def test_nonzero_exit_code_grep_kept(self):
        # grep returning 1 (no matches) is normal, not a failure.
        assert has_failed_command("Exit Code: 1") is True  # generic
        assert has_failed_command("grep foo bar.txt\nExit Code: 1") is False

    def test_zero_exit_code_kept(self):
        assert has_failed_command("ok\nExit Code: 0") is False

    # --- False-positive guards: things the OLD regex wrongly killed ---

    def test_file_contents_with_error_word_kept(self):
        """Reading a source file that mentions 'Error:' must not drop the result."""
        content = (
            "     1\tdef main():\n"
            "     2\t    raise RuntimeError('Error: bad input')\n"
            "     3\t    return 0\n"
        )
        assert has_failed_command(content) is False

    def test_traceback_kept(self):
        """A traceback in tool output is valuable training signal, not a drop reason."""
        content = (
            "Traceback (most recent call last):\n"
            '  File "main.py", line 3, in <module>\n'
            "    raise ValueError('boom')\n"
            "ValueError: boom"
        )
        assert has_failed_command(content) is False

    def test_openapi_enum_failed_kept(self):
        """OpenAPI schemas with enum value 'Failed' must not be dropped."""
        content = '{"status": {"type": "string", "enum": ["Pending", "Failed", "Done"]}}'
        assert has_failed_command(content) is False

    def test_plan_approval_text_kept(self):
        """Claude's plan-approval system message must survive cleaning."""
        content = (
            "User has approved your plan. You can now start coding. Start with the first todo."
        )
        assert has_failed_command(content) is False

    def test_permission_denied_word_kept(self):
        """The bare phrase 'Permission denied' (without exit code) is no longer
        sufficient to drop — it appears in docs, error messages being
        explained, and code samples. Only an actual nonzero exit code or
        command-not-found / no-such-file qualifies."""
        assert has_failed_command("The article explains permission denied errors.") is False

    def test_word_failed_alone_kept(self):
        assert has_failed_command("All 17 tests passed; 0 failed.") is False

    # --- is_error=True is no longer a drop signal ---

    def test_is_error_true_result_kept(self):
        """clean_conversation must keep tool results flagged is_error=True
        as long as their content isn't an unambiguous shell failure. These
        are the failure-recovery trajectories most valuable for agent SFT."""
        conv = {
            "id": "errkeep",
            "source": "claude",
            "messages": [
                {"role": "user", "content": "fix it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "is_error": True,
                    "content": "File contents: raise RuntimeError('Error: bad')",
                },
                {"role": "assistant", "content": "I see the bug, let me fix it."},
            ],
        }
        cleaned, removed = clean_conversation(conv)
        assert removed == 0
        # The tool result must still be present.
        roles = [m["role"] for m in cleaned["messages"]]
        assert roles.count("tool") == 1

    def test_real_shell_failure_still_dropped_in_pipeline(self):
        """End-to-end: an actual ``Exit Code: 127`` tool result is removed
        and the corresponding orphaned tool_call is cleaned up too."""
        conv = {
            "id": "shellfail",
            "source": "claude",
            "messages": [
                {"role": "user", "content": "run it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "bash: nope: command not found\nExit Code: 127",
                },
            ],
        }
        cleaned, removed = clean_conversation(conv)
        assert removed == 1
        # Orphaned tool_call should be stripped from the assistant turn.
        for m in cleaned["messages"]:
            assert m["role"] != "tool"
            if m["role"] == "assistant":
                assert not m.get("tool_calls")


# ---------------------------------------------------------------------------
# Length capping
# ---------------------------------------------------------------------------


class TestCapConversationLength:
    """Only 37% of a real 522-session corpus fit a 32k context and the
    largest record was ~3.8M estimated tokens, so an uncapped dataset
    silently depends on the trainer's overflow behaviour. Truncation keeps
    a prefix (still a valid trajectory) rather than dropping the record.
    """

    def _long_conv(self, n_pairs=40):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        for i in range(n_pairs):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": f"c{i}", "function": {"name": "Bash", "arguments": "{}"}}
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 4000})
        return {"id": "long", "messages": messages}

    def test_short_conversation_untouched(self):
        conv = {"id": "s", "messages": [{"role": "user", "content": "hi"}]}
        out, truncated = cap_conversation_length(conv, max_tokens=131072)
        assert truncated is False
        assert out == conv

    def test_long_conversation_truncated(self):
        out, truncated = cap_conversation_length(self._long_conv(), max_tokens=5000)
        assert truncated is True
        assert len(out["messages"]) < len(self._long_conv()["messages"])

    def test_system_message_always_kept(self):
        out, _ = cap_conversation_length(self._long_conv(), max_tokens=10)
        assert out["messages"][0]["role"] == "system"

    def test_truncation_leaves_no_orphaned_tool_calls(self):
        """A cut mid-pair must not leave a tool_call without its result."""
        for budget in (500, 2000, 5000, 20000, 50000):
            out, _ = cap_conversation_length(self._long_conv(), max_tokens=budget)
            result_ids = {m.get("tool_call_id") for m in out["messages"] if m.get("role") == "tool"}
            for m in out["messages"]:
                for tc in m.get("tool_calls") or []:
                    assert tc["id"] in result_ids, f"orphaned call at budget={budget}"

    def test_truncation_does_not_end_on_tool_result(self):
        """Only applies when a cut was made — an untruncated record keeps
        whatever shape the session actually had."""
        for budget in (500, 2000, 5000, 20000):
            out, truncated = cap_conversation_length(self._long_conv(), max_tokens=budget)
            assert truncated is True
            if out["messages"]:
                assert out["messages"][-1]["role"] != "tool", f"ends on tool at budget={budget}"

    def test_output_respects_budget(self):
        from logminer.pipeline.evaluate import estimate_token_count

        budget = 20000
        out, _ = cap_conversation_length(self._long_conv(), max_tokens=budget)
        total = sum(estimate_token_count(m.get("content") or "") for m in out["messages"])
        assert total <= budget

    def test_original_record_not_mutated(self):
        conv = self._long_conv()
        before = len(conv["messages"])
        cap_conversation_length(conv, max_tokens=5000)
        assert len(conv["messages"]) == before
