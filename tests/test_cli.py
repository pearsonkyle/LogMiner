import argparse
import json
import types

from logminer import cli


def test_filter_parser_accepts_hf_repo():
    args = cli.build_parser().parse_args(
        ["filter", "--input", "in.jsonl", "--output", "out.jsonl", "--hf-repo", "me/data"]
    )
    assert args.hf_repo == "me/data"


def test_run_parser_accepts_hf_repo():
    args = cli.build_parser().parse_args(
        ["run", "--source", "claude", "--output", "out.jsonl", "--hf-repo", "me/data"]
    )
    assert args.hf_repo == "me/data"


def test_filter_parser_accepts_hf_private():
    args = cli.build_parser().parse_args(
        ["filter", "--input", "in.jsonl", "--output", "out.jsonl", "--hf-private"]
    )
    assert args.hf_private is True


def test_run_parser_accepts_hf_private():
    args = cli.build_parser().parse_args(
        ["run", "--source", "claude", "--output", "out.jsonl", "--hf-private"]
    )
    assert args.hf_private is True


def test_upload_dataset_fails_without_token(tmp_path, monkeypatch):
    """An explicit --hf-repo with no token must not exit 0. A soft skip let
    scheduled jobs report green while never publishing anything."""
    import pytest

    for env_var in cli.HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(SystemExit) as exc:
        cli._upload_dataset_to_huggingface(tmp_path / "out.jsonl", "me/data")

    assert "no token found" in str(exc.value)


def _fake_hub(calls, readme_exists=False):
    """A stand-in HfApi that records every call it receives."""

    class FakeHfApi:
        def __init__(self, token: str):
            calls.append(("init", {"token": token}))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def update_repo_settings(self, **kwargs):
            calls.append(("update_repo_settings", kwargs))

        def file_exists(self, **kwargs):
            calls.append(("file_exists", kwargs))
            return readme_exists

        def upload_file(self, **kwargs):
            calls.append(("upload_file", kwargs))

    return FakeHfApi


def _patch_hub(monkeypatch, api_cls):
    """Scope the importlib patch to huggingface_hub only.

    `cli.importlib` is the shared importlib module object, so replacing its
    import_module swaps it out process-wide — any unrelated dynamic import
    during the test would otherwise receive the fake.
    """
    real_import_module = cli.importlib.import_module

    def fake_import_module(name):
        if name == "huggingface_hub":
            return types.SimpleNamespace(HfApi=api_cls)
        return real_import_module(name)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)


def test_upload_dataset_uses_huggingface_hub(tmp_path, monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []
    FakeHfApi = _fake_hub(calls)

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    _patch_hub(monkeypatch, FakeHfApi)

    path = tmp_path / "training.jsonl"
    path.write_text("{}\n")

    uploaded = cli._upload_dataset_to_huggingface(path, "me/data", private=True)

    assert uploaded is True
    assert calls == [
        ("init", {"token": "hf_test_token"}),
        (
            "create_repo",
            {"repo_id": "me/data", "repo_type": "dataset", "private": True, "exist_ok": True},
        ),
        # create_repo(private=…) is creation-time only, so an existing public
        # repo would stay public without this.
        (
            "update_repo_settings",
            {"repo_id": "me/data", "repo_type": "dataset", "private": True},
        ),
        ("file_exists", {"repo_id": "me/data", "filename": "README.md", "repo_type": "dataset"}),
        (
            "upload_file",
            {
                "path_or_fileobj": cli._build_huggingface_dataset_card(path, "me/data").encode(),
                "path_in_repo": "README.md",
                "repo_id": "me/data",
                "repo_type": "dataset",
                "commit_message": "Add dataset card metadata from logminer",
            },
        ),
        (
            "upload_file",
            {
                "path_or_fileobj": str(path),
                "path_in_repo": cli.HF_DATA_PATH,
                "repo_id": "me/data",
                "repo_type": "dataset",
                "commit_message": "Upload training.jsonl from logminer",
            },
        ),
    ]


def test_upload_does_not_overwrite_existing_dataset_card(tmp_path, monkeypatch):
    """A hand-written card on the Hub must survive re-uploads."""
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    _patch_hub(monkeypatch, _fake_hub(calls, readme_exists=True))

    path = tmp_path / "training.jsonl"
    path.write_text("{}\n")
    cli._upload_dataset_to_huggingface(path, "me/data")

    uploads = [kw["path_in_repo"] for name, kw in calls if name == "upload_file"]
    assert uploads == [cli.HF_DATA_PATH]


def test_upload_skips_privacy_update_when_public(tmp_path, monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    _patch_hub(monkeypatch, _fake_hub(calls))

    path = tmp_path / "training.jsonl"
    path.write_text("{}\n")
    cli._upload_dataset_to_huggingface(path, "me/data", private=False)

    assert not any(name == "update_repo_settings" for name, _ in calls)


def test_upload_uses_fixed_repo_path_regardless_of_local_name(tmp_path, monkeypatch):
    """Uploading under the local filename let successive runs accumulate
    separate files that load_dataset() globs into one duplicated split."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    seen = []
    for local_name in ("claude.jsonl", "all.jsonl"):
        calls: list[tuple[str, dict[str, object]]] = []
        _patch_hub(monkeypatch, _fake_hub(calls, readme_exists=True))
        path = tmp_path / local_name
        path.write_text("{}\n")
        cli._upload_dataset_to_huggingface(path, "me/data")
        seen += [kw["path_in_repo"] for name, kw in calls if name == "upload_file"]

    assert seen == [cli.HF_DATA_PATH, cli.HF_DATA_PATH]


def test_build_huggingface_dataset_card_includes_search_tag_and_repo_link(tmp_path):
    card = cli._build_huggingface_dataset_card(tmp_path / "training.jsonl", "me/data")

    assert "tags:" in card
    assert "- logminer" in card
    assert "- coding-agent-logs" in card
    assert cli.LOGMINER_REPO_URL in card
    assert "# me/data" in card


def test_cmd_filter_uploads_when_hf_repo_is_set(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "training"
    input_path.write_text(
        json.dumps(
            {
                "id": "conv-1",
                "source": "claude",
                "score": 0.9,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        )
        + "\n"
    )
    uploads: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(
        cli,
        "_upload_dataset_to_huggingface",
        lambda path, repo_id, private=False: uploads.append((str(path), repo_id, private)) or True,
    )

    cli.cmd_filter(
        argparse.Namespace(
            input=str(input_path),
            output=str(output_path),
            min_score=0.5,
            max_tokens=131072,
            keep_system_boilerplate=True,
            hf_repo="me/data",
            hf_private=True,
        )
    )

    assert uploads == [(str(output_path.with_suffix(".jsonl")), "me/data", True)]


def test_cmd_run_passes_hf_repo_to_filter(tmp_path, monkeypatch):
    seen: list[tuple[str | None, bool]] = []
    monkeypatch.setattr(cli, "cmd_parse", lambda args: None)
    monkeypatch.setattr(cli, "cmd_redact", lambda args: None)
    monkeypatch.setattr(cli, "cmd_clean", lambda args: None)
    monkeypatch.setattr(cli, "cmd_evaluate", lambda args: None)
    monkeypatch.setattr(
        cli,
        "cmd_filter",
        lambda args: seen.append((args.hf_repo, args.hf_private)),
    )

    cli.cmd_run(
        argparse.Namespace(
            source="claude",
            input=None,
            output=str(tmp_path / "training.jsonl"),
            min_score=0.5,
            max_tokens=131072,
            keep_system_boilerplate=True,
            hf_repo="me/data",
            hf_private=True,
        )
    )

    assert seen == [("me/data", True)]


def test_cmd_redact_redacts_top_level_tools(tmp_path):
    """`tools` is not inside `messages`, but `format_for_training` embeds it
    into `messages[0]` downstream — so it reaches the training file and has
    to be redacted too."""
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "redacted.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "conv-1",
                "source": "claude",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "mcp__deckdoctor__lookup",
                            "description": "reads /Users/alice/Programs/DeckDoctor",
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        + "\n"
    )

    cli.cmd_redact(argparse.Namespace(input=str(input_path), output=str(output_path)))

    written = output_path.read_text()
    assert "deckdoctor" not in written.lower()
    assert "/Users/alice" not in written
    assert isinstance(json.loads(written)["tools"], list)


def test_cmd_redact_anonymizes_mcp_tool_names(tmp_path):
    """MCP server names carry project identity on both the assistant's
    tool_call and the tool result that answers it."""
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "redacted.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "conv-1",
                "messages": [
                    {"role": "user", "content": "look in /Users/alice/Programs/DeckDoctor"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "mcp__deckdoctor__lookup",
                                    "arguments": {"q": "x"},
                                },
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "Bash", "arguments": {"command": "ls"}},
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "c1",
                        "name": "mcp__deckdoctor__lookup",
                        "content": "ok",
                    },
                ],
            }
        )
        + "\n"
    )

    cli.cmd_redact(argparse.Namespace(input=str(input_path), output=str(output_path)))

    written = output_path.read_text()
    assert "deckdoctor" not in written.lower()
    # Built-in tool names are the model's real API surface — leave them alone.
    assert '"name": "Bash"' in written
