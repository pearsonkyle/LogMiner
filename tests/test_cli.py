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


def test_upload_dataset_skips_without_token(tmp_path, capsys, monkeypatch):
    for env_var in cli.HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    uploaded = cli._upload_dataset_to_huggingface(tmp_path / "out.jsonl", "me/data")

    assert uploaded is False
    assert "Skipping Hugging Face upload" in capsys.readouterr().err


def test_upload_dataset_uses_huggingface_hub(tmp_path, monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeHfApi:
        def __init__(self, token: str):
            calls.append(("init", {"token": token}))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("upload_file", kwargs))

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(HfApi=FakeHfApi),
    )

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
                "path_in_repo": "training.jsonl",
                "repo_id": "me/data",
                "repo_type": "dataset",
                "commit_message": "Upload training.jsonl from logminer",
            },
        ),
    ]


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
            hf_repo="me/data",
            hf_private=True,
        )
    )

    assert seen == [("me/data", True)]
