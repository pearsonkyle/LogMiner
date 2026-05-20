import argparse
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")
LOGMINER_DATASET_TAGS = ("logminer", "coding-agent-logs")
LOGMINER_REPO_URL = "https://github.com/pearsonkyle/LogMiner"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    results.append(json.loads(line))
    return results


def _write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _ensure_jsonl(path: str) -> Path:
    """Append .jsonl if the user gave no suffix. Other suffixes are left alone."""
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(".jsonl")
        print(f"Note: appended .jsonl → {p}", file=sys.stderr)
    return p


def _get_huggingface_token() -> tuple[str | None, str | None]:
    for env_var in HF_TOKEN_ENV_VARS:
        token = os.environ.get(env_var)
        if token:
            return env_var, token
    return None, None


def _build_huggingface_dataset_card(path: Path, repo_id: str) -> str:
    tags = "\n".join(f"- {tag}" for tag in LOGMINER_DATASET_TAGS)
    return (
        "---\n"
        "tags:\n"
        f"{tags}\n"
        "---\n\n"
        f"# {repo_id}\n\n"
        f"This dataset was uploaded with [LogMiner]({LOGMINER_REPO_URL}).\n\n"
        "## Provenance\n\n"
        f"- Source repository: [{LOGMINER_REPO_URL}]({LOGMINER_REPO_URL})\n"
        f"- Uploaded file: `{path.name}`\n"
        "- Upload tool: `logminer --hf-repo`\n"
    )


def _upload_dataset_to_huggingface(path: Path, repo_id: str, private: bool = False) -> bool:
    token_env_var, token = _get_huggingface_token()
    if not token:
        env_vars = ", ".join(HF_TOKEN_ENV_VARS)
        print(
            f"Skipping Hugging Face upload for {repo_id}: no token found in {env_vars}.",
            file=sys.stderr,
        )
        return False

    try:
        hub_module = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face upload requires the optional dependency: pip install -e '.[huggingface]'"
        ) from exc

    api = hub_module.HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=_build_huggingface_dataset_card(path, repo_id).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card metadata from logminer",
    )
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {path.name} from logminer",
    )
    print(f"Uploaded {path.name} to https://huggingface.co/datasets/{repo_id} ({token_env_var})")
    return True


def cmd_parse(args: argparse.Namespace) -> None:
    from logminer.parsers import REGISTRY, get_parser

    sources = list(REGISTRY) if args.source == "all" else [args.source]
    output_path = _ensure_jsonl(args.output)
    all_results: list[dict[str, Any]] = []

    for source in sources:
        parser = get_parser(source)
        input_path = Path(args.input) if args.input else parser.DEFAULT_LOG_DIR
        print(f"Parsing {source} from {input_path} ...")
        results = parser.parse_all(input_path)
        print(f"  {len(results)} sessions parsed")
        all_results.extend(results)

    _write_jsonl(output_path, all_results)
    print(f"Written {len(all_results)} records to {output_path}")


def cmd_redact(args: argparse.Namespace) -> None:
    from logminer.redaction.anonymizer import Anonymizer
    from logminer.redaction.secrets import redact_text

    anon = Anonymizer()
    records = _load_jsonl(Path(args.input))
    total_redacted = 0

    def redact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal total_redacted
        cleaned = []
        for msg in messages:
            m = dict(msg)
            if m.get("content"):
                text, count = redact_text(anon.text(str(m["content"])))
                m["content"] = text
                total_redacted += count
            # Also redact tool call arguments (preserve JSON structure)
            if m.get("tool_calls"):
                new_calls = []
                for tc in m["tool_calls"]:
                    tc = dict(tc)
                    func = tc.get("function", {})
                    if func and func.get("arguments"):
                        args = func["arguments"]
                        # Serialize to JSON string for redaction, then parse back
                        args_json = json.dumps(args) if isinstance(args, dict) else str(args)
                        redacted_json, count = redact_text(anon.text(args_json))
                        total_redacted += count
                        # Try to recover dict; fall back to redacted string
                        try:
                            args_out = json.loads(redacted_json)
                        except (json.JSONDecodeError, TypeError):
                            args_out = redacted_json
                        tc["function"] = {**func, "arguments": args_out}
                    new_calls.append(tc)
                m["tool_calls"] = new_calls
            cleaned.append(m)
        return cleaned

    output = []
    for rec in records:
        r = dict(rec)
        if "messages" in r:
            r["messages"] = redact_messages(r["messages"])
        output.append(r)

    output_path = _ensure_jsonl(args.output)
    _write_jsonl(output_path, output)
    print(f"Redacted {total_redacted} secrets. Written to {output_path}")


def cmd_clean(args: argparse.Namespace) -> None:
    from logminer.pipeline.cleanup import clean_file

    clean_file(Path(args.input), _ensure_jsonl(args.output))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from logminer.pipeline.evaluate import evaluate_file

    evaluate_file(
        Path(args.input),
        _ensure_jsonl(args.output),
        min_turns=args.min_turns,
        min_token_count=args.min_token_count,
    )


def _to_arrow_safe(record: dict[str, Any]) -> dict[str, Any]:
    """Make a training record loadable by ``datasets.load_dataset("json", …)``.

    PyArrow infers a single schema across all rows in a JSONL and rejects the
    file when columns disagree on type. Two shapes in the post-``filter``
    output trip it up:

    1. ``messages[0]["tools"]`` carries an OpenAI tool schema whose
       ``parameters.properties`` struct varies session-to-session (different
       sessions used different tools), so Arrow can't unify it across rows.
    2. ``tool_calls[*].function.arguments`` is usually a dict but is left as
       a string when JSON parsing failed upstream — mixed dict/string in the
       same column is a hard Arrow error.

    The trainer (``LLMTrainer._prepare_messages_and_tools``) already handles
    ``messages`` as either ``list[dict]`` or ``list[str]`` (JSON-encoded),
    so the safe move is to serialize each message to a JSON string here.
    That collapses the column to ``list[string]`` — uniform across all rows
    — and the trainer parses it back transparently.

    The same heterogeneous-schema problem also hits the top-level ``tools``
    and ``metadata`` columns. Both are redundant on disk: the trainer does
    ``select_columns(["messages"])`` and recovers tools from ``messages[0]``
    (where ``format_for_training`` already embedded them). So we drop them
    here rather than serializing — keeping the file lean and the schema
    obviously uniform.
    """
    messages = record.get("messages", [])
    out = {k: v for k, v in record.items() if k not in ("tools", "metadata")}
    out["messages"] = [json.dumps(m) for m in messages]
    return out


def cmd_filter(args: argparse.Namespace) -> None:
    from logminer.pipeline.cleanup import format_for_training

    records = _load_jsonl(Path(args.input))
    filtered = [r for r in records if r.get("score", 0) >= args.min_score]
    formatted = [_to_arrow_safe(format_for_training(r)) for r in filtered]
    output_path = _ensure_jsonl(args.output)
    _write_jsonl(output_path, formatted)
    print(f"Filtered {len(records)} → {len(filtered)} records (min-score={args.min_score})")
    if args.hf_repo:
        _upload_dataset_to_huggingface(output_path, args.hf_repo, private=args.hf_private)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate parsed sessions against a Qwen chat template with tool support."""
    from transformers import AutoTokenizer

    from logminer.pipeline.cleanup import format_for_training

    # Load input: either pre-parsed JSONL or parse from source
    if args.input:
        records = _load_jsonl(Path(args.input))
        print(f"Loaded {len(records)} records from {args.input}")
    elif args.source:
        from logminer.parsers import REGISTRY, get_parser

        sources = list(REGISTRY) if args.source == "all" else [args.source]
        records = []
        for source in sources:
            parser = get_parser(source)
            input_path = parser.DEFAULT_LOG_DIR
            print(f"Parsing {source} from {input_path} ...")
            results = parser.parse_all(input_path)
            print(f"  {len(results)} sessions parsed")
            records.extend(results)
    else:
        print("Error: provide --input (parsed JSONL) or --source (parse live)")
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Validate each record
    passed = []
    failed = []
    total_tool_calls = 0
    sessions_with_tools = 0

    for rec in records:
        rec_id = rec.get("id", "unknown")
        formatted = format_for_training(rec)
        messages = formatted.get("messages", [])

        # Extract tools from first message (same as trainer pipeline)
        tools = None
        if messages and isinstance(messages[0], dict):
            tools = messages[0].pop("tools", None)

        # Count tool calls
        n_calls = sum(
            len(m.get("tool_calls", [])) for m in messages if m.get("role") == "assistant"
        )
        total_tool_calls += n_calls
        if n_calls > 0:
            sessions_with_tools += 1

        # Parse arguments from JSON strings to dicts (format_for_training does this)
        for m in messages:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    args_val = func.get("arguments")
                    if isinstance(args_val, str):
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            func["arguments"] = json.loads(args_val)

        # Try apply_chat_template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=False,
            )
            n_tokens = len(tokenizer.encode(text))
            passed.append(
                {
                    **formatted,
                    "messages": messages,
                    "_validation": {
                        "tokens": n_tokens,
                        "tool_calls": n_calls,
                        "has_tools": tools is not None,
                    },
                }
            )
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            failed.append({"id": rec_id, "error": err_msg, "n_messages": len(messages)})
            print(f"  FAIL [{rec_id}]: {err_msg}")

    # Report
    print(f"\n{'=' * 55}")
    print("  Validation Results")
    print(f"{'=' * 55}")
    print(f"  Total sessions:     {len(records)}")
    print(f"  Passed:             {len(passed)}")
    print(f"  Failed:             {len(failed)}")
    print(f"  Sessions with tools:{sessions_with_tools}")
    print(f"  Total tool calls:   {total_tool_calls}")

    if passed:
        token_counts = [r["_validation"]["tokens"] for r in passed]
        print(f"  Token range:        {min(token_counts)} - {max(token_counts)}")
        print(f"  Avg tokens:         {sum(token_counts) / len(token_counts):.0f}")

    if failed:
        print("\n  Failures:")
        for f in failed[:10]:
            print(f"    [{f['id']}] {f['error']}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")
    print(f"{'=' * 55}")

    # Write output
    if args.output:
        output_path = _ensure_jsonl(args.output)
        # Strip _validation metadata before writing
        output_records = []
        for r in passed:
            r.pop("_validation", None)
            output_records.append(r)
        _write_jsonl(output_path, output_records)
        print(f"Written {len(output_records)} validated records to {output_path}")


def cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline: parse → redact → clean → evaluate → filter."""
    final_output = _ensure_jsonl(args.output)
    base = final_output.stem
    parent = final_output.parent

    raw = parent / f"{base}_raw.jsonl"
    redacted = parent / f"{base}_redacted.jsonl"
    cleaned = parent / f"{base}_cleaned.jsonl"
    scored = parent / f"{base}_scored.jsonl"

    parser = build_parser()
    parse_argv = ["parse", "--source", args.source, "--output", str(raw)]
    if args.input:
        parse_argv += ["--input", args.input]
    cmd_parse(parser.parse_args(parse_argv))
    cmd_redact(parser.parse_args(["redact", "--input", str(raw), "--output", str(redacted)]))
    cmd_clean(parser.parse_args(["clean", "--input", str(redacted), "--output", str(cleaned)]))
    cmd_evaluate(parser.parse_args(["evaluate", "--input", str(cleaned), "--output", str(scored)]))
    cmd_filter(
        parser.parse_args(
            [
                "filter",
                "--input",
                str(scored),
                "--output",
                str(final_output),
                "--min-score",
                str(args.min_score),
                *(["--hf-repo", args.hf_repo] if args.hf_repo else []),
                *(["--hf-private"] if args.hf_private else []),
            ]
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logminer",
        description="logminer — extract LLM training logs",
    )
    sub = parser.add_subparsers(dest="command")

    # parse
    p_parse = sub.add_parser("parse", help="Parse provider logs into JSONL")
    p_parse.add_argument(
        "--source",
        required=True,
        help="Provider name (claude, opencode, qwen, all)",
    )
    p_parse.add_argument("--input", default=None, help="Input path (default: provider default)")
    p_parse.add_argument("--output", required=True, help="Output JSONL file")

    # redact
    p_redact = sub.add_parser("redact", help="Redact secrets and anonymize paths")
    p_redact.add_argument("--input", required=True)
    p_redact.add_argument("--output", required=True)

    # clean
    p_clean = sub.add_parser("clean", help="Remove bad tool calls/results")
    p_clean.add_argument("--input", required=True)
    p_clean.add_argument("--output", required=True)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Score conversation quality")
    p_eval.add_argument("--input", required=True)
    p_eval.add_argument("--output", required=True)
    p_eval.add_argument("--min-turns", type=int, default=2)
    p_eval.add_argument("--min-token-count", type=int, default=1000)

    # filter
    p_filter = sub.add_parser("filter", help="Filter by minimum score")
    p_filter.add_argument("--input", required=True)
    p_filter.add_argument("--output", required=True)
    p_filter.add_argument("--min-score", type=float, default=0.5)
    p_filter.add_argument(
        "--hf-repo",
        default=None,
        help=(
            "Optional Hugging Face dataset repo (owner/name) to upload the output to "
            "when HF_TOKEN or HUGGINGFACE_HUB_TOKEN is set"
        ),
    )
    p_filter.add_argument(
        "--hf-private",
        action="store_true",
        help="Create the Hugging Face dataset repo as private when uploading",
    )

    # validate
    p_val = sub.add_parser("validate", help="Validate parsed data against chat template")
    p_val.add_argument("--input", default=None, help="Pre-parsed JSONL file")
    p_val.add_argument("--source", default=None, help="Parse from source (claude, opencode, qwen)")
    p_val.add_argument("--output", default=None, help="Output validated JSONL file")
    p_val.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Tokenizer model")

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Run full pipeline end-to-end")
    p_run.add_argument("--source", required=True)
    p_run.add_argument("--input", default=None)
    p_run.add_argument("--output", required=True)
    p_run.add_argument("--min-score", type=float, default=0.5)
    p_run.add_argument(
        "--hf-repo",
        default=None,
        help="Optional Hugging Face dataset repo (owner/name) to upload the final output to",
    )
    p_run.add_argument(
        "--hf-private",
        action="store_true",
        help="Create the Hugging Face dataset repo as private when uploading",
    )

    return parser


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:] or [
        "run",
        "--source",
        "all",
        "--output",
        "logminer.jsonl",
        "--min-score",
        "0.5",
    ]
    args = parser.parse_args(argv)

    dispatch = {
        "parse": cmd_parse,
        "redact": cmd_redact,
        "clean": cmd_clean,
        "evaluate": cmd_evaluate,
        "filter": cmd_filter,
        "validate": cmd_validate,
        "run": cmd_run,
    }
    dispatch[args.command](args)
