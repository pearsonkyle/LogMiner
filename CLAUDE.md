# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .[test]                     # install with pytest
pip install -e .[validate]                 # install with transformers (only `validate` needs it)
pytest                                     # full suite
pytest tests/test_parsers.py -k claude     # filter to one parser/test
ruff check .                               # lint (CI also runs `ruff format --check`)
ruff format .                              # apply formatting
pre-commit run --all-files                 # run all hooks locally
```

CLI entry point is `python -m logminer <command>` (or the `logminer` script). Subcommands: `parse`, `redact`, `clean`, `evaluate`, `filter`, `validate`, `run`. CI matrix runs against Python 3.10/3.11/3.12.

## Architecture

The pipeline is a chain of JSONL → JSONL transforms. Each stage has its own subcommand and writes a file; `run` chains them all and keeps every intermediate (`*_raw.jsonl`, `*_redacted.jsonl`, `*_cleaned.jsonl`, `*_scored.jsonl`) so stages can be diffed or resumed independently.

```
parse → redact → clean → evaluate → filter
```

Stages are idempotent and read only the previous stage's file, so a halted `run` can be resumed by invoking the next subcommand directly with `--input <partial file>`.

**Core dependency rule:** the parse/redact/clean/evaluate/filter stages must remain stdlib-only (so the CLI runs in network-less sandboxes). Only the `validate` subcommand may import `transformers`. `pyproject.toml` reflects this: `dependencies = []`, with `transformers` behind the `[validate]` extra.

### Layout

- `logminer/cli.py` — argparse dispatcher for all subcommands.
- `logminer/parsers/` — pluggable per-agent parsers. `base.py` defines the `BaseParser` ABC and `build_tool_schema()` helper. `__init__.py` exposes a `REGISTRY` dict mapping `--source` names → parser classes. Adding a new agent is one new file plus one line in `REGISTRY`.
- `logminer/pipeline/cleanup.py` — `clean_conversation` (drops malformed tool calls, orphaned tool results, empty turns) and `format_for_training` (the training-format writer). Keep cleaning and formatting separate.
- `logminer/pipeline/evaluate.py` — quality scoring. New signals should fold into the existing `score` float in `[0, 1]`; `filter` only knows about `score`.
- `cap_conversation_length` (in `pipeline/cleanup.py`, applied by `filter`) truncates to `--max-tokens` (default 131072) by keeping a *prefix* — a prefix of an agent trajectory is still a valid trajectory, whereas dropping loses the long multi-step sessions that are the best signal. It re-runs orphan removal after the cut so truncation can never leave a `tool_calls` entry without its result. Token counts use the scorer's `estimate_token_count`, which over-estimates vs. real BPE, so the cap is conservative.
- `logminer/redaction/secrets.py` — regex-based scrubbing. `redact_text` returns `(redacted, count)`; preserve that signature so CLI reporting stays correct. Shape-only patterns that also match ordinary developer output (`credit_card`, `ethereum_private_key`) are gated on nearby keywords via `_keep_match` — a Luhn checksum alone passes ~10% of arbitrary digit runs, and the bare rule redacted *inside* floats and Unix timestamps.
- `logminer/redaction/anonymizer.py` — path/username/project anonymization, backed by `bip39_wordlist.txt`.

### Anonymizer contract

Each session gets its own persona keyed on `session_key` (pass the record `id`). Two invariants, in tension:

- **Across sessions** personas differ, so no path literal dominates the corpus. The previous exact-`$HOME` → `/home/REDACTED_USER` scheme produced *one* string 20,371 times across 8 distinct prefixes; a model trained on that memorizes it as *the* home directory. Now: 548 prefixes, top one at 1.8%.
- **Within a session** the mapping is a pure function of `(session_key, original)`, so a trajectory that `cd`s somewhere and later reads under it stays coherent. Never randomize per occurrence.

`cmd_redact` runs `harvest_identities` over the **whole corpus** before anonymizing any record, because evidence that a name is identifying is unevenly spread — a username appears as `/home/<name>/…` in a few sessions and only as `github.com/<name>/…` in the rest.

Identity reaches the training file through more than message content. All of these must stay redacted: `content`, `tool_calls[].function.{name,arguments}`, `name` on tool results, and the **top-level `tools`** (not part of `messages`, but `format_for_training` embeds it into `messages[0]`). Built-in tool names (`Bash`, `Read`) are deliberately left alone — they are the model's real API surface.

### Parser contract

Every parser subclasses `BaseParser`, sets `SOURCE` (CLI name) and `DEFAULT_LOG_DIR`, and implements `discover_sessions()` and `parse_session()`. `parse_session` returns either a dict shaped like the record schema below, or `None` to silently drop. The pipeline stages are agent-agnostic; faithful normalization in the parser is the only contract.

Record shape after `parse`:
```json
{"id": "...", "source": "claude",
 "messages": [{"role": "user|assistant|tool", "content": "...", "tool_calls": [...], "tool_call_id": "..."}]}
```

Practical parser invariants worth preserving:
- Sort turns by timestamp at the top of `parse_session` — most agent logs interleave out of order.
- Assistant `tool_calls` must pair with `role: "tool"` messages sharing the same `tool_call_id`. `clean` drops orphans, but parsers should never emit them.
- Return nothing for empty/telemetry/system-housekeeping turns rather than emitting blank strings.

### Training-format writer (subtle but load-bearing)

`format_for_training` (in `pipeline/cleanup.py`) makes two non-obvious choices so the output loads via vanilla `datasets.load_dataset("json", …)`:

1. `messages` is serialized to `list[str]` (each turn JSON-encoded). PyArrow can't unify heterogeneous `list[dict]` schemas across sessions; a uniform string column can.
2. Top-level `tools` and `metadata` are dropped on write. Tools are re-embedded inside `messages[0]["tools"]`. Don't reintroduce top-level `tools`/`metadata` without solving the Arrow schema-unification problem.

## Extension points (in order of frequency)

1. **New agent** — add `logminer/parsers/<name>.py` + entry in `parsers/__init__.py:REGISTRY` + a happy-path test in `tests/test_parsers.py`. No pipeline changes needed.
2. **New redaction rule** — add a regex in `redaction/secrets.py`. Rerun just `redact` against an existing `_raw.jsonl` to validate without re-parsing.
3. **New scoring signal** — edit `pipeline/evaluate.py`; fold into `score`, do not introduce a parallel field.
4. **New cleaning rule** — edit `clean_conversation` in `pipeline/cleanup.py`; keep `format_for_training` untouched.

## Output-path convention

`--output` paths without a suffix are auto-completed to `.jsonl` (a one-line `Note:` is printed to stderr). Existing suffixes are left alone. Parent directories are created automatically.
