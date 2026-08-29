"""Tests for the redaction module."""

from logminer.redaction.anonymizer import Anonymizer, harvest_identities
from logminer.redaction.secrets import redact_text, scan_text


def test_anthropic_api_key_detected():
    text = "key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "anthropic_api_key" in names


def test_openai_api_key_detected():
    text = "Authorization: sk-abcdefghijklmnopqrstuvwx"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "openai_api_key" in names


def test_github_token_detected():
    text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "github_token" in names


def test_jwt_token_detected():
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"  # noqa: E501
    findings = scan_text(jwt)
    names = [f.pattern_name for f in findings]
    assert "jwt_token" in names


def test_email_detected():
    text = "Contact alice@somecompany.io for details"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "email" in names


def test_email_allowlist_skipped():
    text = "noreply@github.com should not be flagged"
    findings = scan_text(text)
    email_findings = [f for f in findings if f.pattern_name == "email"]
    assert len(email_findings) == 0


def test_pem_private_key_detected():
    key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA1234567890abcdef\n"
        "-----END RSA PRIVATE KEY-----"
    )
    findings = scan_text(key)
    names = [f.pattern_name for f in findings]
    assert "private_key" in names


def test_redact_text_replaces_secrets():
    text = "My key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"
    redacted, count = redact_text(text)
    assert count >= 1
    assert "sk-ant-api03" not in redacted
    assert "[REDACTED_ANTHROPIC_API_KEY]" in redacted


def test_redact_text_no_secrets():
    text = "Hello world, this is a normal sentence."
    redacted, count = redact_text(text)
    assert count == 0
    assert redacted == text


def test_anonymizer_username_replacement(monkeypatch):
    monkeypatch.setenv("USER", "johndoe")
    anon = Anonymizer()
    result = anon.text("Hello from johndoe on this machine")
    assert "johndoe" not in result


def test_anonymizer_home_dir_replacement():
    home = str(__import__("pathlib").Path.home())
    anon = Anonymizer()
    result = anon.text(f"file is at {home}/config.yaml")
    assert home not in result


def test_anonymizer_short_username_not_replaced(monkeypatch):
    monkeypatch.setenv("USER", "ab")
    anon = Anonymizer()
    result = anon.text("user ab logged in")
    assert result == "user ab logged in"


def test_anonymizer_extra_usernames():
    anon = Anonymizer(extra_usernames=["secretuser"])
    result = anon.text("path /home/secretuser/data")
    assert "secretuser" not in result


# --- Per-session path variance -------------------------------------------


def test_different_sessions_get_different_homes():
    """The whole point of the rewrite: no single memorizable path literal."""
    text = "/Users/alice/Programs/Widget/main.py"
    outs = {Anonymizer(session_key=f"session-{i}").text(text) for i in range(25)}
    assert len(outs) > 15, f"expected varied homes across sessions, got {len(outs)}"


def test_same_session_is_internally_consistent():
    """A trajectory that cds somewhere and later reads under it must agree."""
    anon = Anonymizer(session_key="s1")
    a = anon.text("cd /Users/alice/Programs/Widget")
    b = anon.text("read /Users/alice/Programs/Widget/main.py")
    assert a.split("cd ")[1] in b


def test_anonymization_is_deterministic_for_a_key():
    text = "/home/bob/Code/Thing/x.py"
    assert Anonymizer(session_key="k").text(text) == Anonymizer(session_key="k").text(text)


def test_non_local_home_is_anonymized():
    """Exact-$HOME matching missed remote/homelab paths entirely."""
    result = Anonymizer(session_key="s").text("model_path: /home/pearsonkyle/llm/gguf/m.gguf")
    assert "pearsonkyle" not in result


def test_system_dirs_preserved():
    """Renaming ~/.claude/projects would make the transcript wrong, not private."""
    result = Anonymizer(session_key="s").text("/Users/alice/.claude/projects/foo")
    assert "/.claude/projects/" in result


def test_deep_source_paths_untouched():
    """Below three segments the path is source-tree structure, not identity.

    Three is tuned to the real `~/Programs/<Org>/<Repo>/<package>/...`
    layout: the container dir is preserved by name (`Programs` is a
    system dir), org and repo are renamed, and the package directory is
    left alone so it still matches the `from cardagent import ...` lines
    quoted in the same transcript.
    """
    result = Anonymizer(session_key="s").text(
        "/Users/alice/Programs/DeckDoctor/Card-Agent/cardagent/scripts/sim.py"
    )
    assert result.endswith("/cardagent/scripts/sim.py")
    assert "DeckDoctor" not in result
    assert "Card-Agent" not in result
    assert "/Programs/" in result


def test_project_name_replaced_outside_paths():
    anon = Anonymizer(session_key="s")
    anon.prescan("/Users/alice/Programs/DeckDoctor/x.py")
    result = anon.text("the DeckDoctor repo and mcp__deckdoctor__lookup_card")
    assert "eckdoctor" not in result.lower()


def test_project_name_maps_consistently_across_casing():
    """`DeckDoctor` in a path and `deckdoctor` in an MCP tool name are one project."""
    anon = Anonymizer(session_key="s")
    path_out = anon.text("/Users/alice/Programs/DeckDoctor")
    bare_out = anon.text("mcp__deckdoctor__lookup")
    fake = path_out.rsplit("/", 1)[1]
    assert fake.lower() in bare_out.lower()


def test_short_project_names_not_bare_replaced():
    """Substituting every occurrence of a 2-char dir name would shred the text."""
    anon = Anonymizer(session_key="s")
    anon.prescan("/Users/alice/ai/notes.md")
    assert "ai" in anon.text("this ai is a normal word in prose")


def test_huggingface_token_detected():
    text = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "huggingface_token" in names


def test_aws_access_key_detected():
    text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "aws_access_key" in names


def test_prefixed_hub_token_var_redacted():
    text = "HUGGING_FACE_HUB_TOKEN=zzkdmflerkjf83jrkemfkemr8348"
    redacted, count = redact_text(text)
    assert count == 1
    assert "zzkdmflerkjf83jrkemfkemr8348" not in redacted
    assert "HUGGING_FACE_HUB_TOKEN=" in redacted


def test_prefixed_api_key_var_redacted():
    text = "OPENAI_API_KEY=sk-proj-aaaaaaaaaaaaaaaaaaaabbbbbbbbb1234"
    redacted, count = redact_text(text)
    assert count == 1
    assert "sk-proj-aaaaaaaaaaaaaaaaaaaa" not in redacted
    assert "OPENAI_API_KEY=" in redacted


def test_client_secret_with_colon_separator():
    text = "AZURE_CLIENT_SECRET: aBcD1234EfGh5678IjKl9012MnOpQrSt"
    redacted, count = redact_text(text)
    assert count == 1
    assert "aBcD1234EfGh5678IjKl9012MnOpQrSt" not in redacted


def test_github_fine_grained_pat_detected():
    text = "GITHUB_PAT=github_pat_11ABCDEFG0abcdefghijklZyXwVuTsRqPoNmLkJiHgFeDcBa"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "github_fine_pat" in names


def test_google_oauth_client_secret_detected():
    text = "GOOGLE_CLIENT_SECRET=GOCSPX-aBcDeFgHiJkLmNoPqRsTuVwXyZ"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "google_oauth_client_secret" in names


def test_stripe_secret_key_detected():
    text = "STRIPE_SECRET=" + "sk_" + "live_" + "abcdefghijklmnopqrstuvwx"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "stripe_secret_key" in names


def test_json_style_api_key_redacted():
    text = '{"api_key": "sk-fakefakefakefakefakefake"}'
    redacted, count = redact_text(text)
    assert count >= 1
    assert "sk-fakefakefakefakefakefake" not in redacted


def test_password_with_special_chars_redacted():
    text = 'MY_DB_PASSWORD = "superSecret123!"'
    redacted, count = redact_text(text)
    assert count == 1
    assert "superSecret123!" not in redacted


def test_env_secret_value_only_replaced():
    """Var name should be preserved; only the value is redacted."""
    text = "GOOGLE_CLIENT_SECRET=GOCSPX-aBcDeFgHiJkLmNoPqRsTuVwXyZ"
    redacted, _ = redact_text(text)
    assert redacted.startswith("GOOGLE_CLIENT_SECRET=")


def test_low_entropy_value_not_flagged():
    text = "SOME_TOKEN=aaaaaaaa"
    redacted, count = redact_text(text)
    assert count == 0
    assert redacted == text


def test_normal_text_not_flagged():
    text = "the cat sat on the keyboard near the door"
    _, count = redact_text(text)
    assert count == 0


# --- Crypto wallet keys ---------------------------------------------------


def test_ethereum_private_key_detected():
    text = "PK=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    redacted, count = redact_text(text)
    assert count >= 1
    assert "ac0974bec39a17e36ba4a6b4d238ff94" not in redacted


def test_bitcoin_wif_compressed_detected():
    text = "wif: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "bitcoin_wif" in names


def test_bitcoin_wif_uncompressed_detected():
    text = "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "bitcoin_wif" in names


def test_bip39_mnemonic_12_words_detected():
    text = (
        "seed: abandon abandon abandon abandon abandon abandon "
        "abandon abandon abandon abandon abandon about"
    )
    redacted, count = redact_text(text)
    assert count == 1
    assert "[REDACTED_BIP39_MNEMONIC]" in redacted
    assert "abandon abandon" not in redacted


def test_bip39_mnemonic_24_words_detected():
    text = (
        "legal winner thank year wave sausage worth useful legal winner "
        "thank year wave sausage worth useful legal winner thank year "
        "wave sausage worth title"
    )
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "bip39_mnemonic" in names


def test_bip39_wrong_length_not_flagged():
    # 13 words — not a valid mnemonic length.
    text = " ".join(["abandon"] * 13)
    findings = scan_text(text)
    assert all(f.pattern_name != "bip39_mnemonic" for f in findings)


def test_bip39_non_wordlist_prose_not_flagged():
    # 12 lowercase short words but not all in the BIP-39 wordlist.
    text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
    findings = scan_text(text)
    assert all(f.pattern_name != "bip39_mnemonic" for f in findings)


# --- PEM / PGP private keys ----------------------------------------------


def test_pgp_private_key_block_detected():
    text = (
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\n"
        "lQHYBGabcdEF...\n"
        "-----END PGP PRIVATE KEY BLOCK-----"
    )
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "pgp_private_key" in names


def test_encrypted_pem_private_key_detected():
    text = "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIE...\n-----END ENCRYPTED PRIVATE KEY-----"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "private_key" in names


# --- Service tokens -------------------------------------------------------


def test_pypi_token_detected():
    text = "pypi-AgEIcHlwaS5vcmcCJDEyMzQ1Njc4LWFiY2QtZWZnaC1pamtsLW1ub3BxcnN0dXZ3eA"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "pypi_token" in names


def test_sendgrid_api_key_detected():
    text = "SG.abcdefghijklmnopqrstuv.abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "sendgrid_api_key" in names


def test_gitlab_pat_detected():
    text = "GL=glpat-abcdefghij1234567890"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "gitlab_pat" in names


# --- HTTP auth / SSH ------------------------------------------------------


def test_basic_auth_header_detected():
    text = "Authorization: Basic dXNlcjpwYXNzd29yZA=="
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "basic_auth" in names


def test_ssh_public_key_detected():
    text = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBabcdefghijklmnopqrstuvwxyz user@host"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "ssh_public_key" in names


# --- Credit cards (Luhn) --------------------------------------------------


def test_valid_credit_card_redacted():
    text = "card: 4532015112830366"  # passes Luhn
    redacted, count = redact_text(text)
    assert count == 1
    assert "4532015112830366" not in redacted


def test_invalid_credit_card_not_flagged():
    text = "card: 4532015112830367"  # one digit off, fails Luhn
    _, count = redact_text(text)
    assert count == 0


def test_long_digit_id_not_flagged_as_card():
    text = "transaction id: 123456789012345"
    findings = scan_text(text)
    assert all(f.pattern_name != "credit_card" for f in findings)


def test_separated_card_redacted_without_keyword():
    """4-4-4-4 grouping is corroboration on its own."""
    redacted, count = redact_text("4532 0151 1283 0366")
    assert count == 1
    assert "4532" not in redacted


# Regression: on a 522-session corpus every one of the 194 `credit_card`
# hits was one of these two shapes, and the replacement landed *inside* a
# number (`0.4848…` → `0.[REDACTED_CREDIT_CARD]`).


def test_float_digits_not_flagged_as_card():
    text = "{'recall@10': 0.691025641025641, 'mrr@10': 0.48486721611721617}"
    redacted, count = redact_text(text)
    assert count == 0
    assert redacted == text


def test_unix_ms_timestamp_not_flagged_as_card():
    text = '"startedAt": 1787585109642, "queuedAt": 1787585109631, "attempt": 1'
    _, count = redact_text(text)
    assert count == 0


def test_sha256_with_0x_prefix_not_flagged_as_eth_key():
    text = "commit hash 0x" + "ab12cd34" * 8 + " verified"
    findings = scan_text(text)
    assert all(f.pattern_name != "ethereum_private_key" for f in findings)


def test_eth_key_with_wallet_context_still_flagged():
    text = "wallet private key: 0x" + "ac0974be" * 8
    findings = scan_text(text)
    assert "ethereum_private_key" in [f.pattern_name for f in findings]


# --- Corpus-wide identity harvesting --------------------------------------


def test_harvest_identities_from_path():
    users, projects = harvest_identities("/Users/alice/Programs/DeckDoctor/main.py")
    assert "alice" in users
    assert "deckdoctor" in projects
    assert "programs" not in projects  # generic container dir


def test_harvest_identities_from_mcp_tool_name():
    """The project leaks via MCP server names in sessions with no path to it."""
    _, projects = harvest_identities('{"name": "mcp__deckdoctor__lookup_card"}')
    assert "deckdoctor" in projects


def test_mcp_server_name_anonymized():
    anon = Anonymizer(session_key="s", extra_projects=["deckdoctor"])
    result = anon.text('{"name": "mcp__deckdoctor__lookup_card"}')
    assert "deckdoctor" not in result
    assert result.startswith('{"name": "mcp__')


def test_extra_projects_substituted_without_local_evidence():
    """A session that only mentions the project in prose still gets it."""
    anon = Anonymizer(session_key="s", extra_projects=["deckdoctor"])
    assert "DeckDoctor" not in anon.text("bring this engine into DeckDoctor as a mode")


def test_generic_account_names_not_bare_replaced():
    """`/home/user` is a real home dir but "user" is not an identity."""
    anon = Anonymizer(session_key="s", extra_usernames=["user", "claude", "runner"])
    text = "the user asked claude to run it on a runner"
    assert anon.text(text) == text


def test_generic_account_home_still_path_anonymized():
    result = Anonymizer(session_key="s").text("/home/user/Card-Agent/README.md")
    assert "/home/user/" not in result


def test_name_after_literal_escape_sequence_anonymized():
    """Agent logs embed JSON in JSON, so a name can follow a literal `\\n`
    (backslash + `n`) rather than a real newline. A plain word-boundary
    lookbehind sees the `n` and skips the match."""
    anon = Anonymizer(session_key="s", extra_projects=["card-agent"])
    for prefix in ("\\n", "\\t", "\\r", "\n", " ", '"'):
        assert "Card-Agent" not in anon.text(f"into{prefix}Card-Agent as a mode")


def test_escape_boundary_does_not_match_mid_word():
    anon = Anonymizer(session_key="s", extra_projects=["widget"])
    assert anon.text("a superwidget here") == "a superwidget here"


def test_long_username_glued_to_preceding_letter_still_replaced():
    """Terminal logs mangle prompts: a real qwen session contained
    `(llm) Akpearson:~/…` where a carriage-return overwrite glued the
    username to a preceding letter, and a left boundary let it leak."""
    anon = Anonymizer(session_key="s", extra_usernames=["kpearson"])
    assert "kpearson" not in anon.text("(llm) Akpearson:~/Programs/ai$ ls").lower()


def test_short_username_keeps_strict_boundary():
    """Below 6 chars, substring matching would corrupt ordinary words."""
    anon = Anonymizer(session_key="s", extra_usernames=["mark"])
    assert anon.text("add a bookmark here") == "add a bookmark here"


def test_username_prefix_of_longer_name_not_replaced():
    """The right boundary still holds: `kpearsons` is a different name."""
    anon = Anonymizer(session_key="s", extra_usernames=["kpearson"])
    assert "kpearsons" in anon.text("the kpearsons account")
