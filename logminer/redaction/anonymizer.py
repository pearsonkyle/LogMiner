"""Identity anonymization for paths, usernames, and project names.

The previous implementation replaced ``$HOME`` with the single literal
``/home/REDACTED_USER`` and ``$USER`` with one deterministic hash.
Measured on a 522-session Claude corpus that produced **20,371
occurrences of one constant string** and 8 distinct path prefixes in the
entire dataset. A model fine-tuned on that memorizes
``/home/REDACTED_USER`` as *the* home directory instead of learning that
home directories vary — the anonymization succeeded at hiding the user
and failed at producing usable training data.

This version gives each session its own persona — root, username, and
project directory names — drawn deterministically from the vendored
BIP-39 wordlist and keyed on the session id. Two properties matter, and
they pull in opposite directions:

- **Across sessions** the personas differ, so no single path literal
  dominates the corpus.
- **Within a session** the mapping is a pure function of
  ``(session_key, original)``, so every occurrence of a path maps to the
  same replacement. A trajectory that ``cd``s into a directory and later
  reads a file under it stays coherent. Randomizing per *occurrence*
  would trade memorization for incoherence, which is strictly worse.

Matching is pattern-based rather than exact-``$HOME``, so it also covers
homes that are not the local one — remote boxes, containers, CI runners
— which the exact-match version missed entirely (it leaked
``/home/pearsonkyle/...`` from a homelab machine 15 times).
"""

import hashlib
import os
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

# Directory names directly under $HOME that carry no personal identity and
# whose literal value is load-bearing in an agent transcript. Renaming
# `~/.claude/projects/` or `~/miniconda3/` would make the log *wrong*
# rather than private, so these pass through untouched. Common generic
# container dirs (Programs, src, work) are here for the same reason: they
# are not identifying, and preserving them keeps paths readable.
_SYSTEM_DIRS = frozenset(
    {
        ".android", ".aws", ".cache", ".cargo", ".claude", ".config", ".conda",
        ".docker", ".gem", ".git", ".gnupg", ".gradle", ".ipython", ".jupyter",
        ".kube", ".lmstudio", ".local", ".m2", ".npm", ".nvm", ".oh-my-zsh",
        ".pyenv", ".rbenv", ".ssh", ".terraform", ".vim", ".vscode", ".yarn",
        "anaconda3", "applications", "bin", "code", "desktop", "developer",
        "development", "documents", "downloads", "dev", "git", "go", "lib",
        "library", "miniconda3", "miniforge3", "movies", "music", "opt",
        "pictures", "profiles", "programs", "project", "projects", "public",
        "repos", "sandbox", "sites", "src", "tmp", "venv", "work", "workspace",
    }
)  # fmt: skip

_ROOTS = ("/home", "/Users")

# Up to three path segments under the home directory are rewritten. Deeper
# segments are source-tree structure (`cardagent/scripts/sim_coverage.py`),
# not identity — and renaming them would desynchronize the paths from the
# file contents quoted alongside them in the same transcript.
_MAX_RENAMED_DEPTH = 3

_HOME_PATH_RX = re.compile(
    r"(?P<root>/home|/Users)/(?P<user>[A-Za-z0-9._-]{2,32})"
    rf"(?P<rest>(?:/[A-Za-z0-9._+@-]+){{0,{_MAX_RENAMED_DEPTH}}})"
)

# Usernames at least this long are substituted even when glued to a
# preceding letter. Terminal logs mangle prompts (a carriage-return
# overwrite produced a literal `Akpearson:~/…` in a real qwen session), and
# a left word boundary refuses to match inside what looks like a longer
# identifier — leaking the username. At 6+ characters the odds of a real
# username occurring inside an unrelated word are negligible, and the right
# boundary still prevents matching a prefix of a longer name. Shorter
# usernames keep the strict boundary, where that risk is real.
_UNBOUNDED_USERNAME_MIN_LEN = 6

# Minimum length before a project name is also replaced as a bare word
# (outside a path). Short names are neither identifying nor safe to
# substitute — replacing every occurrence of "ai" would shred the text.
_BARE_TOKEN_MIN_LEN = 5

# Bare-token boundaries use explicit lookarounds rather than `\b` so that
# `_` counts as a boundary: MCP tool names embed the project name as
# `mcp__deckdoctor__lookup_card_by_name`, and `\b` does not match between
# `_` and a letter.
# Agent logs frequently embed JSON inside JSON, so a name can sit directly
# after a *literal* two-character escape (`\` + `n`) rather than a real
# newline. A plain `(?<![A-Za-z0-9])` lookbehind sees the `n` and refuses
# to match, leaving `\nCard-Agent` un-anonymized, so escape sequences count
# as boundaries too.
_BOUNDARY_L = r"(?:(?<![A-Za-z0-9])|(?<=\\n)|(?<=\\t)|(?<=\\r))"
_BOUNDARY_R = r"(?![A-Za-z0-9])"

# Locally-configured MCP servers are named after the project they serve, so
# `mcp__deckdoctor__lookup_card_by_name` leaks the project in sessions where
# no filesystem path to it ever appeared.
_MCP_TOOL_RX = re.compile(r"(mcp__)([A-Za-z0-9][A-Za-z0-9_-]{2,})(__)")

# Shared/service account names. These are real home directories and get
# path-renamed like any other, but they are not *identifying*, and
# substituting every bare occurrence of "user", "claude" or "root" in prose
# would shred the text.
_GENERIC_ACCOUNTS = frozenset(
    {
        "admin", "app", "build", "ci", "claude", "daemon", "debian", "dev",
        "developer", "docker", "ec2-user", "guest", "home", "jenkins", "node",
        "nobody", "root", "runner", "sandbox", "test", "ubuntu", "user",
        "users", "vagrant", "www-data",
    }
)  # fmt: skip


def harvest_identities(s: str) -> tuple[set[str], set[str]]:
    """Collect ``(usernames, project_names)`` appearing in one text.

    Run over the whole corpus before anonymizing any of it. Identity
    evidence is unevenly distributed across sessions: a homelab username
    shows up as ``/home/pearsonkyle/...`` in a handful of sessions and
    only as ``github.com/pearsonkyle/...`` in the rest, and a project
    appears as a path in some sessions and only as an MCP tool name in
    others. A purely per-session scan sees the URL or the tool name with
    no way to know it is identifying; a corpus-wide pass does.
    """
    users: set[str] = set()
    projects: set[str] = set()
    if not s:
        return users, projects
    for match in _HOME_PATH_RX.finditer(s):
        users.add(match.group("user").lower())
        for segment in (match.group("rest") or "").split("/")[1:]:
            if segment and segment not in (".", "..") and segment.lower() not in _SYSTEM_DIRS:
                projects.add(segment.lower())
    for match in _MCP_TOOL_RX.finditer(s):
        projects.add(match.group(2).lower())
    return users, projects


@lru_cache(maxsize=1)
def _wordlist() -> tuple[str, ...]:
    """The BIP-39 English wordlist as a stable, indexable tuple."""
    data = resources.files("logminer.redaction").joinpath("bip39_wordlist.txt")
    return tuple(sorted({w.strip() for w in data.read_text().splitlines() if w.strip()}))


class Anonymizer:
    """Replaces identifying paths and names with a per-session persona.

    ``session_key`` seeds the persona. Pass the session/conversation id so
    that two different sessions get different fake identities while a
    single session stays internally consistent. With no key (the default)
    the whole corpus shares one persona, which reproduces the old
    zero-variance behaviour — callers that care should pass one.
    """

    def __init__(
        self,
        extra_usernames: list[str] | None = None,
        session_key: str = "",
        extra_projects: list[str] | None = None,
    ):
        self._session_key = session_key
        self._local_username = os.environ.get("USER") or os.environ.get("USERNAME") or ""
        self._local_home = str(Path.home())
        self._extra = [u for u in (extra_usernames or []) if u]
        self._user_map: dict[str, str] = {}
        self._project_map: dict[str, str] = {}
        self._bare_rx: re.Pattern[str] | None = None
        self._bare_dirty = True
        # Seed the maps so names harvested elsewhere in the corpus are
        # substituted here even if this session never shows them in a path.
        for project in extra_projects or []:
            self._fake_project(project)

    # --- persona derivation ------------------------------------------

    def _digest(self, kind: str, value: str) -> int:
        seed = f"{self._session_key}\x00{kind}\x00{value}".encode()
        return int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")

    def _root(self) -> str:
        return _ROOTS[self._digest("root", "") % len(_ROOTS)]

    def _fake_username(self, original: str) -> str:
        key = original.lower()
        if key not in self._user_map:
            n = self._digest("user", key)
            words = _wordlist()
            w1 = words[n % len(words)]
            w2 = words[(n >> 24) % len(words)]
            style = (n >> 52) % 4
            if style == 0:
                fake = w1
            elif style == 1:
                fake = f"{w1}{n % 100:02d}"
            elif style == 2:
                fake = f"{w1}.{w2}"
            else:
                fake = f"{w1[0]}{w2}"
            self._user_map[key] = fake
        return self._user_map[key]

    def _fake_project(self, original: str) -> str:
        # Keyed case-insensitively: the same project shows up as `DeckDoctor`
        # in a path and `mcp__deckdoctor__lookup` in a tool name, and both
        # must resolve to the same replacement.
        key = original.lower()
        if key not in self._project_map:
            n = self._digest("project", key)
            words = _wordlist()
            w1 = words[n % len(words)]
            w2 = words[(n >> 24) % len(words)]
            style = (n >> 52) % 4
            if style == 0:
                fake = f"{w1}-{w2}"
            elif style == 1:
                fake = f"{w1.capitalize()}{w2.capitalize()}"
            elif style == 2:
                fake = f"{w1}_{w2}"
            else:
                fake = w1
            # Preserve the leading dot of hidden directories.
            if key.startswith("."):
                fake = f".{fake}"
            self._project_map[key] = fake
            self._bare_dirty = True
        return self._project_map[key]

    # --- path rewriting ----------------------------------------------

    def _rewrite_segment(self, segment: str) -> str:
        if not segment or segment in (".", "..") or segment.lower() in _SYSTEM_DIRS:
            return segment
        return self._fake_project(segment)

    def _rewrite_home_path(self, match: re.Match[str]) -> str:
        user = match.group("user")
        rest = match.group("rest") or ""
        # `/home/user/...` where "user" is a system dir means this is not a
        # home path at all (e.g. `/home/src`); leave the segment logic to
        # handle it uniformly.
        out = f"{self._root()}/{self._fake_username(user)}"
        for segment in rest.split("/")[1:]:
            out += "/" + self._rewrite_segment(segment)
        return out

    # --- public API ---------------------------------------------------

    def prescan(self, s: str) -> None:
        """Populate the project/user maps without rewriting.

        Bare-token replacement can only be applied once every project name
        in the *record* is known, otherwise a name that first appears in a
        path late in the conversation would be left un-replaced in the
        prose earlier in it. Callers should ``prescan`` every message in a
        record before calling :meth:`text` on any of them.
        """
        if not s:
            return
        for match in _HOME_PATH_RX.finditer(s):
            self._rewrite_home_path(match)

    def _bare_token_rx(self) -> re.Pattern[str] | None:
        """Regex matching identifying names outside of a path context."""
        if self._bare_dirty:
            self._bare_dirty = False
            names = [
                n
                for n in self._project_map
                if len(n) >= _BARE_TOKEN_MIN_LEN and n.lower() not in _SYSTEM_DIRS
            ]
            # Longest first so `Card-Agent` wins over a hypothetical `Card`.
            names.sort(key=len, reverse=True)
            self._bare_rx = (
                re.compile(
                    _BOUNDARY_L + "(" + "|".join(re.escape(n) for n in names) + ")" + _BOUNDARY_R,
                    re.IGNORECASE,
                )
                if names
                else None
            )
        return self._bare_rx

    def text(self, s: str) -> str:
        """Apply all anonymization to a text string."""
        if not s:
            return s

        # Homes that don't live under /home or /Users (`/var/root`, or a
        # relocated $HOME) can't be found by pattern, so substitute the
        # literal first and let the pattern pass handle everything else.
        if self._local_home and not self._local_home.startswith(_ROOTS):
            s = s.replace(self._local_home, f"{self._root()}/{self._fake_username('local')}")

        s = _HOME_PATH_RX.sub(self._rewrite_home_path, s)

        # MCP server names, which are named after the project they serve.
        s = _MCP_TOOL_RX.sub(
            lambda m: f"{m.group(1)}{self._fake_project(m.group(2)).lower()}{m.group(3)}", s
        )

        # Bare project names: `the DeckDoctor repo`, `mcp__deckdoctor__x`,
        # and Claude's own encoded project dirs (`-Users-me-Programs-X`).
        bare_rx = self._bare_token_rx()
        if bare_rx is not None:
            s = bare_rx.sub(lambda m: self._fake_project(m.group(1)), s)

        # Bare usernames anywhere in the text, including ones only ever seen
        # in a path. The 4-char floor avoids substituting initials.
        candidates = {self._local_username, *self._extra, *self._user_map}
        for username in sorted(candidates, key=len, reverse=True):
            if username and len(username) >= 4 and username.lower() not in _GENERIC_ACCOUNTS:
                fake = self._fake_username(username)
                left = "" if len(username) >= _UNBOUNDED_USERNAME_MIN_LEN else _BOUNDARY_L
                s = re.sub(
                    left + re.escape(username) + _BOUNDARY_R,
                    lambda _m, fake=fake: fake,
                    s,
                    flags=re.IGNORECASE,
                )
        return s

    def path(self, s: str) -> str:
        """Anonymize a file path."""
        return self.text(s)
