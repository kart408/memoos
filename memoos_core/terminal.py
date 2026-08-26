"""
The terminal memory layer.

A terminal forgets everything the moment you close it. Scrollback is not
memory — it is a transcript with no index, no structure, and no idea what
mattered. Reopen the window tomorrow and the work you were three hours
into is gone.

This module closes that gap in two stages, deliberately kept apart:

  journal   Every command, git action and Claude Code turn is appended
            raw, in under a millisecond, by `journal.Journal`. Nothing
            is interpreted. Nothing is allowed to be slow.

  distil    Later — at session end, or on demand — a session's events are
            rewritten as a short digest and pushed through the extraction
            pipeline, which turns a hundred shell lines into a handful of
            atomic memories and the entity graph connecting them.

The split is the whole design. Interpretation is expensive and belongs
nowhere near the prompt you are waiting on; recall needs structure, and
structure is exactly what raw scrollback lacks.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import config
from .journal import CLAUDE, COMMAND, GIT, NOTE, Journal, container_for

# Commands that say nothing about what you were doing. Looking around a
# directory is navigation, not work, and journalling it as memory buries
# the few lines that actually mattered under a hundred that didn't.
NOISE = {
    "ls", "ll", "la", "cd", "pwd", "clear", "exit", "history", "man",
    "which", "whoami", "date", "top", "htop", "tree", "echo", "printenv",
    "env", "less", "more", "head", "tail", "wc", "source", "export",
    # Process and session management. Starting, stopping and waiting on
    # things is how you *operate* a machine, not what you did on it —
    # `sleep 60` and `kill 23643 && pkill ollama` both became permanent
    # memories, and `sleep` and `kill` both became graph entities.
    "sleep", "kill", "pkill", "killall", "jobs", "bg", "fg", "wait",
    "ps", "uptime", "df", "du", "free", "open", "say", "clear",
    # Searching and filtering. Looking for a thing is not doing anything
    # to it — `find` and a `ps aux | grep` pipeline both became permanent
    # memories, and both became graph entities on top of that.
    #
    # `sed` is deliberately absent: `sed -i` edits in place, so it is a
    # write wearing an inspection's clothes. Same reason `xargs` is out —
    # what it does depends entirely on what follows it.
    "find", "grep", "egrep", "fgrep", "rg", "ag", "locate", "awk",
    "sort", "uniq", "cut", "diff", "file", "stat", "basename",
    "dirname", "realpath", "column", "jq", "cat",
}

# Commands whose *arguments* turn an inspection into a write. `find .`
# is looking around; `find . -delete` is not.
_NOT_REALLY_LOOKING = {
    "find": ("-delete", "-exec", "-execdir", "-ok"),
}

def _split_stages(text: str) -> tuple[List[str], bool]:
    """
    Split a command line into its stages, and say whether it redirects.

    Written as a scanner rather than a regex because the separators live
    inside quoted arguments too, and splitting on those is how
    `ps aux | grep -E "uvicorn|api.py|main.py"` came apart into stages
    called `"uvicorn` and `api.py` — neither of which is a command, so
    the whole pipeline read as real work and became a memory.

    Returns (stages, redirects). A redirect is reported separately
    because it settles the question on its own: something was written.
    """
    stages: List[str] = []
    current: List[str] = []
    quote = ""
    redirects = False
    index = 0

    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            index += 1
            continue
        if char == ">":
            redirects = True
            index += 1
            continue
        if text.startswith(("&&", "||"), index):
            stages.append("".join(current))
            current = []
            index += 2
            continue
        if char in "|;":
            stages.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1

    stages.append("".join(current))
    return [s.strip() for s in stages if s.strip()], redirects
NOISE_PAIRS = {
    ("git", "status"), ("git", "log"), ("git", "diff"), ("git", "branch"),
    ("git", "show"), ("git", "stash"), ("ls", "-la"),
}

# Commands whose *failure* is the interesting part.
_FILE_LIKE = re.compile(r"[\w./-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|sql|md|json|ya?ml|toml|html|css)")


# zsh's AUTO_CD, where a bare directory path *is* the command. There is
# no command word for NOISE to match, so `cd /Users/me` was filtered as
# navigation and `/Users/me` was not — and the difference became the
# permanent memory "The user ran `/Users/karthikreddy`", plus a graph
# node for a directory on one machine.
#
# An extension is what separates the two cases: `./deploy.sh` and
# `~/bin/build.sh` are things being run, `~/projects/memoos` and `..`
# are somewhere being gone to. Anything carrying an argument is work
# whatever it looks like.
_BARE_PATH = re.compile(r"^(?:/|~|\.\.)[\w./~-]*$")


def _is_navigation(stage_parts: List[str]) -> bool:
    if len(stage_parts) != 1:
        return False
    target = stage_parts[0]
    return bool(_BARE_PATH.match(target)) and not os.path.splitext(target)[1]


def _stage_is_noise(stage: str) -> bool:
    """Is one command in a pipeline pure looking-around?"""
    parts = stage.split()
    if not parts:
        return True
    if _is_navigation(parts):
        return True
    head = os.path.basename(parts[0])
    if len(parts) >= 2 and (head, parts[1]) in NOISE_PAIRS:
        return True
    if head not in NOISE:
        return False
    forbidden = _NOT_REALLY_LOOKING.get(head, ())
    return not any(flag in parts for flag in forbidden)


# Exit codes that mean "somebody stopped this", not "this broke".
#
# A shell reports a signalled process as 128 + signal. 130 is Ctrl-C,
# which is how you stop a server you started on purpose — and it was
# being written into memory as "the demo script failed with exit code
# 130", which reads as a bug in the demo. Only the user-initiated
# signals are listed: 134 (SIGABRT) and 139 (SIGSEGV) are genuine
# crashes and stay failures.
INTERRUPTED = {129, 130, 143}   # SIGHUP, SIGINT, SIGTERM


def is_noise(command: str) -> bool:
    """
    Is this command pure navigation, with nothing to remember?

    Judged per stage, because a command line is usually several commands
    and the old rule gave up on the first `|`. That is how
    `ps aux | grep -E "uvicorn|api.py" | grep -v grep` became a permanent
    memory: a pipe was read as "this does something", when every stage of
    that one is looking around. A pipeline is noise exactly when all of
    its stages are — `cat data.json | python load.py` still is not,
    because `python` is work.

    A redirect is the real signal that something was written, and it
    short-circuits the whole line: `cat api.py` is inspection,
    `cat api.py > backup.py` is not.
    """
    text = command.strip()
    if not text:
        return True
    stages, redirects = _split_stages(text)
    if redirects:
        return False
    return bool(stages) and all(_stage_is_noise(s) for s in stages)


def files_in(text: str) -> List[str]:
    """Filenames mentioned anywhere in a command or message."""
    return list(dict.fromkeys(_FILE_LIKE.findall(text)))


# ---------------------------------------------------------------- digest


def _clock(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%H:%M")
    except (ValueError, TypeError):
        return "??:??"


def _one_line(text: str, limit: int = 240) -> str:
    """
    Flatten and clip a prompt to one readable sentence's worth.

    Enforced here, at the point of use, and not only when the transcript
    is imported. The journal is append-only and holds whatever the import
    rules were on the day it was written; the digest has a line budget
    today. Clipping where the constraint actually lives means changing it
    never requires rewriting history.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def _quoted(text: str) -> str:
    """Pull the message out of `-m "..."` style arguments."""
    match = re.search(r"""["']([^"']{3,120})["']""", text)
    return match.group(1) if match else ""


def describe_command(command: str) -> str:
    """
    Say what a command *did*, in English.

    This exists because of what the alternative produces. Feed raw shell
    lines to an extractor and you get memories about shell lines — "User
    ran commands: git checkout -b fix-auth, pip install pyjwt" — which is
    a transcript with extra steps. Nobody reads that later and learns
    anything. Rewriting each line as a sentence first means the memory
    that comes out the other end is "User created the git branch
    fix-auth", which you can read on its own, months later, and
    understand.

    Returns a verb phrase with no subject; the digest supplies one.
    """
    parts = command.strip().split()
    if not parts:
        return ""
    tool = os.path.basename(parts[0])
    rest = parts[1:]
    tail = " ".join(rest)

    if tool == "git" and rest:
        action, args = rest[0], rest[1:]
        target = " ".join(args)
        if action == "checkout" and "-b" in args:
            branch = [a for a in args if a != "-b"]
            return f"created and switched to the git branch {' '.join(branch)}"
        if action in {"checkout", "switch"}:
            return f"switched to the git branch {target}"
        if action == "commit":
            message = _quoted(target)
            return f'committed "{message}"' if message else "made a git commit"
        if action == "merge":
            return f"merged the branch {target}"
        if action == "rebase":
            return f"rebased onto {target}"
        if action == "push":
            return "pushed the branch to the remote"
        if action == "pull":
            return "pulled the latest changes from the remote"
        if action == "clone":
            return f"cloned the repository {target}"
        if action in {"revert", "reset"}:
            return f"{action} the working tree to {target}" if target else f"ran a git {action}"
        return f"ran git {action} {target}".strip()

    if tool in {"pip", "pip3"} and rest:
        if rest[0] == "install":
            packages = [a for a in rest[1:] if not a.startswith("-")]
            if "-r" in rest:
                return f"installed the Python dependencies listed in {' '.join(packages)}"
            return f"installed the Python package {', '.join(packages)}" if packages \
                else "installed Python dependencies"
        if rest[0] == "uninstall":
            return f"removed the Python package {' '.join(rest[1:])}"

    if tool in {"npm", "yarn", "pnpm"} and rest:
        if rest[0] in {"install", "add", "i"}:
            packages = [a for a in rest[1:] if not a.startswith("-")]
            return (f"installed the {tool} package {', '.join(packages)}" if packages
                    else f"installed {tool} dependencies")
        if rest[0] in {"run", "start", "build", "test"}:
            return f"ran the {tool} script {' '.join(rest[1:]) or rest[0]}"

    if tool in {"pytest", "jest", "vitest", "go"} and (tool != "go" or rest[:1] == ["test"]):
        target = " ".join(a for a in rest if not a.startswith("-")) or "the whole suite"
        return f"ran the tests in {target}"

    if tool in {"python", "python3"} and rest:
        if rest[0] == "-m":
            return f"ran the module {' '.join(rest[1:2])}"
        return f"ran the script {rest[0]}"

    if tool in {"uvicorn", "gunicorn", "flask", "fastapi"}:
        return f"started the {tool} server {tail}".strip()

    if tool == "docker" and rest:
        return f"ran docker {rest[0]} {' '.join(rest[1:])}".strip()

    if tool == "make":
        return f"ran make {tail}".strip() if tail else "ran make"

    if tool in {"mkdir", "touch", "rm", "mv", "cp", "chmod"}:
        verb = {"mkdir": "created the directory", "touch": "created the file",
                "rm": "deleted", "mv": "moved", "cp": "copied",
                "chmod": "changed permissions on"}[tool]
        return f"{verb} {' '.join(a for a in rest if not a.startswith('-'))}".strip()

    if tool in {"vim", "nvim", "nano", "code", "subl"} and rest:
        return f"edited {' '.join(rest)}"

    if tool == "curl" and rest:
        url = next((a for a in rest if a.startswith("http")), "")
        return f"made an HTTP request to {url}" if url else "made an HTTP request"

    if tool in {"ollama"} and rest:
        return f"ran ollama {' '.join(rest)}"

    return f"ran `{command.strip()}`"


def build_digest(container: str, events: List[Dict[str, Any]]) -> str:
    """
    Rewrite a session's raw events as prose an extractor can work with.

    Two rules, both learned the hard way. Every line is a whole sentence,
    because a bulleted list under a heading produces memories that quote
    the heading — "Files involved: api.py" is not something you can be
    reminded of. And every line is one fact, because the extractor splits
    on ideas, and a line holding three commands becomes one memory
    holding three commands.
    """
    lines: List[str] = []
    seen: set = set()

    def say(sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        if not sentence.endswith("."):
            sentence += "."
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            lines.append(sentence)

    for event in events:
        text = (event.get("text") or "").strip()
        if not text:
            continue
        kind = event.get("kind")

        if kind == CLAUDE:
            text = _one_line(text)
            # Marked as a request, not stated as a fact.
            #
            # This line used to read `The user wanted to "..."`, and it
            # was the single biggest source of junk in the store: every
            # prompt became an assertion that the user *wanted* something,
            # and the extractor faithfully recorded the wanting. Thirty of
            # fifty-one memories in a real store were things like "User
            # wanted to commit the changes" — true for thirty seconds,
            # stored forever, and ranked against facts that still hold.
            #
            # A request is context for reading the commands that follow,
            # never a memory in itself. The prompt is told to treat this
            # prefix that way; naming it explicitly is what makes that
            # instruction land.
            #
            # Still not "asked Claude Code to ...": naming the tool in
            # every line makes it an entity in every memory, and an entity
            # attached to everything is attached to nothing.
            say(f'The user asked for: {text.rstrip(".")}')
        elif kind == NOTE:
            say(f"The user noted that {text[0].lower()}{text[1:]}"
                if text[:1].isupper() else f"The user noted that {text}")
        elif kind == GIT:
            say(f"In version control, {text}")
        elif kind == COMMAND:
            if is_noise(text):
                continue
            phrase = describe_command(text)
            code = event.get("exit_code")
            if code in INTERRUPTED:
                # Stopping a thing you started is not the thing failing.
                say(f"The user {phrase}")
            elif code not in (0, None):
                # Same past-tense phrase either way. "tried to ran the
                # tests" is what you get from bolting an infinitive frame
                # onto a past-tense verb, and a memory that reads wrong
                # reads as untrustworthy.
                say(f"The user {phrase}, and it failed with exit code {code}")
            else:
                say(f"The user {phrase}")

    # Deliberately no "on <date>, in project <x>" header. It reads like a
    # sentence, so the extractor dutifully turns it into a memory — "User
    # worked on the project memoos on 2026-08-24" — which restates the
    # container and the row's own created_at and displaces something real.
    # Provenance belongs in columns, not in prose.
    return "\n".join(lines)


# -------------------------------------------------------------- the layer


class TerminalMemory:
    """
    One project's terminal memory.

    `MemoOS` is built on first use rather than in the constructor, so
    journalling and listing stay in the millisecond range — only the
    paths that genuinely need embeddings or the extraction model pay for
    loading them.
    """

    def __init__(self, container: Optional[str] = None,
                 persist_path: Optional[str] = None):
        self.container = config.safe_container(container) if container \
            else container_for()
        self.persist_path = persist_path or config.DATA_DIR
        self.journal = Journal(data_dir=self.persist_path)
        self._memo = None

    @property
    def memo(self):
        if self._memo is None:
            from .memory import MemoOS  # deferred: this is the expensive import
            self._memo = MemoOS(container=self.container,
                                persist_path=self.persist_path)
        return self._memo

    # ------------------------------------------------------------ write

    def record_command(self, command: str, *, session_id: str, cwd: str,
                       exit_code: Optional[int] = None) -> str:
        return self.journal.record(COMMAND, command, container=self.container,
                                   session_id=session_id, cwd=cwd,
                                   exit_code=exit_code)

    def record_note(self, text: str, *, session_id: str = "notes") -> str:
        return self.journal.record(NOTE, text, container=self.container,
                                   session_id=session_id)

    # --------------------------------------------------------- distilling

    def distill(self, session_id: Optional[str] = None,
                limit: int = 400) -> Dict[str, Any]:
        """
        Turn journalled events into memories and graph structure.

        Only undistilled events are considered, so running this twice on
        the same session is a no-op rather than a duplicate — which
        matters because it is wired to run automatically when a shell
        exits, and shells exit in all sorts of ways.
        """
        events = self.journal.events(self.container, session_id=session_id,
                                     undistilled_only=True, limit=limit)
        if not events:
            return {"distilled": 0, "created": [], "digest": "", "skipped": "nothing new"}

        digest = build_digest(self.container, events)
        # An empty digest means the session was all navigation. Mark it
        # done and spend nothing on it.
        if not digest.strip():
            self.journal.mark_distilled([e["id"] for e in events],
                                        self.container)
            return {"distilled": len(events), "created": [], "digest": digest,
                    "skipped": "no signal in this session"}

        title = f"terminal session {session_id or 'recent'} — {self.container}"
        # The event ids travel with the document, which is what lets a
        # memory be traced back past the digest to the commands that
        # produced it. Without them the trail stops at "some session".
        result = self.memo.ingest_document(
            digest, title=title, source="terminal",
            metadata={"event_ids": [e["id"] for e in events],
                      "session_id": session_id,
                      "session_ids": sorted({e["session_id"] for e in events})},
        )

        # Only retire events the model actually read. If a chunk timed out,
        # leaving its events pending costs a re-run; marking them done
        # costs the work itself, silently and permanently. The journal is
        # the source of truth precisely so this is recoverable.
        if result.complete:
            self.journal.mark_distilled([e["id"] for e in events],
                                        self.container)

        return {
            "distilled": len(events) if result.complete else 0,
            "digest": digest,
            "created": result.created,
            "superseded": result.superseded,
            "entities": result.entities,
            "summary": result.summary,
            "chunks_total": result.chunks_total,
            "chunks_failed": result.chunks_failed,
            "retained": 0 if result.complete else len(events),
        }

    def ingest_kb(self, path: str) -> Dict[str, Any]:
        """
        Add a document to what this project knows about itself.

        Notes, specs, a README, anything describing how you work — the
        same pipeline as a session digest, so a document's facts land in
        the same graph as the terminal's and can be retrieved together.
        """
        resolved = os.path.abspath(os.path.expanduser(path))
        with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        if not text.strip():
            return {"created": [], "summary": "file was empty", "path": resolved}

        result = self.memo.ingest_document(
            text, title=os.path.basename(resolved), uri=resolved, source="kb"
        )
        return {"path": resolved, "created": result.created,
                "entities": result.entities, "summary": result.summary}

    # ------------------------------------------------------------- read

    def recall(self, query: Optional[str] = None, top_k: int = 8) -> Dict[str, Any]:
        """
        What a new terminal should know about this project.

        With a query this is ordinary hybrid search. Without one it is
        the session-open case: the most recent work, plus whatever is
        still sitting in the journal undistilled.
        """
        sessions = self.journal.sessions(self.container, limit=5)
        pending = self.journal.events(self.container, undistilled_only=True, limit=40)

        if query:
            hits = self.memo.search(query, top_k=top_k, touch=False)
            memories = [h.memory for h in hits]
        else:
            hits = []
            memories = self.memo.all(limit=top_k)

        return {
            "container": self.container,
            "query": query,
            "hits": hits,
            "memories": memories,
            "sessions": sessions,
            "pending": pending,
            "entities": self.memo.entities(limit=15),
        }

    def stats(self) -> Dict[str, Any]:
        base = self.memo.stats()
        base["sessions"] = len(self.journal.sessions(self.container, limit=1000))
        base["events"] = len(self.journal.events(self.container, limit=100_000))
        return base


# ------------------------------------------------- Claude Code transcripts


def claude_project_dir(project_path: Optional[str] = None) -> str:
    """
    Where Claude Code keeps this project's transcripts.

    It slugifies the absolute path by replacing every separator with a
    dash, so /Users/me/memoos becomes -Users-me-memoos.
    """
    path = os.path.abspath(project_path or os.getcwd())
    slug = path.replace(os.sep, "-")
    return os.path.expanduser(os.path.join("~/.claude/projects", slug))


def _text_of(message: Any) -> str:
    """Pull readable text out of a transcript message, whatever its shape."""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


# A prompt that carries no content of its own. Most of what gets typed at
# a coding agent is steering — "continue", "fix it", "yes" — which means
# something only in reply to a turn we are not keeping. Journalling them
# produces memories like "User said fix it", which is worse than useless:
# it is noise that outranks real memories on recency.
FILLER = {
    "continue", "go on", "go ahead", "yes", "yeah", "yep", "no", "ok", "okay",
    "sure", "do it", "fix it", "fix that", "fix both", "run it", "try again",
    "again", "next", "stop", "wait", "thanks", "thank you", "please", "good",
    "nice", "perfect", "great", "cool", "done", "proceed", "keep going",
    "carry on", "resume", "retry", "same", "both", "all of them", "y", "n",
    "hi", "hey", "hello", "yo", "test", "testing",
}

# Lines the harness writes into the transcript as if the user had typed
# them. They are session bookkeeping, not something anybody asked for.
HARNESS_MARKERS = ("[Request interrupted", "[The user", "<command-", "<local-command")

MIN_PROMPT_CHARS = 18


def is_filler(text: str) -> bool:
    """Is this prompt pure steering, with no content of its own?"""
    stripped = text.strip().strip(".!?,").lower()
    if stripped in FILLER:
        return True
    # Short *and* starting with a filler word: "yes, fix that too" says no
    # more than "yes" does. A short prompt that starts with something else
    # ("deploy to staging") is kept — brevity is not emptiness.
    if len(stripped) < MIN_PROMPT_CHARS:
        first = stripped.split(",")[0].split()[:2]
        if not first:
            return True   # no words at all is as empty as steering gets
        return " ".join(first) in FILLER or first[0] in FILLER
    return False


def read_claude_turns(transcript: str, *, max_chars: int = 240) -> List[Dict[str, Any]]:
    """
    The prompts a user typed in one Claude Code session, minus the filler.

    Only the human turns. The assistant's replies are long, largely
    restate the request, and would swamp extraction; what a session was
    *about* is what was asked of it. Tool calls and system lines are
    skipped for the same reason.
    """
    turns: List[Dict[str, Any]] = []
    try:
        handle = open(transcript, "r", encoding="utf-8", errors="replace")
    except OSError:
        return turns

    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "user" or row.get("isSidechain"):
                continue
            text = _text_of(row.get("message")).strip()
            # Command output and pasted attachments arrive as user turns
            # too; they are not things anybody asked for.
            if not text or text.startswith("<") or text.startswith("Caveat:"):
                continue
            if text.startswith(HARNESS_MARKERS) or is_filler(text):
                continue
            # Pasted documents arrive as user turns too. Collapsed to one
            # line and clipped, because the digest gives each event a
            # single sentence and a 2,000-word spec pasted into the middle
            # of one is not a sentence — it is the rest of the chunk.
            flat = " ".join(text.split())
            if len(flat) > max_chars:
                flat = flat[:max_chars].rsplit(" ", 1)[0] + "…"
            turns.append({
                "text": flat,
                "session_id": row.get("sessionId", os.path.basename(transcript)[:8]),
                "cwd": row.get("cwd"),
                "branch": row.get("gitBranch"),
                "timestamp": row.get("timestamp"),
            })
    return turns


def turn_key(session_id: str, text: str) -> str:
    """
    A stable identity for one imported prompt.

    Keyed on content, not position. The obvious key is "session id plus
    index", and it is wrong: the index is a position in the *filtered*
    list, so the day the filter changes — one more filler word recognised
    — every prompt after it shifts by one, stops matching what was already
    imported, and the whole transcript lands a second time. Hashing the
    text makes re-importing idempotent regardless of what the filter does.
    """
    digest = hashlib.sha1(f"{session_id}\x00{text}".encode("utf-8")).hexdigest()
    return f"{session_id}:{digest[:12]}"


def import_claude_sessions(memory: TerminalMemory, *,
                           project_path: Optional[str] = None,
                           newest: int = 3) -> Dict[str, Any]:
    """
    Journal what was asked of Claude Code in this project's recent sessions.

    Re-importing is safe: every turn carries its transcript uuid, and
    ones already journalled are skipped, so this can be run on every
    shell start without the log growing a duplicate each time.
    """
    directory = claude_project_dir(project_path)
    if not os.path.isdir(directory):
        return {"imported": 0, "sessions": 0, "detail": f"no transcripts at {directory}"}

    transcripts = sorted(
        (os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jsonl")),
        key=os.path.getmtime, reverse=True,
    )[:newest]

    known = {
        event["metadata"].get("turn")
        for event in memory.journal.events(memory.container, kind=CLAUDE, limit=5000)
    }

    imported = 0
    for transcript in transcripts:
        for turn in read_claude_turns(transcript):
            key = turn_key(turn["session_id"], turn["text"])
            if key in known:
                continue
            memory.journal.record(
                CLAUDE, turn["text"], container=memory.container,
                session_id=f"claude-{turn['session_id'][:8]}",
                cwd=turn.get("cwd"),
                metadata={"turn": key, "branch": turn.get("branch"),
                          "at": turn.get("timestamp")},
            )
            imported += 1

    return {"imported": imported, "sessions": len(transcripts), "detail": directory}
