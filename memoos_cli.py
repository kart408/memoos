#!/usr/bin/env python3
"""
memoos — memory for your terminal.

A shell forgets everything when you close it. This puts a memory layer
underneath it: every command, git action and Claude Code prompt is
journalled instantly, folded into structured memories later, and handed
back when you open a new terminal in the same project.

    memoos install        wire it into your shell
    memoos recall         what was I doing here?
    memoos distill        fold this session into memory
    memoos ingest FILE    teach it about you or the project
    memoos graph          what it knows, and how it connects
    memoos serve          the dashboard

Every command is scoped to a *container*, which is the git repo you are
standing in. Projects never see each other's memory.
"""

import os
import sys

# ---------------------------------------------------------------- hot path
# `memoos log` runs from a shell hook on every command you type. It gets
# handled before argparse or anything else is imported, because the only
# thing that matters here is that it is over quickly.
if len(sys.argv) >= 3 and sys.argv[1] == "log":
    from memoos_core.connection import is_connected
    # The hook checks this too, and checks it more cheaply. This is the
    # backstop for an out-of-date hook, or anything else calling `log`
    # directly: disconnected must mean disconnected, not mostly.
    if not is_connected():
        sys.exit(0)
    from memoos_core.journal import Journal, container_for
    _exit_code = int(sys.argv[2]) if sys.argv[2].lstrip("-").isdigit() else None
    _command = " ".join(sys.argv[3:])
    _cwd = os.environ.get("MEMOOS_CWD") or os.getcwd()
    Journal().record(
        "command", _command, container=container_for(_cwd),
        session_id=os.environ.get("MEMOOS_SESSION", "shell"),
        cwd=_cwd, exit_code=_exit_code,
    )
    sys.exit(0)

import argparse  # noqa: E402
import json  # noqa: E402

from memoos_core import quick  # noqa: E402
from memoos_core.journal import Journal, container_for  # noqa: E402

# ------------------------------------------------------------------ output

_TTY = sys.stdout.isatty()


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


dim = lambda s: paint(s, "2")
bold = lambda s: paint(s, "1")
cyan = lambda s: paint(s, "36")
green = lambda s: paint(s, "32")
yellow = lambda s: paint(s, "33")
magenta = lambda s: paint(s, "35")


def rule(title: str) -> None:
    print(f"\n{bold(title)}")


def clock(iso: str) -> str:
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d %b %H:%M")
    except (ValueError, TypeError):
        return "?"


# ---------------------------------------------------------------- commands


def cmd_recall(args) -> int:
    """
    What this project remembers.

    Without a query this stays on the fast path — plain SQL, no vector
    store, no model — because it is meant to run when a terminal opens
    and nobody wants to wait three seconds for a shell prompt.
    """
    container = args.container or container_for()
    journal = Journal()

    if args.query:
        from memoos_core.terminal import TerminalMemory
        memory = TerminalMemory(container=container)
        hits = memory.memo.search(" ".join(args.query), top_k=args.limit, touch=False)
        rule(f"{container} · recall")
        if not hits:
            print(dim("  nothing relevant remembered yet"))
            return 0
        for hit in hits:
            why = ",".join(hit.matched_by)
            sim = f"{hit.vector_score:.2f}" if hit.vector_score is not None else " -- "
            print(f"  {cyan(sim)} {hit.memory.text}  {dim('(' + why + ')')}")
        return 0

    counts = quick.counts(container)
    memories = quick.recent_memories(container, limit=args.limit)
    entities = quick.top_entities(container, limit=10)
    sessions = journal.sessions(container, limit=3)
    pending = journal.events(container, undistilled_only=True, limit=200)

    tally = (f"{counts['memories']} memories · {counts['entities']} entities · "
             f"{counts['relations']} relations")
    print(f"\n{bold('memoos')} {dim('·')} {cyan(container)}  {dim(tally)}")

    if memories:
        rule("Last remembered")
        for m in memories:
            print(f"  {magenta('[' + m['memory_type'][:4] + ']')} {m['text']}")
    else:
        print(dim("\n  nothing distilled yet — run `memoos distill`"))

    if entities:
        rule("Working on")
        print("  " + "  ".join(f"{e['name']}{dim('·' + str(e['mention_count']))}"
                               for e in entities))

    if sessions:
        rule("Recent sessions")
        for s in sessions:
            flag = yellow(f"{s['failures']} failed") if s["failures"] else green("clean")
            print(f"  {clock(s['ended_at'])}  {s['events']:>4} events  {flag}"
                  f"  {dim(s['session_id'])}")

    if pending:
        print(f"\n{yellow(str(len(pending)))} events not yet folded into memory — "
              f"{dim('memoos distill')}")
    print()
    return 0


def cmd_note(args) -> int:
    text = " ".join(args.text)
    Journal().record("note", text, container=args.container or container_for(),
                     session_id=os.environ.get("MEMOOS_SESSION", "notes"))
    print(green("noted."))
    return 0


def cmd_distill(args) -> int:
    from memoos_core.terminal import TerminalMemory
    memory = TerminalMemory(container=args.container)
    print(dim(f"distilling {memory.container}… (this runs the extraction model)"))
    result = memory.distill(session_id=args.session)

    failed = result.get("chunks_failed", 0)
    if failed:
        total = result.get("chunks_total", 0)
        print(yellow(f"  {failed} of {total} chunks could not be read") +
              dim(" — the extraction model timed out or returned nothing usable"))
        print(dim(f"  {result.get('retained', 0)} events kept in the journal; "
                  f"run `memoos distill` again to retry them"))
        print(dim(f"  if this keeps happening, raise MEMOOS_LLM_TIMEOUT "
                  f"(currently {__import__('memoos_core.config', fromlist=['x']).LLM_TIMEOUT}s)"))

    if not result["created"]:
        if not failed:
            print(dim(f"  {result.get('skipped', 'nothing to do')} "
                      f"({result['distilled']} events)"))
        return 1 if failed else 0

    rule(f"{memory.container} · distilled {result['distilled']} events")
    for m in result["created"]:
        print(f"  {green('+')} {magenta('[' + m.memory_type.value + ']')} {m.text}")
    for m in result.get("superseded", []):
        print(f"  {yellow('~')} superseded: {dim(m.text)}")
    names = [e.name for e in result.get("entities", [])]
    if names:
        print(f"\n  entities: {', '.join(names)}")
    print()
    return 0


def cmd_ingest(args) -> int:
    from memoos_core.terminal import TerminalMemory
    memory = TerminalMemory(container=args.container)
    for path in args.paths:
        if not os.path.exists(os.path.expanduser(path)):
            print(yellow(f"  skipped (not found): {path}"))
            continue
        print(dim(f"reading {path}…"))
        result = memory.ingest_kb(path)
        rule(f"{os.path.basename(result['path'])} → {memory.container}")
        for m in result["created"]:
            print(f"  {green('+')} {m.text}")
        print(dim(f"  {result['summary']}"))
    print()
    return 0


def cmd_claude(args) -> int:
    from memoos_core.terminal import TerminalMemory, import_claude_sessions
    memory = TerminalMemory(container=args.container)
    result = import_claude_sessions(memory, newest=args.newest)
    print(f"imported {green(str(result['imported']))} prompts from "
          f"{result['sessions']} transcript(s)")
    if not result["imported"]:
        print(dim(f"  {result.get('detail', '')}"))
    elif not args.no_distill:
        return cmd_distill(args)
    return 0


def cmd_sessions(args) -> int:
    container = args.container or container_for()
    rows = Journal().sessions(container, limit=args.limit)
    if not rows:
        print(dim("no sessions journalled yet"))
        return 0
    rule(f"{container} · sessions")
    for s in rows:
        state = green("distilled") if not s["pending"] else yellow(f"{s['pending']} pending")
        print(f"  {clock(s['started_at'])} → {clock(s['ended_at'])}  "
              f"{s['events']:>4} events  {state}  {dim(s['session_id'])}")
    print()
    return 0


def cmd_graph(args) -> int:
    container = args.container or container_for()
    entities = quick.top_entities(container, limit=args.limit)
    edges = quick.relations(container, limit=args.limit * 3)

    if not entities:
        print(dim("graph is empty — nothing distilled yet"))
        return 0

    rule(f"{container} · entities")
    for e in entities:
        print(f"  {cyan(e['name']):<28} {dim(e['entity_type'])}  ×{e['mention_count']}")
    if edges:
        rule("relations")
        for r in edges:
            print(f"  {r['subject']} {magenta('—' + r['predicate'] + '→')} {r['object']}")
    print()
    return 0


def cmd_stats(args) -> int:
    container = args.container or container_for()
    data = quick.counts(container)
    data["sessions"] = len(Journal().sessions(container, limit=10_000))
    print(json.dumps(data, indent=2))
    return 0


def cmd_doctor(args) -> int:
    """
    Check every part of the chain, in the order it actually runs.

    The layer has two speeds and they fail differently. Journalling is
    stdlib and SQLite, so it works or the disk is broken. Distillation
    needs Ollama and a model, and it runs *in the background when your
    terminal closes* — which is the worst possible place for a failure,
    because nobody is watching. A session whose distill fails is not
    lost (its events stay pending and the next run picks them up), but
    you would never know to look. This is where you find out.
    """
    import urllib.error
    import urllib.request

    from memoos_core import config, connection

    ok = True

    def line(good: bool, label: str, detail: str = "", fatal: bool = True) -> None:
        nonlocal ok
        if good:
            mark = green("✓")
        else:
            mark = yellow("!") if not fatal else "\033[31m✗\033[0m"
            if fatal:
                ok = False
        print(f"  {mark} {label:<26} {dim(detail)}")

    print(f"\n{bold('attach')}")
    state = connection.status()
    line(state["hook_installed"], "shell hook installed",
         os.path.join(os.path.expanduser("~/.memoos"), "memoos.zsh"))

    zshrc = os.path.expanduser("~/.zshrc")
    sourced = False
    if os.path.exists(zshrc):
        with open(zshrc, encoding="utf-8", errors="replace") as handle:
            sourced = "memoos.zsh" in handle.read()
    line(sourced, "sourced from ~/.zshrc",
         "" if sourced else "add: source ~/.memoos/memoos.zsh")

    # Set by the hook itself, so its presence proves this shell attached.
    attached = bool(os.environ.get("MEMOOS_SESSION"))
    line(attached, "attached to this shell",
         os.environ.get("MEMOOS_SESSION", "open a new terminal"), fatal=False)
    line(state["connected"], "recording",
         "disconnected" if not state["connected"] else "")

    print(f"\n{bold('store')}")
    data_dir = os.path.abspath(config.DATA_DIR)
    pinned = bool(os.environ.get("MEMOOS_DATA_DIR"))
    line(pinned, "MEMOOS_DATA_DIR pinned",
         data_dir if pinned else "unpinned — every directory grows its own store")
    writable = os.access(data_dir, os.W_OK) if os.path.isdir(data_dir) \
        else os.access(os.path.dirname(data_dir) or ".", os.W_OK)
    line(writable, "writable", data_dir)

    container = args.container or container_for()
    path = config.db_path(container=container)
    line(True, "container", f"{container} → {path}")

    print(f"\n{bold('distillation')}   {dim('runs in the background when a terminal closes')}")
    models = []
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=3) as response:
            models = [m["name"] for m in json.load(response).get("models", [])]
        line(True, "ollama reachable", config.OLLAMA_URL)
    except (urllib.error.URLError, OSError, ValueError) as error:
        line(False, "ollama reachable", f"{config.OLLAMA_URL} — {error}")

    if models:
        wanted = config.EXTRACT_MODEL
        present = wanted in models or f"{wanted}:latest" in models \
            or any(m.split(":")[0] == wanted.split(":")[0] for m in models)
        line(present, "extraction model",
             wanted if present else f"{wanted} missing — ollama pull {wanted}")

    pending = len(Journal().events(container, undistilled_only=True, limit=10_000))
    line(True, "events awaiting distill", str(pending) if pending else "none")

    counts = quick.counts(container)
    print(f"\n  {counts['memories']} memories · {counts['entities']} entities · "
          f"{counts['relations']} relations\n")

    if not ok:
        print(dim("  something above is broken — journalling still works, "
                  "but closing a terminal will not fold it into memory\n"))
    return 0 if ok else 1


def cmd_files(args) -> int:
    """
    Where each user's data actually is, and how big it is.

    One container is one SQLite file holding that user's entire state, so
    this doubles as the answer to "what do I hand to the cloud?" — every
    path printed here is independently copyable, replicable and
    deletable.
    """
    from memoos_core import config

    names = config.stored_containers()
    if not names:
        print(dim("\n  no data stored yet\n"))
        return 0

    directory = os.path.abspath(config.containers_dir())
    print(f"\n{bold('stored data')}  {dim(directory)}\n")

    total = 0
    for name in names:
        path = config.db_path(container=name)
        # A live writer leaves a -wal beside the database; it is part of
        # the store until SQLite checkpoints it, so count it in the size
        # rather than under-reporting what has to be copied.
        size = sum(os.path.getsize(path + suffix)
                   for suffix in ("", "-wal", "-shm")
                   if os.path.exists(path + suffix))
        total += size
        counts = quick.counts(name)
        print(f"  {cyan(name):<28} {_human(size):>9}   "
              f"{dim(f'{counts["memories"]} memories · {counts["entities"]} entities')}")

    print(f"\n  {len(names)} file(s), {_human(total)} total\n")
    return 0


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}GB"


def cmd_connect(args) -> int:
    from memoos_core import connection
    changed = connection.connect()
    print(green("connected") + dim(" — this terminal is feeding the memory layer"))
    if not changed:
        print(dim("  (it already was)"))
    if connection.status()["shell_off"]:
        print(yellow("  note: MEMOOS_OFF is set in this shell, so this window "
                     "stays off until you run `memoos connect --here`"))
    return 0


def cmd_disconnect(args) -> int:
    from memoos_core import connection
    changed = connection.disconnect()
    print(yellow("disconnected") + dim(" — nothing new will be journalled"))
    print(dim("  recall, search and the graph keep working; only recording stops"))
    if not changed:
        print(dim("  (it already was)"))
    return 0


def cmd_status(args) -> int:
    from memoos_core import connection
    state = connection.status()
    container = args.container or container_for()
    counts = quick.counts(container)
    pending = len(Journal().events(container, undistilled_only=True, limit=10_000))

    mark = green("● connected") if state["connected"] else yellow("○ disconnected")
    print(f"\n{mark}   {dim('project')} {cyan(container)}")
    if state["shell_off"]:
        print(dim("  off for this shell only (MEMOOS_OFF is set)"))
    elif state["globally_off"]:
        print(dim(f"  off everywhere — flag at {state['flag_file']}"))
    if not state["hook_installed"]:
        print(dim("  shell hook not installed — run `memoos install`"))

    print(f"\n  {counts['memories']} memories · {counts['entities']} entities · "
          f"{counts['relations']} relations")
    if pending:
        print(f"  {yellow(str(pending))} events waiting to be distilled")
    print()
    return 0


def cmd_install(args) -> int:
    from memoos_core.hook import ZSH_HOOK, install_hook
    if args.print_only:
        print(ZSH_HOOK)
        return 0
    path = install_hook()
    print(f"{green('installed')} → {path}")
    print(dim("  add this to ~/.zshrc (once):"))
    print(f"  source {path}")
    print(dim("\n  then open a new terminal, or: ") + f"source {path}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    print(dim(f"dashboard → http://{args.host}:{args.port}/"))
    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
    return 0


# ------------------------------------------------------------------ parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memoos", description="memory for your terminal")
    parser.add_argument("--container", help="override the project container")
    # Not required: bare `memoos` attaches this terminal and reports the
    # state, because that is what typing the name of a thing should do.
    subs = parser.add_subparsers(dest="command")

    p = subs.add_parser("recall", help="what this project remembers")
    p.add_argument("query", nargs="*", help="optional question (uses semantic search)")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=cmd_recall)

    p = subs.add_parser("note", help="journal something yourself")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_note)

    p = subs.add_parser("distill", help="fold journalled events into memory")
    p.add_argument("--session", help="only this session id")
    p.set_defaults(func=cmd_distill)

    p = subs.add_parser("ingest", help="add knowledge-base documents")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_ingest)

    p = subs.add_parser("claude", help="import Claude Code prompts for this project")
    p.add_argument("--newest", type=int, default=3, help="how many transcripts")
    p.add_argument("--no-distill", action="store_true")
    p.add_argument("--session", default=None, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_claude)

    p = subs.add_parser("sessions", help="list journalled sessions")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_sessions)

    p = subs.add_parser("graph", help="entities and relations")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_graph)

    p = subs.add_parser("stats", help="counts, as JSON")
    p.set_defaults(func=cmd_stats)

    p = subs.add_parser("connect", help="start feeding the memory layer")
    p.set_defaults(func=cmd_connect)

    p = subs.add_parser("disconnect", help="stop recording (reading still works)")
    p.set_defaults(func=cmd_disconnect)

    p = subs.add_parser("status", help="connected or not, and what is stored")
    p.set_defaults(func=cmd_status)

    p = subs.add_parser("install", help="wire memoos into your shell")
    p.add_argument("--print-only", action="store_true", help="print the hook, install nothing")
    p.set_defaults(func=cmd_install)

    p = subs.add_parser("doctor", help="check every part of the chain")
    p.add_argument("--container")
    p.set_defaults(func=cmd_doctor)

    p = subs.add_parser("files", help="where each user's data is stored")
    p.set_defaults(func=cmd_files)

    p = subs.add_parser("serve", help="run the dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    return parser


def cmd_default(args) -> int:
    """
    Bare `memoos`: attach this terminal, then show what it knows.

    Running the name of a tool should switch it on, not print a usage
    error. Connecting is idempotent, so typing this twice is harmless.
    """
    from memoos_core import connection
    if connection.connect():
        print(green("connected") + dim(" — this terminal now feeds the memory layer"))
    if not connection.status()["hook_installed"]:
        print(yellow("shell hook not installed") +
              dim(" — run `memoos install` to journal commands automatically"))
    return cmd_status(args)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "func", None) is None:
        args.func = cmd_default
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError as err:
        print(yellow(f"not found: {err}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
