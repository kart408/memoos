#!/usr/bin/env python3
"""
memoos — memory for your terminal.

A shell forgets everything when you close it. This puts a memory layer
underneath it: every command, git action and Claude Code prompt is
journalled instantly, folded into structured memories later, and handed
back when you open a new terminal in the same project.

    memoos install        wire it into your shell
    memoos start          ollama + the dashboard, from anywhere
    memoos recall         what was I doing here?
    memoos context TASK   what an agent should know before starting
    memoos signals        what has gone wrong here before, and what fixed it
    memoos trace ID       where a memory came from
    memoos distill        fold this session into memory
    memoos ingest FILE    teach it about you or the project
    memoos graph          what it knows, and how it connects
    memoos clear          forget a project, or every project
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
    _container = container_for(_cwd)
    _journal = Journal()
    _journal.record(
        "command", _command, container=_container,
        session_id=os.environ.get("MEMOOS_SESSION", "shell"),
        cwd=_cwd, exit_code=_exit_code,
    )

    # A terminal you never close never reaches `zshexit`, so everything it
    # journals waits forever. Once enough has piled up, fold some of it in
    # now. This runs in the process the hook already backgrounded and
    # disowned, so the prompt pays nothing for it — one COUNT(*) on an
    # indexed column, and only then a spawn.
    from memoos_core import config as _config
    if _config.AUTODISTILL_AFTER > 0:
        try:
            if _journal.pending_count(_container) >= _config.AUTODISTILL_AFTER:
                from memoos_core.autodistill import spawn_if_idle
                spawn_if_idle(_container)
        except Exception:
            # Journalling has already succeeded, and it is the part that
            # must not fail. A problem starting the distil costs a delay,
            # never an event.
            pass
    sys.exit(0)

import argparse  # noqa: E402
import threading  # noqa: E402
import webbrowser  # noqa: E402
import json  # noqa: E402
from typing import Dict, List, Sequence  # noqa: E402

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
red = lambda s: paint(s, "31")
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


def type_labels(types: Sequence[str]) -> List[str]:
    """
    `[type]` tags for a column, padded to the widest one on screen.

    Padded rather than truncated. Cutting to a fixed four characters
    lined the column up and printed `[even]` and `[prob]`, which are not
    words — and `[pref]` and `[prob]` differ by one letter at a glance,
    which is the opposite of what a type label is for. Padding is applied
    here, before colour: ANSI codes are invisible but `ljust` counts them.
    """
    width = max((len(t) for t in types), default=0) + 2
    return [("[" + t + "]").ljust(width) for t in types]


def entity_names(entities: Sequence) -> List[str]:
    """
    The distinct nodes a distil touched, in the order it touched them.

    One entry arrives per *attachment*, so a node mentioned by three
    memories came back three times and printed as `start_demo.sh,
    start_demo.sh` — which reads as two nodes sitting side by side when
    the graph holds one, the exact confusion the store works to avoid.

    Deduped by id rather than by name, because the node is the thing
    being reported: `upsert_entity` matches on the normalised name, so
    two spellings that normalise alike are one node arriving twice under
    different labels.
    """
    seen: Dict[str, str] = {}
    for entity in entities:
        seen.setdefault(entity.id, entity.name)
    return list(seen.values())


def rejection_note(result: Dict) -> str:
    """
    What extraction's guards threw away, or "" if they threw away nothing.

    Zero memories has two unrelated causes. The model found nothing worth
    saying, which is what an uneventful session looks like and is fine —
    or it said things and every one was rejected here, which means the
    prompt and its own validator disagree and somebody has to look. Both
    printed the same line, so the second was invisible: a guard checking
    for the literal word "user" silently dropped four of the extraction
    prompt's own worked examples, and nothing anywhere said a candidate
    had ever existed.
    """
    candidates = result.get("candidates", 0)
    reasons = result.get("rejected_summary", "")
    if not candidates or not reasons:
        return ""
    kept = len(result.get("created", []))
    if kept:
        return dim(f"  {candidates} candidates, {kept} kept — dropped: {reasons}")
    return (yellow(f"  {candidates} candidates, none kept") +
            dim(f" — {reasons}"))


def search_window(limit: int, ceiling: int) -> int:
    """
    How many candidates to fetch to fill a page of `limit` after gating.

    Fetching exactly `limit` and then removing the weak ones is what made
    `--limit 8` return 7, or 2: the bar ate slots that results further
    down were ready to fill. Widening the slice is close to free, because
    `search` scores a fixed candidate pool and `top_k` only slices the
    ranked end of it — the retrieval work is already done. `ceiling` is
    that pool, past which there is nothing left to ask for.
    """
    return max(limit, min(limit * 4, ceiling))


def gated_page(hits: Sequence, passing: Sequence, limit: int) -> tuple:
    """
    The page to show, and how many results the bar took off it.

    The count is over the window the user asked for, not over everything
    fetched to fill it. Saying "18 below the bar" because the tail of a
    60-deep pool is weak tells them nothing they asked about; the useful
    number is how many results that would have been on *this page* were
    removed from it.
    """
    kept = {result.memory.id for result in passing}
    suppressed = sum(1 for result in hits[:limit]
                     if result.memory.id not in kept)
    return list(passing[:limit]), suppressed


def suppression_note(suppressed: int, *, any_shown: bool) -> str:
    """
    What the confidence bar removed, or "" if it removed nothing.

    Suppressed is not the same as absent. `recall` is the command you
    reach for when you are asking why the store said what it said, so
    the count the bar took stays on screen and `--all` brings the
    results themselves back — hiding them silently would trade one
    quiet failure for another.

    Dim in both cases, unlike `rejection_note`'s warning, because these
    are different events wearing the same shape. Every candidate being
    rejected means the prompt and its own validator disagree and
    somebody has to look; everything falling below the bar is the
    *correct* answer to a question this container knows nothing about.
    """
    if not suppressed:
        return ""
    if any_shown:
        return dim(f"  {suppressed} more below the confidence bar — `--all`")
    return dim(f"  {suppressed} below the confidence bar — `--all` to see them")


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
        from memoos_core import config
        from memoos_core.retrieval import in_scope
        from memoos_core.terminal import TerminalMemory
        query = " ".join(args.query)
        memory = TerminalMemory(container=container)

        # `--limit` is how many results to *show*, so the bar has to run
        # before the count is taken. Asking `search` for exactly the
        # limit and then removing the weak ones meant a `--limit 8` that
        # returned 7, or 2 — the bar ate slots that results further down
        # were ready to fill.
        #
        # Widening the slice is close to free: `search` scores a fixed
        # candidate pool (VECTOR_CANDIDATES + KEYWORD_CANDIDATES) and
        # `top_k` only slices the ranked end of it, so the retrieval work
        # is already done. The pool is the ceiling because past it there
        # is nothing more to ask for.
        ceiling = config.VECTOR_CANDIDATES + config.KEYWORD_CANDIDATES
        hits = memory.memo.search(
            query, top_k=search_window(args.limit, ceiling), touch=False)

        # `search` returns its nearest neighbours however distant — that
        # is what nearest-neighbour means, and it is the right contract
        # for a caller that can act on the scores. A person reading three
        # lines in a terminal is not that caller: asked something this
        # container knows nothing about, it answered with its three
        # least-bad guesses and a number beside each one. The bars that
        # turn that into "I don't know about that" already existed and
        # were calibrated; only `context` was using them.
        if args.all:
            shown, suppressed = hits[:args.limit], 0
        else:
            shown, suppressed = gated_page(
                hits, in_scope(hits, query=query), args.limit)

        rule(f"{container} · recall")
        if not shown:
            print(dim("  nothing relevant remembered yet"))
            note = suppression_note(suppressed, any_shown=False)
            if note:
                print(note)
            return 0
        for hit in shown:
            why = ",".join(hit.matched_by)
            sim = f"{hit.vector_score:.2f}" if hit.vector_score is not None else " -- "
            print(f"  {cyan(sim)} {hit.memory.text}  {dim('(' + why + ')')}")
        note = suppression_note(suppressed, any_shown=True)
        if note:
            print(note)
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
        tags = type_labels([m["memory_type"] for m in memories])
        for tag, m in zip(tags, memories):
            print(f"  {magenta(tag)} {m['text']}")
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


def cmd_context(args) -> int:
    """
    The handover: what an agent should know before it starts this task.

    Prints the context block and stops. MemoOS does not answer — the
    agent that asked is the thing that knows what you are trying to do,
    and this is the part it was missing.

    `--json` for a program on the other end, plain text for a human or
    for pasting straight into a prompt.
    """
    from memoos_core.terminal import TerminalMemory

    container = args.container or container_for()
    task = " ".join(args.task).strip()
    if not task:
        print(yellow("context for what?"), file=sys.stderr)
        return 1

    result = TerminalMemory(container=container).memo.context_for(
        task, top_k=args.limit, include_stale=args.include_stale)

    if args.json:
        print(json.dumps({
            "container": result["container"],
            "query": result["query"],
            "plan": result["plan"],
            "context": result["context"],
            "memories": [
                {"id": h.memory.id, "text": h.memory.text,
                 "type": h.memory.memory_type.value,
                 "score": h.score, "matched_by": h.matched_by,
                 "current": h.memory.is_current(),
                 "importance": h.memory.importance,
                 "confidence": h.memory.confidence}
                for h in result["results"]
            ],
            # Past signals, handed over rather than acted on.
            "signals": [e.as_dict() for e in result.get("signals", [])],
        }, indent=2))
        return 0 if result["results"] else 1

    plan = result["plan"]
    if plan["concepts"] and not args.quiet:
        # Showing the expansion matters: it is the one retrieval stage
        # that rewrites the query, and a search that silently searched
        # for something else is a search you cannot debug.
        print(f"\n{dim('searching for')} {', '.join(plan['terms'])}"
              f"{dim(' + ')}{cyan(', '.join(plan['concepts']))}")

    if not result["results"]:
        print(dim(f"\n  nothing remembered about that in {container}\n"))
        return 1

    print()
    for line in result["context"].splitlines():
        print(f"  {line}" if line else "")
    if not args.quiet:
        print(dim("\n  hand this to your agent — memoos stops here"))
    print()
    return 0


def cmd_signals(args) -> int:
    """
    What has gone wrong in this project, and what fixed it.

    Reported, not acted on. Nothing here retires a memory or reorders a
    ranking — deciding what to do about a known failure needs to know
    what you are trying to do, and that is not something a memory layer
    can see.
    """
    from memoos_core.terminal import TerminalMemory

    container = args.container or container_for()
    episodes = TerminalMemory(container=container).memo.signals(limit=args.limit)
    if not episodes:
        print(dim(f"\n  nothing has gone wrong in {container} yet — "
                  f"or nothing was distilled as a problem\n"))
        return 0

    unresolved = [e for e in episodes if not e.resolved]
    resolved = [e for e in episodes if e.resolved]

    if unresolved:
        rule(f"{container} · open problems")
        for episode in unresolved:
            print(f"  {yellow('!')} {episode.problem.text}")
            print(f"    {dim(episode.problem.id)}")

    if resolved and not args.open_only:
        rule("known failures, and what fixed them")
        for episode in resolved:
            print(f"  {red('✗')} {episode.problem.text}")
            for solution in episode.solutions:
                print(f"    {green('↳')} {solution.text}")
            if episode.shared:
                print(f"    {dim('both mention: ' + ', '.join(episode.shared))}")
    print()
    return 0


def cmd_trace(args) -> int:
    """Follow a memory back to the raw input it was distilled from."""
    from memoos_core.terminal import TerminalMemory

    container = args.container or container_for()
    trail = TerminalMemory(container=container).memo.source(args.memory_id)
    if trail is None:
        print(yellow(f"no memory {args.memory_id!r} in {container}"), file=sys.stderr)
        return 1

    memory = trail["memory"]
    rule(f"{magenta('[' + memory.memory_type.value + ']')} {memory.text}")
    window = clock(memory.valid_from.isoformat()) if memory.valid_from else "?"
    until = (clock(memory.valid_until.isoformat())
             if memory.valid_until else green("current"))
    print(f"  {dim('valid'):<12} {window} → {until}")
    print(f"  {dim('confidence'):<12} {memory.confidence:.2f}"
          f"   {dim('importance')} {memory.importance:.2f}")

    document = trail["document"]
    if document:
        print(f"  {dim('from'):<12} {document.title or document.source}"
              f"{dim(' · ' + (document.uri or document.source))}")
    if trail["chunk"]:
        rule("passage the model read")
        for line in trail["chunk"].splitlines()[:12]:
            print(f"  {dim(line)}")

    events = trail["events"]
    if events:
        rule(f"{len(events)} journalled event(s) behind it")
        for event in events[:args.limit]:
            code = event.get("exit_code")
            flag = red("✗") if code not in (0, None) else green("✓")
            print(f"  {flag} {clock(event['created_at'])}  {event['text']}")
    print()
    return 0


def cmd_clear(args) -> int:
    """
    Forget a project entirely, or every project.

    One container is one file, so this is a delete rather than a sweep
    across tables — which is the honest thing to do. A DELETE would
    leave the journal, the entities and the vectors to be cleaned up
    separately, and missing one of them is how a "cleared" project comes
    back still knowing things.

    Destructive and unrecoverable, so it names what it is about to
    remove and waits for you to agree.
    """
    from memoos_core import config

    if args.all:
        targets = config.stored_containers()
    else:
        container = args.container or container_for()
        targets = [container] if os.path.exists(
            config.db_path(container=container)) else []

    if not targets:
        print(dim("\n  nothing stored to clear\n"))
        return 0

    print(f"\n{bold('about to delete')}")
    total = 0
    for name in targets:
        path = config.db_path(container=name)
        size = sum(os.path.getsize(path + suffix)
                   for suffix in ("", "-wal", "-shm")
                   if os.path.exists(path + suffix))
        total += size
        counts = quick.counts(name)
        tally = f"{counts['memories']} memories · {counts['entities']} entities"
        print(f"  {cyan(name):<28} {_human(size):>9}   {dim(tally)}")
    print(f"\n  {len(targets)} file(s), {_human(total)} — {yellow('this cannot be undone')}")

    if not args.yes:
        try:
            answer = input("  type 'yes' to confirm: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "yes":
            print(dim("  cancelled\n"))
            return 1

    for name in targets:
        path = config.db_path(container=name)
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
    print(f"  {green('cleared')} {len(targets)} project(s)\n")
    return 0


def cmd_distill(args) -> int:
    from memoos_core import autodistill

    # Only set when the background logger started this one. Owning the
    # lock for the whole run is what stops a busy session from spawning a
    # second distil on top of this one.
    lock = getattr(args, "lock", None)
    autodistill.claim(lock)
    if lock:
        autodistill.install_signal_cleanup()
    try:
        return _distill(args)
    finally:
        autodistill.release(lock)


def _distill(args) -> int:
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
            print(rejection_note(result) or
                  dim(f"  {result.get('skipped', 'nothing to do')} "
                      f"({result['distilled']} events)"))
        return 1 if failed else 0

    rule(f"{memory.container} · distilled {result['distilled']} events")
    for m in result["created"]:
        print(f"  {green('+')} {magenta('[' + m.memory_type.value + ']')} {m.text}")
    for m in result.get("superseded", []):
        print(f"  {yellow('~')} superseded: {dim(m.text)}")
    names = entity_names(result.get("entities", []))
    if names:
        print(f"\n  entities: {', '.join(names)}")
    note = rejection_note(result)
    if note:
        print(note)
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
        note = rejection_note(result)
        if note:
            print(note)
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
            mark = yellow("!") if not fatal else red("✗")
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
         "" if state["connected"] else "disconnected — run `memoos connect`",
         fatal=False)

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
        print(dim("  something above is broken — closing a terminal will not "
                  "fold its session into memory\n"))
    elif not state["connected"]:
        print(dim("  the chain is intact, but recording is switched off — "
                  "nothing new is being journalled\n"))
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
        tally = f"{counts['memories']} memories · {counts['entities']} entities"
        print(f"  {cyan(name):<28} {_human(size):>9}   {dim(tally)}")

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


def cmd_start(args) -> int:
    """
    Everything needed to see MemoOS running, from any directory.

    `start_demo.sh` does this too, but a script can only be run by its
    path — and a new Terminal window opens in your home directory, where
    `./start_demo.sh` means a file that is not there. `memoos` is on your
    PATH everywhere the shell hook is loaded, so this is the form that
    works from wherever you happen to be.

    The difference from `serve` is Ollama: this starts it if it is not
    already up, and refuses if the extraction model is missing, because
    without it a closing terminal has nothing to fold its session with.
    """
    import shutil as _shutil
    import subprocess
    import time

    if not _ollama_reachable():
        if not _shutil.which("ollama"):
            print("\n  " + red("ollama is not installed") +
                  dim(" — see https://ollama.com/download\n"))
            return 1
        print(dim("  starting ollama..."))
        with open("/tmp/ollama.log", "ab") as log:
            # Detached: it must outlive this command, which exits as soon
            # as you stop the server.
            subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log,
                             start_new_session=True)
        for _ in range(20):
            if _ollama_reachable():
                break
            time.sleep(0.5)
        else:
            print("\n  " + red("ollama did not come up") +
                  dim(" — check /tmp/ollama.log\n"))
            return 1

    from memoos_core import config
    if not _has_model(config.EXTRACT_MODEL):
        print("\n  " + red(f"extraction model {config.EXTRACT_MODEL} is missing"))
        print(dim(f"  run: ollama pull {config.EXTRACT_MODEL}\n"))
        return 1

    print(f"  {green('ollama')} {dim(config.OLLAMA_URL)}   "
          f"{green('model')} {dim(config.EXTRACT_MODEL)}")
    return cmd_serve(args)


def _has_model(wanted: str) -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{_ollama_url()}/api/tags", timeout=3) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return False
    # `mistral` should match `mistral:latest`, and the other way round.
    stem = wanted.split(":")[0]
    return any(name == wanted or name.split(":")[0] == stem for name in names)


def cmd_serve(args) -> int:
    """
    Run the dashboard, from wherever you happen to be standing.

    The shell hook puts `memoos` on your PATH globally, so every command
    has to work from any directory — but uvicorn resolves "api:app" as an
    *import*, which only succeeds from the project root. Without the path
    insert below, `memoos serve` worked in ~/memoos and nowhere else,
    which is the opposite of what a globally-available command should do.
    """
    import uvicorn

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    # Not fatal. The dashboard reads memories, entities and the graph
    # straight out of SQLite, all of which work with Ollama down; only
    # distilling and semantic search need it. Better to say so and serve
    # than to refuse over a dependency half the page does not use.
    if not _ollama_reachable():
        print(yellow("  ollama unreachable") +
              dim(f" at {_ollama_url()} — the dashboard will read fine, "
                  "but distilling needs it"))

    url = f"http://{args.host}:{args.port}/"
    print(dim(f"dashboard → {url}"))
    if not args.no_open:
        # Deferred until the server is actually up, otherwise the browser
        # races uvicorn's bind and lands on a connection error.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
    return 0


def _ollama_url() -> str:
    from memoos_core import config
    return config.OLLAMA_URL


def _ollama_reachable() -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{_ollama_url()}/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


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
    p.add_argument("--limit", type=int, default=8,
                   help="how many results to show (after the confidence bar)")
    p.add_argument("--all", action="store_true",
                   help="include results the store is not confident about")
    p.set_defaults(func=cmd_recall)

    p = subs.add_parser("context", help="what an agent should know before this task")
    p.add_argument("task", nargs="+")
    p.add_argument("--json", action="store_true",
                   help="machine-readable, for an agent on the other end")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--include-stale", action="store_true",
                   help="keep memories whose validity window has closed")
    p.add_argument("--quiet", action="store_true", help="the block on its own")
    p.set_defaults(func=cmd_context)

    p = subs.add_parser("signals", help="what has gone wrong here before")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--open", dest="open_only", action="store_true",
                   help="only problems with no recorded fix")
    p.set_defaults(func=cmd_signals)

    p = subs.add_parser("trace", help="where a memory came from")
    p.add_argument("memory_id")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_trace)

    p = subs.add_parser("clear", help="forget a project, or every project")
    # No --container here: it is a global option, and redefining it on a
    # subparser silently overwrites the global value with None. For a
    # command that deletes files that meant `memoos --container other
    # clear` erasing the project you were standing in instead.
    p.add_argument("--all", action="store_true", help="every project, not just this one")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    p.set_defaults(func=cmd_clear)

    p = subs.add_parser("distill", help="fold journalled events into memory")
    p.add_argument("--session", help="only this session id")
    p.add_argument("--lock", help=argparse.SUPPRESS)   # set by the auto-distil
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
    p.set_defaults(func=cmd_doctor)

    p = subs.add_parser("files", help="where each user's data is stored")
    p.set_defaults(func=cmd_files)

    p = subs.add_parser("start", help="start ollama and the dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    p.set_defaults(func=cmd_start)

    p = subs.add_parser("serve", help="run the dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
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
