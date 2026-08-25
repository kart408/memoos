"""
Folding a long session in as it goes, rather than only when it ends.

Distillation normally happens at `zshexit`. That covers the ordinary
life of a terminal — you type `exit`, or you close the window, and the
session becomes memory. It does not cover the terminal you opened on
Monday and are still using on Friday: its events are journalled and
folded into nothing, and a reboot or a `kill -9` means `zshexit` never
runs at all. Nothing is lost — the journal is the source of truth and
those events stay pending — but the memory is unavailable until the
window finally closes.

So once enough events have piled up, the background logger starts a
distil itself. Two things make that safe to do from the hot path:

It is already in the background. The shell hook spawns and disowns the
logger for every command, so the work here is never between you and your
prompt.

It cannot stack. Distillation runs a language model and takes tens of
seconds; commands arrive far faster than that. Without a lock, a busy
session would spawn a new distil on every command until the machine gave
up. The lock below is a file holding a pid, which is the one form that
survives the thing holding it being killed — a stale lock from a
process that no longer exists is detectable, and gets taken over rather
than deadlocking forever.

Stdlib only, and imported only once the threshold is actually crossed.
"""

import os
import signal
import subprocess
import sys
from typing import Optional

LOCK_DIR = os.path.expanduser("~/.memoos/locks")


def _lock_path(container: str) -> str:
    # Per container: two projects distilling at once is fine, and in fact
    # the thing you want when a session spans several repositories.
    from . import config
    return os.path.join(LOCK_DIR, config.safe_container(container) + ".pid")


def _alive(pid: int) -> bool:
    """Is this process still running, and still ours to reason about?"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists but belongs to someone else. The pid has almost
        # certainly been recycled; treat the lock as stale.
        return False
    return True


def _held_by(container: str) -> Optional[int]:
    """The pid of a live distil for this container, if there is one."""
    path = _lock_path(container)
    try:
        with open(path, encoding="utf-8") as handle:
            pid = int(handle.read().strip())
    except (FileNotFoundError, ValueError):
        return None
    if _alive(pid):
        return pid
    # Stale: whoever held this died without cleaning up.
    try:
        os.remove(path)
    except OSError:
        pass
    return None


def spawn_if_idle(container: str) -> bool:
    """
    Start a background distil for `container` unless one is running.

    Returns whether it started one. The child writes its own pid to the
    lock and removes it on the way out, so a crash leaves a stale lock
    rather than a permanent one.
    """
    if _held_by(container) is not None:
        return False

    os.makedirs(LOCK_DIR, exist_ok=True)
    cli = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "memoos_cli.py")

    # Detached: this logger exits in milliseconds and the distil takes
    # tens of seconds, so it must not be a child that dies with us.
    with open(os.devnull, "wb") as devnull:
        child = subprocess.Popen(
            # `--container` is a global option, so it belongs before the
            # subcommand — after it, argparse rejects the whole call and
            # the child dies before it distils anything.
            [sys.executable, cli, "--container", container,
             "distill", "--lock", _lock_path(container)],
            stdout=devnull, stderr=devnull, start_new_session=True,
        )

    # Written by the parent as well as the child: the child needs a
    # moment to start, and a second logger arriving in that window would
    # otherwise see no lock and spawn a duplicate.
    try:
        with open(_lock_path(container), "w", encoding="utf-8") as handle:
            handle.write(str(child.pid))
    except OSError:
        pass
    return True


def claim(path: Optional[str]) -> None:
    """Called by the distil itself, to own the lock it was handed."""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except OSError:
        pass


def release(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _terminate(signum, frame):  # pragma: no cover - signal path
    raise SystemExit(1)


def install_signal_cleanup() -> None:
    """Make a terminated distil release its lock rather than orphan it."""
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, _terminate)
        except (ValueError, OSError):
            pass
