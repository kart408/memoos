"""
The switch: whether the terminal is currently feeding the memory layer.

Attaching memory to a shell should not be a commitment. Some work is
worth remembering and some is not — a scratch directory, someone else's
machine, an afternoon of poking at credentials — and the honest way to
handle that is a switch you can flip, not a config file you have to edit
and re-source.

Two scopes, because they answer different questions:

  global   A flag file under ~/.memoos. Every shell, including ones
           already open, respects it the moment it changes. This is the
           "stop recording me" switch.

  shell    The MEMOOS_OFF environment variable, set by the shell wrapper
           in the current terminal only. This is the "not this window"
           switch, and it cannot be set by a subprocess — only by the
           shell function, which is why it lives in the hook.

Recording is what gets switched off. Reading never does: `recall`,
`search` and the graph keep working while disconnected, because being
unwilling to record today says nothing about what you learned yesterday.

Stdlib only — the hot path imports this on every command.
"""

import os
from typing import Any, Dict

STATE_DIR = os.path.expanduser("~/.memoos")
FLAG_FILE = os.path.join(STATE_DIR, "disconnected")

# Set by the shell wrapper for a single terminal. Any non-empty value
# other than "0"/"false" counts as off.
ENV_FLAG = "MEMOOS_OFF"


def _env_off() -> bool:
    value = os.environ.get(ENV_FLAG, "").strip().lower()
    return bool(value) and value not in {"0", "false", "no"}


def is_connected() -> bool:
    """Should this process journal what it sees?"""
    return not _env_off() and not os.path.exists(FLAG_FILE)


def connect() -> bool:
    """Resume journalling everywhere. Returns True if anything changed."""
    if not os.path.exists(FLAG_FILE):
        return False
    os.remove(FLAG_FILE)
    return True


def disconnect() -> bool:
    """
    Stop journalling everywhere, immediately.

    Existing shells pick this up on their next command — the hook tests
    for the flag rather than caching it, precisely so that turning this
    off does not require opening a new terminal.
    """
    if os.path.exists(FLAG_FILE):
        return False
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(FLAG_FILE, "w", encoding="utf-8") as handle:
        handle.write("memoos is disconnected; delete this file or run `memoos connect`\n")
    return True


def status() -> Dict[str, Any]:
    return {
        "connected": is_connected(),
        "globally_off": os.path.exists(FLAG_FILE),
        "shell_off": _env_off(),
        "flag_file": FLAG_FILE,
        "hook_installed": os.path.exists(os.path.join(STATE_DIR, "memoos.zsh")),
    }
