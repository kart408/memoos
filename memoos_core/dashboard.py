"""
The dashboard's lifetime is the terminal's lifetime.

`memoos serve` runs in the foreground of the window you started it from,
which reads like a process that dies with that window. It does not
always. Closing a terminal sends SIGHUP to its foreground process group,
and uvicorn installs handlers for SIGINT and SIGTERM and nothing else —
so the shutdown depends on a signal arriving that a force-quit, a lost
tty, or a detached start never sends. When it does not arrive the server
is orphaned onto init, still bound to its port. The next `memoos start`
then fails with "address already in use" against a dashboard nobody can
see, whose browser tab still answers, from a session that ended days ago.

So the lifetime is enforced rather than assumed, from two directions:

  signals   SIGHUP is handled, which turns the ordinary close from a
            default kill that skips teardown into a graceful shutdown
            that gets to run it.

  parent    A watchdog thread reads getppid(). When the shell that
            started us dies the kernel reparents us, and that change is
            the one notice that cannot go missing. This is the half that
            catches everything the signal does not.

And the tab goes with it. A dashboard whose server has stopped is a page
that will not reload, and leaving it open leaves a corpse on the screen.
Shutdown closes the browser tabs pointing at it — only those: the script
matches this dashboard's own origin, in browsers that are already
running, and touches nothing else on the way past.

Closing a tab is macOS-only and best-effort, both on purpose. It is
scripted through osascript because there is no equivalent that works
everywhere, and a browser that will not be scripted — automation
permission refused, an unresponsive window, a browser we do not know —
must cost the shutdown nothing. Every failure here is silent, because
the server stopping is the part that matters and it has already
happened by the time any of this runs.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import List, Optional

# The same dashboard, spelled differently. We open the tab with one of
# these, but a page reached by typing `localhost:8000` is the same
# server and has earned the same cleanup.
LOOPBACK = ("127.0.0.1", "localhost", "0.0.0.0", "::1", "[::1]")

# Everything built on Chromium shares one scripting dictionary, and
# Safari's is close enough that the same script drives both: windows,
# tabs, `URL of tab`, `close`. Names are as they appear in `ps -Ac`,
# which is how we tell what is running without asking a browser.
BROWSERS = (
    "Google Chrome",
    "Google Chrome Canary",
    "Chromium",
    "Brave Browser",
    "Microsoft Edge",
    "Vivaldi",
    "Safari",
)

# Iterating tabs downward matters: closing tab 3 renumbers everything
# after it, so a forward loop skips the tab that slid into the gap. The
# inner `try`s are for windows that have no tabs — a Safari downloads
# window, a Chrome app window — which would otherwise abort the sweep
# for every window behind them.
_CLOSE_TABS = '''
on run
  set targets to __TARGETS__
  set closedCount to 0
  try
    if application __APP__ is running then
      tell application __APP__
        repeat with w in (every window)
          try
            set i to (count of tabs of w)
            repeat while i > 0
              try
                set u to ((URL of tab i of w) as text)
                repeat with t in targets
                  if u is t or u starts with (t & "/") then
                    close tab i of w
                    set closedCount to closedCount + 1
                    exit repeat
                  end if
                end repeat
              end try
              set i to i - 1
            end repeat
          end try
        end repeat
      end tell
    end if
  end try
  return closedCount as text
end run
'''


def _applescript_string(value: str) -> str:
    """A Python string as an AppleScript literal, quoted and escaped."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def origins(host: str, port: int) -> List[str]:
    """
    Every URL prefix that means "this dashboard".

    A loopback bind answers to three names and the browser records
    whichever one was typed, so closing only the one we opened would
    leave a tab the user reached by hand. A bind to a real interface
    gets exactly its own address and no aliasing guesswork.
    """
    hosts = list(LOOPBACK) if host in LOOPBACK else [host]
    seen: List[str] = []
    for name in hosts:
        origin = f"http://{name}:{port}"
        if origin not in seen:
            seen.append(origin)
    return seen


def _running_apps() -> set:
    """
    Which applications are up, in one call and without asking them.

    Scripting a browser to find out whether it is worth scripting costs
    an osascript launch each, and on a Mac with automation permissions
    unset it costs a consent dialog each — during shutdown, which is the
    worst possible moment to interrupt someone. `ps` answers for free.
    """
    try:
        result = subprocess.run(["ps", "-Ac", "-o", "comm="],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def close_tabs(host: str, port: int) -> int:
    """
    Close the browser tabs showing this dashboard. Returns how many.

    Never raises, and never launches a browser to do it: a browser that
    is not running has no tab to close, and starting one in order to
    find that out would be the rudest possible way to end a session.
    """
    if sys.platform != "darwin":
        return 0

    targets = "{" + ", ".join(_applescript_string(o) for o in origins(host, port)) + "}"
    running = _running_apps()
    closed = 0

    for app in BROWSERS:
        if app not in running:
            continue
        script = (_CLOSE_TABS
                  .replace("__TARGETS__", targets)
                  .replace("__APP__", _applescript_string(app)))
        try:
            result = subprocess.run(["osascript", "-e", script],
                                    capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        # A refused automation prompt, a browser mid-crash, a dictionary
        # that turned out not to match: all of them mean "no tab closed
        # here", none of them mean the shutdown went wrong.
        if result.returncode != 0:
            continue
        try:
            closed += int(result.stdout.strip() or 0)
        except ValueError:
            continue

    return closed


def stop_when_terminal_closes(stop, interval: float = 1.0) -> None:
    """
    Call `stop` once the terminal that started this process is gone.

    `stop` is a plain callback rather than a signal, and that is the
    whole point. Interrupting the main thread looks like the obvious
    move and is a trap: `os.kill(getpid(), SIGINT)` hands the signal to
    whichever thread will take it, and a process started as a background
    job inherits SIGINT set to SIG_IGN from its shell — so Python never
    installs a handler for it and the interrupt lands nowhere at all. A
    callback that flips uvicorn's own `should_exit` is checked by its
    main loop every tick and cannot be ignored by anybody.

    Two triggers, because either one alone has a hole in it:

      SIGHUP    the ordinary close. Handled only so that the default
                action — die here, now, skipping the teardown — does not
                get to run. This is the fast path when it arrives.

      getppid   the backstop. When SIGHUP never comes, the reparenting
                still does; nothing can suppress it, and polling for it
                costs one syscall a second.
    """
    stopped = threading.Event()

    def once() -> None:
        if not stopped.is_set():
            stopped.set()
            stop()

    def hangup(signum, frame):   # noqa: ARG001 - signal handler signature
        once()

    try:
        signal.signal(signal.SIGHUP, hangup)
    except (AttributeError, OSError, ValueError):
        # No SIGHUP here, or not the main thread. The watchdog below
        # covers the same ground, a second later.
        pass

    original = os.getppid()
    if original <= 1:
        # Already owned by init, so there is no terminal to outlive and
        # no ppid change that will ever come. Watching would be a thread
        # that never fires.
        return

    def watch() -> None:
        while os.getppid() == original and not stopped.is_set():
            time.sleep(interval)
        once()

    threading.Thread(target=watch, name="memoos-parent-watch",
                     daemon=True).start()


def port_in_use(host: str, port: int) -> bool:
    """
    Would uvicorn fail to bind here?

    Asked the way uvicorn asks it, SO_REUSEADDR included, so that a
    socket sitting in TIME_WAIT is not reported as a server that is
    still there.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return True
    return False


def listening_pid(port: int) -> Optional[int]:
    """Who is holding the port, if we can find out cheaply."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    for token in result.stdout.split():
        try:
            return int(token)
        except ValueError:
            continue
    return None
