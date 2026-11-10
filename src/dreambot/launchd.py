"""Running Dreambot in the background with launchd, macOS's service manager.

Dreambot installs a per-user *LaunchAgent* (not a system daemon): sending iMessages goes through
the Messages app, which only works inside your logged-in desktop session, and a LaunchAgent is
exactly "run this while I am logged in".
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "com.dreambot.agent"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def interpreter() -> str:
    """The real Python binary behind this environment.

    macOS grants privacy permissions (Full Disk Access, Automation) to this exact file, so it
    is what the service runs and what ``dreambot doctor`` tells people to add.
    """
    return os.path.realpath(sys.executable)


def build_plist(log_dir: Path, *, extra_env: dict[str, str] | None = None) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        # launchd starts with a bare PATH; `sips`, `osascript` and friends live in these.
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    for name in ("HF_HOME", "HF_HUB_CACHE", "DREAMBOT_HOME", "DREAMBOT_OUTPUT_DIR"):
        if value := os.environ.get(name):
            environment[name] = value
    environment.update(extra_env or {})
    return {
        "Label": LABEL,
        # sys.executable (inside the virtual environment) so the right packages are found;
        # it resolves to the interpreter() binary that holds the privacy permissions.
        "ProgramArguments": [sys.executable, "-m", "dreambot", "run"],
        "RunAtLoad": True,
        # Restart after a crash, but stay stopped after a clean exit (e.g. nothing is configured).
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 20,
        "ProcessType": "Interactive",
        "StandardOutPath": str(log_dir / "dreambot.log"),
        "StandardErrorPath": str(log_dir / "dreambot.log"),
        "EnvironmentVariables": environment,
        "WorkingDirectory": str(Path.home()),
    }


def _launchctl(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/bin/launchctl", *arguments], capture_output=True, text=True, check=False)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(log_dir: Path) -> Path:
    """Write the LaunchAgent and start it. Returns the plist path."""
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_loaded():
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    with path.open("wb") as handle:
        plistlib.dump(build_plist(log_dir), handle)
    result = _launchctl("bootstrap", _domain(), str(path))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "launchctl bootstrap failed")
    return path


def uninstall() -> bool:
    """Stop the service and remove the LaunchAgent. Returns False if it was not installed."""
    path = plist_path()
    loaded = is_loaded()
    if loaded:
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    existed = path.exists()
    path.unlink(missing_ok=True)
    return loaded or existed


def is_loaded() -> bool:
    return _launchctl("print", f"{_domain()}/{LABEL}").returncode == 0


def status() -> dict[str, str | bool | None]:
    result = _launchctl("print", f"{_domain()}/{LABEL}")
    info: dict[str, str | bool | None] = {
        "installed": plist_path().exists(),
        "loaded": result.returncode == 0,
        "pid": None,
        "state": None,
    }
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            key, _, value = line.strip().partition(" = ")
            if key == "pid":
                info["pid"] = value
            elif key == "state":
                info["state"] = value
    return info
