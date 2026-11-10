"""Running Dreambot in the background with launchd, macOS's service manager.

Dreambot installs a per-user *LaunchAgent* (not a system daemon): sending iMessages goes through
the Messages app, which only works inside your logged-in desktop session, and a LaunchAgent is
exactly "run this while I am logged in".

One macOS rule shapes this module. Desktop, Documents and Downloads are guarded: the first time
a background program reads a file there, macOS stops it and asks on screen whether it may. For
a service that is fatal in a quiet way — Python cannot even read its own virtual environment,
so it hangs before its first line of code, showing a dialog to a Mac nobody may be looking at.
A checkout in ``~/Documents`` is the most ordinary place for one to be, so when Dreambot's code
lives in a guarded folder the service runs from a copy kept with Dreambot's own data instead.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "com.dreambot.agent"
#: Folders in the home directory that macOS guards with a consent prompt per program.
GUARDED = ("Desktop", "Documents", "Downloads")


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def interpreter() -> str:
    """The real Python binary behind this environment.

    macOS grants privacy permissions (Full Disk Access, Automation) to this exact file, so it
    is what the service runs and what ``dreambot doctor`` tells people to add.
    """
    return os.path.realpath(sys.executable)


def guarded_folder(path: Path | str) -> str | None:
    """The privacy-guarded place a path is in (``Documents``, an external volume...), or ``None``."""
    absolute = Path(os.path.abspath(path))
    try:
        parts = absolute.relative_to(Path.home()).parts
    except ValueError:
        # Removable and network volumes are guarded in the same way.
        return "an external volume" if absolute.parts[:2] == ("/", "Volumes") else None
    if parts[:2] == ("Library", "Mobile Documents"):
        return "iCloud Drive"
    return parts[0] if parts and parts[0] in GUARDED else None


def code_locations() -> list[Path]:
    """Where the running Dreambot is read from: its environment and its package."""
    import dreambot

    return [Path(sys.prefix), Path(dreambot.__file__).parent]


def code_is_guarded() -> str | None:
    """The guarded folder Dreambot's own code is in, if it is in one."""
    return next((folder for location in code_locations() if (folder := guarded_folder(location))), None)


def project_root() -> Path | None:
    """The checkout this Dreambot runs from, if it runs from one."""
    import dreambot

    root = Path(dreambot.__file__).parent.parent.parent
    return root if (root / "pyproject.toml").is_file() else None


def runtime_dir(home: Path) -> Path:
    return home / "runtime"


def build_runtime(home: Path) -> Path:
    """Install a copy of this Dreambot under ``home`` and return its Python.

    The copy has exactly the packages the checkout is locked to. It is made with ``uv`` from
    the terminal, which is allowed to read the checkout; the service then never has to.
    """
    uv = shutil.which("uv")
    root = project_root()
    if uv is None or root is None:
        raise RuntimeError(
            "Dreambot's code is in a folder macOS guards, and I can't make a copy for the service "
            + ("without `uv` on the PATH." if uv is None else "because this is not a source checkout.")
            + " Move it out of Desktop, Documents and Downloads, or install it with `uv tool install`."
        )
    target = runtime_dir(home)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    environment = target / "venv"
    python = environment / "bin" / "python"

    def run(*arguments: str) -> None:
        result = subprocess.run([uv, *arguments], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            shutil.rmtree(target, ignore_errors=True)
            detail = (result.stderr or result.stdout).strip().splitlines()[-3:]
            raise RuntimeError("could not build the service's copy of Dreambot: " + " ".join(detail))

    run("venv", "--quiet", "--python", interpreter(), str(environment))
    if (root / "uv.lock").is_file():
        requirements = target / "requirements.txt"
        run(
            "export", "--quiet", "--project", str(root), "--frozen", "--no-dev",
            "--no-emit-project", "--no-hashes", "--output-file", str(requirements),
        )  # fmt: skip
        run("pip", "install", "--quiet", "--python", str(python), "--requirements", str(requirements))
        run("pip", "install", "--quiet", "--python", str(python), "--no-deps", str(root))
    else:
        run("pip", "install", "--quiet", "--python", str(python), str(root))
    return python


def build_plist(log_dir: Path, *, python: str | None = None, extra_env: dict[str, str] | None = None) -> dict:
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
        # The Python of a virtual environment, so the right packages are found; it resolves
        # to the interpreter() binary that holds the privacy permissions.
        "ProgramArguments": [python or sys.executable, "-m", "dreambot", "run"],
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


def install(log_dir: Path, home: Path) -> tuple[Path, Path | None]:
    """Write the LaunchAgent and start it.

    Returns the plist path and, when the service had to be given its own copy of Dreambot
    (see the module docstring), where that copy is.
    """
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_loaded():
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    python: str | None = None
    runtime: Path | None = None
    if code_is_guarded():
        python = str(build_runtime(home))
        runtime = runtime_dir(home)
    with path.open("wb") as handle:
        plistlib.dump(build_plist(log_dir, python=python), handle)
    result = _launchctl("bootstrap", _domain(), str(path))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "launchctl bootstrap failed")
    return path, runtime


def uninstall(home: Path | None = None) -> bool:
    """Stop the service and remove the LaunchAgent. Returns False if it was not installed."""
    path = plist_path()
    loaded = is_loaded()
    if loaded:
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    existed = path.exists()
    path.unlink(missing_ok=True)
    if home is not None:
        shutil.rmtree(runtime_dir(home), ignore_errors=True)
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
