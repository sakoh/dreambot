"""The background service: where it may run from, and how its private copy is built.

Nothing here talks to launchd or runs uv; both are replaced by recorders.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dreambot import cli, launchd

HOME = Path.home()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (HOME / "Documents" / "work" / "dreambot" / ".venv", "Documents"),
        (HOME / "Desktop" / "dreambot", "Desktop"),
        (HOME / "Downloads" / "dreambot-main" / "src" / "dreambot", "Downloads"),
        (HOME / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "dreambot", "iCloud Drive"),
        (Path("/Volumes/External/dreambot"), "an external volume"),
        (HOME / "src" / "dreambot", None),
        (HOME / "Developer" / "Documents" / "dreambot", None),
        (HOME / "Library" / "Application Support" / "Dreambot" / "runtime", None),
        (HOME / ".local" / "share" / "uv" / "tools" / "dreambot", None),
        (Path("/opt/dreambot"), None),
    ],
)
def test_folders_macos_guards_are_recognised(path: Path, expected: str | None):
    assert launchd.guarded_folder(path) == expected


class Launchctl:
    """Stands in for /bin/launchctl."""

    def __init__(self, loaded: bool = False):
        self.calls: list[tuple[str, ...]] = []
        self.loaded = loaded

    def __call__(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(arguments)
        if arguments[0] == "print":
            return subprocess.CompletedProcess(arguments, 0 if self.loaded else 113, "", "")
        self.loaded = arguments[0] == "bootstrap"
        return subprocess.CompletedProcess(arguments, 0, "", "")


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Launchctl:
    launchctl = Launchctl()
    monkeypatch.setattr(launchd, "_launchctl", launchctl)
    monkeypatch.setattr(launchd, "plist_path", lambda: tmp_path / "LaunchAgents" / "agent.plist")
    return launchctl


def test_a_checkout_outside_guarded_folders_runs_in_place(tmp_path: Path, service: Launchctl, monkeypatch):
    monkeypatch.setattr(launchd, "code_is_guarded", lambda: None)
    monkeypatch.setattr(launchd, "build_runtime", lambda home: pytest.fail("no copy is needed"))
    path, runtime = launchd.install(tmp_path / "logs", tmp_path / "home")
    plist = plistlib.loads(path.read_bytes())
    assert runtime is None and plist["ProgramArguments"] == [sys.executable, "-m", "dreambot", "run"]
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] == {"SuccessfulExit": False}
    assert [call[0] for call in service.calls] == ["print", "bootstrap"]


def test_a_checkout_in_documents_gets_a_copy_for_the_service(tmp_path: Path, service: Launchctl, monkeypatch):
    home = tmp_path / "home"
    copied = home / "runtime" / "venv" / "bin" / "python"
    monkeypatch.setattr(launchd, "code_is_guarded", lambda: "Documents")
    monkeypatch.setattr(launchd, "build_runtime", lambda where: copied)
    service.loaded = True  # an older service is running: it is stopped before the new one starts
    path, runtime = launchd.install(tmp_path / "logs", home)
    plist = plistlib.loads(path.read_bytes())
    assert plist["ProgramArguments"] == [str(copied), "-m", "dreambot", "run"]
    assert runtime == home / "runtime"
    assert [call[0] for call in service.calls] == ["print", "bootout", "bootstrap"]

    (home / "runtime").mkdir(parents=True)
    assert launchd.uninstall(home) is True
    assert not path.exists() and not (home / "runtime").exists(), "the copy goes with the service"


def uv_recorder(monkeypatch: pytest.MonkeyPatch, fail_on: str | None = None) -> list[list[str]]:
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(list(command))
        failed = fail_on is not None and fail_on in command
        return subprocess.CompletedProcess(
            command, 1 if failed else 0, "", "error: no such package\n" if failed else ""
        )

    monkeypatch.setattr(launchd.shutil, "which", lambda name: "/opt/uv/bin/uv")
    monkeypatch.setattr(launchd.subprocess, "run", run)
    return commands


def test_the_copy_is_built_from_the_lockfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "Documents" / "dreambot"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'dreambot'\n")
    (root / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(launchd, "project_root", lambda: root)
    commands = uv_recorder(monkeypatch)
    home = tmp_path / "home"
    (home / "runtime" / "stale").mkdir(parents=True)

    python = launchd.build_runtime(home)
    assert python == home / "runtime" / "venv" / "bin" / "python"
    assert not (home / "runtime" / "stale").exists(), "an older copy is replaced, not added to"
    venv, export, packages, project = commands
    assert (
        venv[:2] == ["/opt/uv/bin/uv", "venv"]
        and launchd.interpreter() in venv
        and venv[-1] == str(python.parent.parent)
    )
    assert export[1] == "export" and {"--frozen", "--no-dev", "--no-emit-project"} <= set(export)
    assert str(root) in export, "the locked versions come from the checkout"
    requirements = home / "runtime" / "requirements.txt"
    assert (
        str(requirements) in export and packages[1:3] == ["pip", "install"] and str(requirements) in packages
    )
    assert project[1:3] == ["pip", "install"] and "--no-deps" in project and project[-1] == str(root)
    assert all(str(python) in command for command in (packages, project))

    # Without a lockfile the project is installed with whatever versions resolve.
    (root / "uv.lock").unlink()
    commands.clear()
    launchd.build_runtime(home)
    assert [command[1] for command in commands] == ["venv", "pip"] and commands[1][-1] == str(root)


def test_a_failed_copy_leaves_nothing_behind_and_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "dreambot"
    root.mkdir()
    (root / "pyproject.toml").write_text("")
    monkeypatch.setattr(launchd, "project_root", lambda: root)
    uv_recorder(monkeypatch, fail_on="pip")
    with pytest.raises(RuntimeError, match=r"could not build the service's copy.*no such package"):
        launchd.build_runtime(tmp_path / "home")
    assert not (tmp_path / "home" / "runtime").exists()

    monkeypatch.setattr(launchd.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="without `uv` on the PATH"):
        launchd.build_runtime(tmp_path / "home")
    monkeypatch.setattr(launchd.shutil, "which", lambda name: "/opt/uv/bin/uv")
    monkeypatch.setattr(launchd, "project_root", lambda: None)
    with pytest.raises(RuntimeError, match="not a source checkout"):
        launchd.build_runtime(tmp_path / "home")


def test_service_install_explains_the_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DREAMBOT_HOME", str(tmp_path / "home"))
    runtime = tmp_path / "home" / "runtime"
    monkeypatch.setattr(launchd, "code_is_guarded", lambda: "Documents")
    monkeypatch.setattr(launchd, "install", lambda log_dir, home: (tmp_path / "agent.plist", runtime))
    result = CliRunner().invoke(cli.app, ["service", "install"])
    said = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "your Documents folder" in said and "Making a copy for the service" in said
    assert "run `dreambot service install` again" in said

    def broken(log_dir, home):
        raise RuntimeError("could not build the service's copy of Dreambot: boom")

    monkeypatch.setattr(launchd, "install", broken)
    result = CliRunner().invoke(cli.app, ["service", "install"])
    assert result.exit_code == 1 and "boom" in " ".join(result.output.split())
