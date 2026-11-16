"""The command line: setup and doctor, without a network, a Keychain or the Messages database."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from dreambot import cli, launchd, secrets

TOKEN = "123456789:AAExampleExampleExampleExample_12345"
runner = CliRunner()


def said(result) -> str:
    """A command's output on one line (the console wraps long lines wherever it likes)."""
    return " ".join(result.output.split())


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty Dreambot home and model cache, and a Keychain that only remembers."""
    monkeypatch.setenv("DREAMBOT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    (tmp_path / "hub").mkdir()
    stored: dict[str, str] = {}
    monkeypatch.setattr(secrets, "set_token", lambda name, value: stored.__setitem__(name, value))
    monkeypatch.setattr(secrets, "get_token", lambda name: stored.get(name))
    monkeypatch.setattr(secrets, "token_source", lambda name: "keychain" if name in stored else None)
    monkeypatch.setattr(cli, "_stored_for_test", stored, raising=False)
    # ...and no launchd: these tests must never see or restart a real background service.
    launchctl: list[tuple[str, ...]] = []

    def no_service(*arguments: str) -> subprocess.CompletedProcess[str]:
        launchctl.append(arguments)
        return subprocess.CompletedProcess(arguments, 113, "", "Could not find service")

    monkeypatch.setattr(launchd, "_launchctl", no_service)
    monkeypatch.setattr(cli, "_launchctl_for_test", launchctl, raising=False)
    return tmp_path / "home"


def telegram_answers(
    monkeypatch: pytest.MonkeyPatch, status: int, body: dict | None, error: Exception | None = None
):
    asked: list[str] = []

    def get(url: str, **kwargs) -> httpx.Response:
        asked.append(url)
        if error is not None:
            raise error
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    return asked


def test_a_telegram_token_is_checked_before_it_is_saved(home: Path, monkeypatch: pytest.MonkeyPatch):
    asked = telegram_answers(
        monkeypatch, 200, {"ok": True, "result": {"id": 1, "is_bot": True, "username": "my_dream_bot"}}
    )
    result = runner.invoke(cli.app, ["setup", "telegram"], input=f"{TOKEN}\n")
    assert result.exit_code == 0, result.output
    assert "That is @my_dream_bot" in said(result) and "https://t.me/my_dream_bot" in said(result)
    assert asked == [f"https://api.telegram.org/bot{TOKEN}/getMe"]
    assert cli._stored_for_test == {"telegram": TOKEN}
    assert tomllib.loads((home / "config.toml").read_text())["telegram"] == {"enabled": True}
    assert TOKEN not in result.output and TOKEN not in said(result), "a token is never echoed"
    assert "Discord" not in said(result) and "iMessage" not in said(result), (
        "only the step that was asked for"
    )


def test_a_rejected_or_malformed_token_is_not_saved(home: Path, monkeypatch: pytest.MonkeyPatch):
    asked = telegram_answers(
        monkeypatch, 401, {"ok": False, "error_code": 401, "description": "Unauthorized"}
    )
    result = runner.invoke(cli.app, ["setup", "telegram"], input=f"{TOKEN}\n")
    assert "did not accept that token: Unauthorized" in said(result) and "Nothing was saved" in said(result)
    assert cli._stored_for_test == {} and not (home / "config.toml").exists()

    asked.clear()
    result = runner.invoke(cli.app, ["setup", "telegram"], input="not-a-token\n")
    assert "bot tokens look like" in said(result) and asked == [], "nothing is sent for an obvious typo"
    assert cli._stored_for_test == {}

    result = runner.invoke(cli.app, ["setup", "telegram"], input="\n")
    assert result.exit_code == 0 and cli._stored_for_test == {}, "Enter skips the step"


def test_an_unreachable_telegram_does_not_block_setup_or_leak_the_token(
    home: Path, monkeypatch: pytest.MonkeyPatch
):
    error = httpx.ConnectError(f"could not connect to https://api.telegram.org/bot{TOKEN}/getMe")
    telegram_answers(monkeypatch, 0, None, error=error)
    result = runner.invoke(cli.app, ["setup", "telegram"], input=f"{TOKEN}\n")
    assert result.exit_code == 0 and "could not be reached to check it (ConnectError)" in said(result)
    assert cli._stored_for_test == {"telegram": TOKEN}
    assert TOKEN not in result.output and TOKEN not in said(result)


def test_setup_knows_its_frontends(home: Path):
    result = runner.invoke(cli.app, ["setup", "whatsapp"])
    assert result.exit_code == 2 and "telegram, discord, imessage, web" in said(result)


def test_setup_skips_everything_on_enter_and_never_opens_messages(
    home: Path, monkeypatch: pytest.MonkeyPatch
):
    from dreambot.frontends.imessage import chatdb

    def forbidden(self) -> None:
        raise AssertionError("the Messages database was opened")

    monkeypatch.setattr(chatdb.ChatDB, "open", forbidden)
    result = runner.invoke(cli.app, ["setup"], input="\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert not (home / "config.toml").exists()
    for heading in ("Telegram", "Discord", "iMessage", "Web app", "Start it"):
        assert heading in said(result)


def test_doctor_leaves_messages_alone_until_imessage_is_enabled(home: Path, monkeypatch: pytest.MonkeyPatch):
    from dreambot import launchd
    from dreambot.frontends.imessage import chatdb

    opened: list[bool] = []
    monkeypatch.setattr(chatdb.ChatDB, "open", lambda self: opened.append(True))
    monkeypatch.setattr(chatdb.ChatDB, "close", lambda self: None)
    monkeypatch.setattr(launchd, "status", lambda: {"installed": False, "loaded": False})

    result = runner.invoke(cli.app, ["doctor"])
    assert "iMessage: off" in said(result) and opened == []
    assert "Language model: none" in said(result) and "dreambot models pull describe" in said(result)

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text('[imessage]\nenabled = true\nmode = "dedicated"\n')
    result = runner.invoke(cli.app, ["doctor"])
    assert "iMessage: enabled (dedicated mode), Messages database is readable" in said(result) and opened == [
        True
    ]


DISCORD_TOKEN = "MTIzNDU2Nzg5MDEyMzQ1Njc4OQ.GabcDE.exampleExampleExampleExample_1234567"


def discord_answers(monkeypatch: pytest.MonkeyPatch, status: int, body: dict):
    asked: list[dict] = []

    def get(url: str, **kwargs) -> httpx.Response:
        asked.append({"url": url, **kwargs})
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    return asked


def test_a_discord_token_is_checked_and_the_invite_link_shown(home: Path, monkeypatch: pytest.MonkeyPatch):
    asked = discord_answers(
        monkeypatch, 200, {"id": "1234567890123456789", "username": "Dreambot", "bot": True}
    )
    result = runner.invoke(cli.app, ["setup", "discord"], input=f"{DISCORD_TOKEN}\n")
    assert result.exit_code == 0, result.output
    assert "That is Dreambot" in said(result) and "Discord enabled" in said(result)
    assert asked[0]["url"] == "https://discord.com/api/v10/users/@me"
    assert asked[0]["headers"]["Authorization"] == f"Bot {DISCORD_TOKEN}"
    assert cli._stored_for_test == {"discord": DISCORD_TOKEN}
    assert tomllib.loads((home / "config.toml").read_text())["discord"] == {"enabled": True}
    assert "client_id=1234567890123456789" in said(result).replace(" ", ""), "the add-to-server link"
    assert DISCORD_TOKEN not in result.output

    rejected = discord_answers(monkeypatch, 401, {"message": "401: Unauthorized", "code": 0})
    cli._stored_for_test.clear()
    result = runner.invoke(cli.app, ["setup", "discord"], input=f"{DISCORD_TOKEN}\n")
    assert "did not accept that token: 401: Unauthorized" in said(result) and cli._stored_for_test == {}
    rejected.clear()
    result = runner.invoke(cli.app, ["setup", "discord"], input="nonsense\n")
    assert "three runs of letters and digits" in said(result) and rejected == []


def test_setup_restarts_a_running_service_only_when_something_changed(
    home: Path, monkeypatch: pytest.MonkeyPatch
):
    telegram_answers(monkeypatch, 200, {"ok": True, "result": {"username": "my_dream_bot"}})
    calls: list[tuple[str, ...]] = []

    def running(*arguments: str) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(launchd, "_launchctl", running)
    result = runner.invoke(cli.app, ["setup", "telegram"], input="\n")
    assert "restarted" not in said(result) and not [call for call in calls if call[0] == "kickstart"]

    result = runner.invoke(cli.app, ["setup", "telegram"], input=f"{TOKEN}\n")
    assert "The background service was restarted" in said(result)
    assert [call[:2] for call in calls if call[0] == "kickstart"] == [("kickstart", "-k")]


def test_service_restart_needs_a_running_service(home: Path, monkeypatch: pytest.MonkeyPatch):
    result = runner.invoke(cli.app, ["service", "restart"])
    assert result.exit_code == 1 and "not running" in said(result)
    assert not [call for call in cli._launchctl_for_test if call[0] == "kickstart"]


def test_setup_imessage_points_at_the_service_when_one_is_running(
    home: Path, monkeypatch: pytest.MonkeyPatch
):
    from dreambot.frontends.imessage import chatdb

    monkeypatch.setattr(
        chatdb.ChatDB, "open", lambda self: pytest.fail("the terminal's own access is beside the point")
    )
    monkeypatch.setattr(
        launchd, "_launchctl", lambda *arguments: subprocess.CompletedProcess(arguments, 0, "", "")
    )
    result = runner.invoke(cli.app, ["setup", "imessage"], input="dedicated\n")
    assert result.exit_code == 0, result.output
    assert "iMessage enabled in dedicated mode" in said(result)
    assert "The background service reads the Messages database" in said(result)
    assert "The background service was restarted" in said(result)
    assert tomllib.loads((home / "config.toml").read_text())["imessage"] == {
        "enabled": True,
        "mode": "dedicated",
    }


def test_setup_warns_when_discord_would_refuse_the_invite_link(home: Path, monkeypatch: pytest.MonkeyPatch):
    """The wall a real setup ran into: "Integration requires code grant"."""
    answers = {
        "/users/@me": {"id": "1234567890123456789", "username": "Dreambot", "bot": True},
        "/applications/@me": {"id": "1234567890123456789", "bot_require_code_grant": True},
    }

    def get(url: str, **kwargs) -> httpx.Response:
        body = next(value for path, value in answers.items() if url.endswith(path))
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    result = runner.invoke(cli.app, ["setup", "discord"], input=f"{DISCORD_TOKEN}\n")
    assert "Requires OAuth2 Code Grant" in said(result) and "Switch it off" in said(result)
    assert "cannot be added as a friend" in said(result)

    answers["/applications/@me"]["bot_require_code_grant"] = False
    cli._stored_for_test.clear()
    result = runner.invoke(cli.app, ["setup", "discord"], input=f"{DISCORD_TOKEN}\n")
    assert "Requires OAuth2 Code Grant" not in said(result) and "client_id=" in said(result).replace(" ", "")


def test_a_restart_waits_for_running_jobs_unless_told_not_to(home: Path, monkeypatch: pytest.MonkeyPatch):
    """Restarting mid-job throws the job away (it happened to a live Telegram pan)."""
    import time

    from dreambot.core.jobs import Job, JobKind, JobStatus
    from dreambot.core.store import Store

    calls: list[tuple[str, ...]] = []

    def running(*arguments: str) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(launchd, "_launchctl", running)
    home.mkdir(parents=True, exist_ok=True)
    database = cli._config().database_path
    store = Store(database)
    job = Job(
        id="BUSY01",
        kind=JobKind.PAN,
        user="telegram:1",
        channel="telegram:1",
        prompt="a fox",
        status=JobStatus.RUNNING,
    )
    store.save_job(job)

    naps: list[float] = []

    def nap(seconds: float) -> None:
        naps.append(seconds)
        if len(naps) == 3:  # the job finishes while the command waits
            job.status = JobStatus.DONE
            store.save_job(job)

    monkeypatch.setattr(time, "sleep", nap)
    result = runner.invoke(cli.app, ["service", "restart"])
    assert result.exit_code == 0, result.output
    assert "busy with 1 job(s)" in said(result) and "Restarted." in said(result)
    assert len(naps) == 3 and [call[0] for call in calls].count("kickstart") == 1

    # --now does not wait.
    job.status = JobStatus.RUNNING
    store.save_job(job)
    naps.clear()
    result = runner.invoke(cli.app, ["service", "restart", "--now"])
    assert "busy" not in said(result) and naps == [] and "Restarted." in said(result)

    # A job that never finishes does not hold the restart for ever.
    monkeypatch.setattr(cli, "RESTART_PATIENCE", 0.0)
    monkeypatch.setattr(cli._wait_for_jobs, "__kwdefaults__", {"now": False, "patience": 0.0})
    result = runner.invoke(cli.app, ["service", "restart"])
    assert "restarting anyway" in said(result) and "Restarted." in said(result)
    store.close()


def test_setup_web_switches_the_web_app_on_and_off(home: Path):
    result = runner.invoke(cli.app, ["setup", "web"], input="\n")
    assert result.exit_code == 0 and not (home / "config.toml").exists(), "Enter changes nothing"
    result = runner.invoke(cli.app, ["setup", "web"], input="yes\n")
    assert "Switched on" in said(result) and "http://127.0.0.1:7373/" in said(result)
    assert "enabled = true" in (home / "config.toml").read_text()
    result = runner.invoke(cli.app, ["setup", "web"], input="\n")
    assert "It is on" in said(result) and "enabled = true" in (home / "config.toml").read_text()
    result = runner.invoke(cli.app, ["setup", "web"], input="off\n")
    assert "Switched off" in said(result) and "enabled = false" in (home / "config.toml").read_text()
    assert "Web app: off" in said(runner.invoke(cli.app, ["doctor"]))
