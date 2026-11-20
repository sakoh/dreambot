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


def test_doctor_shows_folders_and_reasons_as_they_are(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The folders are the person's to name, and a failed check gives its reason in someone else's
    # words. Square brackets in either were read as the console's markup: with everything kept
    # in a folder called [lab], every path `dreambot doctor` printed was missing that folder.
    import importlib.metadata

    lab = tmp_path / "[lab]"
    (lab / "home").mkdir(parents=True)
    (lab / "vision").mkdir()
    (lab / "home" / "config.toml").write_text(f'[assist]\nvision_model = "{lab / "vision"}"\n')
    monkeypatch.setenv("DREAMBOT_HOME", str(lab / "home"))
    monkeypatch.setenv("DREAMBOT_OUTPUT_DIR", str(lab / "pictures"))
    monkeypatch.setenv("HF_HUB_CACHE", str(lab / "hub"))
    monkeypatch.setattr(launchd, "status", lambda: {"installed": False, "loaded": False})

    def unreadable(name: str) -> str:
        raise OSError(f"cannot read the metadata of {name} [permission denied]")

    monkeypatch.setattr(importlib.metadata, "version", unreadable)
    result = runner.invoke(cli.app, ["doctor"])
    assert "MLX — cannot read the metadata of mlx [permission denied]" in said(result)
    # The console breaks a long path wherever the line ends; put it back together.
    unbroken = result.output.replace("\n", "")
    assert f"Language model — {lab}/vision; prompt enhancer" in unbroken
    assert f"Hugging Face cache: {lab}/hub" in unbroken
    assert f"Config: {lab}/home/config.toml" in unbroken
    assert f"Images are saved to: {lab}/pictures" in unbroken


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


def test_what_telegram_and_discord_answer_is_shown_as_it_is(home: Path, monkeypatch: pytest.MonkeyPatch):
    # A bot's name and the reason a token is refused are Telegram's and Discord's words, not
    # Dreambot's. Square brackets in them were read as the console's markup and left out: a
    # Discord bot called Dreambot [dev] was greeted as "Dreambot ".
    discord_answers(
        monkeypatch, 200, {"id": "1234567890123456789", "username": "Dreambot [dev]", "bot": True}
    )
    result = runner.invoke(cli.app, ["setup", "discord"], input=f"{DISCORD_TOKEN}\n")
    assert result.exit_code == 0, result.output
    assert "That is Dreambot [dev]. Token saved to your Keychain" in said(result)

    cli._stored_for_test.clear()
    discord_answers(monkeypatch, 401, {"message": "401: Unauthorized [token was reset]", "code": 0})
    result = runner.invoke(cli.app, ["setup", "discord"], input=f"{DISCORD_TOKEN}\n")
    assert "did not accept that token: 401: Unauthorized [token was reset]. Nothing" in said(result)
    assert cli._stored_for_test == {}

    telegram_answers(monkeypatch, 401, {"ok": False, "description": "Unauthorized [bot was deleted]"})
    result = runner.invoke(cli.app, ["setup", "telegram"], input=f"{TOKEN}\n")
    assert "did not accept that token: Unauthorized [bot was deleted]. Nothing" in said(result)

    telegram_answers(monkeypatch, 200, {"ok": True, "result": {"username": "my_[dream]_bot"}})
    result = runner.invoke(cli.app, ["setup", "telegram"], input=f"{TOKEN}\n")
    assert result.exit_code == 0, result.output
    assert "That is @my_[dream]_bot. Token saved" in said(result)
    assert "Your bot: https://t.me/my_[dream]_bot" in said(result)


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


def test_the_background_services_files_are_named_as_they_are(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Where the service is installed, what it runs and where it writes its log are paths on this
    # Mac. A folder in them with square brackets in its name, [lab] say, was read as the
    # console's markup and left out, so the path that was printed led nowhere.
    from dreambot.frontends.imessage import chatdb

    lab = tmp_path / "[lab]"
    monkeypatch.setenv("HOME", str(lab))  # the log is kept in the home folder's Library/Logs
    monkeypatch.setenv("DREAMBOT_HOME", str(lab / "dreambot"))
    monkeypatch.setattr(launchd, "interpreter", lambda: f"{lab}/python/bin/python3.12")
    monkeypatch.setattr(launchd, "code_is_guarded", lambda: "Documents")
    agent = lab / "Library" / "LaunchAgents" / "agent.plist"
    monkeypatch.setattr(launchd, "install", lambda log_dir, home: (agent, home / "runtime"))

    result = runner.invoke(cli.app, ["service", "install"])
    assert result.exit_code == 0, result.output
    # The console breaks a long path wherever the line ends; put it back together.
    unbroken = result.output.replace("\n", "")
    assert f"Installed and started: {agent}" in unbroken
    assert f"It runs the copy in: {lab}/dreambot/runtime" in unbroken
    assert f"Logs: {lab}/Library/Logs/Dreambot/dreambot.log" in unbroken
    assert f"grant Full Disk Access to: {lab}/python/bin/python3.12" in unbroken

    # `dreambot doctor` points at the same log when only the service may read Messages.
    def refused(self) -> None:
        raise chatdb.ChatDBError("macOS is blocking access to your Messages database.")

    monkeypatch.setattr(chatdb.ChatDB, "open", refused)
    monkeypatch.setattr(
        launchd, "_launchctl", lambda *arguments: subprocess.CompletedProcess(arguments, 0, "", "")
    )
    (lab / "dreambot").mkdir(parents=True)
    (lab / "dreambot" / "config.toml").write_text("[imessage]\nenabled = true\n")
    result = runner.invoke(cli.app, ["doctor"])
    assert "the background service may be allowed to" in said(result)
    assert f"Its log says: {lab}/Library/Logs/Dreambot/dreambot.log" in result.output.replace("\n", "")


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


def test_why_messages_cannot_be_read_is_said_in_full(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # When the Messages database cannot be read, the message ends with what SQLite said, in
    # square brackets. Those were read as the console's markup, so `dreambot doctor` and
    # `dreambot setup imessage` gave the advice without the reason. The program to give Full
    # Disk Access to is named by its path, which lost any folder with brackets in its name.
    from dreambot.frontends.imessage import chatdb

    not_a_database = tmp_path / "chat.db"
    not_a_database.write_text("this is not a database")
    monkeypatch.setattr(chatdb.ChatDB.__init__, "__defaults__", (not_a_database,))
    monkeypatch.setattr(launchd, "interpreter", lambda: "/Users/me/[tools]/python/bin/python3.12")
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text("[imessage]\nenabled = true\n")

    result = runner.invoke(cli.app, ["doctor"])
    assert "cannot read the Messages database yet" in said(result)
    assert "shows exactly which program to add.) [file is not a database]" in said(result)
    assert "grant Full Disk Access to: /Users/me/[tools]/python/bin/python3.12" in said(result)

    result = runner.invoke(cli.app, ["setup", "imessage"], input="self\n")
    assert result.exit_code == 0, result.output
    assert "shows exactly which program to add.) [file is not a database]" in said(result)
    assert "also give Full Disk Access to: /Users/me/[tools]/python/bin/python3.12" in said(result)

    # With the background service running, it is the service's program that is named.
    monkeypatch.setattr(
        launchd, "_launchctl", lambda *arguments: subprocess.CompletedProcess(arguments, 0, "", "")
    )
    result = runner.invoke(cli.app, ["setup", "imessage"], input="self\n")
    assert "then ⌘⇧G and paste): /Users/me/[tools]/python/bin/python3.12" in said(result)


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


def test_a_message_is_printed_as_it_is_square_brackets_and_all(home: Path, monkeypatch: pytest.MonkeyPatch):
    # Square brackets are the console's own markup. Printed inside a coloured line as it was, a
    # hint that names a section of the config file lost the section, a name lost its brackets,
    # and a name that held a closing tag stopped the command with a traceback.
    import socket

    monkeypatch.setenv("DREAMBOT_BACKEND", "fake")
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        result = runner.invoke(cli.app, ["web", "--port", str(port), "--no-open"])
    assert result.exit_code == 1, result.output
    assert f"port {port} is in use" in said(result)
    assert "choose another with `[web] port` in the config file" in said(result)

    result = runner.invoke(cli.app, ["loras", "remove", "[web]"])
    assert result.exit_code == 2 and "`[web]` can't be a LoRA's name" in said(result)
    result = runner.invoke(cli.app, ["loras", "remove", "x[/red]"])
    assert result.exit_code == 2 and "`x[/red]` can't be a LoRA's name" in said(result)
    result = runner.invoke(cli.app, ["setup", "[bold]telegram"])
    assert result.exit_code == 2 and "not `[bold]telegram`" in said(result)
    result = runner.invoke(cli.app, ["models", "pull", "[web]"])
    assert result.exit_code == 2 and "Unknown model '[web]'" in said(result)


def test_a_prompt_and_the_paths_of_what_was_made_are_printed_as_they_are(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A prompt is the person's own words, and the folders are theirs to name. Square brackets in
    # either were read as the console's markup: a sign that says [open] lost its word in the
    # caption, a folder called [lab] went missing from every path that was printed, and a
    # prompt that held [/dim] ended the command with a traceback once its pictures were made.
    from tests.test_adapters import make_repo

    out, cache = tmp_path / "[lab]" / "pictures", tmp_path / "[lab]" / "hub"
    monkeypatch.setenv("DREAMBOT_OUTPUT_DIR", str(out))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("DREAMBOT_BACKEND", "fake")

    result = runner.invoke(cli.app, ["imagine", "a neon sign that says [open] --ar 1:1", "--no-open"])
    assert result.exit_code == 0, result.output
    assert "a neon sign that says [open]" in said(result)
    # The console breaks a long path wherever the line ends; put it back together.
    unbroken = result.output.replace("\n", "")
    assert f"1. {out}/" in unbroken, "where each picture is"
    assert f"grid: {out}/" in unbroken, "where the grid is"
    result = runner.invoke(cli.app, ["imagine", "a fox [/dim] in the snow --ar 1:1", "--no-open"])
    assert result.exit_code == 0 and "a fox [/dim] in the snow" in said(result), result.output

    result = runner.invoke(cli.app, ["chat", "--no-open"], input="a red fox in the snow --ar 1:1\n")
    assert result.exit_code == 0, result.output
    assert f"  {out}/" in result.output.replace("\n", ""), "chat prints where the grid is"

    # An installed model is shown with the folder it is in, and a copy of one that somebody
    # else exported is listed under the name its folder has.
    make_repo(cache, "mflux-community/z-image-turbo-mflux-q8", {"transformer/0.safetensors": b"x"})
    make_repo(cache, "someone/Z-Image-Turbo-mflux-q8[mine]", {"transformer/0.safetensors": b"x"})
    result = runner.invoke(cli.app, ["models", "info", "z-image-turbo"])
    assert f"{cache}/models--mflux-community--z-image-turbo-mflux-q8/" in result.output.replace("\n", "")
    assert "Installed: someone/Z-Image-Turbo-mflux-q8[mine] — q8" in said(result)
