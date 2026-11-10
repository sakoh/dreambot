"""The command line: setup and doctor, without a network, a Keychain or the Messages database."""

from __future__ import annotations

import tomllib
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from dreambot import cli, secrets

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
    assert result.exit_code == 2 and "telegram, discord, imessage" in said(result)


def test_setup_skips_everything_on_enter_and_never_opens_messages(
    home: Path, monkeypatch: pytest.MonkeyPatch
):
    from dreambot.frontends.imessage import chatdb

    def forbidden(self) -> None:
        raise AssertionError("the Messages database was opened")

    monkeypatch.setattr(chatdb.ChatDB, "open", forbidden)
    result = runner.invoke(cli.app, ["setup"], input="\n\n\n")
    assert result.exit_code == 0, result.output
    assert not (home / "config.toml").exists()
    for heading in ("Telegram", "Discord", "iMessage", "Start it"):
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
