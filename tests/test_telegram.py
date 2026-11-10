"""The Telegram frontend against a fake Bot API: real python-telegram-bot objects and handlers,
no network and no token."""

from __future__ import annotations

import asyncio
import itertools
import json
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from telegram import Update
from telegram.request import BaseRequest, RequestData

from dreambot.core.jobs import JobKind, JobStatus
from dreambot.core.service import Dreambot
from dreambot.engine.client import Engine
from dreambot.frontends.telegram_bot import TelegramFrontend
from tests.test_service import make_config

ALICE = 1001
BOT = {"id": 42, "is_bot": True, "first_name": "Dreambot", "username": "dreambot_test_bot"}


class FakeBotAPI(BaseRequest):
    """Answers Bot API calls locally and records them."""

    def __init__(self, files: dict[str, Path] | None = None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._ids = itertools.count(500)
        self.files = files or {}

    @property
    def read_timeout(self) -> float | None:
        return 5.0

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def do_request(
        self, url: str, method: str, request_data: RequestData | None = None, *args, **kwargs
    ):
        endpoint = url.rsplit("/", 1)[-1]
        if "/file/bot" in url:  # a file download
            return 200, self.files[url.rsplit("/", 1)[-1]].read_bytes()
        params = dict(request_data.parameters) if request_data else {}
        self.calls.append((endpoint, params))
        return 200, json.dumps({"ok": True, "result": self._result(endpoint, params)}).encode()

    def _message(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "message_id": next(self._ids),
            "date": 0,
            "chat": {"id": int(params.get("chat_id", ALICE)), "type": "private"},
            "from": BOT,
            "text": params.get("text", ""),
        }

    def _result(self, endpoint: str, params: dict[str, Any]) -> Any:
        if endpoint == "getMe":
            return BOT
        if endpoint in ("sendMessage", "sendPhoto", "sendDocument", "editMessageText"):
            return self._message(params)
        if endpoint == "sendMediaGroup":
            return [self._message(params) for _ in params.get("media", [0])]
        if endpoint == "getFile":
            return {
                "file_id": params["file_id"],
                "file_unique_id": "u",
                "file_path": f"photos/{params['file_id']}",
            }
        return True

    def of(self, endpoint: str) -> list[dict[str, Any]]:
        return [params for name, params in self.calls if name == endpoint]

    def labels(self, params: dict[str, Any]) -> list[str]:
        markup = params.get("reply_markup") or {}
        markup = json.loads(markup) if isinstance(markup, str) else markup
        return [button["text"] for row in markup.get("inline_keyboard", []) for button in row]

    def payload(self, params: dict[str, Any], label: str) -> str:
        markup = params.get("reply_markup") or {}
        markup = json.loads(markup) if isinstance(markup, str) else markup
        return next(
            b["callback_data"] for row in markup["inline_keyboard"] for b in row if b["text"] == label
        )


class Chat:
    def __init__(self, service: Dreambot, front: TelegramFrontend, api: FakeBotAPI):
        self.service, self.front, self.api = service, front, api
        self._update_ids = itertools.count(1)
        self._message_ids = itertools.count(100)

    async def settle(self) -> None:
        for _ in range(400):
            await asyncio.sleep(0.02)
            jobs = self.service.store.recent_jobs(limit=50)
            if self.service.queue_length == 0 and all(job.status.finished for job in jobs):
                await asyncio.sleep(0.05)
                return
        raise AssertionError("jobs did not finish")

    async def feed(self, payload: dict[str, Any]) -> None:
        update = Update.de_json({"update_id": next(self._update_ids), **payload}, self.front.app.bot)
        await self.front.app.process_update(update)
        await asyncio.sleep(0)

    async def message(self, text: str = "", *, user: int = ALICE, chat: dict | None = None, **extra) -> None:
        body = {
            "message_id": next(self._message_ids),
            "date": 0,
            "chat": chat or {"id": user, "type": "private"},
            "from": {"id": user, "is_bot": False, "first_name": "Alice"},
            **extra,
        }
        if text:
            body["caption" if ("photo" in extra or "document" in extra) else "text"] = text
        await self.feed({"message": body})
        if "media_group_id" not in extra:
            await self.settle()

    async def press(self, sent: dict[str, Any], label: str, *, user: int = ALICE) -> None:
        await self.feed(
            {
                "callback_query": {
                    "id": "cb",
                    "from": {"id": user, "is_bot": False, "first_name": "Alice"},
                    "chat_instance": "ci",
                    "data": self.api.payload(sent, label),
                    "message": {
                        "message_id": sent.get("_id", 1),
                        "date": 0,
                        "chat": {"id": int(sent["chat_id"]), "type": "private"},
                        "from": BOT,
                    },
                }
            }
        )
        await self.settle()


@pytest.fixture
async def chat(tmp_path: Path) -> AsyncIterator[Chat]:
    upload = tmp_path / "upload.jpg"
    Image.new("RGB", (640, 480), (30, 120, 200)).save(upload)
    api = FakeBotAPI(files={"PHOTO_A": upload, "PHOTO_B": upload})
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path), engine=engine) as service:
        service.access.add_owner(f"telegram:{ALICE}")
        front = TelegramFrontend(service, "123:TEST", request=api, polling=False)
        await front.start()
        try:
            yield Chat(service, front, api)
        finally:
            await front.stop()


async def test_startup_registers_the_command_menu(chat: Chat):
    commands = chat.api.of("setMyCommands")[0]["commands"]
    assert {"imagine", "describe", "blend", "settings", "help"} <= {
        command["command"] for command in commands
    }


async def test_private_message_becomes_a_grid_with_buttons(chat: Chat):
    await chat.message("a lighthouse at dusk --ar 16:9")
    job = chat.service.store.recent_jobs(limit=1)[0]
    assert (
        job.status is JobStatus.DONE
        and job.user == f"telegram:{ALICE}"
        and job.channel == f"telegram:{ALICE}"
    )

    status = chat.api.of("sendMessage")[0]
    assert status["text"].startswith("⏳") and chat.api.labels(status) == ["Cancel"]
    assert chat.api.of("deleteMessage"), "the status message is removed when the result arrives"

    photo = chat.api.of("sendPhoto")[-1]
    assert "a lighthouse at dusk" in photo["caption"] and f"#{job.id}" in photo["caption"]
    assert chat.api.labels(photo) == ["U1", "U2", "U3", "U4", "🔄", "V1", "V2", "V3", "V4", "⬇️ Originals"]
    assert all(
        len(b.encode()) <= 64 for b in [chat.api.payload(photo, label) for label in chat.api.labels(photo)]
    )


async def test_buttons_and_original_files(chat: Chat):
    await chat.message("a fox")
    job = chat.service.store.recent_jobs(limit=1)[0]
    grid = chat.api.of("sendPhoto")[-1]

    await chat.press(grid, "U2")
    picked = chat.api.of("sendPhoto")[-1]
    assert "Image 2" in picked["caption"]
    assert "Upscale (Subtle)" in chat.api.labels(picked) and chat.api.labels(picked)[-1] == "⬇️ Original"

    await chat.press(picked, "⬇️ Original")
    assert len(chat.api.of("sendDocument")) == 1

    await chat.press(grid, "⬇️ Originals")
    assert len(chat.api.of("sendMediaGroup")[-1]["media"]) == 4

    await chat.press(grid, "V4")
    varied = chat.service.store.recent_jobs(limit=1)[0]
    assert varied.kind is JobKind.VARIATION and (varied.parent, varied.parent_index) == (job.id, 4)
    assert chat.api.of("answerCallbackQuery"), "every button press is acknowledged"


async def test_photo_with_caption_is_downloaded_and_used(chat: Chat):
    photo = [{"file_id": "PHOTO_A", "file_unique_id": "a", "width": 640, "height": 480}]
    await chat.message("make it night --edit --n 1", photo=photo)
    job = chat.service.store.recent_jobs(limit=1)[0]
    assert job.status is JobStatus.DONE and job.inputs["attachments"]
    with Image.open(job.inputs["attachments"][0]) as attached:
        assert attached.size == (640, 480)


async def test_album_is_handled_as_one_message(chat: Chat):
    for file_id in ("PHOTO_A", "PHOTO_B"):
        await chat.message(
            "/blend" if file_id == "PHOTO_A" else "",
            photo=[{"file_id": file_id, "file_unique_id": file_id, "width": 640, "height": 480}],
            media_group_id="album1",
        )
    await asyncio.sleep(1.0)
    await chat.settle()
    job = chat.service.store.recent_jobs(limit=1)[0]
    assert job.kind is JobKind.BLEND and len(job.inputs["attachments"]) == 2


async def test_reply_and_reaction_target_a_message(chat: Chat):
    await chat.message("a fox")
    job = chat.service.store.recent_jobs(limit=1)[0]
    # The grid was sent as the message with the id the fake API handed back.
    grid_ref = next(ref for ref, _ in chat.service.store.messages_for_job("telegram", job.id))
    grid_id = int(grid_ref.split("/")[1])

    reply_to = {"message_id": grid_id, "date": 0, "chat": {"id": ALICE, "type": "private"}, "from": BOT}
    await chat.message("upscale 3", reply_to_message=reply_to)
    assert chat.service.store.recent_jobs(limit=1)[0].parent_index == 3

    await chat.feed(
        {
            "message_reaction": {
                "chat": {"id": ALICE, "type": "private"},
                "message_id": grid_id,
                "date": 0,
                "user": {"id": ALICE, "is_bot": False, "first_name": "Alice"},
                "old_reaction": [],
                "new_reaction": [{"type": "emoji", "emoji": "👎"}],
            }
        }
    )
    await chat.settle()
    assert "Nothing is running" in chat.api.of("sendMessage")[-1]["text"]


async def test_groups_only_react_to_commands_and_strangers_are_refused(chat: Chat):
    group = {"id": -100500, "type": "supergroup", "title": "Friends"}
    before = len(chat.api.calls)
    await chat.message("what a nice day", chat=group)
    assert len(chat.api.calls) == before

    await chat.message("/imagine@dreambot_test_bot a fox --n 1", chat=group)
    job = chat.service.store.recent_jobs(limit=1)[0]
    assert job.channel == "telegram:-100500" and job.status is JobStatus.DONE

    await chat.message("a fox", user=2002)
    assert (
        "private" in chat.api.of("sendMessage")[-1]["text"]
        and "/allow telegram:2002" in chat.api.of("sendMessage")[-1]["text"]
    )


async def test_button_that_needs_words_uses_force_reply(chat: Chat):
    await chat.message("a fox --n 1")
    single = chat.api.of("sendPhoto")[-1]
    await chat.press(single, "Edit")
    question = chat.api.of("sendMessage")[-1]
    markup = question["reply_markup"]
    markup = json.loads(markup) if isinstance(markup, str) else markup
    assert markup.get("force_reply") is True
    await chat.message("make it night")
    assert chat.service.store.recent_jobs(limit=1)[0].kind is JobKind.EDIT
