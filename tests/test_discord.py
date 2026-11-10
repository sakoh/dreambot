"""The Discord frontend, with stand-ins for Discord's interaction and channel objects.

The command tree, views, modals and files are the real discord.py ones; only the network-facing
objects (an interaction's response, a channel) are faked.
"""

from __future__ import annotations

import asyncio
import itertools
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import discord
import pytest
from PIL import Image

from dreambot.core.jobs import JobKind, JobStatus
from dreambot.core.service import Dreambot
from dreambot.engine.client import Engine
from dreambot.frontends.discord_bot import DiscordFrontend, _TextModal
from tests.test_service import make_config

ALICE = 7001
CHANNEL = 9000
_ids = itertools.count(1000)


class FakeMessage:
    def __init__(self, log: list[FakeMessage], content, files=None, view=None, ephemeral=False):
        self.id = next(_ids)
        self.content = content or ""
        self.files = [f.filename for f in files or []]
        self.view = view
        self.ephemeral = ephemeral
        self.deleted = False
        self.edits: list[str] = []
        log.append(self)

    async def edit(self, content=None, view=None, **_):
        self.edits.append(content)
        self.content, self.view = content, view

    async def delete(self):
        self.deleted = True

    def labels(self) -> list[str]:
        return [
            item.label
            for item in (self.view.children if self.view else [])
            if isinstance(item, discord.ui.Button)
        ]

    def custom_id(self, label: str) -> str:
        return next(item.custom_id for item in self.view.children if getattr(item, "label", None) == label)


class FakeChannel:
    def __init__(self, log: list[FakeMessage]):
        self.log = log
        self.id = CHANNEL

    async def send(self, content=None, files=None, view=None, reference=None, **_):
        return FakeMessage(self.log, content, files, view)

    def get_partial_message(self, message_id: int):
        return next(message for message in self.log if message.id == message_id)


class FakeResponse:
    def __init__(self, interaction: FakeInteraction):
        self.interaction = interaction
        self.done = False
        self.modal: Any = None
        self.deferred = False
        self.thinking = False
        self.thinking_private = False
        self.edited: tuple[str, Any] | None = None

    def is_done(self) -> bool:
        return self.done

    async def send_message(self, content=None, files=None, view=None, ephemeral=False, **_):
        assert not self.done, "an interaction can only be answered once"
        self.done = True
        self.interaction.original = FakeMessage(self.interaction.log, content, files, view, ephemeral)

    async def send_modal(self, modal):
        assert not self.done, "a modal has to be the first answer to an interaction"
        self.done, self.modal = True, modal

    async def defer(self, *, ephemeral: bool = False, thinking: bool = False):
        assert not self.done, "an interaction can only be answered once"
        self.done = self.deferred = True
        #: A "thinking" defer shows a placeholder that the first follow-up replaces, and that
        #: follow-up is private or public as the defer was, whatever it asks for itself.
        self.thinking, self.thinking_private = thinking, ephemeral

    async def edit_message(self, content=None, view=None, **_):
        assert not self.done, "an interaction can only be answered once"
        self.done, self.edited = True, (content, view)


class FakeInteraction:
    def __init__(self, log: list[FakeMessage], user: int = ALICE, data: dict | None = None):
        self.log = log
        self.user = SimpleNamespace(id=user)
        self.channel_id = CHANNEL
        self.data = data or {}
        self.response = FakeResponse(self)
        self.original: FakeMessage | None = None
        self.followup = SimpleNamespace(send=self._followup)
        self.followups: list[FakeMessage] = []
        self.extras: dict[Any, Any] = {}

    def is_expired(self) -> bool:
        return False

    async def original_response(self) -> FakeMessage:
        assert self.original is not None
        return self.original

    async def _followup(self, content=None, files=None, view=None, ephemeral=False, **_):
        assert self.response.done, "a follow-up needs the interaction to have been answered first"
        if self.response.thinking and not self.followups:
            ephemeral = self.response.thinking_private
        message = FakeMessage(self.log, content, files, view, ephemeral)
        self.followups.append(message)
        return message


class Server:
    def __init__(self, service: Dreambot, front: DiscordFrontend):
        self.service, self.front = service, front
        self.log: list[FakeMessage] = []
        channel = FakeChannel(self.log)

        async def get_channel(conversation: str):
            return channel

        front._channel = get_channel  # type: ignore[method-assign]

    async def settle(self) -> None:
        for _ in range(400):
            await asyncio.sleep(0.02)
            jobs = self.service.store.recent_jobs(limit=50)
            if self.service.queue_length == 0 and all(job.status.finished for job in jobs):
                await asyncio.sleep(0.05)
                return
        raise AssertionError("jobs did not finish")

    async def slash(self, name: str, *, as_user: int = ALICE, **params) -> FakeInteraction:
        interaction = FakeInteraction(self.log, as_user)
        command = self.front.tree.get_command(name.split()[0])
        for part in name.split()[1:]:
            command = command.get_command(part)
        await command.callback(interaction, **params)
        await self.settle()
        return interaction

    async def press(
        self, message: FakeMessage, label: str, *, user: int = ALICE, values=None
    ) -> FakeInteraction:
        data = {"custom_id": message.custom_id(label)}
        if values:
            data["values"] = values
        interaction = FakeInteraction(self.log, user, data)
        await self.front._on_component(interaction)
        await self.settle()
        return interaction

    @property
    def visible(self) -> list[FakeMessage]:
        return [message for message in self.log if not message.deleted]


class FakeAttachment:
    def __init__(self, source: Path, content_type: str = "image/png"):
        self.filename = source.name
        self.content_type = content_type
        self._source = source

    async def save(self, target: Path) -> None:
        Path(target).write_bytes(self._source.read_bytes())


@pytest.fixture
async def server(tmp_path: Path) -> AsyncIterator[Server]:
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path), engine=engine) as service:
        service.access.add_owner(f"discord:{ALICE}")
        front = DiscordFrontend(service, "token")
        await ChatFrontendStart.start(front)
        yield Server(service, front)


class ChatFrontendStart:
    """Start only the platform-independent part: no login, no gateway."""

    @staticmethod
    async def start(front: DiscordFrontend) -> None:
        from dreambot.frontends.base import ChatFrontend

        await ChatFrontend.start(front)


async def test_slash_imagine_answers_the_interaction_then_posts_the_grid(server: Server):
    interaction = await server.slash("imagine", prompt="a lighthouse at dusk --ar 16:9")
    job = server.service.store.recent_jobs(limit=1)[0]
    assert (
        job.status is JobStatus.DONE
        and job.user == f"discord:{ALICE}"
        and job.channel == f"discord:{CHANNEL}"
    )

    status = interaction.original
    assert status is not None and status.deleted and not status.ephemeral
    assert status.edits or status.content.startswith("⏳")

    result = server.visible[-1]
    assert result.files == [Path(job.grid).name]
    assert "a lighthouse at dusk" in result.content and f"#{job.id}" in result.content
    assert result.labels() == ["U1", "U2", "U3", "U4", "🔄", "V1", "V2", "V3", "V4"]
    rows = {item.row for item in result.view.children}
    assert rows == {0, 1}


async def test_errors_and_information_are_private(server: Server):
    bad = await server.slash("imagine", prompt="a fox --chaos 999")
    assert bad.original.ephemeral and "between 0 and 100" in bad.original.content
    info = await server.slash("info")
    assert info.original.ephemeral and "Jobs completed" in info.original.content
    listing = await server.slash("show")
    assert listing.original.ephemeral


async def test_buttons_select_and_generate(server: Server):
    await server.slash("imagine", prompt="a fox")
    job = server.service.store.recent_jobs(limit=1)[0]
    grid = server.visible[-1]

    picked = await server.press(grid, "U3")
    # The press is acknowledged at once (Discord allows 3 seconds); the picture follows.
    assert picked.response.deferred and picked.original is None
    (single,) = picked.followups
    assert single.files == [Path(job.image(3).path).name] and "Upscale (Subtle)" in single.labels()
    assert len({item.row for item in single.view.children}) <= 5

    varied = await server.press(grid, "V2")
    assert varied.response.deferred and varied.followups, "the status line follows the press"
    latest = server.service.store.recent_jobs(limit=1)[0]
    assert latest.kind is JobKind.VARIATION and latest.parent_index == 2

    favorite = await server.press(single, "❤️")
    assert favorite.followups[-1].ephemeral and favorite.followups[-1].content == "Added to favorites."


async def test_buttons_that_need_text_open_a_modal(server: Server):
    await server.slash("imagine", prompt="a fox --n 1")
    single = server.visible[-1]
    job = server.service.store.recent_jobs(limit=1)[0]

    pressed = await server.press(single, "Vary (Region)")
    modal = pressed.response.modal
    assert isinstance(modal, _TextModal)
    assert [label for label, _ in modal.fields] == ["Which part?", "What should be there?"]
    submit = FakeInteraction(server.log)
    await server.front._on_modal(submit, modal.action, ["the sky", "an aurora"])
    await server.settle()
    edited = server.service.store.recent_jobs(limit=1)[0]
    assert edited.kind is JobKind.EDIT and edited.prompt == "an aurora" and edited.parent == job.id

    zoom = (await server.press(single, "Custom Zoom")).response.modal
    assert zoom.fields[1][1].default == job.prompt  # the prompt is pre-filled, as in Midjourney
    await server.front._on_modal(FakeInteraction(server.log), zoom.action, ["1.4", ""])
    await server.settle()
    assert server.service.store.recent_jobs(limit=1)[0].detail["factor"] == 1.4


async def test_remix_modal_is_prefilled_with_the_prompt(server: Server):
    await server.slash("prefer remix")
    await server.slash("imagine", prompt="a fox --ar 3:2")
    grid = server.visible[-1]
    modal = (await server.press(grid, "V1")).response.modal
    assert modal.title == "Remix Prompt" and modal.fields[0][1].default == "a fox --aspect 3:2"
    await server.front._on_modal(FakeInteraction(server.log), modal.action, ["a wolf --ar 3:2"])
    await server.settle()
    assert server.service.store.recent_jobs(limit=1)[0].text == "a wolf"


async def test_attachments_describe_and_blend(server: Server, sample_image: Path, tmp_path: Path):
    other = tmp_path / "other.png"
    Image.new("RGB", (256, 256), (0, 200, 0)).save(other)

    asked = await server.slash("describe", image=FakeAttachment(sample_image))
    # Fetching an attachment can outlast Discord's 3 seconds, so the command says "thinking" first.
    assert asked.response.deferred and asked.response.thinking and not asked.response.thinking_private
    assert asked.followups and asked.followups[0].content.startswith("⏳"), "the status line replaces it"
    described = server.visible[-1]
    assert described.labels() == ["1", "2", "3", "4"]

    await server.slash(
        "blend",
        image1=FakeAttachment(sample_image),
        image2=FakeAttachment(other),
        dimensions=SimpleNamespace(value="landscape"),
    )
    blended = server.service.store.recent_jobs(limit=1)[0]
    assert blended.kind is JobKind.BLEND and (blended.width, blended.height) == (1344, 896)

    await server.slash("edit", instruction="make it night", image=FakeAttachment(sample_image))
    assert server.service.store.recent_jobs(limit=1)[0].status is JobStatus.DONE

    # Something that is not an image is refused privately, before anything is fetched or deferred.
    refused = await server.slash("describe", image=FakeAttachment(tmp_path / "notes.txt", "text/plain"))
    assert refused.original.ephemeral and "isn't an image" in refused.original.content
    assert not refused.response.deferred


async def test_settings_panel(server: Server):
    opened = await server.slash("settings")
    panel = opened.original
    assert panel.ephemeral and "RAW Mode" in panel.labels() and "Stylize high" in panel.labels()
    select = next(item for item in panel.view.children if isinstance(item, discord.ui.Select))
    assert "z-image-turbo" in [option.value for option in select.options]

    pressed = await server.press(panel, "Stylize high")
    assert server.service.resolved_settings(f"discord:{ALICE}").stylize == 250
    _content, view = pressed.response.edited
    high = next(item for item in view.children if getattr(item, "label", "") == "Stylize high")
    assert high.style is discord.ButtonStyle.success

    await server.press(panel, "RAW Mode")
    await server.press(panel, "Subtle Variation Mode")
    resolved = server.service.resolved_settings(f"discord:{ALICE}")
    assert resolved.raw and resolved.variation == "subtle"

    model = FakeInteraction(server.log, data={"custom_id": "set:model", "values": ["qwen-image"]})
    await server.front._on_component(model)
    assert server.service.resolved_settings(f"discord:{ALICE}").model == "qwen-image"

    await server.press(panel, "Reset Settings")
    assert server.service.resolved_settings(f"discord:{ALICE}").stylize == 100


async def test_strangers_are_refused_privately_and_pairing_works(server: Server):
    refused = await server.slash("imagine", as_user=8002, prompt="a fox")
    assert refused.original.ephemeral and "private" in refused.original.content
    await server.slash("imagine", prompt="a fox")
    grid = server.visible[-1]
    pressed = await server.press(grid, "V1", user=8002)
    assert pressed.followups[-1].ephemeral and "private" in pressed.followups[-1].content

    code = server.service.access.new_pairing_code()
    paired = await server.slash("pair", as_user=8002, code=code)
    assert "Paired" in paired.original.content and server.service.access.is_owner("discord:8002")

    allowed = await server.slash("allow", user=SimpleNamespace(id=8003))
    assert "discord:8003 can now use" in allowed.original.content


async def test_direct_message_and_mentions(server: Server, sample_image: Path):
    bot_user = SimpleNamespace(id=1, bot=True)
    server.front.client._connection.user = bot_user  # what discord.py sets after login

    def message(content: str, *, guild=None, mentions=(), attachments=(), reference=None):
        return SimpleNamespace(
            author=SimpleNamespace(id=ALICE, bot=False),
            guild=guild,
            mentions=list(mentions),
            content=content,
            attachments=list(attachments),
            reference=reference,
            channel=SimpleNamespace(id=CHANNEL, send=None),
        )

    await server.front._on_message(message("a fox --n 1"))
    await server.settle()
    assert server.service.store.recent_jobs(limit=1)[0].text == "a fox"

    before = len(server.service.store.recent_jobs(limit=50))
    await server.front._on_message(message("just chatting", guild=object()))
    await server.settle()
    assert len(server.service.store.recent_jobs(limit=50)) == before

    await server.front._on_message(message("<@1> a wolf --n 1", guild=object(), mentions=[bot_user]))
    await server.settle()
    assert server.service.store.recent_jobs(limit=1)[0].text == "a wolf"

    await server.front._on_message(
        message("make it night --edit --n 1", attachments=[FakeAttachment(sample_image)])
    )
    await server.settle()
    assert server.service.store.recent_jobs(limit=1)[0].inputs["attachments"]


async def test_oversized_files_are_sent_as_jpeg(server: Server, tmp_path: Path, monkeypatch):
    import dreambot.frontends.discord_bot as module

    monkeypatch.setattr(module, "UPLOAD_LIMIT", 1000)
    await server.slash("imagine", prompt="a fox --n 1")
    assert server.visible[-1].files[0].endswith(".jpg")


# --------------------------------------------------------------------------------------
# What only the real Discord does
# --------------------------------------------------------------------------------------


def http_error(status: int, text: str) -> discord.HTTPException:
    return discord.HTTPException(SimpleNamespace(status=status, reason=text), text)


async def test_a_plain_slash_command_is_answered_directly(server: Server):
    interaction = await server.slash("imagine", prompt="a fox --n 1")
    assert not interaction.response.deferred, "nothing slow happens first, so the status line is the answer"
    assert interaction.original is not None and interaction.extras.get("answered")


async def test_status_edits_that_fail_do_not_lose_the_result(
    server: Server, caplog: pytest.LogCaptureFixture
):
    async def expired(self, content=None, view=None, **_):
        raise http_error(401, "Invalid Webhook Token")

    original_edit, original_delete = FakeMessage.edit, FakeMessage.delete
    FakeMessage.edit = FakeMessage.delete = expired  # type: ignore[method-assign]
    try:
        with caplog.at_level("WARNING"):
            await server.slash("imagine", prompt="a fox --n 1")
    finally:
        FakeMessage.edit, FakeMessage.delete = original_edit, original_delete  # type: ignore[method-assign]
    job = server.service.store.recent_jobs(limit=1)[0]
    assert job.status is JobStatus.DONE and server.visible[-1].files == [Path(job.images[0].path).name]
    assert not [record for record in caplog.records if record.levelname == "ERROR"]


async def test_an_old_interaction_reply_is_reached_through_the_channel(server: Server):
    import time

    from dreambot.frontends.discord_bot import INTERACTION_LIFETIME

    class OldReply(discord.WebhookMessage):
        def __init__(self) -> None:  # the real one is built from a payload; only its type matters here
            pass

    in_channel = FakeMessage(server.log, "⏳ 40% — a fox")
    ref = str(in_channel.id)
    stale = OldReply()
    server.front._sent[ref] = (stale, time.monotonic() - INTERACTION_LIFETIME - 1)
    assert await server.front._message(str(CHANNEL), ref) is in_channel, "the interaction has expired"
    fresh = OldReply()
    server.front._sent[ref] = (fresh, time.monotonic())
    assert await server.front._message(str(CHANNEL), ref) is fresh
    # A message sent to the channel in the first place never expires.
    server.front._sent[ref] = (in_channel, time.monotonic() - 10 * INTERACTION_LIFETIME)
    assert await server.front._message(str(CHANNEL), ref) is in_channel


async def test_a_dead_gateway_connection_is_logged(server: Server, caplog: pytest.LogCaptureFixture):
    async def refused() -> None:
        raise discord.PrivilegedIntentsRequired(None)

    task = asyncio.get_running_loop().create_task(refused())
    await asyncio.wait([task])
    with caplog.at_level("ERROR"):
        server.front._connection_ended(task)
    assert "the connection ended: PrivilegedIntentsRequired" in caplog.text
