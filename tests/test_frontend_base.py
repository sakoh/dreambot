"""The platform-independent half of every chat frontend, driven through an in-memory platform."""

from __future__ import annotations

import asyncio
import itertools
import subprocess
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dreambot.core.actions import Action, ActionKind, encode_action
from dreambot.core.jobs import EventType, JobEvent, JobKind, JobStatus
from dreambot.core.service import Dreambot
from dreambot.core.settings import UserSettings
from dreambot.engine.client import Engine
from dreambot.frontends.base import Buttons, ChatFrontend, Incoming
from tests.test_service import make_config


@dataclass
class Sent:
    ref: str
    conversation: str
    text: str
    images: list[str]
    buttons: Buttons | None
    ask_reply: bool = False
    deleted: bool = False
    edits: list[str] = field(default_factory=list)

    def labels(self) -> list[str]:
        return [label for row in self.buttons or [] for label, _ in row]

    def payload(self, label: str) -> str:
        return next(payload for row in self.buttons or [] for text, payload in row if text == label)


class MemoryFrontend(ChatFrontend):
    """A platform that keeps its messages in a list."""

    name = "mem"

    def __init__(self, service: Dreambot, **capabilities):
        super().__init__(service)
        for key, value in capabilities.items():
            setattr(self, key, value)
        self.messages: list[Sent] = []
        self._ids = itertools.count(1)
        self._done: dict[str, asyncio.Event] = {}
        service.subscribe(self._watch)

    def _watch(self, event: JobEvent) -> None:
        if event.type in (EventType.FINISHED, EventType.FAILED, EventType.CANCELLED):
            self._done.setdefault(event.job.id, asyncio.Event()).set()

    async def send(
        self,
        conversation,
        text="",
        images: Sequence[str] = (),
        buttons=None,
        *,
        reply_to=None,
        ask_reply=False,
    ):
        refs = []
        # One message per image when a platform cannot group them with buttons, else one message.
        groups = [[image] for image in images] if (images and not self.grid_as_one_image) else [list(images)]
        for group in groups:
            ref = f"m{next(self._ids)}"
            self.messages.append(Sent(ref, conversation, text, group, buttons, ask_reply))
            refs.append(ref)
        return refs

    async def edit(self, conversation, ref, text, buttons=None):
        message = self.get(ref)
        message.edits.append(text)
        message.text, message.buttons = text, buttons

    async def delete(self, conversation, ref):
        self.get(ref).deleted = True

    def get(self, ref: str) -> Sent:
        return next(message for message in self.messages if message.ref == ref)

    @property
    def visible(self) -> list[Sent]:
        return [message for message in self.messages if not message.deleted]

    async def settle(self) -> None:
        """Wait until every job has finished and its result has been delivered."""
        for _ in range(400):
            await asyncio.sleep(0.02)
            jobs = self.service.store.recent_jobs(limit=50)
            if self.service.queue_length == 0 and all(job.status.finished for job in jobs):
                await asyncio.sleep(0.05)
                return
        raise AssertionError("jobs did not finish")

    async def say(self, text: str, *, user: str = "alice", conversation: str = "room", **kwargs) -> None:
        await self.handle_message(Incoming(user=user, conversation=conversation, text=text, **kwargs))
        await self.settle()

    async def press(self, message: Sent, label: str, *, user: str = "alice") -> str | None:
        notice = await self.handle_button(user, message.conversation, message.payload(label))
        await self.settle()
        return notice


@pytest.fixture
async def service(tmp_path: Path) -> AsyncIterator[Dreambot]:
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path), engine=engine) as bot:
        bot.access.add_owner("mem:alice")
        yield bot


async def test_prompt_shows_live_status_then_a_grid_with_buttons(service: Dreambot):
    front = MemoryFrontend(service)
    await front.start()
    await front.say("a lighthouse at dusk --ar 16:9")

    status, result = front.messages
    assert status.deleted and status.text.startswith("⏳")
    assert status.labels() == ["Cancel"] or status.edits  # it carried a Cancel button while running
    job = service.store.recent_jobs(limit=1)[0]
    assert result.images == [job.grid]
    assert "a lighthouse at dusk" in result.text and f"#{job.id}" in result.text
    assert result.labels() == ["U1", "U2", "U3", "U4", "🔄", "V1", "V2", "V3", "V4"]
    # The grid message is linked to the job, so a reply to it can find the job.
    assert service.store.find_message("mem", f"room/{result.ref}").job == job.id


async def test_buttons_select_vary_and_upscale(service: Dreambot):
    front = MemoryFrontend(service)
    await front.start()
    await front.say("a fox")
    grid = front.visible[-1]
    job = service.store.recent_jobs(limit=1)[0]

    await front.press(grid, "U2")
    picked = front.visible[-1]
    assert picked.images == [job.image(2).path]
    assert (
        "Upscale (Subtle)" in picked.labels() and "Zoom Out 2x" in picked.labels() and "⬅️" in picked.labels()
    )
    assert "Make Square" not in picked.labels()  # it is already square
    link = service.store.find_message("mem", f"room/{picked.ref}")
    assert (link.job, link.image) == (job.id, 2)

    await front.press(grid, "V3")
    varied = service.store.recent_jobs(limit=1)[0]
    assert varied.kind is JobKind.VARIATION and varied.parent_index == 3
    assert front.visible[-1].images == [varied.grid]

    await front.press(picked, "Upscale (Subtle)")
    upscaled = service.store.recent_jobs(limit=1)[0]
    assert upscaled.kind is JobKind.UPSCALE and upscaled.parent_index == 2
    final = front.visible[-1]
    assert final.images == [upscaled.images[0].path]
    assert "Upscale (Subtle)" not in final.labels() and "❤️" in final.labels()

    notice = await front.press(picked, "❤️")
    assert notice == "Added to favorites."


async def test_buttons_that_need_words_ask_and_use_the_next_message(service: Dreambot):
    front = MemoryFrontend(service)
    await front.start()
    await front.say("a fox --n 1")
    single = front.visible[-1]

    await front.press(single, "Vary (Region)")
    question = front.visible[-1]
    assert question.ask_reply and "B2-D4: a red balloon" in question.text
    await front.say("the sky: an aurora")
    edited = service.store.recent_jobs(limit=1)[0]
    assert edited.kind is JobKind.EDIT and edited.prompt == "an aurora"

    await front.press(single, "Custom Zoom")
    await front.say("1.3 a fox in a huge forest")
    zoomed = service.store.recent_jobs(limit=1)[0]
    assert (
        zoomed.kind is JobKind.ZOOM
        and zoomed.detail["factor"] == 1.3
        and zoomed.text == "a fox in a huge forest"
    )

    await front.press(single, "Custom Zoom")
    await front.say("ten")
    assert "number above 1" in front.visible[-1].text

    await front.press(single, "Edit")
    await front.say("make it night")
    assert service.store.recent_jobs(limit=1)[0].text == "make it night"


async def test_a_pending_question_does_not_swallow_commands_or_other_people(service: Dreambot):
    service.access.add_owner("mem:bob")
    front = MemoryFrontend(service)
    await front.start()
    await front.say("a fox --n 1")
    single = front.visible[-1]
    await front.press(single, "Edit")

    # Bob's message is his own prompt, not the answer to Alice's question.
    await front.say("a wolf --n 1", user="bob")
    assert service.store.recent_jobs(limit=1)[0].text == "a wolf"
    # A command from Alice is still a command...
    await front.say("/info")
    assert "Jobs completed" in front.visible[-1].text
    # ...and her next plain message answers the question.
    await front.say("make it night")
    latest = service.store.recent_jobs(limit=1)[0]
    assert latest.kind is JobKind.EDIT and latest.text == "make it night"


async def test_remix_mode_asks_for_a_prompt(service: Dreambot):
    service.save_settings("mem:alice", UserSettings(remix=True))
    front = MemoryFrontend(service)
    await front.start()
    await front.say("a fox")
    grid = front.visible[-1]
    await front.press(grid, "V1")
    assert "Remix" in front.visible[-1].text and "a fox" in front.visible[-1].text
    await front.say("a wolf")
    remixed = service.store.recent_jobs(limit=1)[0]
    assert remixed.kind is JobKind.VARIATION and remixed.text == "a wolf"


async def test_replies_and_reactions_target_the_message_they_are_on(service: Dreambot):
    front = MemoryFrontend(service)
    await front.start()
    await front.say("a fox")
    first_grid = front.visible[-1]
    first = service.store.recent_jobs(limit=1)[0]
    await front.say("a wolf")

    await front.say("V2", reply_to=first_grid.ref)
    varied = service.store.recent_jobs(limit=1)[0]
    assert (varied.parent, varied.parent_index) == (first.id, 2)

    await front.press(first_grid, "U4")
    picked = front.visible[-1]
    await front.handle_reaction("alice", "room", picked.ref, "❤️")
    assert service.store.get_job(first.id).image(4).favorite
    assert front.visible[-1].text == "Added to favorites."

    await front.handle_reaction("alice", "room", picked.ref, "👍")
    await front.settle()
    upscaled = service.store.recent_jobs(limit=1)[0]
    assert upscaled.kind is JobKind.UPSCALE and (upscaled.parent, upscaled.parent_index) == (first.id, 4)

    # Reactions on unknown messages, unknown emoji, or from strangers do nothing.
    before = len(front.messages)
    await front.handle_reaction("alice", "room", "nope", "❤️")
    await front.handle_reaction("alice", "room", picked.ref, "🦄")
    await front.handle_reaction("mallory", "room", picked.ref, "❤️")
    assert len(front.messages) == before


async def test_describe_offers_numbered_prompts(service: Dreambot, sample_image: Path):
    front = MemoryFrontend(service)
    await front.start()
    await front.say("/describe", attachments=[str(sample_image)])
    described = front.visible[-1]
    assert described.labels() == ["1", "2", "3", "4"] and "1. " in described.text
    await front.press(described, "3")
    job = service.store.recent_jobs(limit=1)[0]
    assert job.kind is JobKind.IMAGINE and "description #3" in job.text


async def test_platform_without_buttons_or_edits_sends_one_message_per_image(service: Dreambot):
    front = MemoryFrontend(
        service, has_buttons=False, can_edit=False, grid_as_one_image=False, acknowledge=True
    )
    await front.start()
    await front.say("a fox")
    job = service.store.recent_jobs(limit=1)[0]

    ack, *pictures, caption = front.messages
    assert ack.text.startswith("🎨") and not ack.deleted and not ack.edits
    assert [p.images for p in pictures] == [[image.path] for image in job.images]
    assert all(p.buttons is None and p.text == "" for p in pictures)
    assert "a fox" in caption.text and "U1–U4" in caption.text and caption.buttons is None
    # Every picture is individually addressable.
    for picture, image in zip(pictures, job.images, strict=True):
        link = service.store.find_message("mem", f"room/{picture.ref}")
        assert (link.job, link.image) == (job.id, image.index)

    # Replying to one picture with words edits that picture; a tapback favourites it.
    await front.say("make it snow", reply_to=pictures[2].ref)
    edited = service.store.recent_jobs(limit=1)[0]
    assert edited.kind is JobKind.EDIT and (edited.parent, edited.parent_index) == (job.id, 3)
    await front.handle_reaction("alice", "room", pictures[0].ref, "❤")
    assert service.store.get_job(job.id).image(1).favorite


async def test_failures_and_cancellations_are_reported(tmp_path: Path):
    engine = Engine(
        backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL, env={"DREAMBOT_FAKE_STEP_DELAY": "0.05"}
    )
    async with Dreambot(make_config(tmp_path), engine=engine) as service:
        service.access.add_owner("mem:alice")
        front = MemoryFrontend(service)
        await front.start()
        await front.handle_message(Incoming(user="alice", conversation="room", text="slow --steps 200"))
        await asyncio.sleep(0.6)
        status = front.messages[0]
        assert status.labels() == ["Cancel"]
        assert any("%" in edit for edit in status.edits)  # live progress reached the status message
        notice = await front.handle_button("alice", "room", status.payload("Cancel"))
        await front.settle()
        assert "Cancelled 1" in notice
        assert status.text.startswith("⚠️ Cancelled") and not status.deleted
        assert service.store.recent_jobs(limit=1)[0].status is JobStatus.CANCELLED


async def test_strangers_cannot_press_buttons_and_only_own_conversations_are_served(service: Dreambot):
    front = MemoryFrontend(service)
    other = MemoryFrontend(service)
    other.name = "other"
    await front.start()
    await other.start()
    await front.say("a fox")
    grid = front.visible[-1]
    notice = await front.handle_button("mallory", "room", grid.payload("V1"))
    assert "private" in notice
    assert await front.handle_button("alice", "room", "garbage") == "That button is no longer valid."
    expired = encode_action(Action(ActionKind.SELECT, "ZZZZZZ", 1))
    assert "can't find a job" in await front.handle_button("alice", "room", expired)
    # The second frontend saw the same job events but they were not for it.
    assert other.messages == []


async def test_long_captions_are_trimmed_for_the_platform(service: Dreambot):
    front = MemoryFrontend(service, max_caption=120)
    await front.start()
    await front.say("a fox " + "very " * 80 + "detailed --n 1")
    assert len(front.visible[-1].text) <= 120 and front.visible[-1].text.endswith("…")
