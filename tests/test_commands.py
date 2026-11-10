"""The typed interface shared by iMessage, Telegram and the console."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from PIL import Image

from dreambot.core.actions import Action, ActionKind, decode_action, encode_action, parse_action
from dreambot.core.commands import Commands, Reply, job_reply
from dreambot.core.jobs import EventType, Job, JobEvent, JobKind, JobStatus
from dreambot.core.service import Dreambot
from dreambot.core.store import MessageLink
from dreambot.engine.client import Engine
from tests.test_service import make_config

USER = "imessage:+15550001111"
CHANNEL = "imessage:chat1"


class Harness:
    def __init__(self, service: Dreambot):
        self.service = service
        self.commands = Commands(service)
        self._done: dict[str, asyncio.Event] = {}
        service.subscribe(self._on_event)

    def _on_event(self, event: JobEvent) -> None:
        if event.type in (EventType.FINISHED, EventType.FAILED, EventType.CANCELLED):
            self._done.setdefault(event.job.id, asyncio.Event()).set()

    async def say(self, text: str, *, wait: bool = True, **kwargs) -> list[Reply]:
        replies = await self.commands.handle(user=USER, channel=CHANNEL, text=text, **kwargs)
        if wait:
            for reply in replies:
                for job in reply.queued:
                    await asyncio.wait_for(self._done.setdefault(job.id, asyncio.Event()).wait(), 30)
        return replies

    async def job(self, text: str, **kwargs) -> Job:
        replies = await self.say(text, **kwargs)
        assert len(replies) == 1 and len(replies[0].queued) == 1, replies
        return replies[0].queued[0]


@pytest.fixture
async def chat(tmp_path: Path) -> AsyncIterator[Harness]:
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path), engine=engine) as service:
        service.access.add_owner(USER)
        yield Harness(service)


# --------------------------------------------------------------------------------------
# Typed shortcut grammar
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("U2", Action(ActionKind.SELECT, index=2)),
        ("u 4", Action(ActionKind.SELECT, index=4)),
        ("/u1", Action(ActionKind.SELECT, index=1)),
        ("V3", Action(ActionKind.VARY, index=3)),
        ("v2: a red hat --ar 1:1", Action(ActionKind.VARY, index=2, prompt="a red hat --ar 1:1")),
        ("V2: A Red Hat", Action(ActionKind.VARY, index=2, prompt="A Red Hat")),
        ("r", Action(ActionKind.REROLL)),
        ("Reroll", Action(ActionKind.REROLL)),
        ("again", Action(ActionKind.REROLL)),
        ("🔄", Action(ActionKind.REROLL)),
        ("upscale 2", Action(ActionKind.UPSCALE_SUBTLE, index=2)),
        ("upscale", Action(ActionKind.UPSCALE_SUBTLE)),
        ("upscale 3 creative", Action(ActionKind.UPSCALE_CREATIVE, index=3)),
        ("upscale creative", Action(ActionKind.UPSCALE_CREATIVE)),
        ("upscale creative 2", Action(ActionKind.UPSCALE_CREATIVE, index=2)),
        ("vary 2", Action(ActionKind.VARY, index=2)),
        ("vary subtle", Action(ActionKind.VARY_SUBTLE)),
        ("vary 1 strong", Action(ActionKind.VARY_STRONG, index=1)),
        ("vary strong 4", Action(ActionKind.VARY_STRONG, index=4)),
        ("zoom", Action(ActionKind.ZOOM, value="2")),
        ("zoom out", Action(ActionKind.ZOOM, value="2")),
        ("zoom 1.5x", Action(ActionKind.ZOOM, value="1.5")),
        ("zoom out 2x", Action(ActionKind.ZOOM, value="2")),
        ("zoom 3 1.5x", Action(ActionKind.ZOOM, index=3, value="1.5")),
        ("zoom 2 2", Action(ActionKind.ZOOM, index=2, value="2")),
        ("pan left", Action(ActionKind.PAN, value="left")),
        ("pan 2 up", Action(ActionKind.PAN, index=2, value="up")),
        ("pan down 3", Action(ActionKind.PAN, index=3, value="down")),
        ("➡️", Action(ActionKind.PAN, value="right")),
        ("⬆️ 2", Action(ActionKind.PAN, index=2, value="up")),
        ("square", Action(ActionKind.SQUARE)),
        ("make square 2", Action(ActionKind.SQUARE, index=2)),
        ("fav 2", Action(ActionKind.FAVORITE, index=2)),
        ("❤️", Action(ActionKind.FAVORITE)),
        ("info 3", Action(ActionKind.INFO, index=3)),
        ("seed", Action(ActionKind.INFO)),
        ("cancel", Action(ActionKind.CANCEL)),
        ("stop", Action(ActionKind.CANCEL)),
        ("animate 2 high", Action(ActionKind.ANIMATE, index=2, value="high")),
        (
            "region 2 B2-D4: a red balloon",
            Action(ActionKind.VARY_REGION, index=2, value="B2-D4", prompt="a red balloon"),
        ),
        ("region the sky: an aurora", Action(ActionKind.VARY_REGION, value="the sky", prompt="an aurora")),
        ("edit 3: make it night", Action(ActionKind.EDIT, index=3, prompt="make it night")),
        ("edit: Make It Night", Action(ActionKind.EDIT, prompt="Make It Night")),
    ],
)
def test_shortcuts(text: str, expected: Action):
    assert parse_action(text) == expected


@pytest.mark.parametrize(
    "prompt",
    [
        "zoom lens photo of a hawk",
        "pan fried dumplings on a plate",
        "pan across a mountain range",
        "square dance in a barn",
        "upscale restaurant interior",
        "vary the colors of autumn",
        "stop sign in the desert",
        "cancel culture protest illustration",
        "u2 concert poster",
        "v8 engine closeup",
        "v2 rocket on a launchpad",
        "r2d2 style robot",
        "info graphic about bees",
        "seed pods on a branch",
        "love letters on a desk",
        "again and again, a recursive hallway",
        "region of france, vineyards",
        "edit suite with monitors",
        "a",
        "",
    ],
)
def test_prompts_are_never_mistaken_for_shortcuts(prompt: str):
    assert parse_action(prompt) is None


def test_out_of_range_shortcut_values_are_not_actions():
    assert parse_action("zoom 9x") is None
    assert parse_action("zoom 1x") is None
    assert parse_action("u0") is None


def test_button_payloads_round_trip_and_fit_platform_limits():
    for action in [
        Action(ActionKind.SELECT, "7QK2MD", 3),
        Action(ActionKind.ZOOM, "7QK2MD", 1, "1.5"),
        Action(ActionKind.PAN, "7QK2MD", 4, "left"),
        Action(ActionKind.REROLL, "7QK2MD"),
        Action(ActionKind.UPSCALE_CREATIVE, "7QK2MD", 16),
    ]:
        payload = encode_action(action)
        assert len(payload.encode()) <= 64  # Telegram's callback_data limit
        assert decode_action(payload) == action
    assert decode_action("something else") is None
    assert decode_action("db:zz:JOB") is None
    with pytest.raises(ValueError):
        encode_action(Action(ActionKind.REROLL))


# --------------------------------------------------------------------------------------
# Conversations
# --------------------------------------------------------------------------------------


async def test_bare_text_in_a_direct_chat_is_a_prompt(chat: Harness):
    job = await chat.job("a lighthouse at dusk —ar 16:9")  # em dash, as an iPhone would send it
    assert job.status is JobStatus.DONE and (job.width, job.height) == (1456, 816)
    reply = job_reply(job)
    assert reply.images == [job.grid] and "a lighthouse at dusk" in reply.text and f"#{job.id}" in reply.text


async def test_bare_text_in_a_group_is_ignored_but_commands_work(chat: Harness):
    assert await chat.say("hey did you see the game last night", direct=False) == []
    job = await chat.job("/imagine a fox --n 1", direct=False)
    assert job.status is JobStatus.DONE
    # Telegram group form, and Midjourney's "prompt:" habit.
    assert (await chat.job("/imagine@dreambot_bot prompt: a wolf --n 1", direct=False)).text == "a wolf"


async def test_shortcuts_apply_to_the_latest_job(chat: Harness):
    job = await chat.job("a fox")
    picked = await chat.say("U2")
    assert picked[0].images == [job.image(2).path] and picked[0].image.index == 2
    varied = await chat.job("V3")
    assert varied.kind is JobKind.VARIATION and (varied.parent, varied.parent_index) == (job.id, 3)


async def test_replying_to_a_picture_targets_that_picture(chat: Harness):
    first = await chat.job("a fox")
    await chat.job("a wolf")  # a newer job that a bare shortcut would otherwise pick
    target = MessageLink(first.id, 4)

    upscaled = await chat.job("upscale", target=target)
    assert (upscaled.parent, upscaled.parent_index) == (first.id, 4)

    # Words in reply to a picture are an edit instruction for it.
    edited = await chat.job("make it snow", target=target)
    assert edited.kind is JobKind.EDIT and edited.text == "make it snow" and edited.parent == first.id

    # A reply to the grid message (no single image) still needs a number.
    reply = await chat.say("upscale", target=MessageLink(first.id, None))
    assert "Which image" in reply[0].text
    assert (await chat.job("upscale 2", target=MessageLink(first.id, None))).parent_index == 2


async def test_errors_come_back_as_replies_not_exceptions(chat: Harness):
    for text, fragment in [
        ("a fox --chaos 999", "between 0 and 100"),
        ("U1", "nothing here"),
        ("/nonsense", "don't know that command"),
        ("/describe", "Attach the image"),
        ("/blend", "Attach 2 to 5 images"),
        ("/show ZZZZZZ", "can't find a job"),
        ("/settings stylize purple", "Stylize is"),
        ("/settings model nope", "Unknown model"),
        ("/shorten", "Give me a prompt to shorten"),
    ]:
        replies = await chat.say(text)
        assert len(replies) == 1 and replies[0].text.startswith("⚠️") and fragment in replies[0].text, text


async def test_attachment_without_words_gets_guidance(chat: Harness, sample_image: Path):
    replies = await chat.say("", attachments=[sample_image])
    assert "describe" in replies[0].text and not replies[0].queued
    assert await chat.say("") == []


async def test_describe_edit_and_blend_with_attachments(chat: Harness, sample_image: Path, tmp_path: Path):
    other = tmp_path / "other.png"
    Image.new("RGB", (320, 320), (0, 90, 200)).save(other)

    described = await chat.job("/describe", attachments=[sample_image])
    assert described.kind is JobKind.DESCRIBE and len(described.texts) == 4
    assert "1. " in job_reply(described).text

    edited = await chat.job("/edit make it night", attachments=[sample_image])
    assert edited.status is JobStatus.DONE

    blended = await chat.job("/blend portrait", attachments=[sample_image, other])
    assert blended.kind is JobKind.BLEND and (blended.width, blended.height) == (896, 1344)


async def test_settings_and_prefer(chat: Harness):
    shown = (await chat.say("/settings"))[0].text
    assert "stylize     100 (med)" in shown and "variation   strong" in shown

    assert "250 (high)" in (await chat.say("/settings stylize high"))[0].text
    assert "variation   subtle" in (await chat.say("/settings variation subtle"))[0].text
    assert "count       2 images" in (await chat.say("/settings count 2"))[0].text
    assert "model       qwen-image" in (await chat.say("/settings model qwen"))[0].text
    job = await chat.job("a fox")
    assert job.model == "qwen-image" and len(job.images) == 2

    assert "now ends with" in (await chat.say("/prefer suffix --ar 3:2"))[0].text
    assert ((await chat.job("a fox")).width, 896) == (1344, 896)
    assert "⚠️" in (await chat.say("/prefer suffix --bogus 1"))[0].text
    assert "cleared" in (await chat.say("/prefer suffix"))[0].text

    assert "`--wide` now means" in (await chat.say("/prefer option set wide --ar 21:9 --stylize 300"))[0].text
    assert "--wide: --ar 21:9" in (await chat.say("/prefer option list"))[0].text
    wide = await chat.job("a fox --wide")
    assert wide.width / wide.height == pytest.approx(21 / 9, rel=0.02)
    assert "already a built-in" in (await chat.say("/prefer option set chaos --seed 1"))[0].text
    assert "Removed" in (await chat.say("/prefer option set wide"))[0].text

    assert "now on" in (await chat.say("/prefer remix"))[0].text
    assert "now off" in (await chat.say("/prefer remix"))[0].text
    assert "strong variations" in (await chat.say("/prefer variability"))[0].text

    assert "reset" in (await chat.say("/settings reset"))[0].text
    assert (await chat.job("a fox")).model == "z-image-turbo"


async def test_show_info_models_help_and_noops(chat: Harness):
    assert "No jobs yet" in (await chat.say("/show"))[0].text
    job = await chat.job("a fox")
    assert job.id in (await chat.say("/show"))[0].text
    shown = (await chat.say(f"/show {job.id.lower()}"))[0]
    assert shown.images == [job.grid] and shown.job.id == job.id

    assert "Jobs completed: 1" in (await chat.say("/info"))[0].text
    assert "z-image-turbo" in (await chat.say("/models"))[0].text
    assert "--stylize" in (await chat.say("/params"))[0].text
    assert "U1–U4" in (await chat.say("/help"))[0].text
    assert "full speed" in (await chat.say("/relax"))[0].text
    assert "private" in (await chat.say("/stealth"))[0].text


async def test_region_overlay_helps_choose_cells(chat: Harness):
    job = await chat.job("a fox --n 1")
    replies = await chat.say("region")
    assert replies[0].images and Path(replies[0].images[0]).is_file() and "B2-D4" in replies[0].text
    with Image.open(replies[0].images[0]) as overlay:
        assert overlay.size == (job.width, job.height)
    grid_job = await chat.job("a fox")
    assert "Pick one image first" in (await chat.say("region"))[0].text
    assert grid_job.status is JobStatus.DONE


async def test_cancel_command(chat: Harness):
    assert "Nothing is running" in (await chat.say("/cancel"))[0].text


# --------------------------------------------------------------------------------------
# Access
# --------------------------------------------------------------------------------------

STRANGER = "telegram:999"


async def stranger_says(chat: Harness, text: str, **kwargs) -> list[Reply]:
    return await chat.commands.handle(user=STRANGER, channel="telegram:999", text=text, **kwargs)


async def test_a_dreambot_is_private_until_someone_is_let_in(chat: Harness):
    refused = await stranger_says(chat, "a fox")
    assert len(refused) == 1 and not refused[0].queued
    assert "private" in refused[0].text and "/allow telegram:999" in refused[0].text
    assert (await stranger_says(chat, "/imagine a fox"))[0].text == refused[0].text
    # In a group, a stranger's ordinary chatter gets no reply at all.
    assert await stranger_says(chat, "what's for lunch", direct=False) == []

    assert (
        "Only an owner" in (await stranger_says(chat, "/allow telegram:999"))[0].text
        or "private" in (await stranger_says(chat, "/allow telegram:999"))[0].text
    )
    assert "can now use" in (await chat.say(f"/allow {STRANGER}"))[0].text
    allowed = await stranger_says(chat, "a fox --n 1")
    assert allowed[0].queued
    assert "Only an owner" in (await stranger_says(chat, "/allow telegram:1"))[0].text
    assert "member" in (await stranger_says(chat, "/whoami"))[0].text

    assert STRANGER in (await chat.say("/access"))[0].text
    assert "Removed" in (await chat.say(f"/revoke {STRANGER}"))[0].text
    assert "private" in (await stranger_says(chat, "a fox"))[0].text


async def test_pairing_claims_ownership_with_a_one_time_code(tmp_path: Path):
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path), engine=engine) as service:
        commands = Commands(service)

        async def say(text: str) -> str:
            return (await commands.handle(user=STRANGER, channel="telegram:999", text=text))[0].text

        assert "hasn't been claimed yet" in await say("a fox")
        assert "isn't valid" in await say("/pair 000000")
        code = service.access.new_pairing_code()
        assert "isn't valid" in await say("/pair " + ("111111" if code != "111111" else "222222"))
        assert "Paired" in await say(f"/pair {code}")
        assert service.access.is_owner(STRANGER)
        # The code cannot be used twice, by anyone.
        other = await commands.handle(user="discord:5", channel="discord:5", text=f"/pair {code}")
        assert "isn't valid" in other[0].text
        assert "already own" in await say("/pair 123456")


async def test_expired_pairing_codes_are_rejected(tmp_path: Path):
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path), engine=engine) as service:
        code = service.access.new_pairing_code()
        pending = service.store.get_state("access.pairing")
        service.store.set_state("access.pairing", {**pending, "expires": 0})
        assert not service.access.pair(STRANGER, code)


async def test_open_mode_and_config_lists(tmp_path: Path):
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    config = make_config(
        tmp_path, access={"owners": ["Discord:42"], "allowed": ["imessage:+1 (555) 000-2222"]}
    )
    async with Dreambot(config, engine=engine) as service:
        access = service.access
        assert access.is_owner("discord:42") and access.role("imessage:+15550002222") == "member"
        assert not access.is_allowed(STRANGER)
        assert access.is_owner("console:me")  # the Mac's own terminal is always its owner
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path / "open", access={"open": True}), engine=engine) as service:
        assert service.access.role(STRANGER) == "member" and not service.access.is_owner(STRANGER)


async def test_shorten_proposes_shorter_prompts_and_keeps_parameters(chat: Harness):
    long_prompt = (
        "a majestic red fox, sitting in fresh snow, golden hour light, highly detailed, 8k, masterpiece, "
        "soft bokeh background, wildlife photography --ar 16:9 --chaos 10"
    )
    job = await chat.job(f"/shorten {long_prompt}")
    assert job.kind is JobKind.SHORTEN and job.status is JobStatus.DONE and 2 <= len(job.texts) <= 5
    assert all(text.endswith("--aspect 16:9 --chaos 10") for text in job.texts)
    lengths = [len(text) for text in job.texts]
    assert lengths == sorted(lengths, reverse=True) and lengths[-1] < lengths[0]
    assert "masterpiece" not in job.texts[0] and "8k" not in job.texts[0]
    assert job.texts[-1].startswith("a majestic red fox")
    reply = job_reply(job)
    assert reply.text.startswith("1. ") and "imagine it" in reply.text
