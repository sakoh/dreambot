"""Lookbooks: named sets of pictures used with ``--p`` — their codes, how each image of a grid
leans on a different picture, and the service that keeps them."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from dreambot.config import Config
from dreambot.core import fingerprint, lookbook
from dreambot.core.fingerprint import StyleFingerprint, Swatch
from dreambot.core.jobs import EventType, JobStatus
from dreambot.core.lookbook import Member
from dreambot.core.planner import ResolvedRefs, StyleSource
from dreambot.core.prompt import parse_prompt
from dreambot.core.service import DreambotError
from dreambot.core.settings import UserSettings
from dreambot.engine.protocol import Op
from tests.test_planner import planner
from tests.test_service import CHANNEL, USER, bot, imagine  # noqa: F401  (bot is a fixture)

SETTINGS = UserSettings(enhance=False).resolved(Config().defaults)


def picture(tmp_path: Path, name: str, colour: tuple[int, int, int]) -> str:
    path = tmp_path / f"{name}.png"
    Image.new("RGB", (320, 240), colour).save(path)
    return str(path)


def member(tmp_path: Path, name: str, colour: tuple[int, int, int], medium: str) -> Member:
    path = picture(tmp_path, name, colour)
    print_ = StyleFingerprint(
        medium=medium, light=f"{name} light", swatches=(Swatch(colour, 1.0),), name=name
    )
    return Member(fingerprint.image_key(path), path, print_)


async def settle(service, timeout: float = 20) -> None:
    """Wait until the queue and its idle chores have run dry."""
    for _ in range(int(timeout / 0.02)):
        if service.idle:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the service did not settle")


# --------------------------------------------------------------------------------------
# Codes, names and tokens
# --------------------------------------------------------------------------------------


def test_a_code_stands_for_a_set_of_pictures():
    code = lookbook.version_code("abc123", ["k1", "k2"])
    assert lookbook.looks_like_code(code) and len(code) == 7 and code[0].isalpha() and code == code.lower()
    assert lookbook.version_code("abc123", ["k2", "k1", "k1"]) == code, "the same pictures, the same code"
    assert lookbook.version_code("abc123", ["k1", "k2", "k3"]) != code
    assert lookbook.version_code("other", ["k1", "k2"]) != code
    assert not lookbook.looks_like_code("4821") and not lookbook.looks_like_code("Dunes")
    assert not lookbook.looks_like_code("1234567"), (
        "a code starts with a letter: never mistaken for a style code"
    )


def test_names_and_tokens():
    assert lookbook.slug("Autumn  Diner!") == "autumn-diner"
    assert lookbook.clean_name("  Autumn \n Diner ") == "Autumn Diner"
    assert lookbook.parse_token("dunes") == ("dunes", 1.0)
    assert lookbook.parse_token("dusty-noir::2.5") == ("dusty-noir", 2.5)
    assert lookbook.parse_token("odd::name") == ("odd::name", 1.0)


# --------------------------------------------------------------------------------------
# The cast: which picture leads which image
# --------------------------------------------------------------------------------------


def test_one_picture_leads_each_image_and_the_lead_rotates():
    assert lookbook.cast(0, 1, 0) == [] and lookbook.cast(1, 1, 3) == [1.0]
    for count in (2, 3, 5, 12):
        leads = []
        for index in range(count):
            weights = lookbook.cast(count, 99, index)
            lead = max(range(count), key=weights.__getitem__)
            assert weights[lead] / sum(weights) == pytest.approx(lookbook.LEAD_SHARE)
            assert all(weight == 1.0 for position, weight in enumerate(weights) if position != lead)
            leads.append(lead)
        assert sorted(leads) == list(range(count)), "every picture gets its turn before one leads twice"
    assert lookbook.cast(5, 99, 0) == lookbook.cast(5, 99, 5), "then the order repeats"
    assert lookbook.cast(5, 99, 2) == lookbook.cast(5, 99, 2)
    assert [lookbook.cast(8, seed, 0) for seed in range(6)] != [lookbook.cast(8, 0, 0)] * 6, (
        "the seed sets the order"
    )


def test_each_image_of_a_grid_leans_on_a_different_picture(tmp_path: Path):
    members = [
        member(tmp_path, "rust", (170, 70, 30), "gouache illustration"),
        member(tmp_path, "teal", (20, 110, 120), "linocut print"),
        member(tmp_path, "navy", (13, 59, 102), "35mm film photograph"),
    ]
    refs = ResolvedRefs(sources=[StyleSource("look", "Lookbook Trio", 1.0, code="abcdefg", members=members)])
    p = planner(tmp_path, "z-image-turbo")
    plan = p.imagine(parse_prompt("a harbour --p trio --n 3 --seed 4"), SETTINGS, refs)
    assert plan.op is Op.TXT2IMG and len(plan.renders) == 3
    media = sorted(render.prompt.split("in the style of a ")[1].split(",")[0] for render in plan.renders)
    assert media == ["35mm film photograph", "gouache illustration", "linocut print"], "one lead per image"
    for render in plan.renders:
        lead = next(m for m in members if m.print.medium in render.prompt)
        assert lead.print.light in render.prompt, "the lead brings its light with its medium"
        assert sum(other.print.medium in render.prompt for other in members) == 1
    style = plan.detail["style"]
    assert style["sources"] == [{"kind": "look", "label": "Lookbook Trio", "share": 1.0, "code": "abcdefg"}]
    assert len(style["swatches"]) == 3, "the job as a whole is described by what the book has in common"

    again = p.imagine(parse_prompt("a harbour --p trio --n 3 --seed 4"), SETTINGS, refs)
    assert [r.prompt for r in again.renders] == [r.prompt for r in plan.renders], (
        "the same seed, the same leads"
    )

    faint = p.imagine(parse_prompt("a harbour --p trio --n 1 --seed 4 --stylize 20"), SETTINGS, refs)
    assert "with a hint of" in faint.renders[0].prompt, "--stylize sets how strongly a Lookbook is applied"
    off = p.imagine(parse_prompt("a harbour --p trio --n 1 --seed 4 --stylize 0"), SETTINGS, refs)
    assert off.renders[0].prompt == "a harbour" and "style" not in off.detail
    unaffected = p.imagine(parse_prompt("a harbour --p trio --n 1 --seed 4 --sw 0"), SETTINGS, refs)
    assert "in the style of" in unaffected.renders[0].prompt, "--sw is for style references, not Lookbooks"


def test_a_lookbook_board_gives_the_lead_most_of_the_pieces(tmp_path: Path):
    members = [
        member(tmp_path, "rust", (170, 70, 30), "gouache illustration"),
        member(tmp_path, "teal", (20, 110, 120), "gouache illustration"),
    ]
    refs = ResolvedRefs(sources=[StyleSource("look", "Lookbook Pair", 1.0, code="abcdefg", members=members)])
    p = planner(tmp_path, "z-image-turbo", "flux2-klein-4b")
    plan = p.imagine(parse_prompt("a harbour --p pair --n 2 --seed 4"), SETTINGS, refs)
    assert plan.op is Op.EDIT and len(plan.renders) == 2, "each image has a board of its own"
    leads = []
    for render in plan.renders:
        weights = dict(render.board.tiles)
        assert sum(weights.values()) == pytest.approx(100) and max(weights.values()) == pytest.approx(60)
        leads.append(max(weights, key=weights.get))
        assert "is a board of style swatches" in render.prompt
    assert sorted(leads) == sorted(m.path for m in members)


def test_a_style_reference_sets_the_medium_and_a_lookbook_colours_it(tmp_path: Path):
    book = [member(tmp_path, "teal", (20, 110, 120), "linocut print")]
    reference = picture(tmp_path, "ref", (170, 70, 30))
    refs = ResolvedRefs(
        style=[reference],
        sources=[
            StyleSource(
                "image", "image 1", 1.0, pictures=[reference], print=StyleFingerprint(medium="oil painting")
            ),
            StyleSource("look", "Lookbook Teal", 1.0, code="abcdefg", members=book),
        ],
    )
    words = planner(tmp_path, "z-image-turbo").imagine(
        parse_prompt("a harbour --p teal --sref --n 1"), SETTINGS, refs
    )
    prompt = words.renders[0].prompt
    assert "in the style of an oil painting" in prompt and "linocut" not in prompt
    assert "deep teal" in prompt and "teal light" in prompt, "the Lookbook adds its colour and light"

    boarded = planner(tmp_path, "z-image-turbo", "flux2-klein-4b").imagine(
        parse_prompt("a harbour --p teal --sref --n 1 --sw 300 --stylize 100"), SETTINGS, refs
    )
    weights = dict(boarded.renders[0].board.tiles)
    assert weights[reference] == pytest.approx(300) and weights[book[0].path] == pytest.approx(100), (
        "the stronger dial gets more of the board"
    )


# --------------------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------------------


async def test_keeping_a_lookbook(bot, tmp_path: Path):  # noqa: F811
    service, _ = bot
    book = service.create_lookbook(user=USER, name="  Autumn   Diner ")
    assert book["name"] == "Autumn Diner" and book["slug"] == "autumn-diner" and book["count"] == 0
    assert book["code"] is None and book["style"] is None
    for bad in ("", "default", "Liked", "autumn diner"):
        with pytest.raises(DreambotError):
            service.create_lookbook(user=USER, name=bad)

    rust, teal = picture(tmp_path, "rust", (170, 70, 30)), picture(tmp_path, "teal", (20, 110, 120))
    one = await service.add_to_lookbook(user=USER, lookbook_id=book["id"], source=rust)
    assert one["count"] == 1 and lookbook.looks_like_code(one["code"])
    kept = Path(one["images"][0]["path"])
    assert kept.is_file() and service.config.home in kept.parents, (
        "the picture is copied into Dreambot's folder"
    )
    same = await service.add_to_lookbook(user=USER, lookbook_id=book["id"], source=rust)
    assert same["count"] == 1 and same["code"] == one["code"], "a picture is only held once"
    two = await service.add_to_lookbook(user=USER, lookbook_id=book["id"], source=teal)
    assert two["count"] == 2 and two["code"] != one["code"], "a changed book has a new code"
    assert [swatch["hex"] for swatch in two["style"]["swatches"]] == ["#aa461e", "#146e78"]

    await settle(service)
    read = service.lookbooks(USER)[0]
    assert read["unread"] == 0 and all(image["style"]["read"] for image in read["images"]), (
        "the vision model reads the pictures while the GPU is idle"
    )

    back = service.remove_from_lookbook(user=USER, lookbook_id=book["id"], image_id=two["images"][1]["id"])
    assert back["count"] == 1 and back["code"] == one["code"], (
        "the earlier pictures give back the earlier code"
    )
    assert Path(two["images"][1]["path"]).is_file(), "the file stays: the other code still stands for it"
    with pytest.raises(DreambotError, match="isn't in this Lookbook"):
        service.remove_from_lookbook(user=USER, lookbook_id=book["id"], image_id="nope")

    renamed = service.rename_lookbook(user=USER, lookbook_id=book["id"], name="Diner at dusk")
    assert renamed["slug"] == "diner-at-dusk" and renamed["code"] == one["code"]
    assert service.set_default_lookbooks(user=USER, lookbook_ids=[book["id"], "gone", book["id"]]) == [
        book["id"]
    ]
    assert service.lookbooks(USER)[0]["default"] and not service.lookbooks("discord:9")[0]["default"]

    with pytest.raises(DreambotError, match="isn't a public address"):
        await service.add_to_lookbook(user=USER, lookbook_id=book["id"], source="http://127.0.0.1/x.png")
    with pytest.raises(DreambotError, match="can't find that Lookbook"):
        await service.add_to_lookbook(user=USER, lookbook_id="nope", source=rust)


async def test_prompts_use_lookbooks_by_name_code_and_default(bot, tmp_path: Path):  # noqa: F811
    service, recorder = bot
    with pytest.raises(DreambotError, match="don't know a Lookbook called `dunes`"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="a fox --p dunes")
    with pytest.raises(DreambotError, match="no default Lookbook yet"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="a fox --p")

    book = service.create_lookbook(user=USER, name="Dunes")
    with pytest.raises(DreambotError, match="no pictures in it yet"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="a fox --p dunes")
    for name, colour in (("a", (200, 40, 40)), ("b", (40, 40, 201))):
        book = await service.add_to_lookbook(
            user=USER, lookbook_id=book["id"], source=picture(tmp_path, name, colour)
        )
    code = book["code"]

    job = await imagine(service, recorder, "a fox --look Dunes --n 2 --seed 3 --sv 3")
    assert job.status is JobStatus.DONE and job.model == "z-image-turbo"
    assert job.prompt == f"a fox --personalize {code} --count 2 --seed 3 --sv 3", (
        "the name is pinned to the code"
    )
    assert job.detail["style"]["sources"] == [
        {"kind": "look", "label": "Lookbook Dunes", "share": 1.0, "code": code}
    ]
    assert job.images[0].prompt != job.images[1].prompt, "each image leans on a different picture"
    assert all("in the style of a" in image.prompt for image in job.images)

    # Editing the book gives it a new code; the old code still means the old pictures.
    third = await service.add_to_lookbook(
        user=USER, lookbook_id=book["id"], source=picture(tmp_path, "c", (30, 160, 60))
    )
    assert third["code"] != code
    old = await imagine(service, recorder, f"a fox --p {code} --n 2 --seed 3 --sv 3")
    assert [image.prompt for image in old.images] == [image.prompt for image in job.images]
    service.delete_lookbook(lookbook_id=book["id"])
    assert service.lookbooks(USER) == []
    still = await imagine(service, recorder, f"a fox --p {code}::2 --n 1 --sv 3")
    assert still.status is JobStatus.DONE and f"--personalize {code}::2" in still.prompt, (
        "a code outlives its book"
    )
    with pytest.raises(DreambotError, match="don't know a Lookbook called `dunes`"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="a fox --p dunes")


async def test_the_personalize_switch_and_the_liked_book(bot, tmp_path: Path):  # noqa: F811
    service, recorder = bot
    settings = service.settings(USER)
    settings.personalize = True
    service.save_settings(USER, settings)
    plain = await imagine(service, recorder, "a fox --n 1")
    assert plain.status is JobStatus.DONE and "style" not in plain.detail, "nothing set up, nothing said"

    # Favourites become the `liked` book, which a bare --p falls back on.
    result = await service.act(user=USER, channel=CHANNEL, action=_fav(plain.id))
    assert result.image.favorite
    liked = await imagine(service, recorder, "a harbour --n 1 --sv 3")
    assert liked.detail["style"]["sources"][0]["label"] == "Lookbook Liked"
    assert "--personalize" not in liked.prompt, "the switch adds the look, not the parameter"

    book = service.create_lookbook(user=USER, name="Dunes")
    await service.add_to_lookbook(
        user=USER, lookbook_id=book["id"], source=picture(tmp_path, "a", (200, 40, 40))
    )
    service.set_default_lookbooks(user=USER, lookbook_ids=[book["id"]])
    chosen = await imagine(service, recorder, "a harbour --n 1 --sv 3")
    assert chosen.detail["style"]["sources"][0]["label"] == "Lookbook Dunes"
    named = await imagine(service, recorder, "a harbour --p liked --n 1 --sv 3")
    assert named.detail["style"]["sources"][0]["label"] == "Lookbook Liked"

    boarded = await imagine(service, recorder, "a harbour --n 1")
    assert boarded.model.startswith("flux2-klein") and boarded.detail["style"]["board"]
    assert EventType.THINKING not in recorder.types(boarded), "the book was read while the GPU was idle"


def _fav(job_id: str):
    from dreambot.core.actions import Action, ActionKind

    return Action(ActionKind.FAVORITE, job_id, 1)


# --------------------------------------------------------------------------------------
# From a chat
# --------------------------------------------------------------------------------------


def test_the_look_shortcut_needs_its_colon():
    from dreambot.core.actions import Action, ActionKind, decode_action, encode_action, parse_action

    assert parse_action("look 2: Autumn Diner") == Action(ActionKind.LOOK, index=2, prompt="Autumn Diner")
    assert parse_action("Look: dunes") == Action(ActionKind.LOOK, prompt="dunes")
    assert parse_action("/lookbook 1 : dunes") == Action(ActionKind.LOOK, index=1, prompt="dunes")
    for prompt in ("look at this fox", "lookbook of autumn coats", "look", "look:"):
        assert parse_action(prompt) is None, f"{prompt!r} is a prompt, not a command"
    # A phone capitalises the first word; the shortcuts that take free text must not mind.
    assert parse_action("Edit: Make it night") == Action(ActionKind.EDIT, prompt="Make it night")
    assert parse_action("Region the sky: an aurora").kind is ActionKind.VARY_REGION
    bound = Action(ActionKind.LOOK, "7QK2MD", 2)
    assert decode_action(encode_action(bound)) == bound


async def test_lookbooks_from_a_conversation(bot):  # noqa: F811
    from dreambot.core.commands import Commands

    service, recorder = bot
    commands = Commands(service)

    async def say(text: str) -> str:
        replies = await commands.handle(user=USER, channel=CHANNEL, text=text)
        return "\n".join(reply.text or "" for reply in replies)

    assert "no Lookbooks yet" in await say("/looks")
    first = await imagine(service, recorder, "a red barn --n 2 --seed 1")
    assert "Pick" in await say("look: Barns") or "which" in (await say("look: Barns")).lower(), (
        "a grid needs an image number"
    )
    assert "Started the Lookbook `Barns` with this picture. Use it with `--p barns`." in await say(
        "look 1: Barns"
    )
    assert "now has 2 pictures" in await say("look 2: barns")
    assert "now has 2 pictures" in await say("look 2: barns"), "the same picture is not added twice"
    assert first.images[0].path != service.lookbooks(USER)[0]["images"][0]["path"]

    listing = await say("/looks")
    assert "· barns — 2 picture(s), code " in listing and "★" not in listing.split("\n", 1)[1]
    assert "★ barns" in await say("/looks default barns")
    assert service.settings(USER).looks == [service.lookbooks(USER)[0]["id"]]
    assert "⚠️ I don't know a Lookbook called `nope`." in await say("/looks default nope")
    assert "· barns" in await say("/looks default none") and service.settings(USER).looks == []
    assert "⚠️" in await say("/looks default") and "⚠️" in await say("/looks what")

    assert "personalize off" in await say("/settings")
    await say("/looks default barns")
    assert "personalize on — every prompt uses barns" in await say("/settings personalize on")
    styled = await imagine(service, recorder, "a harbour --n 1 --sv 3")
    assert styled.detail["style"]["sources"][0]["label"] == "Lookbook Barns"
    assert "personalize off" in await say("/settings personalize off")
