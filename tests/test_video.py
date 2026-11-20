"""Video: a prompt filmed, a picture animated, a video continued.

Everything here runs without weights: the stand-in backend writes real MP4 files, so that what
is checked is everything around the model. What the videos look like on real weights is in the
README; these are the rules: what is asked for how, which model films it and what it is told,
what is kept of the result, and how each app shows it.
"""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageChops

from dreambot.config import Config
from dreambot.core import direct, render, video
from dreambot.core.actions import Action, ActionKind, decode_action, encode_action, parse_action
from dreambot.core.commands import Commands
from dreambot.core.jobs import VIDEO_KINDS, Clip, EventType, Job, JobEvent, JobImage, JobKind, JobStatus
from dreambot.core.params import find_param
from dreambot.core.planner import PlanError, Planner, ResolvedRefs, filmed, started
from dreambot.core.prompt import PromptError, parse_prompt
from dreambot.core.service import DIRECT_HINT, FIRST_SHARE, Dreambot, DreambotError, describe_job
from dreambot.core.sizes import video_dimensions, video_frames
from dreambot.engine import weights
from dreambot.engine.backends.base import BackendError, input_files, require_inputs
from dreambot.engine.client import Engine, EngineError, new_request_id
from dreambot.engine.protocol import VIDEO_OPS, GenerateRequest, Op, Progress
from dreambot.engine.registry import REGISTRY, Registry
from dreambot.frontends.base import is_video
from dreambot.frontends.discord_bot import BUTTON_ROWS, ROW_BUTTONS, _packed
from dreambot.frontends.web.server import WebApp
from tests.test_adapters import make_repo
from tests.test_frontend_base import MemoryFrontend
from tests.test_loras import Shelf, add, config, person, planner, used
from tests.test_service import CHANNEL, USER, Recorder, act, imagine, make_config
from tests.test_web import HOST, ORIGIN, finished, post, upload

WAN = "wan-2.2"
#: A video model that can do what Wan 2.2 cannot: lead a video up to a picture. The planner
#: goes by what a model says of itself, and this one keeps that way of it working.
ABLE = replace(
    REGISTRY.get(WAN),
    id="able",
    title="Able",
    aliases=(),
    video=replace(REGISTRY.get(WAN).video, ends=True),
)
#: Most of what is checked here is about one video. Four are made unless something says
#: otherwise, as in Midjourney; that is checked where it is said.
ONE = {"video": {"count": 1}}


@pytest.fixture
async def bot(tmp_path: Path):
    """A Dreambot on the stand-in backend, set up to make one video for each prompt."""
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    service = Dreambot(make_config(tmp_path, **ONE), engine=engine)
    recorder = Recorder()
    service.subscribe(recorder)
    await service.start()
    try:
        yield service, recorder
    finally:
        await service.close()


@pytest.fixture
def leads_up_to_pictures():
    """Wan 2.2 as it would be if it could lead a video up to a picture.

    The planner goes by what a model says of itself, and the stand-in backend does whatever it
    is asked: this is how the ways of the service that need such a model are kept working.
    """
    spec = REGISTRY.get(WAN)
    as_it_is = spec.video
    object.__setattr__(spec, "video", replace(as_it_is, ends=True))
    try:
        yield spec
    finally:
        object.__setattr__(spec, "video", as_it_is)


def able(cfg: Config, *installed: str) -> Planner:
    """A planner that has :data:`ABLE` to film with, next to the models named."""
    import random

    shelf = Shelf(cfg, ABLE.id, *installed)
    shelf.registry = Registry([*REGISTRY, ABLE])
    return Planner(shelf, cfg, rng=random.Random(1))


def film(p: Planner, prompt: str, settings=None, refs: ResolvedRefs | None = None):
    """Plan a video the way the service does for a prompt."""
    return p.video(parse_prompt(prompt), settings or person(), refs)


def directing(**settings):
    """Somebody who lets a model on the Mac direct their short prompts."""
    return person().__class__(**{**vars(person(**settings)), "enhance": True})


def clip_of(
    path: Path, frames: int = 49, size: tuple[int, int] = (320, 192), tone: float | None = None
) -> Path:
    """A small real video file: a grey picture that gets lighter, frame by frame."""
    shades = (Image.new("RGB", size, (n * 255 // frames,) * 3) for n in range(frames))
    return video.write(shades, path, size=size, fps=24, tone=tone)


# --------------------------------------------------------------------------------------
# How it is asked for
# --------------------------------------------------------------------------------------


def test_the_video_parameters_are_midjourneys_and_two_more():
    params = parse_prompt("a fox --video --motion high --loop --bs 2 --end @1 --seconds 3").params
    assert params.video is True and params.motion == "high" and params.loop is True and params.bs == 2
    assert [(ref.kind, ref.value) for ref in params.end] == [("attachment", "1")]
    # Not Midjourney's: its videos are five seconds long.
    assert params.seconds == 3
    assert parse_prompt("a fox --sec 4").params.seconds == 4
    # (Videos are silent, as Midjourney's are: no model here makes sound, and none is asked for it.)
    with pytest.raises(PromptError):
        parse_prompt("a fox --video --sound")

    # Any of them asks for a video; `--video` only has to be said when none of the others is.
    for asked in ("--video", "--motion low", "--loop", "--end @1", "--bs 1", "--seconds 2", "--from words"):
        assert parse_prompt(f"a fox {asked}").params.wants_video, asked
    assert not parse_prompt("a fox --ar 16:9 --chaos 20").params.wants_video

    for wrong, why in (
        ("--motion wild", "must be one of: low, high"),
        ("--seconds 0", "between 1 and 5"),
        ("--seconds 6", "between 1 and 5"),
        ("--bs 9", "between 1 and 4"),
    ):
        with pytest.raises(PromptError, match=why):
            parse_prompt(f"a fox {wrong}")

    # They are no longer "planned": each does something.
    assert parse_prompt("a fox --video --motion high --loop --bs 2").planned == []
    assert {find_param(name).support for name in ("video", "loop", "bs", "seconds")} == {"native"}
    # Midjourney's ending frame is known and not yet done: the video model cannot be led up to one.
    assert find_param("end").support == "planned"
    # Midjourney's motion is a setting of its model; here it is said to the model in words.
    assert find_param("motion").support == "emulated" and find_param("motion").how


def test_the_typed_forms_are_strict_about_what_follows_the_word():
    def typed(text: str):
        action = parse_action(text)
        return None if action is None else (action.kind.value, action.index, action.value, action.prompt)

    assert typed("animate") == ("animate", None, None, None)
    assert typed("Animate 2") == ("animate", 2, None, None)
    assert typed("/animate 2 high") == ("animate", 2, "high", None)
    assert typed("animate low motion") == ("animate", None, "low", None)
    # Words come after a colon, capitals kept; parameters alone may follow directly.
    assert typed("animate 3: She turns to the Camera --loop") == (
        "animate",
        3,
        None,
        "She turns to the Camera --loop",
    )
    assert typed("animate high: the dog runs off") == ("animate", None, "high", "the dog runs off")
    assert typed("animate 2 --loop --seconds 3") == ("animate", 2, None, "--loop --seconds 3")
    assert typed("extend") == ("extend", None, None, None)
    assert typed("extend high") == ("extend", None, "high", None)
    assert typed("extend 1: it starts to rain --seconds 2") == (
        "extend",
        1,
        None,
        "it starts to rain --seconds 2",
    )
    # A prompt that happens to start with the word is still a prompt.
    for prompt in (
        "animate the statue",
        "extend the bridge over the river",
        "animate 2 a dancing robot",
        "animate 0",
    ):
        assert typed(prompt) is None, prompt

    animate = parse_action("animate 2 high").on("ABC123")
    assert encode_action(animate) == "db:a:ABC123:2:high" and decode_action("db:a:ABC123:2:high") == animate
    extend = parse_action("extend").on("ABC123", 1)
    assert encode_action(extend) == "db:xt:ABC123:1"
    assert decode_action("db:xt:ABC123:1") == Action(ActionKind.EXTEND, "ABC123", 1)


def test_a_video_is_sized_in_the_models_steps():
    # About 0.4 megapixels, in multiples of 32, whatever the shape.
    assert video_dimensions(16 / 9) == (832, 480)
    assert video_dimensions(1) == (608, 608)
    assert video_dimensions(9 / 16) == (480, 832)
    assert video_dimensions(4 / 3) == (736, 544)
    assert video_dimensions(2.4) == (992, 416)
    # `--hd` is one size up: 720p in 16:9.
    assert video_dimensions(16 / 9, scale=1.5) == (1248, 704)
    with pytest.raises(ValueError, match="at most three times"):
        video_dimensions(4)

    # One frame, and then four at a time.
    assert video_frames(5, fps=24, step=4) == 121
    assert video_frames(2, fps=24, step=4) == 49
    assert video_frames(1, fps=24, step=4, minimum=5) == 25
    assert video_frames(30, fps=24, step=4, maximum=121) == 121


# --------------------------------------------------------------------------------------
# What films it, and what it is told
# --------------------------------------------------------------------------------------


def test_a_video_asked_for_in_words_starts_on_a_picture_made_from_them(tmp_path: Path):
    """Midjourney's way: every video starts on a picture. So the words are drawn first."""
    cfg = config(tmp_path)
    add(cfg)  # a LoRA for the picture model
    p = planner(cfg, WAN, "z-image-turbo")
    plan = film(p, "a fox running through snow --video --seed 7")
    (shot,) = plan.renders
    assert plan.model == WAN == shot.model and plan.op is Op.IMG2VID is shot.op
    # Four videos of five seconds, in 16:9, silent: Midjourney's video.
    assert (plan.width, plan.height) == (shot.width, shot.height) == (832, 480)
    assert (shot.frames, shot.fps, shot.seeds) == (121, 24, [7, 8, 9, 10])
    # What each starts on is still to be made: a picture for each, with its seed and in its
    # shape, by the picture model and at the size that one draws best.
    assert shot.image is None and shot.end_image is None and shot.video is None
    first = plan.first
    assert first is not None and (first.model, first.op) == ("z-image-turbo", Op.TXT2IMG)
    assert first.seeds == plan.seeds == [7, 8, 9, 10] and first.finish is None
    assert (first.width, first.height) == (1424, 832) and first.text == "a fox running through snow"
    assert plan.text == "a fox running through snow" and plan.notes == []
    assert plan.detail == {"video": {"motion": "low", "seconds": 5.04, "picture": "z-image-turbo"}}
    # Until a director says more: the words, and what the camera does.
    assert shot.prompt == f"a fox running through snow. {direct.CAMERA_WORDS['low']}"
    # How much moves is also told to the engine, which starts a video differently for it.
    assert shot.extra == {"motion": "low"}

    # Once the pictures are made, each video starts on its own.
    going = started(plan, {n: f"/made/{n}.png" for n in (1, 2, 3, 4)})
    assert going.first is None and going.shoot is plan.shoot
    assert [(shot.image, shot.end_image, shot.seeds, shot.slots) for shot in going.renders] == [
        (f"/made/{n}.png", None, [6 + n], [n]) for n in (1, 2, 3, 4)
    ]

    # What shapes a picture shapes those, and is not said to have done nothing. A picture model
    # named in the prompt is the one that draws them; `--hd` is the size of the video.
    shaped = film(
        p, "a fox --video --bs 2 --stylize 300 --chaos 20 --ar 1:1 --hd --lora pixel --model z-image-turbo"
    )
    assert shaped.model == WAN and shaped.first.model == "z-image-turbo" and shaped.notes == []
    assert (shaped.width, shaped.height) == (928, 928) and len(shaped.first.seeds) == 2
    assert (shaped.first.width, shaped.first.height) == (1024, 1024) and shaped.first.finish is None
    assert [name for name, _ in used(shaped.first.renders[0])] == ["pixel.safetensors"]
    assert shaped.detail["loras"] == shaped.first.detail["loras"]
    assert film(p, "a fox --video --model wan-2.2").first.model == "z-image-turbo"

    # A loop can be asked for in words: it comes back to the picture that is made for it.
    loop = film(p, "a candle burning --loop --bs 1")
    assert loop.detail["video"]["loop"] is True and loop.renders[0].end_image is None
    (lit,) = started(loop, {1: "/made/candle.png"}).renders
    assert lit.image == lit.end_image == "/made/candle.png"


def test_a_prompt_can_be_filmed_from_its_words_alone(tmp_path: Path, sample_image: Path):
    """What Midjourney does not do: the video model is given the words and nothing to start on."""
    p = planner(config(tmp_path), WAN, "z-image-turbo")
    plan = film(p, "a fox running through snow --video --from words --seed 7")
    (shot,) = plan.renders
    assert plan.first is None and plan.model == WAN == shot.model and plan.op is Op.TXT2VID is shot.op
    assert (plan.width, plan.height) == (shot.width, shot.height) == (832, 480)
    assert (shot.frames, shot.fps, shot.seeds) == (121, 24, [7, 8, 9, 10])
    assert shot.image is None and shot.end_image is None and shot.video is None
    # The words, and what the camera does: what happens is the person's to say.
    assert shot.prompt == f"a fox running through snow. {direct.CAMERA_WORDS['low']}"
    assert plan.text == "a fox running through snow"
    assert plan.detail == {"video": {"motion": "low", "seconds": 5.04}} and plan.notes == []

    # `--from` asks for a video by itself, as every parameter of a video's does.
    more = film(p, "a fox --from words --seconds 2 --bs 2 --ar 1:1 --hd --motion high --no people")
    (shot,) = more.renders
    assert (more.width, more.height) == (928, 928) and shot.frames == 49
    assert len(shot.seeds) == 2 and shot.slots == [1, 2]
    assert shot.prompt == f"a fox. {direct.CAMERA_WORDS['high']} Without people."
    assert shot.extra == {"motion": "high"}
    assert more.detail == {"video": {"motion": "high", "seconds": 2.04, "hd": True}}
    assert more.notes == ["Wan 2.2 has no negative prompt, so `--no` is best-effort."]
    # `--n` counts videos too, and the parameters that shape pictures say that they did nothing.
    counted = film(p, "a fox --video --from words --n 3 --chaos 20 --stylize 300")
    assert len(counted.seeds) == 3
    assert counted.notes == ["`--stylize` and `--chaos` shape pictures and did nothing for this video."]

    # A person can have it so for every prompt, and so can the config. The prompt has the last word.
    assert film(p, "a fox --video", settings=person(picture_first=False)).op is Op.TXT2VID
    assert film(p, "a fox --video --from picture", settings=person(picture_first=False)).first is not None
    off = planner(config(tmp_path, video={"picture_first": False}), WAN, "z-image-turbo")
    assert film(off, "a fox --video").first is None
    assert film(off, "a fox --video", settings=person(picture_first=True)).first is not None
    # With no model to draw a picture, the words are filmed, and it says so.
    alone = planner(config(tmp_path), WAN)
    undrawn = film(alone, "a fox --video")
    assert undrawn.first is None and undrawn.op is Op.TXT2VID
    assert undrawn.notes[-1].startswith("No picture could be made to start on, so the words were filmed.")
    with pytest.raises(PlanError, match="No image model is installed"):
        film(alone, "a fox --video --from picture")
    # A video that is told where to end is not given a start that has nothing to do with it.
    # (By a model that can lead a video up to a picture, which Wan 2.2 cannot.)
    ending = film(
        able(config(tmp_path), "z-image-turbo"), "a fox --video", refs=ResolvedRefs(end=[str(sample_image)])
    )
    assert ending.first is None and ending.op is Op.IMG2VID and ending.renders[0].end_image


def test_a_video_starts_on_the_picture_it_is_given_and_has_its_shape(tmp_path: Path, sample_image: Path):
    p = planner(config(tmp_path), WAN)
    start = str(sample_image)  # 512x384
    plan = film(p, "--video", refs=ResolvedRefs(image_prompts=[start]))
    (shot,) = plan.renders
    assert plan.op is Op.IMG2VID and shot.image == start == plan.source and shot.end_image is None
    assert (plan.width, plan.height) == (736, 544), "4:3, like the picture"
    # With nothing said, the picture is all there is to go on: the model is told how it should move.
    assert shot.prompt == direct.MOTION_WORDS["low"]
    assert film(p, "--video --ar 16:9", refs=ResolvedRefs(image_prompts=[start])).width == 832

    # A loop comes back to the frame it starts on.
    loop = film(p, "the fox blinks --loop", refs=ResolvedRefs(image_prompts=[start]))
    assert (
        loop.renders[0].image == loop.renders[0].end_image == start and loop.detail["video"]["loop"] is True
    )
    assert loop.renders[0].frames == 121 and loop.notes == []
    # A video can also be given only where it ends, or where it starts and where it ends, when
    # the model can lead one up to a picture. Wan 2.2 cannot, and says so with what it can do.
    ending = film(able(config(tmp_path)), "a ship sails in --video", refs=ResolvedRefs(end=[start]))
    assert (
        ending.op is Op.IMG2VID and ending.renders[0].image is None and ending.renders[0].end_image == start
    )
    both = film(able(config(tmp_path)), "--video", refs=ResolvedRefs(image_prompts=[start], end=[start]))
    assert both.renders[0].image == both.renders[0].end_image == start and "loop" not in both.detail["video"]
    for refs in (ResolvedRefs(end=[start]), ResolvedRefs(image_prompts=[start], end=[start])):
        with pytest.raises(
            PlanError, match=r"Wan 2\.2 can start a video on a picture but can't lead one up to"
        ):
            film(p, "a ship sails in --video", refs=refs)

    second = film(p, "--video", refs=ResolvedRefs(image_prompts=[start, start]))
    assert second.notes == ["A video starts on one image; the first one was used."]


def test_what_cannot_be_filmed_is_refused_with_a_way_forward(tmp_path: Path, sample_image: Path):
    p = planner(config(tmp_path), WAN)
    start = ResolvedRefs(image_prompts=[str(sample_image)])
    for prompt, refs, why in (
        ("--video", None, "Give me something to film"),
        ("a fox --loop", None, "`--loop` needs an image to start on"),
        ("a fox --video --ar 4:1", None, "at most three times as wide"),
        # What shapes a picture cannot shape one that is already there, or one that is not made.
        (
            "it runs --video --sref 123",
            start,
            "`--sref` shapes a picture.*turn it into a video with `animate`",
        ),
        ("it runs --video --lora x --oref @1", start, "`--oref` and `--lora` shape a picture"),
        ("a fox --video --from words --sref 123", None, "`--sref` shapes a picture"),
        ("a fox --video --edit @1", None, "`--edit` shapes a picture"),
    ):
        with pytest.raises(PlanError, match=why):
            film(p, prompt, refs=refs)
    with pytest.raises(PlanError, match="`--loop` and `--end` can't both be given"):
        film(p, "a fox --loop", refs=ResolvedRefs(image_prompts=start.image_prompts, end=start.image_prompts))

    with pytest.raises(PlanError, match=r"none is installed.*dreambot models pull wan-2\.2"):
        film(planner(config(tmp_path), "z-image-turbo"), "a fox --video")
    # A picture model cannot be asked to film. (Named in a prompt of words alone, it is the one
    # that draws the picture the video starts on.)
    both = planner(config(tmp_path), WAN, "z-image-turbo")
    for prompt, refs in (
        ("it runs --video --model z-image-turbo", start),
        ("a fox --video --from words --model z-image-turbo", None),
    ):
        with pytest.raises(PlanError):
            film(both, prompt, refs=refs)


def test_animate_and_extend_are_the_same_film_with_a_start_or_a_past(tmp_path: Path, sample_image: Path):
    p = planner(config(tmp_path), WAN)
    animated = p.animate(source=str(sample_image), text="", motion="high")
    (shot,) = animated.renders
    assert animated.op is Op.IMG2VID and shot.image == str(sample_image) and shot.frames == 121
    assert shot.prompt == direct.MOTION_WORDS["high"] and animated.detail["video"]["motion"] == "high"
    # `--motion` in the words wins over the button.
    said = p.animate(
        source=str(sample_image), text="it nods", motion="high", params=parse_prompt("x --motion low").params
    )
    assert said.detail["video"]["motion"] == "low" and said.renders[0].prompt.startswith("it nods. ")

    made = Clip(str(tmp_path / "made.mp4"), frames=121, fps=24.0)
    longer = p.extend(
        clip=made, size=(832, 480), text="", model=WAN, motion="high", before="The fox is trotting."
    )
    (shot,) = longer.renders
    assert longer.op is Op.EXTEND and shot.video == made.path and (longer.width, longer.height) == (832, 480)
    # Four seconds are added, four frames at a time.
    assert (shot.frames, shot.fps) == (96, 24.0)
    assert longer.detail == {"video": {"motion": "high", "seconds": 9.04, "added": 4.0}}
    # Left to itself it goes on as it was going, from the prompt it was made from: Midjourney's
    # Extend Auto. Only what the camera does is said anew.
    assert shot.prompt == f"The fox is trotting. {direct.CAMERA_WORDS['high']}" and longer.shoot is None
    assert shot.extra == {"motion": "high"}

    told = p.extend(
        clip=made, size=(832, 480), text="it leaps", params=parse_prompt("x --seconds 2 --ar 1:1").params
    )
    assert told.renders[0].frames == 48 and told.renders[0].prompt.startswith("it leaps. ")
    assert told.notes == ["`--ar` is for a new video; an extension keeps the shape of the one it continues."]
    # One go adds what the model makes in one clip, less the five frames it goes on from.
    most = p.extend(clip=made, size=(832, 480), text="", params=parse_prompt("x --seconds 5").params)
    assert most.renders[0].frames == 116 and most.detail["video"]["added"] == 4.83
    assert most.notes == ["Wan 2.2 adds up to 4.8 seconds at a time, so that is what was added."]
    # An extension can be led up to a picture by a model that can do that; a loop is something
    # a video is from the start.
    landed = able(config(tmp_path)).extend(clip=made, size=(832, 480), text="", end="shore.png")
    assert landed.renders[0].end_image == "shore.png" and landed.renders[0].video == made.path
    # Continued as a loop, it is led back to its own first frame, which whoever asks hands over.
    round_ = able(config(tmp_path)).extend(
        clip=made, size=(832, 480), text="", end="first.png", params=parse_prompt("x --loop").params
    )
    assert round_.renders[0].end_image == "first.png" and round_.detail["video"]["loop"] is True
    assert round_.notes == []
    with pytest.raises(PlanError, match=r"Wan 2\.2 can't lead a video up to a picture"):
        p.extend(clip=made, size=(832, 480), text="", end="shore.png")
    # As many continuations are made as videos of anything else: four, or what the person has set.
    assert len(longer.seeds) == 4 and len(p.extend(clip=made, size=(832, 480), text="", videos=2).seeds) == 2

    with pytest.raises(PlanError, match="too short to extend: it has 4 frames"):
        p.extend(clip=Clip("x.mp4", 4, 24.0), size=(832, 480), text="")
    with pytest.raises(PlanError, match=r"extends a video up to 21\. It is 20 seconds now"):
        p.extend(clip=Clip("x.mp4", 481, 24.0), size=(832, 480), text="")


# --------------------------------------------------------------------------------------
# The director
# --------------------------------------------------------------------------------------


def test_the_director_is_asked_when_there_is_something_left_to_say(tmp_path: Path, sample_image: Path):
    shown = direct.Shoot(text="", count=1, start=str(sample_image))
    assert direct.wanted(shown), "a picture and no words: somebody has to say what happens"
    assert direct.wanted(direct.Shoot(text="a fox", count=2))
    assert not direct.wanted(direct.Shoot(text="", count=1)), "nothing to go on"
    assert not direct.wanted(direct.Shoot(text="word " * direct.DETAILED_WORDS, count=1)), "already directed"

    p = planner(config(tmp_path), WAN)
    assert film(p, "a fox --video").shoot is None, "the person has art direction switched off"
    asked = film(p, "a fox --video --bs 2 --motion high --no people", settings=directing())
    assert asked.shoot == direct.Shoot(
        text="a fox", count=2, motion="high", seconds=121 / 24, avoid=("people",)
    )
    assert film(p, "a fox --video --raw", settings=directing()).shoot is None, "`--raw` films the words alone"
    picture = p.animate(
        source=str(sample_image), text="", motion="low", params=parse_prompt("x --loop").params
    )
    assert picture.shoot is not None and picture.shoot.start == str(sample_image) and picture.shoot.loop
    assert p.animate(source=str(sample_image), text="", directed=False).shoot is None
    # A video that is continued with nothing new said goes on from its own prompt, and nobody is
    # asked. With new words, or with no prompt to go on from, the director looks at where it has got to.
    made = Clip(str(tmp_path / "made.mp4"), frames=121, fps=24.0)
    assert p.extend(clip=made, size=(832, 480), text="", before="The fox is trotting.").shoot is None
    assert not direct.wanted(
        direct.Shoot(text="", count=1, clip=made.path, before=f"The fox is trotting. {direct.STILL_CAMERA}")
    )
    assert direct.wanted(direct.Shoot(text="", count=1, clip=made.path, before=direct.MOTION_WORDS["low"]))
    going = p.extend(clip=made, size=(832, 480), text="it leaps", before="The fox is trotting.")
    assert (
        going.shoot is not None
        and going.shoot.clip == made.path
        and going.shoot.before == "The fox is trotting."
    )
    assert round(going.shoot.seconds) == 4


def test_what_the_director_is_told(sample_image: Path):
    start = str(sample_image)
    low = direct.instruction(direct.Shoot(text="", count=2, motion="low", start=start))
    assert "first frame of a video 5 seconds long" in low and "Write 2 different prompts" in low
    assert "was made from the words" not in low
    told = direct.instruction(direct.Shoot(text="", count=1, start=start, about='a "ginger" cat on a sill'))
    assert "The picture was made from the words \"a 'ginger' cat on a sill\". Write one prompt" in told
    assert "The camera does not move" in low and "Say only what moves" in low
    # What a small model is given as an example it writes into every line, so it is given none
    # of the things that could move; and what it says about the light, the video model overdoes.
    assert "Name nothing that cannot be seen in the picture" in low
    assert "The light, the weather and the time of day stay exactly as they are" in low
    assert low.endswith("Output exactly 2 lines numbered 1. to 2. and nothing else.")
    assert "asked for" not in low and "loop" not in low
    lively = direct.instruction(direct.Shoot(text="", count=1, motion="high", start=start))
    assert "the camera moves with it" in lively and "If there are none, the camera itself moves" in lively

    high = direct.instruction(
        direct.Shoot(
            text='the dog runs "home"',
            count=1,
            motion="high",
            start=start,
            loop=True,
            avoid=("people",),
        )
    )
    # What the person asked for is not toned down to suit how much else moves: told that movements
    # are small, the director wrote a fight as two animals circling each other.
    assert "with all the force and speed their words have" in high and "the camera moves with it" in high
    assert "The movement is large" not in high
    quiet = direct.instruction(direct.Shoot(text="two bears fighting", count=1, start=start))
    assert "do not tone it down" in quiet and "The camera does not move" in quiet
    assert "The movements are small and slow" not in quiet
    # Of a loop only half is filmed, and it has to look right played backwards too.
    assert "first frame of a video 2 seconds long" in high and "played forwards and then backwards" in high
    # The person's words are put before the line as they are, so the director is not to repeat them.
    assert "The person asked for this: \"the dog runs 'home'\"" in high and "do not repeat it" in high
    assert "Leave these out, and do not mention them: people." in high
    assert high.endswith("Output one line, numbered 1., and nothing else.")

    # A video that is continued is directed from where it has got to.
    next_ = direct.instruction(
        direct.Shoot(text="", count=1, clip="made.mp4", before="The fox is trotting.", seconds=4)
    )
    assert "last frame so far" in next_ and "goes on for 4 more seconds" in next_
    assert (
        'Up to here, this is what happened: "The fox is trotting."' in next_ and "what happens next" in next_
    )

    # With nothing to look at, the same is asked of a language model, about the words.
    words = direct.request(direct.Shoot(text="a fox running through snow", count=2, motion="high"))
    assert words.startswith("Prompt: a fox running through snow\n") and "the camera moves with it" in words
    assert "with all the force and speed their words have" in words
    assert words.endswith("Write exactly 2 lines, numbered from 1. Nothing else.")
    assert "one continuous shot" in direct.SYSTEM and "Do not restate the prompt" in direct.SYSTEM
    assert "not even to watch" in direct.SYSTEM
    assert "by the name the prompt gives it" in direct.SYSTEM


def test_the_directors_line_is_added_to_the_persons_words_never_put_in_their_place(sample_image: Path):
    start = str(sample_image)
    auto = direct.Shoot(text="", count=1, start=start)
    assert (
        direct.compose(auto, "The man is lowering his arm")
        == "The man is lowering his arm. The camera stays still."
    )
    # A line that says what the camera does is taken at its word, and so are the person's own.
    assert direct.compose(auto, "The camera is rising slowly.") == "The camera is rising slowly."
    said = direct.Shoot(text="the Dog runs off, camera follows", count=1, start=start)
    assert (
        direct.compose(said, "Its ears are flapping.")
        == "the Dog runs off, camera follows. Its ears are flapping."
    )
    lively = direct.Shoot(text="", count=1, motion="high", start=start, avoid=("people", "cars"))
    assert direct.compose(lively, "The dog is running") == "The dog is running. Without people, cars."

    # Without a line, how much should move is said in a fixed sentence.
    assert direct.compose(auto, "") == direct.MOTION_WORDS["low"]
    # A picture Dreambot made is said to the video model in the words it was made from, first:
    # told only what moves, the model is free with everything else.
    drawn = replace(auto, about="a lighthouse on a cliff at dusk")
    assert direct.compose(drawn, "The waves are breaking") == (
        "a lighthouse on a cliff at dusk. The waves are breaking. The camera stays still."
    )
    assert direct.compose(drawn, "") == f"a lighthouse on a cliff at dusk. {direct.MOTION_WORDS['low']}"
    assert direct.compose(replace(drawn, text="its lamp turns"), "") == (
        "a lighthouse on a cliff at dusk. its lamp turns. The camera holds still."
    )
    # Not twice, where the person's words are those words or hold them (a prompt that is filmed
    # from a picture made of it), and not for a video that is continued.
    for words in ("a lighthouse on a cliff at dusk", "A lighthouse on a cliff at dusk, its lamp turning."):
        assert direct.compose(replace(drawn, text=words), "").count("lighthouse") == 1, words
    assert direct.compose(replace(drawn, text="A lighthouse"), "") == "A lighthouse. The camera holds still."
    continued = direct.Shoot(text="it rains", count=1, clip="made.mp4", about="a lighthouse on a cliff")
    assert direct.compose(continued, "") == "it rains. The camera holds still."
    # After words of the person's own, only the camera is spoken for: a fight is not made gentle.
    fight = direct.Shoot(text="two bears fighting", count=1)
    assert direct.compose(fight, "") == "two bears fighting. The camera holds still."
    assert (
        direct.compose(replace(fight, motion="high"), "")
        == "two bears fighting. The camera moves with the action."
    )
    assert direct.compose(said, "") == "the Dog runs off, camera follows.", "they spoke for the camera"
    assert direct.compose(direct.Shoot(text="", count=1), "") == ""
    # A video that is continued then goes on as it was going, at the pace now asked for.
    going = direct.Shoot(
        text="", count=1, motion="high", clip="made.mp4", before=f"The fox is trotting. {direct.STILL_CAMERA}"
    )
    assert direct.compose(going, "") == f"The fox is trotting. {direct.CAMERA_WORDS['high']}"
    # What it was made from may have said what the camera does, and then that stands.
    panned = direct.Shoot(text="", count=1, clip="made.mp4", before="The fox trots. The camera pans left.")
    assert direct.compose(panned, "") == "The fox trots. The camera pans left."
    # With nothing to go on, it is told how much should move, as a picture is.
    bare = direct.Shoot(text="", count=1, motion="high", clip="made.mp4", before=direct.MOTION_WORDS["low"])
    assert direct.compose(bare, "") == direct.MOTION_WORDS["high"]

    # The person's words go before the line as they are, so a line that opens with them loses them.
    fight = direct.Shoot(text="Two polar bears fighting in the octagon.", count=2, start=start)
    assert direct.parse(
        [
            "Two polar bears fighting in the octagon, the bears are pushing and the paws are striking",
            "two polar bears fighting in the octagon. Two polar bears fighting in the octagon: they are lunging and biting each other",
        ],
        fight,
    ) == ["The bears are pushing and the paws are striking", "They are lunging and biting each other"]
    assert direct.parse(["Two polar bears fighting in the octagonal ring are pushing hard"], fight)[
        0
    ].startswith("Two polar bears fighting in the octagonal ring"), "only their words whole"

    # What comes back is tidied, and a line that is no line is left out.
    two = direct.Shoot(text="", count=3, start=start)
    assert direct.parse(['  "prompt 1: the grass is swaying in the wind"', "ok", "x " * 200], two) == [
        "The grass is swaying in the wind",
        ("x " * direct.MAX_WORDS).strip().capitalize(),
        "",
    ]


def test_the_vision_model_is_the_one_that_directs(tmp_path: Path, monkeypatch):
    """The video model brings no language model that can look at a picture: the one /describe uses is asked."""
    from dreambot.core.planner import Catalog

    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    assert not REGISTRY.get(WAN).helper_repos

    catalog = Catalog(config(tmp_path))
    assert catalog.vision_model() is None, "nothing is installed: videos are filmed as they are planned"
    qwen = make_repo(
        cache, "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", {"model.safetensors": b"x", "config.json": b"{}"}
    )
    assert catalog.vision_model() == str(qwen)


def test_each_video_of_a_job_gets_a_line_of_its_own(tmp_path: Path):
    p = planner(config(tmp_path), WAN)
    plan = film(p, "a fox --video --bs 3 --seed 5", settings=directing())
    assert len(plan.renders) == 1 and plan.renders[0].seeds == [5, 6, 7]

    directed = filmed(plan, ["The fox is trotting to the left", "The fox is sniffing the snow"])
    assert directed.shoot is None and directed.seeds == [5, 6, 7]
    assert [(shot.seeds, shot.slots) for shot in directed.renders] == [([5], [1]), ([6], [2]), ([7], [3])]
    assert [shot.prompt for shot in directed.renders] == [
        f"a fox. The fox is trotting to the left. {direct.STILL_CAMERA}",
        f"a fox. The fox is sniffing the snow. {direct.STILL_CAMERA}",
        # The director came up one short: this one is filmed as it was planned.
        f"a fox. {direct.CAMERA_WORDS['low']}",
    ]
    assert all(shot.frames == 121 and shot.model == WAN for shot in directed.renders)
    undirected = film(p, "a fox --video")
    assert filmed(undirected, ["anything"]) is undirected


# --------------------------------------------------------------------------------------
# Video files
# --------------------------------------------------------------------------------------


def test_video_files_are_written_read_and_finished_by_dreambots_own_ffmpeg(tmp_path: Path):
    assert Path(video.ffmpeg()).is_file()
    made = clip_of(tmp_path / "made.mp4", frames=49)
    info = video.probe(made)
    assert (info.width, info.height, info.frames, info.fps, info.sound) == (320, 192, 49, 24.0, False)
    assert round(info.seconds, 2) == 2.04
    assert video.probe(clip_of(tmp_path / "tone.mp4", frames=25, tone=440)).sound is True

    # The first frame is black and the last nearly white.
    first, last = video.first_frame(made), video.last_frame(made)
    assert first.size == last.size == (320, 192)
    assert first.getpixel((10, 10))[0] < 20 and last.getpixel((10, 10))[0] > 225

    # A frame in the middle, and the frames of its end as they lie in memory.
    assert 100 < video.frame(made, 24).getpixel((10, 10))[0] < 150
    tail = video.pixels(made, last=5)
    assert len(tail) == 5 * 320 * 192 * 3 and tail == video.pixels(made)[-len(tail) :]
    assert tail[0] > 215 and len(video.pixels(made)) == 49 * 320 * 192 * 3

    # Frames can be handed over as raw pixels, at the quality videos are kept at, and what a
    # video was made from is written into it.
    notes = {"prompt": "a fox — in “snow”", "seed": 7}
    shades = (bytes([n * 5]) * (320 * 192 * 3) for n in range(49))
    kept = video.write(
        shades, tmp_path / "kept.mp4", size=(320, 192), fps=24, quality=video.QUALITY, notes=notes
    )
    assert video.probe(kept).frames == 49 and 115 < video.frame(kept, 24).getpixel((10, 10))[0] < 125
    said = subprocess.run([video.ffmpeg(), "-i", str(kept)], capture_output=True, check=False).stderr.decode()
    comment = next(
        line.split(":", 1)[1].strip() for line in said.splitlines() if line.strip().startswith("comment")
    )
    assert json.loads(comment) == notes
    with pytest.raises(video.VideoError, match="not the size of the video"):
        video.write([b"short"], tmp_path / "bad.mp4", size=(320, 192), fps=24)

    # As a GIF, which plays anywhere: smaller in size and slower.
    animated = video.gif(made, tmp_path / "made.gif", width=160, fps=12)
    with Image.open(animated) as opened:
        assert opened.format == "GIF" and opened.size == (160, 96) and 20 <= opened.n_frames <= 26
    assert video.frame(made, 24).getpixel((10, 10))[0] in range(110, 140), "the frame halfway"

    # A copy small enough for a chat app that takes no more. (Noise is what a video file is largest for.)
    noise = (Image.effect_noise((320, 192), 90).convert("RGB") for _ in range(49))
    heavy = video.write(noise, tmp_path / "heavy.mp4", size=(320, 192), fps=24, tone=440)
    limit = 100_000
    assert heavy.stat().st_size > 2 * limit
    small = video.fit(heavy, tmp_path / "small.mp4", limit=limit)
    assert small.stat().st_size <= limit
    assert (video.probe(small).frames, video.probe(small).sound) == (49, True)
    with pytest.raises(video.VideoError, match="too long to be made small enough"):
        video.fit(made, tmp_path / "tiny.mp4", limit=2000)

    for wrong, why in ((tmp_path / "missing.mp4", "couldn't find"), (tmp_path / "text.mp4", "couldn't read")):
        if "text" in wrong.name:
            wrong.write_text("not a video")
        with pytest.raises(video.VideoError, match=why):
            video.probe(wrong)
    with pytest.raises(video.VideoError, match="no frames"):
        video.write(iter(()), tmp_path / "empty.mp4", size=(320, 192), fps=24)
    with pytest.raises(video.VideoError, match="have to be even"):
        video.write(iter(()), tmp_path / "odd.mp4", size=(321, 192), fps=24)

    assert is_video("a/b/clip.MP4") and is_video(Path("x.mov")) and not is_video("x.png")


def test_a_video_is_made_longer_without_encoding_again_what_it_had(tmp_path: Path):
    def shades(first: int, count: int) -> list[bytes]:
        return [bytes([first + n * 4]) * (64 * 32 * 3) for n in range(count)]

    notes = {"prompt": "a fox", "seed": 7}
    made = video.write(shades(0, 25), tmp_path / "made.mp4", size=(64, 32), fps=24, quality=video.QUALITY)
    longer = video.extended(made, shades(100, 12), tmp_path / "longer.mp4", notes=notes)
    info = video.probe(longer)
    assert (info.width, info.height, info.frames, info.fps, info.sound) == (64, 32, 37, 24.0, False)
    before, after = video.pixels(made), video.pixels(longer)
    # The frames it had are the same to the pixel: they were put in front as they were.
    assert after[: len(before)] == before
    assert abs(after[len(before)] - 100) < 4 and abs(after[-1] - 144) < 4
    said = subprocess.run(
        [video.ffmpeg(), "-i", str(longer)], capture_output=True, check=False
    ).stderr.decode()
    assert '"seed": 7' in said
    # Again: three parts end to end.
    longest = video.extended(longer, shades(200, 8), tmp_path / "longest.mp4")
    assert video.probe(longest).frames == 45 and video.pixels(longest)[: len(after)] == after

    # A video that was written another way is encoded again with what is added, and one with
    # sound keeps its sound.
    other = clip_of(tmp_path / "other.mp4", frames=25, size=(64, 32), tone=440)
    assert video._setup(other) != video._setup(made) and video._setup(tmp_path / "missing.mp4") is None
    joined = video.extended(other, shades(100, 12), tmp_path / "joined.mp4")
    assert (video.probe(joined).frames, video.probe(joined).sound) == (37, True)
    start = video.pixels(joined)[: 64 * 32 * 3]
    assert abs(start[0] - video.pixels(other)[0]) < 6


# --------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------


def request(tmp_path: Path, op: Op, **more) -> GenerateRequest:
    values = {
        "id": new_request_id(),
        "model": WAN,
        "op": op,
        "prompt": "a fox",
        "width": 320,
        "height": 192,
        "frames": 25,
        "fps": 24,
        "seeds": [5],
        "out_dir": str(tmp_path / "engine"),
        "basename": "v",
    }
    return GenerateRequest(**{**values, **more})


async def test_the_engine_hands_back_a_video_and_the_picture_it_starts_on(
    engine: Engine, tmp_path: Path, sample_image: Path
):
    seen: list[tuple[int, int, int]] = []

    def watch(event) -> None:
        if isinstance(event, Progress):
            seen.append((event.index, event.step, event.total))

    result = await engine.generate(request(tmp_path, Op.TXT2VID, seeds=[5, 6]), on_event=watch)
    first, second = result.images
    # Each video reports the model's own steps.
    assert seen == [(index, step, 3) for index in (0, 1) for step in (1, 2, 3)]
    for made in (first, second):
        assert Path(made.video).suffix == ".mp4" and Path(made.path).suffix == ".png"
        assert (made.width, made.height, made.frames, made.fps, made.sound) == (320, 192, 25, 24.0, False)
        assert video.probe(made.video).frames == 25
        with Image.open(made.path) as still:
            assert still.size == (320, 192)
    assert first.video != second.video and (first.seed, second.seed) == (5, 6)

    # It starts on the picture it is given and ends on the other. (The stand-in does whatever
    # it is asked; which of it a real model can do is the planner's to know.)
    end = tmp_path / "end.png"
    Image.new("RGB", (512, 384), (20, 200, 40)).save(end)
    started = await engine.generate(
        request(tmp_path, Op.IMG2VID, image=str(sample_image), end_image=str(end), basename="i")
    )
    (made,) = started.images
    assert made.sound is False and video.probe(made.video).sound is False
    red, green, _ = video.first_frame(made.video).getpixel((160, 20))
    assert red > 150 > green, "the sample picture is red"
    red, green, _ = video.last_frame(made.video).getpixel((160, 20))
    assert green > 150 > red, "and the one it ends on green"

    # Continued, it is the whole video: what there was, and what was added.
    longer = await engine.generate(request(tmp_path, Op.EXTEND, video=first.video, frames=16, basename="x"))
    (extended,) = longer.images
    assert extended.frames == 25 + 16 == video.probe(extended.video).frames
    assert (extended.width, extended.height) == (320, 192)


async def test_the_engine_refuses_a_video_it_was_not_given_enough_for(engine: Engine, tmp_path: Path):
    for wrong, why in (
        (request(tmp_path, Op.IMG2VID), "needs an image to start on or to end on"),
        (request(tmp_path, Op.EXTEND), "needs a video to continue"),
        (request(tmp_path, Op.TXT2VID, frames=None), "needs a number of frames and a frame rate"),
        (request(tmp_path, Op.EXTEND, video=str(tmp_path / "gone.mp4")), "gone.mp4"),
        (request(tmp_path, Op.TXT2VID, model="z-image-turbo"), "z-image-turbo"),
    ):
        with pytest.raises(EngineError, match=why):
            await engine.generate(wrong)
    # A video that only says where it ends is one the engine takes.
    require_inputs(request(tmp_path, Op.IMG2VID, end_image="end.png"))
    with pytest.raises(BackendError):
        require_inputs(request(tmp_path, Op.IMG2VID))
    assert input_files(request(tmp_path, Op.EXTEND, video="a.mp4", image="b.png", end_image="c.png")) == [
        "b.png",
        "c.png",
        "a.mp4",
    ]
    assert {Op.TXT2VID, Op.IMG2VID, Op.EXTEND} == VIDEO_OPS


def test_wan_is_installed_when_its_files_and_the_small_autoencoder_are(tmp_path: Path, monkeypatch):
    cache, kept = tmp_path / "hub", tmp_path / "assets"
    cache.mkdir()
    monkeypatch.setenv("DREAMBOT_ASSETS_DIR", str(kept))
    spec = REGISTRY.get(WAN)
    assert spec.runtime == "wan" and spec.video is not None and not spec.video.ends
    assert {Op.TXT2VID, Op.IMG2VID, Op.EXTEND} <= set(spec.ops)
    (asset,) = spec.assets
    assert weights.assets_dir() == kept and weights.asset_path(asset) is None

    index = json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    )
    files = {
        "mlx_dit.safetensors": b"x" * 10,
        "mlx_dit.json": b"{}",
        "text_encoder/config.json": b"{}",
        "text_encoder/model-00001-of-00002.safetensors": b"x" * 10,
        "text_encoder/model-00002-of-00002.safetensors": b"x" * 10,
        "text_encoder/model.safetensors.index.json": index.encode(),
        "tokenizer/tokenizer.json": b"{}",
        # The model's own autoencoder is in the repository too, and is not read.
        "vae/diffusion_pytorch_model.safetensors": b"x" * 1000,
    }
    export = make_repo(cache, "FastVideo/FastMetal-5B-QAD", files)
    assert weights.resolve(spec, cache=cache) is None, "the small autoencoder is missing"

    kept.mkdir()
    (kept / asset.name).write_bytes(b"")
    assert weights.resolve(spec, cache=cache) is None, "a download that never arrived is not a file"
    (kept / asset.name).write_bytes(b"y" * 7)
    assert weights.asset_path(asset) == kept / asset.name
    found = weights.resolve(spec, cache=cache)
    assert (
        found is not None and found.path == export and found.quant == "q8" and found.location == str(export)
    )
    # Only the files it reads are counted: the others in the repository are not part of it.
    assert found.size_bytes == 3 * 10 + 2 + 2 + len(index) + 2 + 7
    assert weights.recommended_repo(spec) == ("FastVideo/FastMetal-5B-QAD", "q8")
    assert weights.recommended_repo(spec, ["q4"]) == ("FastVideo/FastMetal-5B-QAD", "q8"), (
        "it is published once"
    )

    (export / "text_encoder" / "model-00002-of-00002.safetensors").unlink()
    assert weights.resolve(spec, cache=cache) is None, "a shard of the text encoder is missing"
    (export / "text_encoder" / "model-00002-of-00002.safetensors").write_bytes(b"x" * 10)
    (export / "mlx_dit.json").unlink()
    assert weights.resolve(spec, cache=cache) is None, "a file the engine reads is missing"


def test_the_small_autoencoder_is_fetched_checked_and_kept_as_safetensors(tmp_path: Path, monkeypatch):
    import hashlib

    import mlx.core as mx
    import numpy as np

    from dreambot.engine import prepare
    from dreambot.engine.registry import Asset
    from tests.test_wan import checkpoint

    state = {
        "decoder.1.weight": np.arange(16, dtype=np.float16).reshape(2, 2, 2, 2),
        "decoder.1.bias": np.ones(2, dtype=np.float16),
    }
    published = checkpoint(tmp_path / "taew.pth", state)
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    asset = Asset(name="taew.safetensors", url=published.as_uri(), sha256=digest, license="mit")
    spec = replace(REGISTRY.get(WAN), assets=(asset,))
    folder = tmp_path / "assets"

    prepare.fetch_assets(spec, folder)
    kept = mx.load(str(folder / asset.name))
    assert set(kept) == set(state) and (np.array(kept["decoder.1.weight"]) == state["decoder.1.weight"]).all()
    assert weights.asset_path(asset, folder) == folder / asset.name

    # What is there is not fetched again; what is not what was expected is not kept.
    published.unlink()
    prepare.fetch_assets(spec, folder)
    other = tmp_path / "other.pth"
    other.write_bytes(b"something else")
    wrong = replace(asset, name="wrong.safetensors", url=other.as_uri())
    with pytest.raises(SystemExit, match="not the file that was expected"):
        prepare.fetch_assets(replace(spec, assets=(wrong,)), folder)
    assert not (folder / wrong.name).exists()
    gone = replace(asset, name="gone.safetensors", url=(tmp_path / "nowhere.pth").as_uri())
    with pytest.raises(SystemExit, match="could not be downloaded"):
        prepare.fetch_assets(replace(spec, assets=(gone,)), folder)


# --------------------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------------------


async def test_a_video_job_keeps_the_video_and_the_picture_it_starts_on(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox running through snow --video --seconds 2 --bs 2 --seed 9")
    assert job.kind is JobKind.VIDEO and job.status is JobStatus.DONE and job.is_video and job.model == WAN
    assert job.prompt == "a fox running through snow --video --seconds 2 --bs 2 --seed 9"
    # It was asked for in words, so each video starts on a picture made from them (see below).
    assert job.detail["video"] == {
        "motion": "low",
        "seconds": 2.04,
        "picture": "z-image-turbo",
        "drawn": {"1": "a fox running through snow", "2": "a fox running through snow"},
    }
    assert [image.index for image in job.images] == [1, 2] and job.seeds == [9, 10]
    assert job.grid is None, "videos are not put on a grid"
    for image in job.images:
        clip = job.clip(image.index)
        assert clip is not None and (clip.frames, clip.fps, clip.sound) == (49, 24.0, False)
        assert round(clip.seconds, 2) == 2.04 and job.media(image.index) == clip.path
        # The video is what goes into the pictures folder; its first frame is Dreambot's own.
        assert Path(clip.path).parent == service.config.output_dir / Path(clip.path).parent.name
        assert Path(clip.path).name == f"a-fox-running-through-snow_{job.id}_{image.index}.mp4"
        assert Path(image.path).is_relative_to(service.config.home / "stills") and Path(image.path).is_file()
        assert (image.width, image.height) == (832, 480) and video.probe(clip.path).frames == 49
    assert not list(service.config.output_dir.rglob("*.png")), "only the videos are in the pictures folder"

    # Among pictures a video is shown by the frame it ends on: its first frame is the picture it
    # was made from. (The stand-in's videos end with a white square in the corner they stop in.)
    poster = Path(job.clip(1).poster)
    assert poster.is_relative_to(service.config.home / "stills") and poster.suffix == ".jpg"
    with Image.open(poster) as shown, Image.open(job.images[0].path) as first:
        assert shown.size == first.size == (832, 480)
        assert min(shown.getpixel((816, 468))) > 200 > min(first.getpixel((816, 468)))

    # It survives being stored and read back, and a picture job has no videos.
    stored = service.store.get_job(job.id)
    assert stored.clip(2) == job.clip(2) and stored.is_video
    picture = await imagine(service, recorder, "a fox --n 1")
    assert picture.clip(1) is None and not picture.is_video and picture.media(1) == picture.images[0].path
    assert {JobKind.VIDEO, JobKind.ANIMATE, JobKind.EXTEND} == VIDEO_KINDS

    text = describe_job(job, job.images[0])
    assert "Video 1: seed 9 · 832x480 · 2s at 24 frames a second · silent" in text
    assert "Starts on a picture by z-image-turbo: a fox running through snow" in text
    assert f"File: {job.clip(1).path}" in text
    info = service.info(USER)
    assert info["video_model"] == WAN


async def test_each_video_of_a_prompt_starts_on_a_picture_of_its_own(tmp_path: Path):
    """A video asked for in words is made as Midjourney's are: a picture first, then a video of it."""
    cfg = make_config(tmp_path, defaults={"enhance": True})
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(cfg, engine=engine) as service:
        recorder = Recorder()
        service.subscribe(recorder)
        drawn: dict[int, Image.Image] = {}
        videos_by_then: list[int] = []

        def watch(event) -> None:
            if event.type is EventType.PREVIEW and event.slot is not None:
                # The picture a video is going to start on, shown where the video will be.
                with Image.open(event.preview) as picture:
                    drawn[event.slot] = picture.convert("RGB").resize((832, 480))
                videos_by_then.append(len(event.job.images))

        service.subscribe(watch)
        shown: list[tuple[str, str]] = []
        look = service.engine.describe

        async def describe(request):
            shown.append((request.image, request.instruction))
            return await look(request)

        service.engine.describe = describe

        job = await imagine(service, recorder, "a fox running --video --seconds 1 --bs 2 --seed 3")
        assert job.kind is JobKind.VIDEO and job.status is JobStatus.DONE and job.model == WAN
        assert [image.index for image in job.images] == [1, 2] and job.seeds == [3, 4]
        # The pictures came first, both of them, drawn by the picture model as any prompt is:
        # with art direction, and each with its own.
        assert sorted(drawn) == [1, 2] and videos_by_then == [0, 0]
        assert job.detail["video"]["picture"] == "z-image-turbo" and job.detail["enhanced"] is True
        assert job.detail["video"]["drawn"] == {
            str(
                n
            ): f"a fox running. Placeholder shot {n} from a low angle, soft window light, muted earth tones"
            for n in (1, 2)
        }
        assert ImageChops.difference(drawn[1], drawn[2]).getbbox() is not None, "two different pictures"
        # Each video starts on its own picture...
        for image in job.images:
            with Image.open(image.path) as first:
                assert ImageChops.difference(first.convert("RGB"), drawn[image.index]).getbbox() is None
            assert (image.width, image.height) == (832, 480) and job.clip(image.index).frames == 25
        # ...and is directed from it: the director is shown each picture, and told what was asked for.
        assert len(shown) == 2 and len({picture for picture, _ in shown}) == 2
        assert all('The person asked for this: "a fox running"' in asked for _, asked in shown)
        assert [image.prompt for image in job.images] == [
            "a fox running. Placeholder movement 1: something in the 1424x832 picture is slowly turning. "
            f"{direct.STILL_CAMERA}"
        ] * 2
        text = describe_job(job, job.images[1])
        assert (
            "Starts on a picture by z-image-turbo: a fox running. Placeholder shot 2 from a low angle" in text
        )
        # Progress goes one way, and what the pictures took is a small part of it.
        fractions = [
            event.fraction
            for event in recorder.events
            if event.job.id == job.id and event.type in (EventType.PROGRESS, EventType.PREVIEW)
        ]
        assert fractions == sorted(fractions) and 0 < fractions[0] <= FIRST_SHARE < fractions[-1] < 1
        # Only the videos are kept: the pictures were a means, and are gone with the job's scratch.
        assert not list(cfg.output_dir.rglob("*.png")) and len(list(cfg.output_dir.rglob("*.mp4"))) == 2
        assert not list((cfg.scratch_dir / "jobs").glob("*"))

        # A loop can be asked for in words: it comes back to the picture that was made for it.
        loop = await imagine(service, recorder, "a candle burning --loop --seconds 2 --bs 1")
        assert loop.status is JobStatus.DONE and loop.detail["video"]["loop"] is True
        assert loop.detail["video"]["picture"] == "z-image-turbo" and loop.clip(1) is not None
        # Run again, a job is made the same way; told to, the words are filmed with no picture.
        again = await act(service, recorder, "R", job)
        assert again.detail["video"]["picture"] == "z-image-turbo" and len(again.images) == 2
        before = len(shown)
        words = await imagine(service, recorder, "a fox running --video --seconds 1 --bs 1 --from words")
        assert "picture" not in words.detail["video"] and "drawn" not in words.detail["video"]
        assert len(shown) == before, "nothing to show the director: it is asked about the words"

        # A person can have every prompt of theirs filmed from its words.
        commands = Commands(service)
        (reply,) = await commands.handle(user=USER, channel=CHANNEL, text="/settings picture-first off")
        assert "picture-first off" in reply.text
        assert service.settings(USER).picture_first is False
        plain = await imagine(service, recorder, "a fox running --video --seconds 1 --bs 1")
        assert "picture" not in plain.detail["video"]
        asked_for = await imagine(
            service, recorder, "a fox running --video --seconds 1 --bs 1 --from picture"
        )
        assert asked_for.detail["video"]["picture"] == "z-image-turbo"


async def test_four_videos_are_made_unless_something_says_otherwise(tmp_path: Path):
    """Midjourney's batch of four, its `--bs`, and its Video Batch Size and Video Resolution settings."""
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path), engine=engine) as service:
        recorder = Recorder()
        service.subscribe(recorder)
        commands = Commands(service)

        async def setting(words: str) -> str:
            (reply,) = await commands.handle(user=USER, channel=CHANNEL, text=f"/settings {words}")
            return reply.text

        # They are made one after another, and each can be watched as soon as it is made: the
        # job is written down with what it has so far, which is what serves a video to a page.
        kept_so_far: list[tuple[int, int]] = []

        def watch(event) -> None:
            stored = service.store.get_job(event.job.id)
            if event.job.status is JobStatus.RUNNING and stored is not None:
                kept_so_far.append((len(event.job.images), len(stored.images)))

        service.subscribe(watch)
        four = await imagine(service, recorder, "a fox --video --seconds 1")
        assert [image.index for image in four.images] == [1, 2, 3, 4] and len(set(four.seeds)) == 4
        assert all(four.clip(n) is not None for n in (1, 2, 3, 4)) and four.grid is None
        assert {(1, 1), (2, 2), (3, 3)} <= set(kept_so_far), kept_so_far
        # What the prompt says comes first, then what the person has set.
        assert len((await imagine(service, recorder, "a fox --video --seconds 1 --bs 1")).images) == 1
        assert "videos      2 per prompt" in await setting("videos 2")
        two = await imagine(service, recorder, "a fox --video --seconds 1")
        assert len(two.images) == 2 and (two.width, two.height) == (832, 480)
        assert len((await imagine(service, recorder, "a fox --video --seconds 1 --bs 4")).images) == 4

        # A picture is animated, and a video continued, as many times over.
        picture = await imagine(service, recorder, "a fox --n 1")
        animated = await act(service, recorder, "animate", picture)
        assert len(animated.images) == 2
        longer = await act(service, recorder, "extend 2", animated)
        assert len(longer.images) == 2 and all(longer.clip(n).frames == 121 + 96 for n in (1, 2))
        assert longer.parent_index == 2

        # The size is a setting of its own: a picture's HD is another thing.
        assert "video-hd    on" in await setting("video-hd on")
        large = await imagine(service, recorder, "a fox --video --seconds 1")
        assert (large.width, large.height) == (1248, 704) and large.detail["video"]["hd"] is True
        assert (await imagine(service, recorder, "a fox --n 1")).width == 1024, "pictures are as they were"
        for wrong, why in (
            ("videos 3", "Videos per prompt is 1, 2 or 4."),
            ("videos many", "Videos per prompt is 1, 2 or 4."),
            ("video-hd maybe", "`video-hd` is `on` or `off`."),
        ):
            assert why in await setting(wrong)
        await setting("reset")
        assert "videos      4 per prompt" in await setting("") and "video-hd    off" in await setting("")


async def test_a_job_of_a_kind_this_build_does_not_know_is_left_out_not_read_wrong(bot):
    """A newer Dreambot may share this one's database, as this one does with the builds before video."""
    service, recorder = bot
    known = await imagine(service, recorder, "a fox --n 1")
    stored = service.store.get_job(known.id)
    stored.id = service.store.reserve_job_id()
    service.store.save_job(stored)
    with service.store._lock:
        service.store._db.execute(
            'UPDATE jobs SET data = replace(data, \'"kind": "imagine"\', \'"kind": "hologram"\') WHERE id = ?',
            (stored.id,),
        )
        service.store._db.commit()
    assert service.store.get_job(stored.id) is None
    assert [job.id for job in service.store.recent_jobs(limit=10)] == [known.id]
    assert service.store.last_job(channel=CHANNEL).id == known.id
    assert [job.id for job in service.store.jobs_page(limit=10)] == [known.id]


async def test_a_picture_is_animated_and_a_video_extended(bot, sample_image: Path):
    service, recorder = bot
    picture = await imagine(service, recorder, "a fox in the snow --ar 16:9 --n 2")

    animated = await act(service, recorder, "animate 2 high", picture)
    assert animated.kind is JobKind.ANIMATE and (animated.parent, animated.parent_index) == (picture.id, 2)
    assert animated.detail["video"]["motion"] == "high" and animated.clip(1).frames == 121
    # It is shown under the words of the picture it was made of, with what was asked of it.
    assert animated.prompt == "a fox in the snow --motion high" and animated.text == ""
    assert animated.inputs == {"source": picture.image(2).path}
    assert (animated.width, animated.height) == (832, 480)

    said = await act(service, recorder, "animate 1: The fox turns its Head --loop --seconds 1", picture)
    assert said.prompt == "The fox turns its Head --loop --seconds 1" and said.detail["video"]["loop"] is True
    assert said.images[0].prompt.startswith("a fox in the snow. The fox turns its Head. ")
    assert said.clip(1).frames == 25 and said.notes == []

    longer = await act(service, recorder, "extend", animated)
    assert longer.kind is JobKind.EXTEND and (longer.parent, longer.parent_index) == (animated.id, 1)
    assert longer.clip(1).frames == 121 + 96 and longer.detail["video"]["added"] == 4.0
    assert longer.detail["video"]["motion"] == "high", "it goes on at the pace it had"
    assert video.probe(longer.clip(1).path).frames == 217
    assert Path(animated.clip(1).path).is_file(), "the video it continues is kept as it was"

    again = await act(service, recorder, "extend low: it runs off --seconds 2", longer)
    assert again.clip(1).frames == 217 + 48 and again.detail["video"]["motion"] == "low"
    assert again.prompt == "it runs off --seconds 2 --motion low"

    # Rerun: the same again with a new seed, for a prompt and for an action alike.
    rerun = await act(service, recorder, "reroll", said)
    assert rerun.kind is JobKind.ANIMATE and rerun.parent == picture.id and rerun.prompt == said.prompt
    assert rerun.seeds != said.seeds and rerun.detail["video"]["loop"] is True
    filmed_ = await imagine(service, recorder, "a fox running --video --seconds 1")
    refilmed = await act(service, recorder, "reroll", filmed_)
    assert (
        refilmed.kind is JobKind.VIDEO
        and refilmed.prompt == filmed_.prompt
        and refilmed.seeds != filmed_.seeds
    )

    # A video can itself be animated: another one starts on its first frame.
    restarted = await act(service, recorder, "animate", longer)
    assert restarted.kind is JobKind.ANIMATE and restarted.inputs["source"] == longer.images[0].path

    # Pulling one out hands over the video, and it can be liked like a picture.
    picked = await service.act(user=USER, channel=CHANNEL, action=parse_action("U1").on(filmed_.id))
    assert picked.job.media(picked.image.index).endswith(".mp4")
    liked = await service.act(user=USER, channel=CHANNEL, action=parse_action("fav").on(filmed_.id))
    assert liked.image.favorite is True


async def test_what_is_for_pictures_is_refused_on_a_video_and_the_other_way_round(bot):
    service, recorder = bot
    picture = await imagine(service, recorder, "a fox --n 1")
    filmed_ = await imagine(service, recorder, "a fox --video --seconds 1 --bs 2")

    for typed in (
        "V1",
        "upscale 1",
        "zoom 1 2x",
        "pan 1 left",
        "square 1",
        "edit 1: make it night",
        "retexture 1",
    ):
        with pytest.raises(DreambotError, match="That is a video, and this works on pictures"):
            await service.act(user=USER, channel=CHANNEL, action=parse_action(typed).on(filmed_.id))
    with pytest.raises(DreambotError, match=r"That is a picture, not a video\. `animate` turns it into one"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("extend").on(picture.id))
    with pytest.raises(DreambotError, match=r"Which video\? Add its number \(1-2\)"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("extend").on(filmed_.id))

    # What a video can not be made longer than (five seconds, and four more four times, as on
    # Midjourney), and a file that has gone.
    long = await imagine(service, recorder, "a fox --video")
    reached = long
    for _ in range(4):
        reached = await act(service, recorder, "extend 1", reached)
    assert reached.clip(1).frames == 121 + 4 * 96 and round(reached.clip(1).seconds) == 21
    with pytest.raises(DreambotError, match="extends a video up to 21"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("extend 1").on(reached.id))
    Path(long.clip(1).path).unlink()
    with pytest.raises(DreambotError, match="no longer on disk"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("extend").on(long.id))
    assert service.queue_length == 0, "nothing was queued for any of these"


async def test_a_video_starts_on_the_picture_sent_with_it_and_wan_cannot_end_on_one(
    bot, sample_image: Path, tmp_path: Path
):
    service, recorder = bot
    end = tmp_path / "end.png"
    Image.new("RGB", (512, 384), (20, 200, 40)).save(end)

    # An image with a prompt is the frame it starts on, as in Midjourney.
    started = await imagine(service, recorder, "it melts --video --seconds 1", attachments=[sample_image])
    assert started.kind is JobKind.VIDEO and (started.width, started.height) == (736, 544)
    red, green, _ = video.first_frame(started.clip(1).path).getpixel((300, 40))
    assert red > 150 > green

    # Midjourney's ending frame is asked for the same way, and the model says what it can do instead.
    picture = await imagine(service, recorder, "a fox --n 1")
    cannot = r"Wan 2\.2 can start a video on a picture but can't lead one up to a picture"
    with pytest.raises(DreambotError, match=cannot):
        await service.imagine(
            user=USER,
            channel=CHANNEL,
            prompt="it melts --seconds 1 --end @2",
            attachments=[sample_image, end],
        )
    with pytest.raises(DreambotError, match=cannot):
        action = parse_action("animate: it melts --end @1").on(picture.id)
        await service.act(user=USER, channel=CHANNEL, action=action, attachments=[end])
    for typed, attached in (("extend --loop --seconds 1", []), ("extend: it melts --end @1", [end])):
        with pytest.raises(DreambotError, match=r"Wan 2\.2 can't lead a video up to a picture"):
            await service.act(
                user=USER, channel=CHANNEL, action=parse_action(typed).on(started.id), attachments=attached
            )
    assert service.queue_length == 0, "nothing was queued for any of these"
    # A loop is made all the same: it comes back to the picture the way it went out.
    loop = await imagine(service, recorder, "it melts --loop --seconds 1", attachments=[sample_image])
    assert loop.detail["video"]["loop"] is True
    red, green, _ = video.last_frame(loop.clip(1).path).getpixel((300, 40))
    assert red > 150 > green, "the red picture it started on"


async def test_a_video_ends_on_a_picture_where_the_model_can_lead_up_to_one(
    bot, leads_up_to_pictures, sample_image: Path, tmp_path: Path
):
    service, recorder = bot
    end = tmp_path / "end.png"
    Image.new("RGB", (512, 384), (20, 200, 40)).save(end)
    started = await imagine(service, recorder, "it melts --video --seconds 1", attachments=[sample_image])

    both = await imagine(service, recorder, "it melts --seconds 1 --end @2", attachments=[sample_image, end])
    red, green, _ = video.last_frame(both.clip(1).path).getpixel((300, 40))
    assert green > 150 > red, "it ends on the second picture"
    with pytest.raises(DreambotError, match="`--end @3`: this message has 2 attached"):
        await service.imagine(
            user=USER, channel=CHANNEL, prompt="x --end @3", attachments=[sample_image, end]
        )

    # The same from an action: the picture comes with the message that says `animate`.
    picture = await imagine(service, recorder, "a fox --n 1")
    action = parse_action("animate: it melts --seconds 1 --end @1").on(picture.id)
    result = await service.act(user=USER, channel=CHANNEL, action=action, attachments=[end])
    await recorder.wait(*result.jobs)
    (ended,) = result.jobs
    red, green, _ = video.last_frame(ended.clip(1).path).getpixel((300, 40))
    assert green > 150 > red
    # Run again, it ends on that picture again.
    rerun = await act(service, recorder, "reroll", ended)
    red, green, _ = video.last_frame(rerun.clip(1).path).getpixel((300, 40))
    assert green > 150 > red
    # Continued as a loop, a video is led back to the frame it started on.
    round_ = await act(service, recorder, "extend --loop --seconds 1", started)
    assert round_.detail["video"]["loop"] is True and round_.clip(1).frames == 25 + 24
    red, green, _ = video.last_frame(round_.clip(1).path).getpixel((300, 40))
    assert red > 150 > green, "the red picture it started on"
    with pytest.raises(DreambotError, match="`--loop` and `--end` can't both be given"):
        await service.act(
            user=USER,
            channel=CHANNEL,
            action=parse_action("extend --loop --end @1").on(started.id),
            attachments=[end],
        )
    # A video that is continued can be led up to a picture too.
    action = parse_action("extend: it melts --seconds 1 --end @1").on(started.id)
    result = await service.act(user=USER, channel=CHANNEL, action=action, attachments=[end])
    await recorder.wait(*result.jobs)
    (landed,) = result.jobs
    assert landed.kind is JobKind.EXTEND and landed.clip(1).frames == 25 + 24
    red, green, _ = video.last_frame(landed.clip(1).path).getpixel((300, 40))
    assert green > 150 > red
    with pytest.raises(DreambotError, match="`--end @1`: this message has 0 attached"):
        await service.act(
            user=USER, channel=CHANNEL, action=parse_action("animate: x --end @1").on(picture.id)
        )


async def test_a_model_on_the_mac_says_what_happens_unless_told_not_to(tmp_path: Path, sample_image: Path):
    cfg = make_config(tmp_path, defaults={"enhance": True}, **ONE)
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(cfg, engine=engine) as service:
        recorder = Recorder()
        service.subscribe(recorder)
        picture = await imagine(service, recorder, "a fox --n 1 --raw")
        asked: list[str] = []
        look = service.engine.describe

        async def describe(request):
            asked.append(request.instruction)
            return await look(request)

        service.engine.describe = describe

        # A picture and no words: the vision model is shown the picture and says what moves.
        auto = await act(service, recorder, "animate", picture)
        # It is told what the picture was made from, as Midjourney's is, to know what it is looking at.
        assert 'The picture was made from the words "a fox".' in asked[-1]
        # The video model is told what the picture is of in the words it was made from, then what moves.
        assert auto.images[0].prompt == (
            "a fox. Placeholder movement 1: something in the 1024x1024 picture is slowly turning. "
            "The camera stays still."
        )
        assert auto.detail["enhanced"] is True and auto.notes == [DIRECT_HINT]
        assert "thinking" in [kind.value for kind in recorder.types(auto)]

        # Words of the person's own stay as they are, with a line after them; each video has its own.
        said = await act(service, recorder, "animate high: The fox Leaps --bs 2 --seconds 1", picture)
        assert [image.prompt for image in said.images] == [
            f"a fox. The fox Leaps. Placeholder movement {n}: something in the 1024x1024 picture is slowly turning."
            for n in (1, 2)
        ]
        assert said.notes == [], "the note about it is shown once"
        assert [said.clip(n).frames for n in (1, 2)] == [25, 25]

        # Continued with nothing new said, it goes on from the prompt it was made from, as
        # Midjourney's Extend Auto does: nobody is asked.
        before = len(asked)
        longer = await act(service, recorder, "extend", auto)
        assert len(asked) == before and (auto.width, auto.height) == (608, 608)
        assert longer.images[0].prompt == (
            "a fox. Placeholder movement 1: something in the 1024x1024 picture is slowly turning. "
            f"{direct.CAMERA_WORDS['low']}"
        )
        # With new words it is directed from the frame it has got to (that has the video's size,
        # not the picture's), and told what happened up to there.
        leaping = await act(service, recorder, "extend: it leaps", auto)
        assert leaping.images[0].prompt.startswith(
            "it leaps. Placeholder movement 1: something in the 608x608 picture"
        )
        assert "Up to here, this is what happened" in asked[-1] and "it leaps" in asked[-1]

        # Filmed from words alone, the language model is asked about the words.
        filmed_ = await imagine(service, recorder, "a fox running --video --seconds 1 --from words")
        assert filmed_.images[0].prompt == (
            f"a fox running. Placeholder shot 1 from a low angle, soft window light, muted earth tones. "
            f"{direct.STILL_CAMERA}"
        )

        # `--raw` films the words alone, and so does switching art direction off.
        raw = await imagine(service, recorder, "a fox running --video --seconds 1 --raw")
        assert (
            raw.images[0].prompt == f"a fox running. {direct.CAMERA_WORDS['low']}"
            and "enhanced" not in raw.detail
        )
        settings = service.settings(USER)
        settings.enhance = False
        service.save_settings(USER, settings)
        plain = await act(service, recorder, "animate", picture)
        assert plain.images[0].prompt == f"a fox. {direct.MOTION_WORDS['low']}"


# --------------------------------------------------------------------------------------
# What is shown under it
# --------------------------------------------------------------------------------------


def a_job(kind: JobKind = JobKind.IMAGINE, **fields) -> Job:
    values = {
        "id": "ABC123",
        "kind": kind,
        "user": "u",
        "channel": "c",
        "prompt": "a fox in the snow",
        "status": JobStatus.DONE,
        "images": [JobImage(1, "/pictures/fox.png", 7, 896, 512)],
    }
    return Job(**{**values, **fields})


def labels(rows) -> list[list[str]]:
    return [[label for label, _ in row] for row in rows]


def test_a_picture_can_be_animated_and_a_video_extended_from_their_buttons():
    picture = a_job()
    rows = render.image_actions(picture, picture.images[0])
    assert labels(rows)[4] == ["Animate (Low motion)", "Animate (High motion)"]
    low, high = rows[4]
    assert low[1] == Action(ActionKind.ANIMATE, "ABC123", 1, "low") and high[1].value == "high"
    # Make Square sits with the zooms, as in Midjourney.
    assert labels(rows)[2] == ["Zoom Out 2x", "Zoom Out 1.5x", "Custom Zoom", "Make Square"]
    square = a_job(images=[JobImage(1, "/pictures/fox.png", 7, 1024, 1024)])
    assert all("Make Square" not in row for row in labels(render.image_actions(square, square.images[0])))
    upscaled = a_job(JobKind.UPSCALE)
    assert labels(render.image_actions(upscaled, upscaled.images[0]))[1] == [
        "Animate (Low motion)",
        "Animate (High motion)",
    ]

    made = a_job(JobKind.ANIMATE, model=WAN, detail={"video": {"motion": "high", "seconds": 5.04}})
    made.keep_clip(1, Clip("/pictures/fox.mp4", 121, 24.0, True))
    rows = render.image_actions(made, made.images[0])
    assert labels(rows) == [["Extend (Low motion)", "Extend (High motion)"], ["❤️", "🔄"]]
    assert rows[0][0][1] == Action(ActionKind.EXTEND, "ABC123", 1, "low")
    assert rows[1][1][1] == Action(ActionKind.REROLL, "ABC123")

    # Discord shows five rows of five: the sixth row under a picture shares the fifth.
    packed = _packed(
        [
            [(label, f"p{n}") for n, (label, _) in enumerate(row)]
            for row in render.image_actions(picture, picture.images[0])
        ]
    )
    assert len(packed) == BUTTON_ROWS and all(len(row) <= ROW_BUTTONS for row in packed)
    assert [label for label, _ in packed[4]] == [
        "Animate (Low motion)",
        "Animate (High motion)",
        "❤️",
        "Edit",
        "Retexture",
    ]
    assert [[label for label, _ in row] for row in packed[:4]] == labels(
        render.image_actions(picture, picture.images[0])
    )[:4]
    assert _packed([[("a", "1")], []]) == [[("a", "1")]] and _packed(None or []) == []


def test_captions_say_what_kind_of_video_it_is_and_how_long():
    made = a_job(
        JobKind.ANIMATE,
        model=WAN,
        started=0.0,
        finished=41.0,
        prompt="a fox in the snow --motion high",
        detail={"video": {"motion": "high", "seconds": 5.04, "loop": True}},
    )
    made.keep_clip(1, Clip("/pictures/fox.mp4", 121, 24.0, True))
    assert render.job_caption(made) == (
        "a fox in the snow --motion high — Animated (high motion, looping)\n"
        "wan-2.2 · 896×512 · 5s long · made in 41s · #ABC123"
    )
    made.kind, made.detail["video"] = (
        JobKind.EXTEND,
        {"motion": "low", "seconds": 9.04, "added": 4.0},
    )
    made.keep_clip(1, Clip("/pictures/fox.mp4", 217, 24.0))
    assert render.job_caption(made).splitlines() == [
        "a fox in the snow --motion high — Extended by 4s (low motion)",
        "wan-2.2 · 896×512 · 9s long · made in 41s · #ABC123",
    ]
    made.kind = JobKind.VIDEO
    assert "— Video (low motion)" in render.job_caption(made)
    # The help fits the shortest message any of the apps takes (Discord's), with video in it.
    for direct_message in (True, False):
        text = render.help_text(direct=direct_message)
        assert (
            len(text) <= 1900 and "--video" in text and "animate [high]" in text and "extend [high]" in text
        )


# --------------------------------------------------------------------------------------
# In the chat apps
# --------------------------------------------------------------------------------------


@pytest.fixture
async def service(tmp_path: Path):
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path, **ONE), engine=engine) as running:
        running.access.add_owner("mem:alice")
        yield running


async def test_a_video_arrives_as_a_video_with_its_buttons(service: Dreambot):
    front = MemoryFrontend(service, has_buttons=True)
    await front.start()
    await front.say("a fox running --video --seconds 1")
    await front.settle()
    job = service.store.recent_jobs(limit=1)[0]
    sent = front.visible[-1]
    assert sent.images == [job.clip(1).path] and is_video(sent.images[0])
    assert "— Video (low motion)" in sent.text and "1s long" in sent.text and f"#{job.id}" in sent.text
    assert [[label for label, _ in row] for row in sent.buttons] == [
        ["Extend (Low motion)", "Extend (High motion)"],
        ["❤️", "🔄"],
    ]

    # Its button makes it longer; the answer is the whole video.
    await front.press(sent, "Extend (High motion)")
    await front.settle()
    longer = service.store.recent_jobs(limit=1)[0]
    assert (
        longer.kind is JobKind.EXTEND
        and longer.parent == job.id
        and longer.detail["video"]["motion"] == "high"
    )
    assert front.visible[-1].images == [longer.clip(1).path] and "Extended by 4s" in front.visible[-1].text

    # A picture's buttons can animate it.
    await front.say("a fox --n 1")
    await front.settle()
    picture = front.visible[-1]
    await front.press(picture, "Animate (Low motion)")
    await front.settle()
    animated = service.store.recent_jobs(limit=1)[0]
    assert animated.kind is JobKind.ANIMATE and front.visible[-1].images == [animated.clip(1).path]


async def test_several_videos_arrive_one_by_one_and_typed_shortcuts_work_on_them(service: Dreambot):
    front = MemoryFrontend(service, has_buttons=False)
    await front.start()
    await front.say("a fox running --video --seconds 1 --bs 2")
    await front.settle()
    job = service.store.recent_jobs(limit=1)[0]
    first, second, caption = front.visible[-3:]
    assert first.images == [job.clip(1).path] and first.text == "Video 1"
    assert second.images == [job.clip(2).path] and second.text == "Video 2"
    assert caption.images == [] and "— Video (low motion)" in caption.text
    assert "Reply to a video with `extend` to make it longer" in caption.text

    # Replying to one of them is about that one.
    await front.say("extend: it stops and looks back --seconds 2", reply_to=second.ref)
    await front.settle()
    longer = service.store.recent_jobs(limit=1)[0]
    assert longer.kind is JobKind.EXTEND and (longer.parent, longer.parent_index) == (job.id, 2)
    assert longer.clip(1).frames == 25 + 48

    await front.say("V1", reply_to=first.ref)
    assert "That is a video, and this works on pictures" in front.visible[-1].text
    await front.say("U2", reply_to=first.ref)
    picked = front.visible[-1]
    assert picked.images == [job.clip(2).path] and picked.text.startswith(f"Video 2 of #{job.id}")

    # `animate` typed after a picture, with the picture it should end on sent along.
    await front.say("a fox --n 1")
    await front.settle()
    await front.say("animate high")
    await front.settle()
    animated = service.store.recent_jobs(limit=1)[0]
    assert animated.kind is JobKind.ANIMATE and animated.detail["video"]["motion"] == "high"


async def test_the_help_and_the_parameter_list_know_about_video(service: Dreambot):
    commands = Commands(service)
    (reply,) = await commands.handle(user="mem:alice", channel="mem:room", text="/help")
    assert "--video makes a video" in reply.text
    listed = render.params_text()
    for name in ("--video", "--motion", "--loop", "--end", "--bs", "--seconds", "--from"):
        assert name in listed


# --------------------------------------------------------------------------------------
# In the web app
# --------------------------------------------------------------------------------------


@pytest.fixture
async def site(bot):
    service, recorder = bot
    web = WebApp(service, hosts=frozenset({HOST}))
    transport = httpx.ASGITransport(app=web.app)
    async with httpx.AsyncClient(transport=transport, base_url=ORIGIN) as client:
        assert (await client.get("/")).status_code == 200
        yield service, recorder, web, client
    web.close()


async def test_a_video_still_to_come_is_shown_by_the_picture_it_starts_on(site, sample_image: Path):
    service, recorder, web, client = site
    queue = web.hub.open()
    made = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --video --seconds 1"})
    )
    seen = []
    while not queue.empty():
        seen.append(queue.get_nowait())
    # While the job ran, the page was told where the picture was; now that it is done, no longer.
    told = [event["starts"] for event in seen if "starts" in event]
    assert told and told[0] == {"1": f"/media/preview/{made['id']}?slot=1"}
    assert seen[-1]["type"] == "finished" and "starts" not in seen[-1]
    assert (await client.get(f"/media/preview/{made['id']}?slot=1")).status_code == 404, "gone with the job"

    # While a job runs, the picture is there to be shown, at a size fit for a tile.
    job = service.store.get_job(made["id"])
    job.status = JobStatus.RUNNING
    await service._emit(JobEvent(EventType.PREVIEW, job, fraction=0.04, preview=str(sample_image), slot=2))
    event = queue.get_nowait()
    assert event["type"] == "preview" and event["starts"] == {"2": f"/media/preview/{job.id}?slot=2"}
    shown = await client.get(event["starts"]["2"])
    assert shown.status_code == 200 and shown.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(shown.content)) as picture:
        assert picture.size == (512, 384)
    assert (await client.get(f"/media/preview/{job.id}?slot=nine")).status_code == 404
    # Every event of the job says so until it ends, so a page that was loaded late shows it too.
    await service._emit(JobEvent(EventType.PROGRESS, job, fraction=0.5))
    assert queue.get_nowait()["starts"] == {"2": f"/media/preview/{job.id}?slot=2"}
    job.status = JobStatus.DONE
    await service._emit(JobEvent(EventType.FINISHED, job, fraction=1.0))
    assert "starts" not in queue.get_nowait()
    assert (await client.get(f"/media/preview/{job.id}?slot=2")).status_code == 404
    web.hub.close(queue)


async def test_the_web_app_plays_downloads_and_continues_videos(site):
    service, recorder, _, client = site
    state = (await client.get("/api/state")).json()
    assert state["video"] == {
        "model": WAN,
        "ends": False,
        "seconds": 5,
        "count": 1,
        "motion": "low",
    }
    assert (state["settings"]["videos"], state["settings"]["video_hd"]) == (1, False)
    # Midjourney's Video Batch Size and Video Resolution, as settings of the person's own.
    origin = {"Origin": ORIGIN}
    changed = (await client.put("/api/settings", json={"videos": 2, "video_hd": True}, headers=origin)).json()
    assert (changed["settings"]["videos"], changed["settings"]["video_hd"]) == (2, True)
    two = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --video --seconds 1"})
    )
    assert len(two["images"]) == 2 and (two["images"][0]["width"], two["images"][0]["height"]) == (1248, 704)
    refused = await client.put("/api/settings", json={"videos": 3}, headers=origin)
    assert refused.status_code == 400 and "1, 2 or 4" in refused.json()["error"]
    await client.put("/api/settings", json={"videos": 1, "video_hd": False}, headers=origin)
    # Whether a video asked for in words starts on a picture made from them: on, until a person says.
    assert state["settings"]["picture_first"] is True
    words = (await client.put("/api/settings", json={"picture_first": False}, headers=origin)).json()
    assert words["settings"]["picture_first"] is False
    undrawn = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --video --seconds 1"})
    )
    assert "picture" not in undrawn["video"]
    await client.put("/api/settings", json={"picture_first": True}, headers=origin)

    shown = await finished(
        service,
        recorder,
        client,
        await post(client, "/api/imagine", {"prompt": "a fox --video --seconds 1"}),
    )
    assert shown["kind"] == "video" and shown["video"] == {
        "motion": "low",
        "seconds": 1.04,
        "picture": "z-image-turbo",
        "drawn": {"1": "a fox"},
    }
    (image,) = shown["images"]
    assert image["url"] == f"/media/job/{shown['id']}/1" and image["video"] == f"/media/clip/{shown['id']}/1"
    assert image["seconds"] == 1.04 and image["sound"] is False
    # Among pictures it is shown by the frame it ends on, at whatever size is asked for.
    assert image["poster"] == f"/media/poster/{shown['id']}/1"
    poster = await client.get(image["poster"], params={"w": 256})
    assert poster.status_code == 200 and poster.headers["content-type"].startswith("image/")
    assert (await client.get(f"/media/poster/{shown['id']}/9")).status_code == 404

    # The picture is its first frame; the video is served in ranges, which is how a browser plays one.
    still = await client.get(image["url"], params={"w": 256})
    assert still.status_code == 200 and still.headers["content-type"].startswith("image/")
    whole = await client.get(image["video"])
    assert whole.status_code == 200 and whole.headers["content-type"] == "video/mp4"
    assert whole.headers.get("accept-ranges") == "bytes"
    part = await client.get(image["video"], headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100 and part.content == whole.content[:100]
    saved = await client.get(image["video"], params={"download": 1})
    assert (
        "attachment" in saved.headers["content-disposition"]
        and ".mp4" in saved.headers["content-disposition"]
    )
    assert (await client.get(f"/media/clip/{shown['id']}/9")).status_code == 404
    # Midjourney's "Download Gif".
    animated_gif = await client.get(image["video"], params={"as": "gif"})
    assert animated_gif.headers["content-type"] == "image/gif" and animated_gif.content[:6] == b"GIF89a"
    assert ".gif" in animated_gif.headers["content-disposition"]

    # A picture has no video, and its buttons are the chat apps' buttons.
    picture = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --n 1"})
    )
    assert picture["video"] is None and "video" not in picture["images"][0]
    assert (await client.get(f"/media/clip/{picture['id']}/1")).status_code == 404
    animated = await finished(
        service,
        recorder,
        client,
        await post(
            client,
            f"/api/jobs/{picture['id']}/act",
            {"kind": "animate", "index": 1, "value": "high", "prompt": "--seconds 1"},
        ),
    )
    assert animated["kind"] == "animate" and animated["video"]["motion"] == "high"
    # The model cannot lead a video up to a picture, and the page is told why not.
    end = await upload(client, (20, 200, 40))
    ending = {"kind": "animate", "index": 1, "prompt": "--end @1", "attachments": [{"upload": end["id"]}]}
    refused = await post(client, f"/api/jobs/{picture['id']}/act", ending)
    assert refused.status_code == 400 and "can't lead one up to a picture" in refused.json()["error"]
    longer = await finished(
        service,
        recorder,
        client,
        await post(
            client, f"/api/jobs/{animated['id']}/act", {"kind": "extend", "index": 1, "prompt": "--seconds 2"}
        ),
    )
    assert longer["kind"] == "extend" and longer["images"][0]["seconds"] == round((25 + 48) / 24, 2)
    refused = await post(client, f"/api/jobs/{longer['id']}/act", {"kind": "vary_subtle", "index": 1})
    assert refused.status_code == 400 and "That is a video" in refused.json()["error"]

    # In the gallery a video is marked as one, and downloads as the video.
    gallery = (await client.get("/api/images")).json()["images"]
    marked = {(item["job"], item["index"]): item for item in gallery}
    assert marked[(shown["id"], 1)]["video"] == image["video"] and marked[(shown["id"], 1)]["sound"] is False
    assert "video" not in marked[(picture["id"], 1)]
    archive = await client.post(
        "/api/images/download",
        json={"items": [{"job": shown["id"], "index": 1}, {"job": picture["id"], "index": 1}]},
        headers={"Origin": ORIGIN},
    )
    names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
    assert sorted(Path(name).suffix for name in names) == [".mp4", ".png"]


async def test_the_web_app_offers_an_ending_frame_where_the_model_can_lead_up_to_one(
    site, leads_up_to_pictures
):
    service, recorder, _, client = site
    state = (await client.get("/api/state")).json()
    assert state["video"]["ends"] is True
    picture = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --n 1"})
    )
    end = await upload(client, (20, 200, 40))
    ending = {
        "kind": "animate",
        "index": 1,
        "prompt": "--seconds 1 --end @1",
        "attachments": [{"upload": end["id"]}],
    }
    animated = await finished(
        service, recorder, client, await post(client, f"/api/jobs/{picture['id']}/act", ending)
    )
    red, green, _ = video.last_frame(service.store.get_job(animated["id"]).clip(1).path).getpixel((300, 40))
    assert green > 150 > red, "it ends on the picture that was sent with the action"


def test_the_config_says_how_videos_are_made_unless_a_prompt_says_otherwise(tmp_path: Path):
    usual = Config().video
    assert (usual.count, usual.seconds, usual.motion, usual.hd) == (4, 5, "low", False)
    assert Config().prefer.video == ["wan-2.2"]
    cfg = config(tmp_path, video={"count": 2, "seconds": 3, "motion": "high"})
    plan = film(planner(cfg, WAN), "a fox --video")
    assert len(plan.seeds) == 2 and plan.renders[0].frames == 73 and plan.detail["video"]["motion"] == "high"
    for wrong in ({"count": 9}, {"seconds": 0}, {"motion": "wild"}):
        with pytest.raises(ValueError):
            config(tmp_path, video=wrong)
