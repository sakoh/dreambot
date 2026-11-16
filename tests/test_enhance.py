"""The prompt enhancer: which jobs it art-directs, what it asks the language model, how the
answer is checked, and how the service uses it. The fake backend stands in for the model."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dreambot.config import Config
from dreambot.core import enhance, fingerprint, styles
from dreambot.core.commands import Commands
from dreambot.core.enhance import Brief
from dreambot.core.jobs import EventType, JobKind, JobStatus
from dreambot.core.planner import ResolvedRefs
from dreambot.core.prompt import parse_prompt
from dreambot.core.service import ENHANCE_HINT, Dreambot, describe_job
from dreambot.core.settings import ENHANCE_MIN_MEMORY_GB, UserSettings
from dreambot.engine.client import Engine, EngineError, new_request_id
from dreambot.engine.protocol import TextRequest
from tests.test_planner import planner
from tests.test_service import CHANNEL, USER, Recorder, act, imagine, make_config

FOX = "a red fox in the snow"
ON = UserSettings(enhance=True).resolved(Config().defaults)
OFF = UserSettings(enhance=False).resolved(Config().defaults)


# --------------------------------------------------------------------------------------
# Which jobs are art-directed
# --------------------------------------------------------------------------------------


def test_short_prompts_are_directed_and_finished_ones_are_left_alone():
    assert enhance.wanted(Brief(FOX, 4))
    assert not enhance.wanted(Brief("", 4)) and not enhance.wanted(Brief(FOX, 0))
    # --stylize 0 (and --raw, which plans as stylize 0) means: as written.
    assert not enhance.wanted(Brief(FOX, 4, stylize=0))
    # ...unless variety or strangeness was asked for explicitly.
    assert enhance.wanted(Brief(FOX, 4, stylize=0, chaos=30))
    assert enhance.wanted(Brief(FOX, 4, stylize=0, weird=500))
    assert enhance.wanted(Brief(FOX, 4, stylize=0, exp=20))
    assert not enhance.wanted(Brief(FOX, 4, stylize=0, chaos=5)), "below 10, chaos is off"

    finished = " ".join(["a carefully described scene with its own light and lens"] * 6)
    assert len(finished.split()) >= enhance.DETAILED_WORDS
    assert not enhance.wanted(Brief(finished, 4)), "a long prompt has been art-directed by its author"
    assert enhance.wanted(Brief(finished, 4, stylize=250)), "more stylize was asked for"
    assert enhance.wanted(Brief(finished, 4, chaos=50))


@pytest.mark.parametrize(
    "prompt",
    [
        "minimalist logo of a mountain, flat vector",
        "app icon for a weather app",
        "a sticker of a happy avocado",
        "isometric pixel art of a tiny bakery",
        "seamless pattern of lemons",
        "hand lettering of the word hello",
    ],
)
def test_flat_graphics_are_never_given_a_camera_and_a_light(prompt: str):
    assert not enhance.wanted(Brief(prompt, 4))
    assert not enhance.wanted(Brief(prompt, 4, stylize=750, chaos=80))


# --------------------------------------------------------------------------------------
# What the language model is asked
# --------------------------------------------------------------------------------------


def test_the_request_carries_the_prompt_and_the_dials():
    plain = enhance.request(Brief(FOX, 4))
    assert plain.startswith(f"Prompt: {FOX}\n")
    assert "Write exactly 4 lines, numbered 1. to 4., one per image, each of at most 30 words" in plain
    assert "one series" in plain and "every image is a photograph" in plain
    assert "surreal" not in plain and "intricate" not in plain and "Never mention" not in plain

    single = enhance.request(Brief(FOX, 1))
    assert "Write exactly one line, numbered 1., of at most 30 words" in single
    assert "series" not in single, "one image has nothing to differ from"

    assert "different composition and different light" in enhance.request(Brief(FOX, 4, chaos=25))
    assert "very different from one another" in enhance.request(Brief(FOX, 4, chaos=50))
    wild = enhance.request(Brief(FOX, 4, chaos=100))
    assert "different medium or" in wild and "every image is a photograph" not in wild

    assert "offbeat" in enhance.request(Brief(FOX, 4, weird=100))
    assert "surreal" in enhance.request(Brief(FOX, 4, weird=500))
    assert "bizarre" in enhance.request(Brief(FOX, 4, weird=3000))
    assert "fine detail" in enhance.request(Brief(FOX, 4, exp=10))
    assert "experimental compositions" in enhance.request(Brief(FOX, 4, exp=100))

    constrained = enhance.request(
        Brief(FOX, 2, style=("in the style of a linocut print", "jewel tones"), avoid=("people", "cars"))
    )
    assert "in the style of a linocut print, jewel tones." in constrained
    assert "Never mention: people, cars." in constrained


def test_stylize_sets_how_much_is_written():
    budgets = [Brief(FOX, 4, stylize=value).words for value in (0, 50, 100, 250, 750, 1000)]
    assert budgets == [14, 16, 30, 42, 55, 55]
    assert Brief(FOX, 4, weird=500).words == 38, "a twist needs room"
    assert "minimal" in enhance.request(Brief(FOX, 4, stylize=50))
    assert "bold" in enhance.request(Brief(FOX, 4, stylize=250))
    assert "strong and opinionated" in enhance.request(Brief(FOX, 4, stylize=1000))
    assert "Add no style of your own" in enhance.request(Brief(FOX, 4, stylize=0, chaos=40))

    assert (
        enhance.max_tokens(Brief(FOX, 1))
        < enhance.max_tokens(Brief(FOX, 4))
        < enhance.max_tokens(Brief(FOX, 4, stylize=1000))
    )
    assert enhance.max_tokens(Brief(FOX, 16, stylize=1000)) <= 1500
    # Sampling stays tame at any chaos: variety is asked for in words, not by turning up the heat.
    assert enhance.temperature(Brief(FOX, 4)) == 0.6
    assert enhance.temperature(Brief(FOX, 4, chaos=100)) == 0.8
    assert 0 < enhance.TOP_P < 1 < enhance.REPETITION_PENALTY


def test_the_system_message_holds_the_rules_that_protect_the_prompt():
    for rule in (
        "Do not restate the prompt",
        "do not contradict it",
        "Do not add subjects",
        "single still image",
        "Write in English",
    ):
        assert rule in enhance.SYSTEM
    assert FOX not in enhance.SYSTEM


# --------------------------------------------------------------------------------------
# Checking the answer
# --------------------------------------------------------------------------------------


def test_a_good_answer_is_taken_line_by_line():
    answer = (
        "1. close-up of the fox's face, soft morning light from above, warm reds against cool whites\n"
        "2. low angle shot of the fox walking in snowdrifts, diffused light from behind, long shadows\n"
        "3. three-quarter view in profile, warm side light, fine detail on fur\n"
        "4. distant shot in a wide landscape at dawn, pale light from the horizon"
    )
    lines = enhance.parse(answer, Brief(FOX, 4))
    assert len(lines) == 4 and len(set(lines)) == 4
    assert (
        lines[0] == "close-up of the fox's face, soft morning light from above, warm reds against cool whites"
    )
    assert (
        enhance.compose(FOX, lines[0])
        == f"{FOX}. Close-up of the fox's face, soft morning light from above, warm reds against cool whites"
    )


def test_restatements_cross_references_and_camera_moves_are_removed():
    brief = Brief(FOX, 1)

    def one(line: str) -> str:
        return enhance.parse(f"1. {line}", brief)[0]

    # The prompt again, in whole or in part, is not direction.
    assert one("A red fox in the snow, low angle, hard winter sun") == "low angle, hard winter sun"
    assert (
        one("The fox in the snow, a tight frame, soft overcast light") == "a tight frame, soft overcast light"
    )
    # Lines are used on their own, and the picture does not move.
    assert one("the same scene from above, cold blue light, long shadows") == "cold blue light, long shadows"
    assert (
        one("wide shot, the camera pans slowly across the drifts, pale dawn light")
        == "wide shot, pale dawn light"
    )
    assert one("side view, slow pan, warm low sun, deep shadows") == "side view, warm low sun, deep shadows"
    # Formatting noise, labels and sentence capitals.
    assert (
        one("**Image 1: Low angle. Hard winter sun. Long blue shadows.**")
        == "Low angle, hard winter sun, long blue shadows"
    )
    assert (
        one('"overhead view, flat grey light, fine film grain"')
        == "overhead view, flat grey light, fine film grain"
    )
    # A phrase said twice is kept once.
    assert (
        one("low angle, warm light, low angle, warm light, crisp focus")
        == "low angle, warm light, crisp focus"
    )


def test_only_the_person_may_ask_for_paint():
    painted = "1. low angle, soft dawn light, delicate watercolor washes, muted blues"
    assert enhance.parse(painted, Brief(FOX, 1)) == ["low angle, soft dawn light, muted blues"]
    kept = "low angle, soft dawn light, delicate watercolor washes, muted blues"
    assert enhance.parse(painted, Brief("a watercolor of a fox", 1)) == [kept]
    assert enhance.parse(painted, Brief(FOX, 1, chaos=80)) == [kept], "high chaos asks for other mediums"
    assert enhance.parse(painted, Brief(FOX, 1, style=("in the style of a gouache illustration",))) == [kept]


def test_a_painting_is_not_given_a_lens():
    brief = Brief("a lighthouse at dusk, oil painting", 1)
    mixed = "1. low angle with the tower dominating, deep reds and purples, grainy 35mm film, thick impasto strokes"
    assert enhance.parse(mixed, brief) == [
        "low angle with the tower dominating, deep reds and purples, thick impasto strokes"
    ]
    # Seen on the first live job: a camera's focus has no place in a painting either.
    focus = "1. medium shot in profile, soft morning light through trees, cool greens and blues, shallow focus on her face"
    assert enhance.parse(focus, Brief("an oil painting of a woman reading on a bench", 1)) == [
        "medium shot in profile, soft morning light through trees, cool greens and blues"
    ]
    # Someone who asks for both gets both.
    both = Brief("a photograph of an oil painting in a gallery", 1)
    assert "35mm film" in enhance.parse(mixed, both)[0] and "impasto" in enhance.parse(mixed, both)[0]


def test_things_to_avoid_never_come_back_through_the_direction():
    brief = Brief("a city street at night", 2, avoid=("people", "car"))
    answer = "1. wide shot, crowds of people under umbrellas, neon reflections\n2. low angle, parked cars along the kerb, wet asphalt, sodium light"
    assert enhance.parse(answer, brief) == [
        "wide shot, neon reflections",
        "low angle, wet asphalt, sodium light",
    ]


def test_unusable_lines_are_replaced_or_everything_falls_back():
    brief = Brief(FOX, 4)
    # Too few lines: the good ones are reused, so the whole grid is still directed.
    two = enhance.parse("1. low angle, hard winter sun\n2. overhead view, flat grey light", brief)
    assert two == ["low angle, hard winter sun", "overhead view, flat grey light"] * 2
    # A line of commentary or a slip into another script does not poison its neighbours: the
    # commentary is dropped, the Chinese-only line is dropped, and the last line is cut where
    # it changes script.
    mixed = enhance.parse(
        "1. Here is the prompt you asked for\n2. low angle, hard winter sun\n3. 特写镜头\n4. close view, pale dawn light, 细腻的光影",
        brief,
    )
    assert (
        mixed
        == ["low angle, hard winter sun", "low angle, hard winter sun"] + ["close view, pale dawn light"] * 2
    )
    # Nothing usable at all: every entry is empty and the caller keeps its own plan.
    for garbage in ("", "I cannot help with that.", "1. ok\n2. no", "the the the"):
        assert enhance.parse(garbage, brief) == [""] * 4, garbage
    # More lines than images are ignored.
    assert (
        len(
            enhance.parse(
                "\n".join(f"{n}. low angle number {n}, hard winter sun" for n in range(1, 9)), brief
            )
        )
        == 4
    )


def test_a_rambling_line_loses_its_tail():
    brief = Brief(FOX, 1, stylize=50)
    limit = round(brief.words * 1.5) + 4
    phrases = [f"phrase number {n} goes here" for n in range(1, 12)]
    (line,) = enhance.parse("1. " + ", ".join(phrases), brief)
    assert line.startswith("phrase number 1 goes here") and len(line.split()) <= limit
    assert line == ", ".join(phrases[: len(line.split(", "))]), "whole phrases are dropped from the end"
    (unbroken,) = enhance.parse("1. " + " ".join(f"word{n}" for n in range(80)), brief)
    assert len(unbroken.split()) == limit


def test_compose_only_ever_adds():
    assert enhance.compose("a fox", "low angle, hard sun") == "a fox. Low angle, hard sun"
    assert enhance.compose("a fox, ", "low angle") == "a fox. Low angle"
    assert enhance.compose("a fox.", "") == "a fox."
    assert enhance.compose("", "low angle") == "Low angle"
    sign = 'a neon sign that says "OPEN LATE" above a diner door'
    assert enhance.compose(sign, "straight-on shot, cool glow").startswith(sign + ". ")


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------


def brief_of(p, prompt: str, settings=ON, refs: ResolvedRefs | None = None):
    return p.imagine(parse_prompt(prompt), settings, refs).brief


def test_the_planner_briefs_plain_text_prompts_only(tmp_path: Path, sample_image: Path):
    p = planner(tmp_path, "z-image-turbo", "flux2-klein-4b")
    brief = brief_of(p, f"{FOX} --chaos 40 --weird 300 --exp 20 --stylize 250 --no people")
    assert brief == Brief(FOX, 4, stylize=250, chaos=40, weird=300, exp=20, avoid=("people",))
    assert brief_of(p, f"{FOX} --n 2").count == 2
    assert brief_of(p, FOX, OFF) is None, "the person switched it off"
    assert brief_of(p, f"{FOX} --raw") is None and brief_of(p, f"{FOX} --stylize 0") is None
    assert brief_of(p, f"{FOX} --raw --chaos 50") == Brief(FOX, 4, stylize=0, chaos=50), (
        "raw keeps what was asked for"
    )
    assert brief_of(p, "minimalist logo of a fox, flat vector") is None

    picture = str(sample_image)
    for refs in (
        ResolvedRefs(image_prompts=[picture]),
        ResolvedRefs(edit=[picture]),
        ResolvedRefs(subject=[picture]),
        ResolvedRefs(style=[picture]),
    ):
        assert brief_of(p, FOX, ON, refs) is None, "with an image attached the text is used as written"

    legacy = brief_of(p, f"{FOX} --sref 4821 --sv 1", ON, ResolvedRefs(style_codes=["4821"]))
    assert legacy.style == tuple(styles.style_recipe("4821", 100)) and legacy.style
    code = brief_of(p, f"{FOX} --sref 4821", ON, ResolvedRefs(style_codes=["4821"]))
    assert code.style == tuple(fingerprint.phrases(fingerprint.from_code("4821"), 100)) and code.style
    worded = brief_of(p, f"{FOX} --sref --sv 3", ON, ResolvedRefs(style=[picture]))
    assert worded is not None and worded.style, "a style applied in words leaves the text a description"
    assert "every image is a photograph" not in enhance.request(code), "a style names its own medium"
    assert "every image is a photograph" in enhance.request(brief)


def test_directions_become_per_image_prompts(tmp_path: Path):
    p = planner(tmp_path, "z-image-turbo")
    parsed = parse_prompt(f"{FOX} --seed 5")
    first = p.imagine(parsed, ON)
    assert first.brief is not None and [r.prompt for r in first.renders] == [FOX], "renderable as it is"

    directions = [
        "low angle, hard winter sun",
        "overhead view, flat grey light",
        "close on the eyes",
        "wide and distant",
    ]
    directed = p.imagine(parsed, ON, directions=directions, seeds=first.seeds)
    assert directed.brief is None, "already directed"
    assert directed.seeds == first.seeds == [5, 6, 7, 8]
    assert directed.text == FOX, "the job's text stays what the person wrote"
    assert [(r.prompt, r.seeds, r.slots) for r in directed.renders] == [
        (f"{FOX}. Low angle, hard winter sun", [5], [1]),
        (f"{FOX}. Overhead view, flat grey light", [6], [2]),
        (f"{FOX}. Close on the eyes", [7], [3]),
        (f"{FOX}. Wide and distant", [8], [4]),
    ]
    # Images that share a direction still render in one request.
    shared = p.imagine(parsed, ON, directions=["low angle"] * 4, seeds=first.seeds)
    assert [(r.prompt, r.seeds) for r in shared.renders] == [(f"{FOX}. Low angle", [5, 6, 7, 8])]


def test_an_image_without_direction_keeps_the_rule_based_emulation(tmp_path: Path):
    p = planner(tmp_path, "z-image-turbo")
    parsed = parse_prompt(f"{FOX} --chaos 50 --seed 5")
    ruled = [r.prompt for r in p.imagine(parsed, ON).renders]
    assert len(ruled) == 4 and all(prompt.startswith(f"{FOX}, ") for prompt in ruled)
    mixed = p.imagine(
        parsed, ON, directions=["low angle, hard sun", "", "", "wide and distant"], seeds=[5, 6, 7, 8]
    )
    assert [r.prompt for r in mixed.renders] == [
        f"{FOX}. Low angle, hard sun",
        ruled[1],
        ruled[2],
        f"{FOX}. Wide and distant",
    ]


def test_style_codes_and_things_to_avoid_survive_direction(tmp_path: Path):
    p = planner(tmp_path, "z-image-turbo", "qwen-image")
    recipe = ", ".join(fingerprint.phrases(fingerprint.from_code("4821"), 100))
    coded = p.imagine(
        parse_prompt(f"{FOX} --sref 4821 --n 1 --stylize 750"),
        ON,
        ResolvedRefs(style_codes=["4821"]),
        directions=["low angle, hard sun"],
    )
    assert coded.renders[0].prompt == f"{FOX}. Low angle, hard sun, {recipe}", (
        "a code always adds the same look"
    )
    assert "artful composition" not in coded.renders[0].prompt, (
        "the rule-based stylize emphasis is not stacked on top"
    )

    # A distilled model has no negative prompt: the sentence saying what to avoid closes the prompt.
    avoided = p.imagine(
        parse_prompt(f"{FOX} --no people, cars --n 1"), ON, directions=["low angle, hard sun"]
    )
    assert avoided.renders[0].prompt == f"{FOX}. Low angle, hard sun. Without people, cars."
    assert avoided.text == f"{FOX}. Without people, cars." and avoided.negative is None
    ruled = p.imagine(parse_prompt(f"{FOX} --no people --chaos 50 --n 1 --seed 1"), OFF).renders[0].prompt
    assert ruled.startswith(f"{FOX}, ") and ruled.endswith(". Without people."), ruled
    # A model with guidance takes a real negative prompt instead.
    cfg = p.imagine(parse_prompt(f"{FOX} --no people --n 1 --model qwen-image"), ON, directions=["low angle"])
    assert cfg.renders[0].prompt == f"{FOX}. Low angle" and cfg.renders[0].negative == "people"


# --------------------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------------------


@pytest.fixture
async def bot(tmp_path: Path) -> AsyncIterator[tuple[Dreambot, Recorder]]:
    """A bot whose owner-wide default has the enhancer on."""
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    service = Dreambot(make_config(tmp_path, defaults={"enhance": True}), engine=engine)
    recorder = Recorder()
    service.subscribe(recorder)
    await service.start()
    try:
        yield service, recorder
    finally:
        await service.close()


async def test_the_fake_language_model_answers_requests_for_lines(engine: Engine):
    brief = Brief(FOX, 3)
    request = TextRequest(
        id=new_request_id(), weights="fake", system=enhance.SYSTEM, prompt=enhance.request(brief)
    )
    (answer,) = await engine.text(request)
    assert len(enhance.parse(answer, brief)) == 3 and all(enhance.parse(answer, brief))
    (echo,) = await engine.text(TextRequest(id=new_request_id(), weights="fake", prompt="shorten this"))
    assert echo.startswith("[fake llm]")


async def test_a_plain_prompt_is_art_directed_image_by_image(bot):
    service, recorder = bot
    job = await imagine(service, recorder, f"{FOX} --seed 7")
    assert job.status is JobStatus.DONE and job.detail.get("enhanced") is True
    assert job.prompt == f"{FOX} --seed 7" and job.text == FOX, (
        "what the person wrote is what the job is called"
    )
    prompts = [image.prompt for image in job.images]
    assert len(set(prompts)) == 4 and all(prompt.startswith(f"{FOX}. Placeholder shot") for prompt in prompts)
    assert [image.seed for image in job.images] == [7, 8, 9, 10]

    types = recorder.types(job)
    assert EventType.THINKING in types
    assert types.index(EventType.STARTED) < types.index(EventType.THINKING) < types.index(EventType.FINISHED)
    assert f"Rendered from: {prompts[1]}" in describe_job(job, job.images[1])


async def test_the_explanation_is_given_once(bot):
    service, recorder = bot
    first = await imagine(service, recorder, FOX)
    second = await imagine(service, recorder, "a lighthouse at dusk")
    assert first.notes == [ENHANCE_HINT] and second.notes == []
    assert second.detail.get("enhanced") is True
    assert service.settings(USER).hints == ["enhance"]


async def test_raw_stylize_zero_and_the_setting_switch_it_off(bot):
    service, recorder = bot
    for prompt in (f"{FOX} --raw", f"{FOX} --stylize 0"):
        job = await imagine(service, recorder, prompt)
        assert not job.detail.get("enhanced") and EventType.THINKING not in recorder.types(job)
        assert all(image.prompt is None for image in job.images) and job.notes == []

    service.save_settings(USER, UserSettings(enhance=False))
    job = await imagine(service, recorder, FOX)
    assert not job.detail.get("enhanced") and EventType.THINKING not in recorder.types(job)
    service.save_settings(USER, UserSettings(enhance=True, raw=True))
    assert not (await imagine(service, recorder, FOX)).detail.get("enhanced"), "raw mode wins"


async def test_the_same_seed_gives_the_same_direction(bot):
    service, recorder = bot
    one = await imagine(service, recorder, f"{FOX} --seed 3 --chaos 40")
    two = await imagine(service, recorder, f"{FOX} --seed 3 --chaos 40")
    assert [image.prompt for image in one.images] == [image.prompt for image in two.images]
    assert all(image.prompt for image in one.images)


async def test_variations_keep_an_images_direction_and_zooms_do_not(bot):
    service, recorder = bot
    job = await imagine(service, recorder, FOX)
    third = job.image(3).prompt
    variation = await act(service, recorder, "V3", job)
    assert variation.kind is JobKind.VARIATION and variation.text == third
    assert variation.prompt == FOX, "it is still called by the person's prompt"
    # A variation of the variation carries the direction on.
    again = await act(service, recorder, "V1", variation)
    assert again.text == third

    upscaled = await act(service, recorder, "upscale 3", job)
    assert upscaled.text == third
    zoomed = await act(service, recorder, "zoom 3", job)
    assert zoomed.text == FOX, "the framing is about to change, so only the scene is described"
    # Remix replaces the prompt altogether.
    remixed = await act(service, recorder, "V3: a grey wolf", job)
    assert remixed.text == "a grey wolf"


async def test_rerolls_permutations_and_repeats_are_each_directed(bot):
    service, recorder = bot
    jobs = await service.imagine(user=USER, channel=CHANNEL, prompt="a {red, grey} fox --n 1 --repeat 2")
    await recorder.wait(*jobs)
    assert len(jobs) == 4 and all(job.detail.get("enhanced") for job in jobs)
    assert {job.text for job in jobs} == {"a red fox", "a grey fox"}
    # Rerolling one of them runs that prompt once more, not the whole `--repeat`.
    rerolled = await act(service, recorder, "R", jobs[0])
    assert rerolled.detail.get("enhanced") and rerolled.images[0].prompt.startswith("a red fox. ")
    assert "--repeat" not in rerolled.prompt


@pytest.mark.parametrize("failure", ["error", "garbage", "empty", "no-model"])
async def test_a_failing_enhancer_never_fails_the_job(bot, monkeypatch: pytest.MonkeyPatch, failure: str):
    service, recorder = bot

    async def broken(request: TextRequest) -> list[str]:
        if failure == "error":
            raise EngineError("the language model fell over")
        return [] if failure == "empty" else ["Sorry, I can't write prompts."]

    if failure == "no-model":
        monkeypatch.setattr(service.catalog, "vision_model", lambda: None)
    else:
        monkeypatch.setattr(service.engine, "text", broken)
    job = await imagine(service, recorder, f"{FOX} --chaos 50 --seed 2")
    assert job.status is JobStatus.DONE and len(job.images) == 4 and job.error is None
    assert not job.detail.get("enhanced") and job.notes == []
    # The rule-based emulation the plan already had is what gets rendered.
    assert all(image.prompt.startswith(f"{FOX}, ") for image in job.images)
    assert service.settings(USER).hints == [], "nothing was art-directed, so nothing to explain"


async def test_the_enhancer_can_be_cancelled_like_any_other_step(bot, monkeypatch: pytest.MonkeyPatch):
    service, recorder = bot
    from dreambot.engine.client import EngineCancelled

    async def cancelled(request: TextRequest) -> list[str]:
        raise EngineCancelled

    monkeypatch.setattr(service.engine, "text", cancelled)
    (job,) = await service.imagine(user=USER, channel=CHANNEL, prompt=FOX)
    await recorder.wait(job)
    assert job.status is JobStatus.CANCELLED and job.images == []


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


async def test_settings_show_and_change_the_enhancer(bot, monkeypatch: pytest.MonkeyPatch):
    service, _ = bot
    service.access.add_owner(USER)
    commands = Commands(service)

    async def say(text: str) -> str:
        (reply,) = await commands.handle(user=USER, channel=CHANNEL, text=text)
        return reply.text

    assert "enhance     on — a local language model art-directs short prompts" in await say("/settings")
    assert "enhance     off" in await say("/settings enhance off")
    assert service.settings(USER).enhance is False and service.resolved_settings(USER).enhance is False
    assert "enhance     on" in await say("/settings enhance on")
    assert "enhance     on (but raw mode uses prompts as written)" in await say("/settings raw on")
    await say("/settings raw off")
    monkeypatch.setattr(service.catalog, "vision_model", lambda: None)
    assert "enhance     on (needs a language model: dreambot models pull describe)" in await say("/settings")
    assert "I don't have a setting" not in await say("/settings enhance off")


def test_the_default_follows_the_macs_memory(monkeypatch: pytest.MonkeyPatch):
    defaults = Config().defaults
    assert defaults.enhance is None, "unset means: decide by memory"
    for memory, expected in ((16, False), (ENHANCE_MIN_MEMORY_GB, True), (96, True)):
        monkeypatch.setattr("dreambot.core.settings.physical_memory_gb", lambda memory=memory: float(memory))
        monkeypatch.setattr(
            "dreambot.core.settings.auto_enhance", lambda memory=memory: memory >= ENHANCE_MIN_MEMORY_GB
        )
        assert UserSettings().resolved(defaults).enhance is expected
    # An explicit choice, the bot's or the person's, beats the default either way.
    monkeypatch.setattr("dreambot.core.settings.auto_enhance", lambda: False)
    assert (
        UserSettings().resolved(Config.model_validate({"defaults": {"enhance": True}}).defaults).enhance
        is True
    )
    assert UserSettings(enhance=True).resolved(defaults).enhance is True
    monkeypatch.setattr("dreambot.core.settings.auto_enhance", lambda: True)
    assert UserSettings(enhance=False).resolved(defaults).enhance is False
    assert (
        UserSettings().resolved(Config.model_validate({"defaults": {"enhance": False}}).defaults).enhance
        is False
    )
    assert "hints" not in UserSettings().to_dict() and UserSettings(hints=["enhance"]).to_dict() == {
        "hints": ["enhance"]
    }
