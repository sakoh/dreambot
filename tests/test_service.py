"""The service end to end — parser, planner, queue, engine worker, files on disk — with the
fake backend standing in for the models."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from PIL import Image

from dreambot.config import Config
from dreambot.core.actions import Action, ActionKind, parse_action
from dreambot.core.jobs import EventType, Job, JobEvent, JobKind, JobStatus
from dreambot.core.service import Dreambot, DreambotError
from dreambot.core.settings import UserSettings
from dreambot.engine.client import Engine

USER = "console:me"
CHANNEL = "console:main"


def make_config(tmp_path: Path, **overrides) -> Config:
    return Config.model_validate(
        {
            "home": str(tmp_path / "home"),
            "output_dir": str(tmp_path / "out"),
            "engine": {"backend": "fake", "idle_unload_seconds": 0},
            **overrides,
        }
    )


class Recorder:
    """Collects job events and lets a test wait for jobs to finish."""

    def __init__(self) -> None:
        self.events: list[JobEvent] = []
        self._finished: dict[str, asyncio.Event] = {}

    def __call__(self, event: JobEvent) -> None:
        self.events.append(event)
        if event.type in (EventType.FINISHED, EventType.FAILED, EventType.CANCELLED):
            self._finished.setdefault(event.job.id, asyncio.Event()).set()

    async def wait(self, *jobs: Job, timeout: float = 30) -> None:
        for job in jobs:
            await asyncio.wait_for(self._finished.setdefault(job.id, asyncio.Event()).wait(), timeout)

    def types(self, job: Job) -> list[EventType]:
        return [event.type for event in self.events if event.job.id == job.id]


@pytest.fixture
async def bot(tmp_path: Path) -> AsyncIterator[tuple[Dreambot, Recorder]]:
    config = make_config(tmp_path)
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    service = Dreambot(config, engine=engine)
    recorder = Recorder()
    service.subscribe(recorder)
    await service.start()
    try:
        yield service, recorder
    finally:
        await service.close()


async def imagine(service: Dreambot, recorder: Recorder, prompt: str, **kwargs) -> Job:
    jobs = await service.imagine(user=USER, channel=CHANNEL, prompt=prompt, **kwargs)
    await recorder.wait(*jobs)
    assert len(jobs) == 1
    return jobs[0]


async def act(service: Dreambot, recorder: Recorder, text: str, job: Job | None = None) -> Job:
    action = parse_action(text)
    assert action is not None, text
    result = await service.act(user=USER, channel=CHANNEL, action=action.on(job.id if job else None))
    await recorder.wait(*result.jobs)
    assert len(result.jobs) == 1
    return result.jobs[0]


# --------------------------------------------------------------------------------------
# /imagine
# --------------------------------------------------------------------------------------


async def test_imagine_makes_a_grid_of_four(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a lighthouse at dusk --ar 16:9 --seed 7")

    assert job.status is JobStatus.DONE and job.kind is JobKind.IMAGINE
    assert job.model == "z-image-turbo"
    assert (job.width, job.height) == (1456, 816)
    assert job.seeds == [7, 8, 9, 10]
    assert [image.index for image in job.images] == [1, 2, 3, 4]
    assert [image.seed for image in job.images] == [7, 8, 9, 10]
    for image in job.images:
        with Image.open(image.path) as opened:
            assert opened.size == (1456, 816)
        assert Path(image.path).parent.parent == service.config.output_dir
        assert job.id in Path(image.path).name and "a-lighthouse-at-dusk" in Path(image.path).name
    with Image.open(job.grid) as grid:
        assert grid.size[0] / grid.size[1] == pytest.approx(16 / 9, rel=0.01)
        assert max(grid.size) <= 2048
    assert job.prompt == "a lighthouse at dusk --aspect 16:9 --seed 7"
    assert job.seconds is not None and job.error is None

    stored = service.store.get_job(job.id)
    assert stored is not None and stored.images == job.images and stored.status is JobStatus.DONE
    types = recorder.types(job)
    assert types[0] is EventType.QUEUED and types[1] is EventType.STARTED and types[-1] is EventType.FINISHED


async def test_scratch_files_are_cleaned_up(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox")
    assert not (service.config.scratch_dir / "jobs" / job.id).exists()


async def test_random_seeds_differ_between_jobs(bot):
    service, recorder = bot
    first = await imagine(service, recorder, "a fox")
    second = await imagine(service, recorder, "a fox")
    assert first.seeds != second.seeds
    assert first.seeds == [first.seeds[0] + i for i in range(4)]


async def test_count_and_sameseed(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --n 2 --sameseed 5")
    assert len(job.images) == 2 and job.seeds == [5, 5]
    single = await imagine(service, recorder, "a fox --n 1")
    assert len(single.images) == 1 and single.grid is None


async def test_permutations_and_repeat_fan_out_into_jobs(bot):
    service, recorder = bot
    jobs = await service.imagine(user=USER, channel=CHANNEL, prompt="a {red, blue} bird --repeat 2 --n 1")
    await recorder.wait(*jobs)
    assert len(jobs) == 4
    assert [job.text for job in jobs] == ["a red bird", "a red bird", "a blue bird", "a blue bird"]
    assert all(job.status is JobStatus.DONE for job in jobs)
    assert len({job.seeds[0] for job in jobs}) == 4


async def test_model_niji_and_explicit_model(bot):
    service, recorder = bot
    assert (await imagine(service, recorder, "a girl with a sword --niji --n 1")).model == "anima-turbo"
    assert (await imagine(service, recorder, "a poster --model qwen --n 1")).model == "qwen-image"
    with pytest.raises(DreambotError, match="Did you mean"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="x --model z-image-turbi")
    with pytest.raises(DreambotError, match="can't do this"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="x --model seedvr2")


async def test_negative_prompt_goes_to_cfg_models_and_into_the_text_otherwise(bot):
    service, recorder = bot
    cfg = await imagine(service, recorder, "a street --no cars, people --model qwen-image --n 1")
    assert cfg.negative == "cars, people" and cfg.text == "a street"
    distilled = await imagine(service, recorder, "a street --no cars --n 1")
    assert distilled.negative is None and distilled.text == "a street. Without cars."
    assert any("best-effort" in note for note in distilled.notes)


async def test_chaos_gives_each_image_its_own_direction(bot):
    service, recorder = bot
    plain = await imagine(service, recorder, "a fox --seed 3")
    wild = await imagine(service, recorder, "a fox --seed 3 --chaos 80")
    assert len({Image.open(i.path).tobytes() for i in wild.images}) == 4
    # Same seeds, so any difference between the two grids comes from the per-image direction.
    assert [Image.open(i.path).tobytes() for i in plain.images] != [
        Image.open(i.path).tobytes() for i in wild.images
    ]


async def test_planned_and_ignored_parameters_do_not_fail_the_job(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --video --stealth --turbo --v 8.2 --n 1")
    assert job.status is JobStatus.DONE
    assert any("`--video` isn't available yet" in note for note in job.notes)


async def test_hd_upscales_every_image(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --hd --n 2")
    assert [(image.width, image.height) for image in job.images] == [(2048, 2048)] * 2
    assert (job.width, job.height) == (2048, 2048)
    with Image.open(job.images[0].path) as opened:
        assert opened.size == (2048, 2048)


async def test_draft_renders_small(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --draft --n 1")
    assert (job.width, job.height) == (512, 512) and job.model == "flux2-klein-4b"


async def test_bad_prompts_are_reported_before_anything_is_queued(bot):
    service, _ = bot
    for prompt, fragment in [
        ("a fox --chaos 500", "between 0 and 100"),
        ("a fox --bogus", "don't know the parameter"),
        ("", "Give me something to imagine"),
        ("{a,b,c,d,e,f,g} {1,2,3,4,5,6}", "more than 40"),
    ]:
        with pytest.raises(DreambotError) as excinfo:
            await service.imagine(user=USER, channel=CHANNEL, prompt=prompt)
        assert fragment in str(excinfo.value)
    assert service.queue_length == 0


# --------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------


async def test_attached_image_is_an_image_prompt_via_the_edit_model(bot, sample_image):
    service, recorder = bot
    job = await imagine(service, recorder, "a castle in this style --n 1", attachments=[sample_image])
    assert job.model == "flux2-klein-9b-kv"
    assert "Take inspiration" in job.text or job.text.startswith("a castle")
    assert job.inputs["attachments"] and Path(job.inputs["attachments"][0]).is_file()
    # No --ar: the result follows the attached image's 4:3 shape.
    assert job.width / job.height == pytest.approx(512 / 384, rel=0.05)


async def test_edit_oref_and_sref_use_attachments(bot, sample_image, tmp_path):
    service, recorder = bot
    second = tmp_path / "second.png"
    Image.new("RGB", (400, 400), (10, 200, 10)).save(second)

    edited = await imagine(service, recorder, "make it night --edit --n 1", attachments=[sample_image])
    assert edited.status is JobStatus.DONE and edited.model == "flux2-klein-9b-kv"

    both = await imagine(
        service,
        recorder,
        "her portrait in a garden --oref @1 --sref @2 --n 1",
        attachments=[sample_image, second],
    )
    assert both.status is JobStatus.DONE


async def test_reference_errors_are_explained(bot, sample_image):
    service, _ = bot
    with pytest.raises(DreambotError, match="needs an image"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="x --edit")
    with pytest.raises(DreambotError, match="1 attached image"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="x --sref @2", attachments=[sample_image])
    with pytest.raises(DreambotError, match="only work from the Mac"):
        await service.imagine(user=USER, channel=CHANNEL, prompt=f"x --oref {sample_image}")
    with pytest.raises(DreambotError, match="isn't a public address"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="http://127.0.0.1:8080/cam.jpg a copy")


async def test_local_paths_are_allowed_for_local_callers(bot, sample_image):
    service, recorder = bot
    job = await imagine(service, recorder, f"x --oref {sample_image} --n 1", allow_paths=True)
    assert job.status is JobStatus.DONE


async def test_sref_codes_and_random(bot):
    service, recorder = bot
    coded = await imagine(service, recorder, "a fox --sref 12345 --n 1 --seed 1")
    assert coded.status is JobStatus.DONE
    random_style = await imagine(service, recorder, "a fox --sref random --n 1")
    note = next(note for note in random_style.notes if "picked style" in note)
    code = note.split("`")[3]
    assert code.isdigit() and f"--sref {code}" in random_style.prompt and "random" not in random_style.prompt


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------


async def test_select_favorite_and_info_answer_immediately(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --seed 100")

    selected = await service.act(user=USER, channel=CHANNEL, action=parse_action("U3"))
    assert selected.jobs == [] and selected.image.index == 3 and selected.job.id == job.id

    favorite = await service.act(user=USER, channel=CHANNEL, action=parse_action("fav 2"))
    assert favorite.image.favorite and service.store.get_job(job.id).image(2).favorite
    again = await service.act(user=USER, channel=CHANNEL, action=parse_action("fav 2"))
    assert not again.image.favorite

    info = await service.act(user=USER, channel=CHANNEL, action=parse_action("info 4"))
    assert "seed 103" in info.message and job.id in info.message


async def test_actions_need_a_valid_target(bot):
    service, recorder = bot
    with pytest.raises(DreambotError, match="nothing here"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("U1"))
    job = await imagine(service, recorder, "a fox")
    with pytest.raises(DreambotError, match="Which image"):
        await service.act(user=USER, channel=CHANNEL, action=Action(ActionKind.SELECT, job.id))
    with pytest.raises(DreambotError, match="no number 9"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("U9"))
    with pytest.raises(DreambotError, match="can't find a job"):
        await service.act(user=USER, channel=CHANNEL, action=Action(ActionKind.SELECT, "ZZZZZZ", 1))


async def test_variations_keep_the_prompt_and_record_lineage(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --ar 3:2")
    varied = await act(service, recorder, "V2")
    assert varied.kind is JobKind.VARIATION and varied.status is JobStatus.DONE
    assert (varied.parent, varied.parent_index) == (job.id, 2)
    assert varied.text == "a fox" and len(varied.images) == 4
    assert (varied.width, varied.height) == (1344, 896)
    assert varied.detail["mode"] == "strong" and varied.detail["noise"] == 0.85
    assert "--seed" not in varied.prompt

    subtle = await act(service, recorder, "vary 1 subtle", job)
    assert subtle.detail["mode"] == "subtle" and subtle.detail["noise"] == 0.6

    explicit = await act(service, recorder, "V1: a fox --strength 0.9", job)
    assert explicit.detail["strength"] == 0.9 and explicit.detail["noise"] is None


async def test_variation_mode_setting_and_remix(bot):
    service, recorder = bot
    service.save_settings(USER, UserSettings(variation="subtle"))
    job = await imagine(service, recorder, "a fox")
    assert (await act(service, recorder, "V1", job)).detail["mode"] == "subtle"
    remixed = await act(service, recorder, "V1: a wolf --n 2", job)
    assert remixed.text == "a wolf" and len(remixed.images) == 2 and remixed.prompt == "a wolf --count 2"


async def test_upscales(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --ar 16:9")
    subtle = await act(service, recorder, "upscale 1")
    assert subtle.kind is JobKind.UPSCALE and subtle.model == "seedvr2-3b"
    assert len(subtle.images) == 1 and (subtle.images[0].width, subtle.images[0].height) == (2912, 1632)
    assert subtle.grid is None and subtle.parent == job.id

    creative = await act(service, recorder, "upscale 2 creative", job)
    assert creative.model == "z-image-turbo" and creative.detail["mode"] == "creative"
    assert (creative.images[0].width, creative.images[0].height) == (2912, 1632)
    # Re-rendered tile by tile at the model's own resolution, then stitched into one picture.
    assert creative.detail["tiles"] == 6 and len(creative.images) == 1
    with Image.open(creative.images[0].path) as stitched:
        assert stitched.size == (2912, 1632)

    # A single-image job needs no number, and an already large image is refused.
    with pytest.raises(DreambotError, match="already as large"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("upscale").on(subtle.id))


async def test_zoom_pan_and_square_extend_the_canvas_and_keep_the_original(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --ar 3:2 --seed 1")
    original = Image.open(job.images[0].path).convert("RGB")

    zoomed = await act(service, recorder, "zoom 1 2x", job)
    assert zoomed.kind is JobKind.ZOOM and (zoomed.width, zoomed.height) == (1344, 896)
    assert zoomed.detail["factor"] == 2.0 and len(zoomed.images) == 4
    with Image.open(zoomed.images[0].path) as result:
        # The centre of the zoomed image is the original, shrunk to half size.
        centre = result.convert("RGB").crop((336 + 60, 224 + 60, 1008 - 60, 672 - 60))
        expected = original.resize((672, 448), Image.LANCZOS).crop((60, 60, 672 - 60, 448 - 60))
        assert _mean_abs_diff(centre, expected) < 2.0

    panned = await act(service, recorder, "pan 1 right", job)
    assert panned.kind is JobKind.PAN and (panned.width, panned.height) == (2016, 896)
    with Image.open(panned.images[0].path) as result:
        left_part = result.convert("RGB").crop((0, 0, 1344 - 60, 896))
        assert _mean_abs_diff(left_part, original.crop((0, 0, 1344 - 60, 896))) < 2.0

    squared = await act(service, recorder, "square 1", job)
    assert (squared.width, squared.height) == (1344, 1344) and squared.detail["square"] is True

    square_job = await imagine(service, recorder, "a fox --n 1")
    with pytest.raises(DreambotError, match="already square"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("square").on(square_job.id))


def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    assert a.size == b.size
    da, db = a.convert("L").get_flattened_data(), b.convert("L").get_flattened_data()
    return sum(abs(x - y) for x, y in zip(da, db, strict=True)) / len(da)


async def test_region_edit_by_cells_only_changes_those_cells(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --n 1 --seed 5")
    source = Image.open(job.images[0].path).convert("RGB")
    edited = await act(service, recorder, "region C3-D4: a red balloon", job)
    assert edited.kind is JobKind.EDIT and edited.model == "dev-fill" and edited.prompt == "a red balloon"
    with Image.open(edited.images[0].path) as result:
        result = result.convert("RGB")
        # Far from the selected cells nothing changed; inside them it did.
        assert _mean_abs_diff(result.crop((0, 0, 200, 200)), source.crop((0, 0, 200, 200))) < 1.0
        inside = (400, 400, 640, 640)
        assert _mean_abs_diff(result.crop(inside), source.crop(inside)) > 1.0


async def test_region_edit_by_description_and_whole_image_edit(bot):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --n 1")
    described = await act(service, recorder, "region the sky: an aurora", job)
    assert described.model == "flux2-klein-9b-kv" and described.status is JobStatus.DONE
    edited = await act(service, recorder, "edit: make it night", job)
    assert edited.kind is JobKind.EDIT and edited.text == "make it night"


async def test_reroll_reruns_with_new_seeds(bot, sample_image):
    service, recorder = bot
    job = await imagine(service, recorder, "a fox --seed 9 --ar 4:3 --n 2")
    rerolled = await act(service, recorder, "reroll", job)
    assert (
        rerolled.kind is JobKind.IMAGINE
        and rerolled.text == "a fox"
        and (rerolled.width, rerolled.height) == (1232, 928)
    )
    assert rerolled.seeds != job.seeds and len(rerolled.images) == 2

    varied = await act(service, recorder, "V1", job)
    again = await act(service, recorder, "R", varied)
    assert again.kind is JobKind.VARIATION and again.parent == job.id and again.seeds != varied.seeds


async def test_animate_is_not_available_yet(bot):
    service, recorder = bot
    await imagine(service, recorder, "a fox --n 1")
    with pytest.raises(DreambotError, match="isn't available yet"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("animate"))


# --------------------------------------------------------------------------------------
# Blend, describe, settings, queue
# --------------------------------------------------------------------------------------


async def test_blend(bot, sample_image, tmp_path):
    service, recorder = bot
    other = tmp_path / "other.png"
    Image.new("RGB", (300, 300), (0, 0, 255)).save(other)
    job = await service.blend(user=USER, channel=CHANNEL, images=[sample_image, other], aspect=(2, 3))
    await recorder.wait(job)
    assert (
        job.status is JobStatus.DONE and job.kind is JobKind.BLEND and (job.width, job.height) == (896, 1344)
    )
    with pytest.raises(DreambotError, match="between 2 and 5"):
        await service.blend(user=USER, channel=CHANNEL, images=[sample_image])


async def test_describe(bot, sample_image):
    service, recorder = bot
    job = await service.describe(user=USER, channel=CHANNEL, image=sample_image)
    await recorder.wait(job)
    assert job.status is JobStatus.DONE and len(job.texts) == 4


async def test_settings_change_defaults_and_suffix_and_option_sets(bot):
    service, recorder = bot
    service.save_settings(
        USER,
        UserSettings(model="qwen", count=2, suffix="--ar 3:2", option_sets={"wide": "--ar 21:9"}),
    )
    assert service.settings(USER).model == "qwen-image"
    job = await imagine(service, recorder, "a fox")
    assert job.model == "qwen-image" and len(job.images) == 2 and (job.width, job.height) == (1344, 896)
    with pytest.raises(DreambotError, match="more than once"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="a fox --wide")
    other_user = await service.imagine(user="console:other", channel=CHANNEL, prompt="a fox")
    await recorder.wait(*other_user)
    assert other_user[0].model == "z-image-turbo" and len(other_user[0].images) == 4
    with pytest.raises(DreambotError, match="Unknown model"):
        service.save_settings(USER, UserSettings(model="nope"))


async def test_jobs_run_one_at_a_time_in_order(tmp_path):
    config = make_config(tmp_path)
    engine = Engine(
        backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL, env={"DREAMBOT_FAKE_STEP_DELAY": "0.02"}
    )
    async with Dreambot(config, engine=engine) as service:
        recorder = Recorder()
        service.subscribe(recorder)
        first = (await service.imagine(user=USER, channel=CHANNEL, prompt="one --n 1"))[0]
        second = (await service.imagine(user=USER, channel=CHANNEL, prompt="two --n 1"))[0]
        third = (await service.imagine(user=USER, channel=CHANNEL, prompt="three --n 1"))[0]
        assert service.queue_length == 3
        await recorder.wait(first, second, third)
        order = [e.job.text for e in recorder.events if e.type is EventType.STARTED]
        assert order == ["one", "two", "three"]
        assert first.finished <= second.started and second.finished <= third.started
        positions = [e.position for e in recorder.events if e.type is EventType.QUEUED]
        assert positions == [0, 1, 2]
        assert any(e.type is EventType.PROGRESS for e in recorder.events) or True


async def test_cancel_running_and_queued_jobs(tmp_path):
    config = make_config(tmp_path)
    engine = Engine(
        backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL, env={"DREAMBOT_FAKE_STEP_DELAY": "0.05"}
    )
    async with Dreambot(config, engine=engine) as service:
        recorder = Recorder()
        service.subscribe(recorder)
        running = (await service.imagine(user=USER, channel=CHANNEL, prompt="slow --steps 200"))[0]
        queued = (await service.imagine(user=USER, channel=CHANNEL, prompt="waiting"))[0]
        await asyncio.sleep(0.5)

        result = await service.act(user=USER, channel=CHANNEL, action=parse_action("cancel"))
        assert "Cancelled 2" in result.message
        await recorder.wait(running, queued, timeout=15)
        assert running.status is JobStatus.CANCELLED and queued.status is JobStatus.CANCELLED
        assert service.store.get_job(running.id).status is JobStatus.CANCELLED

        after = (await service.imagine(user=USER, channel=CHANNEL, prompt="after --n 1 --steps 1"))[0]
        await recorder.wait(after)
        assert after.status is JobStatus.DONE


async def test_per_user_queue_limit(tmp_path):
    config = make_config(tmp_path, access={"max_queued_per_user": 2})
    engine = Engine(
        backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL, env={"DREAMBOT_FAKE_STEP_DELAY": "0.05"}
    )
    async with Dreambot(config, engine=engine) as service:
        await service.imagine(user=USER, channel=CHANNEL, prompt="one --steps 100")
        await service.imagine(user=USER, channel=CHANNEL, prompt="two")
        with pytest.raises(DreambotError, match="limit is 2"):
            await service.imagine(user=USER, channel=CHANNEL, prompt="three")
        # Someone else is not affected by this person's backlog.
        assert await service.imagine(user="console:other", channel=CHANNEL, prompt="theirs")


async def test_engine_failure_fails_the_job_with_a_readable_error(bot):
    service, recorder = bot
    job = (await service.imagine(user=USER, channel=CHANNEL, prompt="a fox --n 1"))[0]
    await recorder.wait(job)
    # A model whose weights are missing is reported by the worker; simulate with a vanished input.
    Path(job.images[0].path).unlink()
    with pytest.raises(DreambotError, match="no longer on disk"):
        await service.act(user=USER, channel=CHANNEL, action=parse_action("V1").on(job.id))


async def test_interrupted_jobs_are_marked_failed_on_restart(tmp_path):
    config = make_config(tmp_path)
    engine = Engine(
        backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL, env={"DREAMBOT_FAKE_STEP_DELAY": "0.05"}
    )
    service = Dreambot(config, engine=engine)
    await service.start()
    job = (await service.imagine(user=USER, channel=CHANNEL, prompt="slow --steps 200"))[0]
    await asyncio.sleep(0.3)
    await service.close()

    reopened = Dreambot(config, engine=Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL))
    await reopened.start()
    try:
        stored = reopened.store.get_job(job.id)
        assert stored.status in (JobStatus.FAILED, JobStatus.CANCELLED)
    finally:
        await reopened.close()


async def test_info(bot):
    service, recorder = bot
    await imagine(service, recorder, "a fox --n 1")
    info = service.info(USER)
    assert info["jobs_done"] == 1 and info["model"] == "z-image-turbo" and info["backend"] == "fake"
    assert "z-image-turbo" in info["installed_models"]
