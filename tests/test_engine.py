"""The engine client and worker subprocess, exercised end-to-end over the real pipe protocol
with the fake backend (no model weights involved)."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from PIL import Image

from dreambot.engine.client import Engine, EngineCancelled, EngineError, new_request_id
from dreambot.engine.protocol import (
    DescribeRequest,
    Done,
    ErrorKind,
    GenerateRequest,
    ImageDone,
    Loading,
    Op,
    Preview,
    Progress,
    TextRequest,
)


def request(tmp_path: Path, **overrides) -> GenerateRequest:
    values = {
        "id": new_request_id(),
        "model": "z-image-turbo",
        "prompt": "a lighthouse at dusk",
        "seeds": [1, 2, 3, 4],
        "width": 256,
        "height": 256,
        "out_dir": str(tmp_path / "out"),
        "basename": "job",
    }
    values.update(overrides)
    return GenerateRequest(**values)


def slow_engine(step_delay: float = 0.05, **kwargs) -> Engine:
    return Engine(
        backend="fake",
        idle_timeout=None,
        stderr=subprocess.DEVNULL,
        env={"DREAMBOT_FAKE_STEP_DELAY": str(step_delay)},
        **kwargs,
    )


async def test_generates_one_image_per_seed(engine: Engine, tmp_path: Path):
    events = []
    result = await engine.generate(request(tmp_path), on_event=events.append)

    assert [image.seed for image in result.images] == [1, 2, 3, 4]
    assert [image.index for image in result.images] == [0, 1, 2, 3]
    for image in result.images:
        with Image.open(image.path) as opened:
            assert opened.size == (256, 256) == (image.width, image.height)
    # z-image-turbo defaults to 9 steps; every step of every image reports progress.
    progress = [event for event in events if isinstance(event, Progress)]
    assert len(progress) == 4 * 9
    assert {event.total for event in progress} == {9}
    assert isinstance(events[-1], Done)
    assert sum(isinstance(event, ImageDone) for event in events) == 4


async def test_async_event_callbacks_are_awaited(engine: Engine, tmp_path: Path):
    seen = []

    async def on_event(event):
        await asyncio.sleep(0)
        seen.append(type(event).__name__)

    await engine.generate(request(tmp_path, seeds=[7]), on_event=on_event)
    assert seen[-1] == "Done" and "ImageDone" in seen


async def test_images_differ_by_seed_and_are_reproducible(engine: Engine, tmp_path: Path):
    first = await engine.generate(request(tmp_path, seeds=[1, 2], basename="a"))
    second = await engine.generate(request(tmp_path, seeds=[1], basename="b"))
    pixels = [Image.open(image.path).tobytes() for image in (*first.images, *second.images)]
    assert pixels[0] != pixels[1]
    assert pixels[0] == pixels[2]


async def test_stdout_noise_from_model_code_cannot_corrupt_the_protocol(tmp_path: Path):
    async with Engine(
        backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL, env={"DREAMBOT_FAKE_NOISE": "1"}
    ) as engine:
        result = await engine.generate(request(tmp_path))
        assert len(result.images) == 4


async def test_unknown_model_is_an_invalid_request_with_suggestions(engine: Engine, tmp_path: Path):
    with pytest.raises(EngineError) as excinfo:
        await engine.generate(request(tmp_path, model="z-image-turbi"))
    assert excinfo.value.kind == ErrorKind.INVALID
    assert "z-image-turbo" in str(excinfo.value)


async def test_unsupported_operation_is_rejected(engine: Engine, tmp_path: Path, sample_image: Path):
    with pytest.raises(EngineError) as excinfo:
        await engine.generate(request(tmp_path, op=Op.UPSCALE, image=str(sample_image)))
    assert excinfo.value.kind == ErrorKind.INVALID
    assert "does not support 'upscale'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"op": Op.IMG2IMG}, "needs an input image"),
        ({"op": Op.EDIT, "model": "flux2-klein-4b"}, "at least one reference image"),
        ({"op": Op.INPAINT, "model": "dev-fill", "image": "SAMPLE"}, "needs a mask"),
        ({"op": Op.IMG2IMG, "image": "/nonexistent/input.png"}, "not found"),
        ({"op": Op.IMG2IMG, "image": "SAMPLE", "strength": 1.5}, "between 0 and 1"),
    ],
)
async def test_missing_or_bad_inputs_are_rejected(
    engine: Engine, tmp_path: Path, sample_image: Path, overrides: dict, fragment: str
):
    overrides = {key: (str(sample_image) if value == "SAMPLE" else value) for key, value in overrides.items()}
    with pytest.raises(EngineError) as excinfo:
        await engine.generate(request(tmp_path, **overrides))
    assert excinfo.value.kind == ErrorKind.INVALID
    assert fragment in str(excinfo.value)


async def test_too_many_reference_images_is_rejected(engine: Engine, tmp_path: Path, sample_image: Path):
    with pytest.raises(EngineError) as excinfo:
        await engine.generate(
            request(tmp_path, model="dev-kontext", op=Op.EDIT, images=[str(sample_image)] * 2)
        )
    assert "at most 1 reference image" in str(excinfo.value)


async def test_image_operations_run(engine: Engine, tmp_path: Path, sample_image: Path, sample_mask: Path):
    varied = await engine.generate(
        request(
            tmp_path, op=Op.IMG2IMG, image=str(sample_image), strength=0.6, width=None, height=None, seeds=[5]
        )
    )
    assert (varied.images[0].width, varied.images[0].height) == (512, 384)

    upscaled = await engine.generate(
        request(
            tmp_path, model="seedvr2-3b", op=Op.UPSCALE, image=str(sample_image), seeds=[5], basename="up"
        )
    )
    assert (upscaled.images[0].width, upscaled.images[0].height) == (1024, 768)

    inpainted = await engine.generate(
        request(
            tmp_path,
            model="dev-fill",
            op=Op.INPAINT,
            image=str(sample_image),
            mask=str(sample_mask),
            width=512,
            height=384,
            seeds=[5],
            basename="fill",
        )
    )
    with Image.open(inpainted.images[0].path) as result, Image.open(sample_image) as original:
        # Outside the mask the source image must be untouched; inside it must be repainted.
        assert result.getpixel((10, 10)) == original.getpixel((10, 10))
        assert result.getpixel((256, 192)) != original.getpixel((256, 192))


async def test_previews_and_early_stop(engine: Engine, tmp_path: Path):
    events = []
    await engine.generate(request(tmp_path, seeds=[1], steps=10, preview_every=3), on_event=events.append)
    previews = [event for event in events if isinstance(event, Preview)]
    assert [event.step for event in previews] == [3, 6, 9]

    events.clear()
    await engine.generate(request(tmp_path, seeds=[1], steps=10, stop_at=0.5), on_event=events.append)
    assert [event.step for event in events if isinstance(event, Progress)] == [1, 2, 3, 4, 5]


async def test_cancelling_a_running_request(tmp_path: Path):
    async with slow_engine() as engine:
        req = request(tmp_path, steps=400)
        started = asyncio.Event()
        task = asyncio.create_task(
            engine.generate(req, on_event=lambda e: started.set() if isinstance(e, Progress) else None)
        )
        await asyncio.wait_for(started.wait(), 10)
        began = time.perf_counter()
        await engine.cancel(req.id)
        with pytest.raises(EngineCancelled):
            await asyncio.wait_for(task, 10)
        assert time.perf_counter() - began < 2.0
        # The worker survived the cancellation and takes new work straight away.
        result = await engine.generate(request(tmp_path, seeds=[1], steps=1, basename="after"))
        assert len(result.images) == 1


async def test_cancelling_a_queued_request_leaves_the_running_one_alone(tmp_path: Path):
    async with slow_engine(cancel_grace=0.5) as engine:
        first = request(tmp_path, seeds=[1], steps=30, basename="first")
        second = request(tmp_path, seeds=[1], steps=30, basename="second")
        first_task = asyncio.create_task(engine.generate(first))
        await asyncio.sleep(0.2)
        second_task = asyncio.create_task(engine.generate(second))
        await asyncio.sleep(0.1)
        await engine.cancel(second.id)

        # The cancel watchdog (0.5s grace) must not fire while `second` is merely queued:
        # `first` takes ~1.5s and has to finish untouched.
        assert len((await asyncio.wait_for(first_task, 20)).images) == 1
        with pytest.raises(EngineCancelled):
            await asyncio.wait_for(second_task, 10)
        assert not (tmp_path / "out" / "second_0.png").exists()


async def test_abandoned_request_is_cancelled_in_the_worker(tmp_path: Path):
    async with slow_engine() as engine:
        started = asyncio.Event()
        long_task = asyncio.create_task(
            engine.generate(
                request(tmp_path, steps=2000, basename="long"),
                on_event=lambda e: started.set() if isinstance(e, Progress) else None,
            )
        )
        await asyncio.wait_for(started.wait(), 10)
        long_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await long_task

        # Had the worker kept grinding through 4 x 2000 slow steps, this would take minutes.
        began = time.perf_counter()
        result = await asyncio.wait_for(
            engine.generate(request(tmp_path, seeds=[1], steps=1, basename="next")), 15
        )
        assert len(result.images) == 1
        assert time.perf_counter() - began < 5.0


async def test_worker_crash_fails_the_request_and_the_next_one_recovers(tmp_path: Path):
    async with slow_engine() as engine:
        started = asyncio.Event()
        task = asyncio.create_task(
            engine.generate(
                request(tmp_path, steps=2000),
                on_event=lambda e: started.set() if isinstance(e, Progress) else None,
            )
        )
        await asyncio.wait_for(started.wait(), 10)
        assert engine.info is not None
        os.kill(engine.info.pid, signal.SIGKILL)
        with pytest.raises(EngineError) as excinfo:
            await asyncio.wait_for(task, 10)
        assert excinfo.value.kind == ErrorKind.CRASHED
        assert not engine.running

        result = await engine.generate(request(tmp_path, seeds=[1], steps=1, basename="recovered"))
        assert len(result.images) == 1 and engine.running


async def test_a_request_that_ignores_cancellation_gets_the_worker_restarted(tmp_path: Path):
    # One 30-second "step" that never checks for cancellation, like a model stuck in a load.
    async with slow_engine(step_delay=30, cancel_grace=0.5) as engine:
        req = request(tmp_path, seeds=[1], steps=1)
        events = []
        task = asyncio.create_task(engine.generate(req, on_event=events.append))
        await asyncio.sleep(0.5)
        first_pid = engine.info.pid if engine.info else None
        await engine.cancel(req.id)
        with pytest.raises(EngineCancelled):
            await asyncio.wait_for(task, 10)
        assert not engine.running

    async with Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL) as fresh:
        await fresh.generate(request(tmp_path, seeds=[1], steps=1, basename="fresh"))
        assert fresh.info is not None and fresh.info.pid != first_pid


async def test_idle_timeout_stops_the_worker_and_work_restarts_it(tmp_path: Path):
    async with Engine(backend="fake", idle_timeout=0.3, stderr=subprocess.DEVNULL) as engine:
        await engine.generate(request(tmp_path, seeds=[1], steps=1))
        first_pid = engine.info.pid
        assert engine.running
        for _ in range(100):
            if not engine.running:
                break
            await asyncio.sleep(0.05)
        assert not engine.running

        await engine.generate(request(tmp_path, seeds=[1], steps=1, basename="again"))
        assert engine.running and engine.info.pid != first_pid


async def test_idle_timer_does_not_fire_during_a_long_request(tmp_path: Path):
    async with Engine(
        backend="fake",
        idle_timeout=0.2,
        stderr=subprocess.DEVNULL,
        env={"DREAMBOT_FAKE_STEP_DELAY": "0.05"},
    ) as engine:
        result = await engine.generate(request(tmp_path, seeds=[1], steps=20))
        assert len(result.images) == 1


async def test_status_and_unload(engine: Engine, tmp_path: Path):
    assert await engine.status() is None  # no worker yet
    await engine.generate(request(tmp_path, seeds=[1], steps=1))
    report = await engine.status()
    assert report is not None and [loaded.model for loaded in report.loaded] == ["z-image-turbo"]
    await engine.unload()
    report = await engine.status()
    assert report is not None and report.loaded == []


async def test_status_times_out_instead_of_waiting_behind_a_job(tmp_path: Path):
    async with slow_engine() as engine:
        task = asyncio.create_task(engine.generate(request(tmp_path, seeds=[1], steps=40)))
        await asyncio.sleep(0.3)
        assert await engine.status(timeout=0.2) is None
        # ...and asking did not disturb the job in progress.
        assert len((await asyncio.wait_for(task, 20)).images) == 1


async def test_describe_and_text(engine: Engine, sample_image: Path):
    prompts = await engine.describe(DescribeRequest(id=new_request_id(), image=str(sample_image)))
    assert len(prompts) == 4 and "512x384" in prompts[0]
    replies = await engine.text(TextRequest(id=new_request_id(), prompt="hello"))
    assert replies == ["[fake llm] hello"]


async def test_loading_event_is_reported_by_real_backends_only(engine: Engine, tmp_path: Path):
    events = []
    await engine.generate(request(tmp_path, seeds=[1], steps=1), on_event=events.append)
    assert not any(isinstance(event, Loading) for event in events)


async def test_closed_engine_refuses_work(tmp_path: Path):
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    await engine.close()
    with pytest.raises(EngineError):
        await engine.generate(request(tmp_path))


async def test_backend_that_cannot_start_reports_a_clear_error(tmp_path: Path):
    async with Engine(backend="no-such-backend", idle_timeout=None, stderr=subprocess.DEVNULL) as engine:
        with pytest.raises(EngineError) as excinfo:
            await engine.generate(request(tmp_path))
        assert "no-such-backend" in str(excinfo.value)
