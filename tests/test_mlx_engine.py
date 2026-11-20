"""Real MLX inference through the worker, using weights already in the local Hugging Face cache.

These tests are deselected by default (they take a while and need model weights). Run them with::

    uv run pytest -m mlx -s

Each test skips itself when the model it needs is not installed. The worker runs with the
network disabled, exactly as it does in production.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from PIL import Image, ImageChops, ImageStat

from dreambot.engine import weights
from dreambot.engine.client import Engine, EngineCancelled, new_request_id
from dreambot.engine.protocol import GenerateRequest, ImageDone, Loaded, Loading, Op, Preview, Progress
from dreambot.engine.registry import REGISTRY

# One worker (and one loaded model) is shared by the whole module, so everything here runs on a
# single module-scoped event loop.
pytestmark = [pytest.mark.mlx, pytest.mark.asyncio(loop_scope="module")]


def installed(model: str) -> weights.Weights:
    found = weights.resolve(REGISTRY.get(model))
    if found is None:
        pytest.skip(f"{model} is not installed")
    return found


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def mlx_engine() -> AsyncIterator[Engine]:
    engine = Engine(backend="mlx", idle_timeout=None)
    try:
        yield engine
    finally:
        await engine.close()


def request(tmp_path: Path, model: str, **overrides) -> GenerateRequest:
    values = {
        "id": new_request_id(),
        "model": model,
        "weights": installed(model).location,
        "prompt": "a red fox sitting in fresh snow, golden hour, wildlife photograph",
        "seeds": [11],
        "width": 768,
        "height": 768,
        "out_dir": str(tmp_path),
        "basename": "img",
    }
    values.update(overrides)
    return GenerateRequest(**values)


def looks_like_a_picture(path: str) -> bool:
    """Guards against the classic silent failures: all-black, all-grey or pure-noise output."""
    with Image.open(path) as image:
        small = image.convert("L").resize((64, 64))
        pixels = list(small.get_flattened_data())
    mean = sum(pixels) / len(pixels)
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    # Neighbouring pixels of a real picture are correlated; in noise they are not.
    rows = [pixels[i * 64 : (i + 1) * 64] for i in range(64)]
    neighbour_diff = sum(abs(row[x] - row[x + 1]) for row in rows for x in range(63)) / (64 * 63)
    return 10 < mean < 245 and variance > 100 and neighbour_diff < 40


async def test_z_image_turbo_grid_with_progress_and_metadata(mlx_engine: Engine, tmp_path: Path):
    events = []
    req = request(tmp_path, "z-image-turbo", seeds=[1, 2, 3, 4], width=1024, height=1024)
    began = time.perf_counter()
    result = await mlx_engine.generate(req, on_event=events.append)
    elapsed = time.perf_counter() - began

    assert len(result.images) == 4
    assert [image.seed for image in result.images] == [1, 2, 3, 4]
    for image in result.images:
        assert (image.width, image.height) == (1024, 1024)
        assert looks_like_a_picture(image.path)
    assert len({Path(image.path).read_bytes() for image in result.images}) == 4

    progress = [event for event in events if isinstance(event, Progress)]
    assert len(progress) == 4 * 9 and progress[-1].step == progress[-1].total == 9
    assert sum(isinstance(event, Loading) for event in events) == 1
    assert sum(isinstance(event, Loaded) for event in events) == 1

    with Image.open(result.images[0].path) as image:
        meta = json.loads(image.info["dreambot"])
    assert meta["model"] == "z-image-turbo" and meta["seed"] == 1 and meta["steps"] == 9
    assert meta["prompt"] == req.prompt
    print(f"\n[z-image-turbo] 4 x 1024px in {elapsed:.1f}s (load {result.load_seconds:.1f}s)")


async def test_model_stays_warm_and_seeds_reproduce(mlx_engine: Engine, tmp_path: Path):
    events = []
    first = await mlx_engine.generate(
        request(tmp_path, "z-image-turbo", basename="a"), on_event=events.append
    )
    second = await mlx_engine.generate(
        request(tmp_path, "z-image-turbo", basename="b"), on_event=events.append
    )
    # Whichever test ran first paid for the load; by now the model must be served from memory.
    assert second.load_seconds is None
    assert Image.open(first.images[0].path).tobytes() == Image.open(second.images[0].path).tobytes()


async def test_aspect_ratio_and_size_snapping(mlx_engine: Engine, tmp_path: Path):
    result = await mlx_engine.generate(request(tmp_path, "z-image-turbo", width=1456, height=816))
    assert (result.images[0].width, result.images[0].height) == (1456, 816)
    assert looks_like_a_picture(result.images[0].path)

    # Sizes that are not multiples of 16 are snapped rather than rejected.
    snapped = await mlx_engine.generate(
        request(tmp_path, "z-image-turbo", width=1000, height=700, basename="s")
    )
    assert (snapped.images[0].width, snapped.images[0].height) == (1008, 704)


async def test_variation_from_an_init_image(mlx_engine: Engine, tmp_path: Path):
    base = await mlx_engine.generate(request(tmp_path, "z-image-turbo", basename="base"))
    source = base.images[0].path

    def distance(path: str) -> float:
        a = Image.open(source).convert("L").resize((64, 64)).get_flattened_data()
        b = Image.open(path).convert("L").resize((64, 64)).get_flattened_data()
        return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / (64 * 64)

    subtle = await mlx_engine.generate(
        request(
            tmp_path,
            "z-image-turbo",
            op=Op.IMG2IMG,
            image=source,
            strength=0.7,
            seeds=[99],
            basename="subtle",
        )
    )
    strong = await mlx_engine.generate(
        request(
            tmp_path,
            "z-image-turbo",
            op=Op.IMG2IMG,
            image=source,
            strength=0.3,
            seeds=[99],
            basename="strong",
        )
    )
    unrelated = await mlx_engine.generate(
        request(tmp_path, "z-image-turbo", seeds=[99], basename="unrelated")
    )
    for result in (subtle, strong):
        assert looks_like_a_picture(result.images[0].path)
        assert (result.images[0].width, result.images[0].height) == (768, 768)
    d_subtle, d_strong, d_unrelated = (distance(r.images[0].path) for r in (subtle, strong, unrelated))
    print(
        f"\n[variation] distance to source: subtle={d_subtle:.1f} strong={d_strong:.1f} unrelated={d_unrelated:.1f}"
    )
    # Keeping more of the source must stay closer to it, and both must differ from it.
    assert 0 < d_subtle < d_strong
    assert d_subtle < d_unrelated


async def test_previews_and_early_stop(mlx_engine: Engine, tmp_path: Path):
    events = []
    await mlx_engine.generate(request(tmp_path, "z-image-turbo", preview_every=3), on_event=events.append)
    previews = [event for event in events if isinstance(event, Preview)]
    assert [event.step for event in previews] == [3, 6]
    # Preview files are transient: gone once the final image is written.
    assert not Path(previews[0].path).exists()

    events.clear()
    stopped = await mlx_engine.generate(
        request(tmp_path, "z-image-turbo", stop_at=0.5, basename="stopped"), on_event=events.append
    )
    steps = [event.step for event in events if isinstance(event, Progress)]
    assert steps == [1, 2, 3, 4, 5]  # ceil(9 * 0.5) = 5 of 9 steps
    assert (stopped.images[0].width, stopped.images[0].height) == (768, 768)


async def test_cancel_mid_generation_then_keep_working(mlx_engine: Engine, tmp_path: Path):
    req = request(
        tmp_path, "z-image-turbo", seeds=list(range(20)), width=1024, height=1024, basename="cancel"
    )
    first_image = asyncio.Event()
    task = asyncio.create_task(
        mlx_engine.generate(req, on_event=lambda e: first_image.set() if isinstance(e, ImageDone) else None)
    )
    await asyncio.wait_for(first_image.wait(), 120)
    began = time.perf_counter()
    await mlx_engine.cancel(req.id)
    with pytest.raises(EngineCancelled):
        await asyncio.wait_for(task, 60)
    took = time.perf_counter() - began
    print(f"\n[cancel] stopped {took:.2f}s after the request")
    assert took < 10

    after = await mlx_engine.generate(request(tmp_path, "z-image-turbo", basename="after"))
    assert after.load_seconds is None and looks_like_a_picture(after.images[0].path)


async def test_memory_report(mlx_engine: Engine, tmp_path: Path):
    await mlx_engine.generate(request(tmp_path, "z-image-turbo"))
    report = await mlx_engine.status()
    assert report is not None
    assert "z-image-turbo" in [loaded.model for loaded in report.loaded]
    assert report.active_memory_gb > 1
    print(
        f"\n[memory] active={report.active_memory_gb:.1f}GB peak={report.peak_memory_gb:.1f}GB "
        f"cache={report.cache_memory_gb:.1f}GB loaded={[m.model for m in report.loaded]}"
    )


async def test_missing_weights_are_reported_as_not_installed(mlx_engine: Engine, tmp_path: Path):
    from dreambot.engine.client import EngineError
    from dreambot.engine.protocol import ErrorKind

    with pytest.raises(EngineError) as excinfo:
        await mlx_engine.generate(
            request(tmp_path, "z-image-turbo", weights="mflux-community/this-repo-does-not-exist-mflux-q8")
        )
    assert excinfo.value.kind == ErrorKind.NOT_INSTALLED
    assert "dreambot models pull z-image-turbo" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# Other model families (each test skips itself unless that model's weights are installed)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def photo(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A small real photograph-like input: a gradient sky over a dark ground with a bright disc."""
    from PIL import ImageDraw

    path = tmp_path_factory.mktemp("inputs") / "scene.png"
    image = Image.new("RGB", (768, 512), (120, 160, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 340, 768, 512], fill=(60, 90, 50))
    draw.ellipse([520, 60, 640, 180], fill=(250, 230, 150))
    image.save(path)
    return str(path)


async def test_flux2_klein_generates_and_edits(mlx_engine: Engine, tmp_path: Path, photo: str):
    generated = await mlx_engine.generate(
        request(tmp_path, "flux2-klein-4b", width=512, height=512, basename="gen")
    )
    assert (generated.images[0].width, generated.images[0].height) == (512, 512)
    assert looks_like_a_picture(generated.images[0].path)

    edited = await mlx_engine.generate(
        request(
            tmp_path,
            "flux2-klein-4b",
            op=Op.EDIT,
            prompt="make it a night scene with a full moon and stars",
            images=[photo],
            width=768,
            height=512,
            basename="edit",
        )
    )
    assert (edited.images[0].width, edited.images[0].height) == (768, 512)
    assert looks_like_a_picture(edited.images[0].path)
    with Image.open(edited.images[0].path) as night, Image.open(photo) as day:
        brightness = lambda image: sum(image.convert("L").resize((32, 32)).get_flattened_data()) / 1024  # noqa: E731
        assert brightness(night) < brightness(day) - 20, "the edit should have darkened the scene"


async def test_qwen_image_21_inpaints_only_inside_the_mask(mlx_engine: Engine, tmp_path: Path, photo: str):
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (768, 512), 0)
    mask.paste(255, (0, 0, 384, 512))  # repaint the left half
    mask.save(mask_path)
    result = await mlx_engine.generate(
        request(
            tmp_path,
            "qwen-image-2.1",
            op=Op.INPAINT,
            prompt="Replace the masked area with: a dense pine forest",
            image=photo,
            mask=str(mask_path),
            width=768,
            height=512,
            steps=12,
            basename="inpaint",
        )
    )
    with Image.open(result.images[0].path) as output, Image.open(photo) as source:
        output, source = output.convert("RGB"), source.convert("RGB")
        assert output.size == (768, 512)

        def difference(box):
            a = output.crop(box).convert("L").get_flattened_data()
            b = source.crop(box).convert("L").get_flattened_data()
            return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)

        kept, repainted = difference((460, 40, 740, 480)), difference((40, 40, 330, 480))
    print(f"\n[qwen-2.1 inpaint] change outside mask={kept:.1f}, inside mask={repainted:.1f}")
    assert kept < 6 and repainted > kept + 10


@pytest.mark.parametrize("model", ["qwen-image", "ming-image-design"])
async def test_other_installed_models_generate(mlx_engine: Engine, tmp_path: Path, model: str):
    result = await mlx_engine.generate(request(tmp_path, model, width=768, height=768, basename=model))
    assert (result.images[0].width, result.images[0].height) == (768, 768)
    assert looks_like_a_picture(result.images[0].path)
    print(f"\n[{model}] 768px in {result.seconds:.1f}s (load {result.load_seconds or 0:.1f}s)")


async def test_seedvr2_upscales_faithfully(mlx_engine: Engine, tmp_path: Path, photo: str):
    """The dedicated upscaler doubles the size and keeps the picture: Upscale (Subtle) and --hd."""
    found = installed("seedvr2-3b")
    with Image.open(photo) as source:
        source = source.convert("RGB")
        width, height = source.size
    upscale = GenerateRequest(
        id=new_request_id(),
        model="seedvr2-3b",
        op=Op.UPSCALE,
        weights=found.location,
        image=photo,
        seeds=[1],
        out_dir=str(tmp_path),
        basename="up",
        extra={"scale": 2},
    )
    result = await mlx_engine.generate(upscale)
    (image,) = result.images
    assert (image.width, image.height) == (width * 2, height * 2)
    assert looks_like_a_picture(image.path)
    with Image.open(image.path) as big:
        back = big.convert("RGB").resize((width, height), Image.LANCZOS)
    difference = ImageStat.Stat(ImageChops.difference(back, source)).mean
    assert max(difference) < 12, f"the upscale drifted from its source: {difference}"
    print(
        f"\n[seedvr2] {width}x{height} -> {image.width}x{image.height} in {result.seconds:.1f}s, drift {max(difference):.1f}"
    )


async def test_describe_and_text_with_the_local_vision_model(mlx_engine: Engine, photo: str):
    from dreambot.engine.protocol import DescribeRequest, TextRequest

    vision = weights.find_vision_model()
    if vision is None:
        pytest.skip("no vision model is installed")
    prompts = await mlx_engine.describe(
        DescribeRequest(id=new_request_id(), image=photo, weights=str(vision), count=4, seed=1)
    )
    assert len(prompts) == 4 and len(set(prompts)) == 4 and all(len(text) > 15 for text in prompts)
    print("\n[describe] " + " | ".join(prompts)[:300])

    replies = await mlx_engine.text(
        TextRequest(
            id=new_request_id(),
            weights=str(vision),
            prompt="Reply with exactly one word: ready",
            max_tokens=8,
            temperature=0.0,
        )
    )
    assert "ready" in replies[0].lower()


async def test_the_language_model_art_directs_a_prompt(mlx_engine: Engine):
    """The prompt enhancer's request, answered by the real language model and checked by its parser."""
    from dreambot.core import enhance
    from dreambot.core.enhance import Brief
    from dreambot.engine.protocol import TextRequest

    vision = weights.find_vision_model()
    if vision is None:
        pytest.skip("no vision model is installed")

    async def direct(brief: Brief, seed: int) -> list[str]:
        request = TextRequest(
            id=new_request_id(),
            weights=str(vision),
            system=enhance.SYSTEM,
            prompt=enhance.request(brief),
            max_tokens=enhance.max_tokens(brief),
            temperature=enhance.temperature(brief),
            top_p=enhance.TOP_P,
            repetition_penalty=enhance.REPETITION_PENALTY,
            seed=seed,
        )
        (answer,) = await mlx_engine.text(request)
        return enhance.parse(answer, brief)

    brief = Brief("a red fox in the snow", 4)
    lines = await direct(brief, 7)
    assert len(lines) == 4 and all(lines), lines
    assert len(set(lines)) >= 3, "the four images should not all get the same shot"
    for line in lines:
        assert 3 <= len(line.split()) <= round(brief.words * 1.5) + 4, line
        assert line.isascii() or all(ord(char) < 0x250 for char in line), line
    assert lines == await direct(brief, 7), "the same seed gives the same direction"
    print("\n[enhance] " + " | ".join(lines)[:400])

    # Maximum chaos and stylize is where a small model used to fall apart into word salad.
    wild = await direct(Brief("a red fox in the snow", 4, stylize=1000, chaos=100), 7)
    assert all(wild) and len(set(wild)) == 4, wild
    words = " ".join(wild).lower().split()
    assert len(set(words)) > len(words) * 0.4, f"the model is looping: {wild}"


# --------------------------------------------------------------------------------------
# Video: Wan 2.2 through Dreambot's own engine (skipped unless its weights are installed)
# --------------------------------------------------------------------------------------

VIDEO = "wan-2.2"


def video_request(tmp_path: Path, op: Op, **overrides) -> GenerateRequest:
    """A short video at the model's own size: two seconds of 832x480."""
    values = {
        "op": op,
        "prompt": "A red fox trots through fresh snow at golden hour, the camera tracking beside it",
        "width": 832,
        "height": 480,
        "frames": 49,
        "fps": 24,
        "seeds": [5],
        "basename": "video",
    }
    return request(tmp_path, VIDEO, **{**values, **overrides})


def frame_difference(one: Image.Image, other: Image.Image) -> float:
    """How far apart two frames are, from 0 (the same picture) to 255."""
    size = (256, 160)
    stat = ImageStat.Stat(
        ImageChops.difference(one.convert("RGB").resize(size), other.convert("RGB").resize(size))
    )
    return sum(stat.mean) / 3


def colours(picture: Image.Image) -> tuple[float, float]:
    """How much contrast and how much colour a picture has."""
    y, cb, cr = (ImageStat.Stat(band) for band in picture.convert("YCbCr").resize((256, 160)).split())
    return y.stddev[0], (cb.stddev[0] + cr.stddev[0]) / 2


async def test_wan_films_from_words_from_a_picture_and_continues(
    mlx_engine: Engine, tmp_path: Path, photo: str
):
    from dreambot.core import video

    installed(VIDEO)
    events = []
    made = await mlx_engine.generate(
        video_request(tmp_path, Op.TXT2VID, basename="words"), on_event=events.append
    )
    (clip,) = made.images
    info = video.probe(clip.video)
    assert (info.width, info.height, info.frames, info.fps, info.sound) == (832, 480, 49, 24.0, False)
    assert (clip.width, clip.height, clip.frames, clip.sound) == (832, 480, 49, False)
    assert looks_like_a_picture(clip.path), "the picture it starts on"
    moved = frame_difference(video.first_frame(clip.video), video.last_frame(clip.video))
    assert moved > 3, "something happens between the first frame and the last"
    # The three passes are reported as they run; the fourth step is the writing.
    assert [(event.step, event.total) for event in events if isinstance(event, Progress)] == [
        (step, 4) for step in (1, 2, 3)
    ]
    assert [type(event) for event in events[:2]] == [Loading, Loaded]
    with Image.open(clip.path) as still:
        notes = json.loads(still.text["dreambot"])
    assert notes["model"] == VIDEO and notes["seed"] == 5 and notes["frames"] == 49
    print(f"\n[video] words: {made.seconds:.1f}s (load {made.load_seconds or 0:.1f}s), moved {moved:.1f}")

    # The same prompt and seed again give the same video, and the prompt is not read twice.
    began = time.perf_counter()
    again = await mlx_engine.generate(video_request(tmp_path, Op.TXT2VID, basename="again"))
    assert again.load_seconds is None, "the model stays loaded between videos"
    assert frame_difference(video.last_frame(again.images[0].video), video.last_frame(clip.video)) < 1
    print(f"[video] the same again: {time.perf_counter() - began:.1f}s")

    # From a picture: it is the first frame, and the frames after it keep its colours.
    began = time.perf_counter()
    started = await mlx_engine.generate(
        video_request(
            tmp_path, Op.IMG2VID, image=photo, prompt="Clouds drift over the field.", basename="picture"
        )
    )
    (from_picture,) = started.images
    with Image.open(photo) as source:
        # (The picture is 3:2 and the video 16:9: it is scaled to cover the frame and cut to it.)
        shown = source.convert("RGB").resize((832, 555)).crop((0, 37, 832, 517))
    first, last = video.first_frame(from_picture.video), video.last_frame(from_picture.video)
    assert frame_difference(first, shown) < 8, "it starts on the picture"
    (contrast, colour), (contrast_after, colour_after) = colours(first), colours(last)
    print(
        f"[video] picture: {time.perf_counter() - began:.1f}s; contrast {contrast:.0f} -> {contrast_after:.0f}, "
        f"colour {colour:.0f} -> {colour_after:.0f}"
    )
    assert abs(contrast_after - contrast) < 0.35 * contrast and abs(colour_after - colour) < 0.5 * colour + 4

    # Continued: the video as it was, to the pixel, and a second more.
    began = time.perf_counter()
    longer = await mlx_engine.generate(
        video_request(tmp_path, Op.EXTEND, video=clip.video, frames=24, basename="longer")
    )
    (extended,) = longer.images
    assert extended.frames == 49 + 24 == video.probe(extended.video).frames
    before, after = video.pixels(clip.video), video.pixels(extended.video)
    assert after[: len(before)] == before, "what was there is not encoded again"
    size = (832, 480)
    joint = [
        Image.frombytes("RGB", size, after[number * 832 * 480 * 3 : (number + 1) * 832 * 480 * 3])
        for number in (47, 48, 49, 50, 72)
    ]
    step_before, step_at, step_after = (frame_difference(joint[i], joint[i + 1]) for i in range(3))
    print(
        f"[video] continued by a second: {time.perf_counter() - began:.1f}s; frame to frame "
        f"{step_before:.1f} before the joint, {step_at:.1f} at it, {step_after:.1f} after"
    )
    assert step_at < 3 * max(step_before, step_after, 1.0), "it goes on without a jump"
    assert frame_difference(joint[2], joint[4]) > 1, "and something happens in what was added"


async def test_a_video_is_stopped_between_passes_and_the_next_one_still_runs(
    mlx_engine: Engine, tmp_path: Path
):
    from dreambot.core import video

    installed(VIDEO)
    req = video_request(tmp_path, Op.TXT2VID, frames=121, seeds=[1, 2, 3, 4], basename="stopped")
    under_way = asyncio.Event()
    task = asyncio.create_task(
        mlx_engine.generate(
            req, on_event=lambda e: under_way.set() if isinstance(e, Progress) and e.step >= 1 else None
        )
    )
    await asyncio.wait_for(under_way.wait(), 180)
    began = time.perf_counter()
    await mlx_engine.cancel(req.id)
    with pytest.raises(EngineCancelled):
        await asyncio.wait_for(task, 60)
    took = time.perf_counter() - began
    print(f"\n[video] stopped {took:.2f}s after the request")
    assert took < 10, "it stops at the next pass, not at the end of the video"
    assert not list(tmp_path.glob("stopped*.mp4")), "nothing is left of the one that was stopped"

    after = await mlx_engine.generate(video_request(tmp_path, Op.TXT2VID, frames=25, basename="after"))
    assert video.probe(after.images[0].video).frames == 25


async def test_wan_loop_moves_and_comes_back_to_its_first_frame(
    mlx_engine: Engine, tmp_path: Path, photo: str
):
    """The model cannot be led back to a picture, so the way out is filmed and played back."""
    from dreambot.core import video

    installed(VIDEO)
    prompt = "The clouds are drifting to the right, and the grass is swaying."
    loop = await mlx_engine.generate(
        video_request(
            tmp_path, Op.IMG2VID, image=photo, end_image=photo, prompt=prompt, frames=73, basename="loop"
        )
    )
    (clip,) = loop.images
    assert clip.frames == 73
    first, last = video.first_frame(clip.video), video.last_frame(clip.video)
    away, back = frame_difference(first, video.frame(clip.video, 36)), frame_difference(first, last)
    print(f"\n[video] loop: {away:.1f} from its first frame halfway, {back:.1f} at the end")
    assert away > 1.5, "it moves"
    assert back < 1.5, "and ends on the frame it started on"


async def test_wan_makes_the_other_shapes_and_the_larger_size(mlx_engine: Engine, tmp_path: Path):
    from dreambot.core import video

    installed(VIDEO)
    for width, height in ((480, 832), (608, 608), (1248, 704)):
        began = time.perf_counter()
        made = await mlx_engine.generate(
            video_request(tmp_path, Op.TXT2VID, width=width, height=height, frames=25, basename=f"s{width}")
        )
        (clip,) = made.images
        info = video.probe(clip.video)
        assert (info.width, info.height, info.frames) == (width, height, 25)
        assert looks_like_a_picture(clip.path), (width, height)
        print(f"\n[video] {width}x{height}, one second: {time.perf_counter() - began:.1f}s")


async def test_the_vision_model_directs_a_picture(mlx_engine: Engine, photo: str):
    """What the director is asked, answered by the real model: the one /describe uses."""
    from dreambot.config import Config
    from dreambot.core import direct
    from dreambot.core.planner import Catalog
    from dreambot.engine.protocol import DescribeRequest

    director = Catalog(Config()).vision_model()
    if director is None:
        pytest.skip("no vision model is installed")

    async def ask(shoot: direct.Shoot) -> list[str]:
        look = DescribeRequest(
            id=new_request_id(),
            image=photo,
            count=shoot.count,
            weights=director,
            instruction=direct.instruction(shoot),
            max_tokens=direct.LINE_TOKENS,
            seed=3,
        )
        return direct.parse(await mlx_engine.describe(look), shoot)

    low = direct.Shoot(text="", count=2, motion="low", start=photo)
    lines = await ask(low)
    print("\n[director] low: " + " | ".join(lines))
    # A line for each video, or none where the model had no more to say: the picture is a sky,
    # a field and a sun, and a video with no line is given a fixed sentence instead.
    written = [line for line in lines if line]
    assert len(lines) == 2 and written and len(set(written)) == len(written)
    for line in written:
        assert 5 <= len(line.split()) <= direct.MAX_WORDS and "ing" in line, line
    assert lines == await ask(low), "the same seed gives the same direction"

    (high,) = await ask(direct.Shoot(text="a bird flies past", count=1, motion="high", start=photo))
    print("[director] high: " + high)
    assert "camera" in high.lower(), "high motion says what the camera does"
