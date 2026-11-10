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
from PIL import Image

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
