"""Per-model adapters and helper-model discovery, checked without loading any weights."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from PIL import Image

from dreambot.core.text import heuristic_shorten, parse_numbered
from dreambot.engine import weights
from dreambot.engine.backends import mlx_backend
from dreambot.engine.backends.base import BackendError
from dreambot.engine.backends.mlx_backend import _ADAPTERS
from dreambot.engine.protocol import GenerateRequest, Op
from dreambot.engine.registry import REGISTRY


def request(tmp_path: Path, model: str, **overrides) -> GenerateRequest:
    values = {
        "id": "r",
        "model": model,
        "seeds": [1],
        "out_dir": str(tmp_path),
        "basename": "x",
        "prompt": "a fox",
    }
    values.update(overrides)
    return GenerateRequest(**values)


def accepted(model: str, op: Op):
    path = REGISTRY.get(model).variants[op].cls
    module, _, name = path.partition(":")
    cls = getattr(__import__(module, fromlist=[name]), name)
    return inspect.signature(cls.generate_image).parameters


def test_seedvr2_takes_a_scale_or_a_target_and_no_prompt(tmp_path: Path, sample_image: Path):
    spec = REGISTRY.get("seedvr2-3b")
    kwargs = {"prompt": "ignored", "image_path": str(sample_image)}
    _ADAPTERS["seedvr2"](
        kwargs,
        request(tmp_path, spec.id, op=Op.UPSCALE, image=str(sample_image), extra={"scale": 1.5}),
        spec,
        {},
    )
    assert "prompt" not in kwargs and kwargs["softness"] == 0.0
    assert type(kwargs["resolution"]).__name__ == "ScaleFactor" and kwargs["resolution"].value == 1.5

    kwargs = {}
    _ADAPTERS["seedvr2"](
        kwargs, request(tmp_path, spec.id, extra={"resolution": 2160, "softness": 0.5}), spec, {}
    )
    assert kwargs == {"resolution": 2160, "softness": 0.5}


def test_flux_upscaler_renders_at_the_target_size(tmp_path: Path, sample_image: Path):
    spec = REGISTRY.get("dev-controlnet-upscaler")
    kwargs: dict = {}
    _ADAPTERS["flux_upscale"](
        kwargs, request(tmp_path, spec.id, op=Op.UPSCALE, image=str(sample_image)), spec, {}
    )
    assert (kwargs["width"], kwargs["height"], kwargs["controlnet_strength"]) == (1024, 768, 0.4)

    kwargs = {"controlnet_strength": 0.7, "width": 640, "height": 480}
    explicit = request(tmp_path, spec.id, op=Op.UPSCALE, image=str(sample_image), width=640, height=480)
    _ADAPTERS["flux_upscale"](kwargs, explicit, spec, {})
    assert kwargs == {"controlnet_strength": 0.7, "width": 640, "height": 480}
    assert {"controlnet_image_path", "controlnet_strength", "width", "height"} <= set(
        accepted(spec.id, Op.UPSCALE)
    )


def test_zimage_controlnet_builds_typed_controls(tmp_path: Path, sample_image: Path):
    spec = REGISTRY.get("z-image-turbo-controlnet-union-2.1")
    kwargs: dict = {}
    _ADAPTERS["zimage_control"](
        kwargs, request(tmp_path, spec.id, op=Op.CONTROL, image=str(sample_image)), spec, {}
    )
    (control,) = kwargs["controls"]
    assert (
        control.type.value == "depth" and control.strength == 1.0 and Path(control.image_path) == sample_image
    )
    assert (kwargs["width"], kwargs["height"]) == (512, 384)

    kwargs = {}
    pose = request(
        tmp_path, spec.id, op=Op.CONTROL, image=str(sample_image), strength=0.7, extra={"control": "pose"}
    )
    _ADAPTERS["zimage_control"](kwargs, pose, spec, {})
    assert kwargs["controls"][0].type.value == "pose" and kwargs["controls"][0].strength == 0.7
    with pytest.raises(BackendError, match="Unknown control type"):
        bad = request(tmp_path, spec.id, op=Op.CONTROL, image=str(sample_image), extra={"control": "vibes"})
        _ADAPTERS["zimage_control"]({}, bad, spec, {})
    assert "controls" in accepted(spec.id, Op.CONTROL)


class StubVLM:
    calls = 0

    def generate(self, prompt: str, seed: int) -> str:
        StubVLM.calls += 1
        return json.dumps({"short_description": prompt})

    def edit(self, image, edit_instruction: str, use_mask: bool, seed: int) -> str:
        StubVLM.calls += 1
        return json.dumps({"edit_instruction": edit_instruction, "masked": use_mask})


@pytest.fixture
def stub_vlm(monkeypatch):
    StubVLM.calls = 0
    mlx_backend._FIBO_PROMPTS.clear()
    monkeypatch.setattr(mlx_backend, "_fibo_vlm", lambda quantize: StubVLM())
    return StubVLM


def test_fibo_prompts_are_written_once_per_prompt(tmp_path: Path, stub_vlm):
    spec = REGISTRY.get("fibo")
    first: dict = {}
    _ADAPTERS["fibo"](first, request(tmp_path, "fibo"), spec, {})
    second: dict = {}
    _ADAPTERS["fibo"](second, request(tmp_path, "fibo", seeds=[2]), spec, {})
    assert (
        json.loads(first["prompt"]) == {"short_description": "a fox"} and second["prompt"] == first["prompt"]
    )
    assert stub_vlm.calls == 1, "four seeds of one prompt must not load the VLM four times"
    assert first["scheduler"] == "flow_match_euler_discrete"

    ready = json.dumps({"short_description": "already structured"})
    passthrough: dict = {}
    _ADAPTERS["fibo"](passthrough, request(tmp_path, "fibo", prompt=ready), spec, {})
    assert passthrough["prompt"] == ready and stub_vlm.calls == 1


def test_fibo_edit_instruction_mask_and_background_removal(
    tmp_path: Path, sample_image: Path, sample_mask: Path, stub_vlm
):
    spec = REGISTRY.get("fibo-edit")
    kwargs = {"image_path": [str(sample_image)]}
    edit = request(tmp_path, "fibo-edit", op=Op.EDIT, prompt="make it night", images=[str(sample_image)])
    assert _ADAPTERS["fibo_edit"](kwargs, edit, spec, {}) is None
    assert kwargs["image_path"] == str(sample_image)
    assert json.loads(kwargs["prompt"]) == {"edit_instruction": "make it night", "masked": False}

    kwargs = {"image_path": str(sample_image), "mask_path": str(sample_mask)}
    inpaint = request(
        tmp_path,
        "fibo-edit",
        op=Op.INPAINT,
        prompt="a balloon",
        image=str(sample_image),
        mask=str(sample_mask),
    )
    _ADAPTERS["fibo_edit"](kwargs, inpaint, spec, {})
    assert json.loads(kwargs["prompt"])["masked"] is True

    rmbg = REGISTRY.get("fibo-edit-rmbg")
    kwargs = {"image_path": [str(sample_image)]}
    calls_before = stub_vlm.calls
    finish = _ADAPTERS["fibo_edit"](
        kwargs, request(tmp_path, rmbg.id, op=Op.EDIT, prompt="", images=[str(sample_image)]), rmbg, {}
    )
    assert (
        "alpha matte" in json.loads(kwargs["prompt"])["edit_instruction"] and stub_vlm.calls == calls_before
    )
    with Image.open(sample_image) as source:
        cutout = finish(Image.new("L", source.size, 255))
    assert cutout.mode == "RGBA" and cutout.size == (512, 384)


def test_every_adapter_keyword_exists_on_the_model(tmp_path: Path):
    """Keywords the adapters set by hand must be real parameters of the classes they target."""
    expectations = {
        ("seedvr2-3b", Op.UPSCALE): {"resolution", "softness"},
        ("dev-controlnet-upscaler", Op.UPSCALE): {"width", "height", "controlnet_strength"},
        ("z-image-turbo-controlnet-union-2.1", Op.CONTROL): {"controls", "width", "height"},
        ("fibo", Op.TXT2IMG): {"prompt", "scheduler"},
        ("fibo-edit", Op.INPAINT): {"prompt", "scheduler", "image_path", "mask_path"},
        ("qwen-image-2.1", Op.INPAINT): {"image_paths", "mask_image"},
        ("fibo2-edit-turbo", Op.INPAINT): {"image_paths", "mask_path"},
        ("dev-kontext", Op.EDIT): {"image_path"},
    }
    for (model, op), keywords in expectations.items():
        missing = keywords - set(accepted(model, op))
        assert not missing, f"{model} {op.value}: generate_image has no {sorted(missing)}"


def test_numbered_answers_are_parsed_leniently():
    answer = "1. a red fox in snow, wildlife photo\n2) **an oil painting of a fox**\n3: fox close-up, 85mm lens\n**4.** misty winter mood"
    assert parse_numbered(answer, 4) == [
        "a red fox in snow, wildlife photo",
        "an oil painting of a fox",
        "fox close-up, 85mm lens",
        "misty winter mood",
    ]
    assert len(parse_numbered(answer + "\n5. one too many of these", 4)) == 4
    assert parse_numbered(
        "A long first paragraph about the scene.\n\nA second paragraph of description.", 4
    ) == [
        "A long first paragraph about the scene.",
        "A second paragraph of description.",
    ]
    assert parse_numbered("", 4) == []


def make_repo(cache: Path, repo: str, files: dict[str, bytes]) -> Path:
    snapshot = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123"
    for name, data in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return snapshot


def test_installed_weights_are_found_ranked_and_checked_for_completeness(tmp_path: Path):
    cache = tmp_path / "hub"
    cache.mkdir()
    spec = REGISTRY.get("z-image-turbo")
    assert weights.resolve(spec, cache=cache) is None

    q4 = make_repo(cache, "mflux-community/z-image-turbo-mflux-q4", {"transformer/0.safetensors": b"x" * 10})
    q8 = make_repo(cache, "mflux-community/z-image-turbo-mflux-q8", {"transformer/0.safetensors": b"x" * 20})
    upstream = make_repo(cache, "Tongyi-MAI/Z-Image-Turbo", {"transformer/model.safetensors": b"x" * 40})
    found = {item.label: item for item in weights.installed(spec, cache)}
    assert set(found) == {"q4", "q8", "original"} and found["original"].upstream

    best = weights.resolve(spec, prefer=("q8", "q4"), cache=cache)
    assert best.path == q8 and best.location == str(q8) and best.size_bytes == 20
    assert weights.resolve(spec, prefer=("q4", "q8"), cache=cache).path == q4
    # Original weights are addressed by leaving the location unset (mflux loads its pinned revision).
    for path in (q4, q8):
        (path / "transformer" / "0.safetensors").unlink()
    only_upstream = weights.resolve(spec, cache=cache)
    assert only_upstream.path == upstream and only_upstream.location is None

    assert weights.resolve(spec, override=str(upstream), cache=cache).path == upstream
    assert weights.resolve(spec, override="someone/not-cached", cache=cache) is None


def test_incomplete_downloads_do_not_count(tmp_path: Path):
    cache = tmp_path / "hub"
    spec = REGISTRY.get("z-image-turbo")
    snapshot = make_repo(
        cache,
        "mflux-community/z-image-turbo-mflux-q8",
        {
            "transformer/0.safetensors": b"x",
            "transformer/model.safetensors.index.json": json.dumps(
                {"weight_map": {"a": "0.safetensors", "b": "1.safetensors"}}
            ).encode(),
        },
    )
    assert weights.resolve(spec, cache=cache) is None, "a shard named in the index is missing"
    (snapshot / "transformer" / "1.safetensors").write_bytes(b"y")
    assert weights.resolve(spec, cache=cache) is not None

    (snapshot / "vae").mkdir()
    (snapshot / "vae" / "0.safetensors").symlink_to(snapshot / "does-not-exist")
    assert weights.resolve(spec, cache=cache) is None, "a dangling symlink means an interrupted download"
    (snapshot / "vae" / "0.safetensors").unlink()

    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    (blobs / "deadbeef.incomplete").write_bytes(b"")
    assert weights.resolve(spec, cache=cache) is None, "a partial blob means the download is still running"


def test_third_party_exports_extra_repos_and_helpers(tmp_path: Path):
    cache = tmp_path / "hub"
    cache.mkdir()
    ming = REGISTRY.get("ming-image-design")
    make_repo(cache, "someone/Ming-Image-0.1-Design-mflux-q8-te5", {"transformer/0.safetensors": b"x"})
    found = weights.resolve(ming, cache=cache)
    assert found.repo == "someone/Ming-Image-0.1-Design-mflux-q8-te5" and found.quant == "q8"

    canny = REGISTRY.get("dev-controlnet-canny")
    make_repo(cache, "black-forest-labs/FLUX.1-dev", {"transformer/model.safetensors": b"x"})
    assert weights.resolve(canny, cache=cache) is None, "the ControlNet weights are missing"
    make_repo(cache, "InstantX/FLUX.1-dev-Controlnet-Canny", {"model.safetensors": b"x"})
    assert weights.resolve(canny, cache=cache) is not None

    fibo = REGISTRY.get("fibo")
    make_repo(cache, "mflux-community/fibo-mflux-q8", {"transformer/0.safetensors": b"x"})
    assert weights.resolve(fibo, cache=cache) is None, "FIBO cannot run without its prompt-writing model"
    make_repo(cache, "briaai/FIBO-vlm", {"model.safetensors": b"x"})
    assert weights.resolve(fibo, cache=cache) is not None


def test_recommended_download_and_vision_model_lookup(tmp_path: Path):
    assert weights.recommended_repo(REGISTRY.get("flux2-klein-4b"), ("q8",)) == (
        "mflux-community/flux2-klein-4b-mflux-q8",
        "q8",
    )
    assert weights.recommended_repo(REGISTRY.get("seedvr2-3b")) == ("numz/SeedVR2_comfyUI", None)
    assert (
        weights.quant_preference(96)[0] == "q8"
        and weights.quant_preference(32)[0] == "q6"
        and weights.quant_preference(16)[0] == "q4"
    )

    cache = tmp_path / "hub"
    cache.mkdir()
    assert weights.find_vision_model(cache=cache) is None
    make_repo(cache, "mlx-community/Some-Text-LLM-4bit", {"model.safetensors": b"x", "config.json": b"{}"})
    assert weights.find_vision_model(cache=cache) is None
    older = make_repo(
        cache, "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", {"model.safetensors": b"x", "config.json": b"{}"}
    )
    assert weights.find_vision_model(cache=cache) == older
    newer = make_repo(
        cache, "mlx-community/Qwen3-VL-4B-Instruct-4bit", {"model.safetensors": b"x", "config.json": b"{}"}
    )
    assert weights.find_vision_model(cache=cache) == newer, "newer model families are preferred"
    assert weights.find_vision_model("mlx-community/Qwen2.5-VL-7B-Instruct-4bit", cache=cache) == older
    assert weights.find_vision_model(str(older), cache=cache) == older
    assert weights.find_vision_model("mlx-community/not-installed", cache=cache) is None


def test_hf_cache_location_follows_the_environment(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert weights.hf_cache_dir() == Path.home() / ".cache" / "huggingface" / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert weights.hf_cache_dir() == tmp_path / "hf" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere"))
    assert weights.hf_cache_dir() == tmp_path / "elsewhere"


def test_mechanical_shortening_drops_filler_then_trailing_phrases():
    prompt = "a red fox in snow, golden hour, highly detailed, 8k, soft bokeh, wildlife photography"
    versions = heuristic_shorten(prompt, 5)
    assert versions[0] == "a red fox in snow, golden hour, soft bokeh, wildlife photography"
    assert versions[-1] == "a red fox in snow" and len(versions) == 4
    assert [len(v) for v in versions] == sorted((len(v) for v in versions), reverse=True)

    many = heuristic_shorten(", ".join(f"phrase number {i}" for i in range(12)), 5)
    assert len(many) == 5 and many[-1] == "phrase number 0" and many[0].count(",") == 11

    single = heuristic_shorten("a very long single phrase without any commas that keeps going and going", 5)
    assert len(single) >= 3 and len(single[-1]) < len(single[0])
    assert heuristic_shorten("") == [] and heuristic_shorten("fox") == ["fox"]
