"""Dreambot's own Wan 2.2 engine (:mod:`dreambot.engine.wan`) and the backend that drives it.

Three kinds of test:

* Each part on small random weights, which needs no download: the sampler, the transformer,
  the small autoencoder, the reader of PyTorch checkpoints, the colour matching, and the text
  encoder against the umT5 that ``transformers`` has (a small one, with the same weights).
* The whole of a request (a video from words, from a picture, a loop, a continuation) through
  :class:`dreambot.engine.backends.wan.WanVideo`, with small random parts standing in for the
  model's: what is checked is everything around the model.
* The real weights against numbers from FastVideo's MLX runtime, which is what these weights
  were trained to run under (``tests/data/wan_reference.npz``; marked ``mlx`` and skipped
  unless the model is installed). The numbers were made with FastVideo at commit ``1fce9fbc``
  on ``FastVideo/FastMetal-5B-QAD``: its schedule, its position tables, its text encoder's
  reading of a prompt, one pass of its transformer with and without a clean first frame, and
  its TAEHV on a clip and on a picture.
"""

from __future__ import annotations

import json
import math
import pickle
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from dreambot.core import video
from dreambot.engine import weights
from dreambot.engine.backends import wan
from dreambot.engine.backends.base import BackendError, JobCancelled, JobContext
from dreambot.engine.protocol import ErrorKind, GenerateRequest, ImageDone, Op, Progress
from dreambot.engine.registry import REGISTRY
from dreambot.engine.wan import look, sampler, taehv
from dreambot.engine.wan.dit import FORMAT_VERSION, MANIFEST, WEIGHTS, Transformer, WanError
from dreambot.engine.wan.umt5 import LENGTH, TextEncoder

WAN = REGISTRY.get("wan-2.2")
REFERENCE = Path(__file__).parent / "data" / "wan_reference.npz"


# --------------------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------------------


def test_the_three_noise_levels_are_the_ones_the_model_was_distilled_at():
    reference = np.load(REFERENCE)
    schedule = sampler.Schedule.shifted()
    levels = schedule.levels()
    assert levels == pytest.approx(reference["levels"].tolist(), abs=1e-6)
    assert levels == pytest.approx([1000.0, 939.672363, 845.207275], abs=1e-5)
    assert [schedule.sigma(level) for level in levels] == pytest.approx(
        reference["sigmas"].tolist(), abs=1e-7
    )
    # (Another shift moves them: the smaller Wan was distilled under 3.)
    assert sampler.Schedule.shifted(3.0).levels() == pytest.approx([1000.0, 903.341370, 766.144775], abs=1e-5)


def test_positions_turn_a_share_of_each_head_for_time_height_and_width():
    cos, sin = sampler.positions(5, 4, 6, 128)
    assert cos.shape == sin.shape == (5 * 4 * 6, 128) and cos.dtype == sin.dtype == np.float32
    # The first token is turned by nothing. Each pair of dimensions is turned together.
    assert (cos[0] == 1).all() and (sin[0] == 0).all()
    assert (cos[:, 0::2] == cos[:, 1::2]).all() and (sin[:, 0::2] == sin[:, 1::2]).all()
    # 44 dimensions follow time, 42 height and 42 width: one step along an axis turns only its share.
    along_time, along_height, along_width = cos[4 * 6], cos[6], cos[1]
    assert (along_time[44:] == 1).all() and along_time[0] == pytest.approx(math.cos(1.0))
    assert (along_height[:44] == 1).all() and (along_height[86:] == 1).all()
    assert along_height[44] == pytest.approx(math.cos(1.0))
    assert (along_width[:86] == 1).all() and along_width[86] == pytest.approx(math.cos(1.0))
    assert sin[4 * 6][2] == pytest.approx(math.sin(1.0 / 10000 ** (2 / 44)), rel=1e-6)


def test_positions_are_the_reference_runtimes_to_the_last_bit_or_the_one_before():
    reference = np.load(REFERENCE)
    cos, sin = sampler.positions(5, 4, 6, 128)
    assert np.abs(cos - reference["positions_cos"]).max() <= 6e-8
    assert np.abs(sin - reference["positions_sin"]).max() <= 6e-8


class Still:
    """Stands where the transformer does: says there is no noise in what it is shown."""

    channels, patch_size, head_dim = 4, (1, 2, 2), 12

    def __init__(self) -> None:
        self.seen: list[tuple[np.ndarray, np.ndarray, tuple]] = []

    def __call__(self, latents: Any, text: Any, timestep: Any, cos: Any, sin: Any) -> Any:
        assert latents.dtype == mx.float16
        self.seen.append((np.array(latents.astype(mx.float32)), np.array(timestep), tuple(cos.shape)))
        return mx.zeros(latents.shape, dtype=mx.float16)


def test_a_clip_is_made_in_three_passes_from_noise_that_its_seed_decides():
    model, passes = Still(), []
    made = sampler.sample(
        model, None, (3, 4, 6), seed=7, on_step=lambda done, total: passes.append((done, total))
    )
    assert made.shape == (4, 3, 4, 6) and made.dtype == np.float32 and np.isfinite(made).all()
    assert passes == [(1, 3), (2, 3), (3, 3)]
    levels = sampler.Schedule.shifted().levels()
    # Every token is told the level of the pass: 3 frames of 2 x 3 tokens.
    assert [seen[1].shape for seen in model.seen] == [(1, 18)] * 3 and model.seen[0][2] == (18, 12)
    assert [float(seen[1][0, 0]) for seen in model.seen] == pytest.approx(levels, abs=1e-3)
    assert all((seen[1] == seen[1][0, 0]).all() for seen in model.seen)
    # It starts as noise. Told there is none in it, the sampler keeps what it has and mixes in
    # less fresh noise from pass to pass; what the last pass was shown is what comes out.
    spread = [float(seen[0].std()) for seen in model.seen]
    assert 0.9 < spread[0] < 1.1 and spread[0] > spread[1] > spread[2]
    assert (made == model.seen[2][0][0]).all()
    assert (sampler.sample(Still(), None, (3, 4, 6), seed=7) == made).all()
    assert not (sampler.sample(Still(), None, (3, 4, 6), seed=8) == made).all()


def test_a_clip_is_held_to_the_frames_it_starts_on():
    model = Still()
    start = np.full((1, 4, 2, 4, 6), 0.25, dtype=np.float32)
    start[:, :, 1] = -0.5
    made = sampler.sample(model, None, (5, 4, 6), seed=3, start=start)
    assert (made[:, :2] == start[0]).all() and not (made[:, 2:] == 0.25).any()
    for shown, told, _ in model.seen:
        # They are put back before every pass, and their tokens are told they carry no noise.
        assert (shown[0, :, :2] == start[0]).all()
        assert (told[0, : 2 * 6] == 0).all() and (told[0, 2 * 6 :] > 0).all()
    with pytest.raises(ValueError, match="longer"):
        sampler.sample(Still(), None, (2, 4, 6), seed=3, start=start)


def test_a_clip_can_start_from_another_under_the_noise():
    model = Still()
    still = np.full((1, 4, 3, 4, 6), 2.0, dtype=np.float32)
    sampler.sample(model, None, (3, 4, 6), seed=7, still=still, noise_share=0.75)
    from_noise = Still()
    sampler.sample(from_noise, None, (3, 4, 6), seed=7)
    # A quarter of what the first pass is shown is the clip, and three quarters the same noise.
    mixed, noise = model.seen[0][0], from_noise.seen[0][0]
    assert np.abs(mixed - (0.25 * 2.0 + 0.75 * noise)).max() < 2e-3
    # The passes are told the same levels either way.
    assert [float(seen[1][0, 0]) for seen in model.seen] == [float(seen[1][0, 0]) for seen in from_noise.seen]
    whole = Still()
    sampler.sample(whole, None, (3, 4, 6), seed=7, still=still)
    assert (whole.seen[0][0] == noise).all(), "all noise is where it starts unless something says less"


def test_a_clip_can_be_stopped_between_passes():
    model = Still()

    def stop(done: int, total: int) -> None:
        raise JobCancelled()

    with pytest.raises(JobCancelled):
        sampler.sample(model, None, (3, 4, 6), seed=1, on_step=stop)
    assert len(model.seen) == 1


# --------------------------------------------------------------------------------------
# The transformer, on small random weights
# --------------------------------------------------------------------------------------

HEADS, HEAD_DIM, TEXT_DIM, FREQ_DIM, FFN, CHANNELS = 2, 32, 32, 32, 64, 4
DIM = HEADS * HEAD_DIM


def small_transformer(seed: int = 0, blocks: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    """A checkpoint in the form FastMetal's has, a hundredth of the width and two blocks deep."""
    rng = np.random.default_rng(seed)
    arrays: dict[str, Any] = {}

    def linear(name: str, out: int, into: int, bias: bool = True) -> None:
        arrays[f"{name}.weight"] = mx.array(
            (rng.standard_normal((out, into)) / math.sqrt(into)).astype(np.float16)
        )
        if bias:
            arrays[f"{name}.bias"] = mx.array((rng.standard_normal(out) * 0.1).astype(np.float16))

    def vector(name: str, *shape: int, around: float = 0.0) -> None:
        arrays[name] = mx.array((around + rng.standard_normal(shape) * 0.1).astype(np.float32))

    linear("patch_embedding", DIM, CHANNELS * 4)
    linear("condition_embedder.time_embedder.linear_1", DIM, FREQ_DIM)
    linear("condition_embedder.time_embedder.linear_2", DIM, DIM)
    linear("condition_embedder.time_proj", 6 * DIM, DIM)
    linear("condition_embedder.text_embedder.linear_1", DIM, TEXT_DIM)
    linear("condition_embedder.text_embedder.linear_2", DIM, DIM)
    linear("proj_out", CHANNELS * 4, DIM)
    vector("scale_shift_table", 1, 2, DIM)
    for index in range(blocks):
        prefix = f"blocks.{index}."
        for part in (
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out",
        ):
            linear(prefix + part, DIM, DIM)
        for part in ("norm_q", "norm_k", "attn2.norm_q", "attn2.norm_k"):
            vector(f"{prefix}{part}.weight", DIM, around=1.0)
        vector(f"{prefix}self_attn_residual_norm.norm.weight", DIM, around=1.0)
        vector(f"{prefix}self_attn_residual_norm.norm.bias", DIM)
        linear(prefix + "ffn.fc_in", FFN, DIM)
        linear(prefix + "ffn.fc_out", DIM, FFN)
        vector(f"{prefix}scale_shift_table", 1, 6, DIM)
    manifest = {
        "format_version": FORMAT_VERSION,
        "num_blocks": blocks,
        "config": {
            "num_attention_heads": HEADS,
            "attention_head_dim": HEAD_DIM,
            "in_channels": CHANNELS,
            "out_channels": CHANNELS,
            "patch_size": [1, 2, 2],
            "freq_dim": FREQ_DIM,
            "eps": 1e-6,
        },
    }
    return arrays, manifest


def packed(arrays: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The same checkpoint with its square matrices packed to 8 bits, the way FastMetal's are."""
    out, keys = dict(arrays), {}
    for name, value in arrays.items():
        if name.endswith(".weight") and value.ndim == 2 and value.shape[1] % 32 == 0 and "norm" not in name:
            small, scales, biases = mx.quantize(
                value.astype(mx.float32), group_size=32, bits=8, mode="affine"
            )
            out[name], out[f"{name}.scales"], out[f"{name}.biases"] = small, scales, biases
            keys[name] = True
    quantization = {"group_size": 32, "bits": 8, "mode": "affine"}
    return out, {**manifest, "quantized_keys": keys, "quantization": quantization}


def shown(seed: int = 1, frames: int = 3) -> tuple[Any, Any, Any, Any]:
    """Latents of ``frames`` x 4 x 6, a prompt of five tokens, and the clip's position tables."""
    rng = np.random.default_rng(seed)
    latents = mx.array(rng.standard_normal((1, CHANNELS, frames, 4, 6)).astype(np.float16))
    text = mx.array(rng.standard_normal((1, 5, TEXT_DIM)).astype(np.float16))
    cos, sin = (mx.array(table) for table in sampler.positions(frames, 2, 3, HEAD_DIM))
    return latents, text, cos, sin


def told(frames: int, level: float, clean: int = 0) -> Any:
    timestep = np.full((1, frames * 6), level, dtype=np.float32)
    timestep[:, :clean] = 0.0
    return mx.array(timestep)


def test_the_transformer_answers_in_the_shape_it_is_asked_in():
    arrays, manifest = small_transformer()
    model = Transformer(arrays, manifest, compiled=False)
    assert (model.channels, model.head_dim, model.patch_size) == (CHANNELS, HEAD_DIM, (1, 2, 2))
    latents, text, cos, sin = shown()
    out = model(latents, text, told(3, 900.0), cos, sin)
    assert out.shape == latents.shape and out.dtype == mx.float16
    values = np.array(out.astype(mx.float32))
    assert np.isfinite(values).all() and values.std() > 0.01
    # The noise level, the prompt and the latents each change the answer.
    for other in (
        model(latents, text, told(3, 500.0), cos, sin),
        model(latents, text * 0, told(3, 900.0), cos, sin),
        model(latents * 0.5, text, told(3, 900.0), cos, sin),
    ):
        assert np.abs(np.array(other.astype(mx.float32)) - values).max() > 1e-3


def test_the_compiled_transformer_is_the_same_one_and_takes_another_shape_after():
    arrays, manifest = small_transformer()
    plain, traced = Transformer(arrays, manifest, compiled=False), Transformer(arrays, manifest)
    latents, text, cos, sin = shown()
    for clean in (0, 6):
        one = np.array(plain(latents, text, told(3, 800.0, clean), cos, sin).astype(mx.float32))
        other = np.array(traced(latents, text, told(3, 800.0, clean), cos, sin).astype(mx.float32))
        assert np.abs(one - other).max() < 2e-2, clean
        again = np.array(traced(latents, text, told(3, 800.0, clean), cos, sin).astype(mx.float32))
        assert (again == other).all()
    longer = shown(frames=5)
    assert traced(longer[0], longer[1], told(5, 800.0), *longer[2:]).shape == longer[0].shape


def test_frames_held_clean_are_told_so_and_only_at_the_start():
    arrays, manifest = small_transformer()
    model = Transformer(arrays, manifest, compiled=False)
    latents, text, cos, sin = shown()
    noisy = np.array(model(latents, text, told(3, 900.0), cos, sin).astype(mx.float32))
    started = np.array(model(latents, text, told(3, 900.0, clean=6), cos, sin).astype(mx.float32))
    assert np.abs(started - noisy).max() > 1e-3
    late = np.full((1, 18), 900.0, dtype=np.float32)
    late[:, 12:] = 0.0
    with pytest.raises(WanError, match="start of a clip"):
        model(latents, text, mx.array(late), cos, sin)
    with pytest.raises(WanError, match="nothing to make"):
        model(latents, text, mx.zeros((1, 18)), cos, sin)
    two = np.full((1, 18), 900.0, dtype=np.float32)
    two[:, 9:] = 600.0
    with pytest.raises(WanError, match="one level"):
        model(latents, text, mx.array(two), cos, sin)


def test_matrices_packed_to_8_bits_are_multiplied_as_they_are():
    arrays, manifest = small_transformer()
    small, small_manifest = packed(arrays, manifest)
    assert len(small_manifest["quantized_keys"]) > 20
    whole = Transformer(arrays, manifest, compiled=False)
    eight_bit = Transformer(small, small_manifest, compiled=False)
    latents, text, cos, sin = shown()
    one = np.array(whole(latents, text, told(3, 900.0), cos, sin).astype(mx.float32))
    other = np.array(eight_bit(latents, text, told(3, 900.0), cos, sin).astype(mx.float32))
    assert not (one == other).all()
    assert np.corrcoef(one.ravel(), other.ravel())[0, 1] > 0.99


def test_a_checkpoint_is_read_from_its_two_files_or_refused(tmp_path: Path):
    arrays, manifest = packed(*small_transformer())
    with pytest.raises(WanError, match="not a FastMetal checkpoint"):
        Transformer.load(tmp_path)
    mx.save_safetensors(str(tmp_path / WEIGHTS), arrays)
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    model = Transformer.load(tmp_path)
    latents, text, cos, sin = shown()
    assert model(latents, text, told(3, 900.0), cos, sin).shape == latents.shape
    (tmp_path / MANIFEST).write_text(json.dumps({**manifest, "format_version": FORMAT_VERSION + 1}))
    with pytest.raises(WanError, match="format"):
        Transformer.load(tmp_path)
    other = {**manifest, "quantization": {**manifest["quantization"], "mode": "mxfp8"}}
    (tmp_path / MANIFEST).write_text(json.dumps(other))
    with pytest.raises(WanError, match="affine"):
        Transformer.load(tmp_path)


# --------------------------------------------------------------------------------------
# The small autoencoder, on small random weights
# --------------------------------------------------------------------------------------

LATENT, WIDE = 4, 8


def small_autoencoder(
    seed: int = 0, widths: tuple[int, int, int, int] = (16, 8, 8, 8)
) -> dict[str, np.ndarray]:
    """TAEHV's arrays as its checkpoint names and shapes them, an eighth as wide, for 4-channel latents."""
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}

    def conv(name: str, out: int, into: int, size: int = 3, bias: bool = True) -> None:
        scale = 1.0 / math.sqrt(into * size * size)
        arrays[f"{name}.weight"] = (rng.standard_normal((out, into, size, size)) * scale).astype(np.float32)
        if bias:
            arrays[f"{name}.bias"] = (rng.standard_normal(out) * 0.05).astype(np.float32)

    def memory(name: str, width: int) -> None:
        conv(f"{name}.conv.0", width, 2 * width)
        conv(f"{name}.conv.2", width, width)
        conv(f"{name}.conv.4", width, width)

    conv("encoder.0", WIDE, 12)
    for pool, stride in ((2, 2), (7, 2), (12, 1)):
        conv(f"encoder.{pool}.conv", WIDE, WIDE * stride, size=1, bias=False)
        conv(f"encoder.{pool + 1}", WIDE, WIDE, bias=False)
        for index in range(pool + 2, pool + 5):
            memory(f"encoder.{index}", WIDE)
    conv("encoder.17", LATENT, WIDE)
    first, second, third, last = widths
    conv("decoder.1", first, LATENT)
    for at, width, after, stride in ((3, first, second, 1), (9, second, third, 2), (15, third, last, 2)):
        for index in range(at, at + 3):
            memory(f"decoder.{index}", width)
        conv(f"decoder.{at + 4}.conv", width * stride, width, size=1, bias=False)
        conv(f"decoder.{at + 5}", after, width, bias=False)
    conv("decoder.22", 12, last)
    return arrays


@pytest.fixture(scope="module")
def small() -> taehv.TinyAutoencoder:
    arrays = {name: mx.array(value) for name, value in small_autoencoder().items()}
    return taehv.TinyAutoencoder(arrays, half=False)


def test_frames_decoded_a_few_at_a_time_are_the_frames_decoded_at_once(small: taehv.TinyAutoencoder):
    rng = np.random.default_rng(2)
    latents = rng.standard_normal((LATENT, 7, 3, 4)).astype(np.float32)
    whole = small.frames(latents, chunk=7)
    # Four frames for each latent frame, less the three before a clip's first.
    assert whole.shape == (4 * 6 + 1, 48, 64, 3) and whole.dtype == np.uint8
    assert whole.std() > 5, "the frames differ from one another and from grey"
    for chunk in (1, 2, 3, 5):
        # (The same to the level, or the one next to it: how many frames a convolution is
        # given at once decides how its sums are rounded.)
        pieces = small.frames(latents, chunk=chunk)
        assert pieces.shape == whole.shape and np.abs(pieces.astype(np.int16) - whole).max() <= 1, chunk
    # Nothing is remembered from one clip to the next.
    assert (small.frames(latents, chunk=7) == whole).all()
    other = small.frames(rng.standard_normal((LATENT, 7, 3, 4)).astype(np.float32), chunk=2)
    assert np.abs(other.astype(np.int16) - whole).mean() > 5
    assert np.abs(small.frames(latents, chunk=2).astype(np.int16) - whole).max() <= 1
    assert small.frames(latents[:, :1]).shape == (1, 48, 64, 3)


def test_a_picture_and_the_end_of_a_video_are_encoded_as_a_clips_first_latent_frames(
    small: taehv.TinyAutoencoder,
):
    rng = np.random.default_rng(3)
    picture = rng.random((48, 64, 3), dtype=np.float32)
    latent = small.picture_latent(picture)
    assert latent.shape == (1, LATENT, 1, 3, 4) and np.isfinite(latent).all()
    # One frame, then whole groups of four.
    clip = rng.random((5, 48, 64, 3), dtype=np.float32)
    assert small.latents(clip).shape == (1, LATENT, 2, 3, 4)
    assert small.latents(rng.random((9, 48, 64, 3), dtype=np.float32)).shape == (1, LATENT, 3, 3, 4)
    # A clip that stands still on a picture starts with that picture's latent.
    still = small.latents(np.repeat(picture[None], 5, axis=0))
    assert np.allclose(still[:, :, 0], latent[:, :, 0], atol=1e-5)
    # The first latent frame is made from the first four frames, and the last frame fills its group.
    assert not np.allclose(small.latents(clip)[:, :, 0], small.picture_latent(clip[0])[:, :, 0], atol=1e-3)
    ended = np.concatenate([clip, np.repeat(clip[-1:], 3, axis=0)])[:8]
    assert np.allclose(
        small.latents(clip), small.latents(np.concatenate([ended, ended[-1:]]))[:, :, :2], atol=1e-5
    )
    with pytest.raises(ValueError, match="groups of 4"):
        small.latents(clip[:3])
    with pytest.raises(ValueError, match="steps of 16"):
        small.picture_latent(picture[:40])


def test_half_precision_is_within_a_shade_of_full(small: taehv.TinyAutoencoder):
    arrays = {name: mx.array(value) for name, value in small_autoencoder().items()}
    half = taehv.TinyAutoencoder(arrays, half=True)
    latents = np.random.default_rng(4).standard_normal((LATENT, 3, 3, 4)).astype(np.float32)
    one, other = small.frames(latents).astype(np.int16), half.frames(latents).astype(np.int16)
    assert np.abs(one - other).max() <= 2


def checkpoint(path: Path, state: dict[str, np.ndarray], pickled: bytes | None = None) -> Path:
    """A file laid out as ``torch.save`` lays out a state dict, written without PyTorch."""
    kinds = {np.dtype(np.float16): "HalfStorage", np.dtype(np.float32): "FloatStorage"}

    class Stored:
        def __init__(self, key: str, value: np.ndarray):
            self.key, self.value = key, value

    class Writer(pickle.Pickler):
        def persistent_id(self, obj: Any) -> Any:
            if isinstance(obj, Stored):
                return ("storage", Kind(kinds[obj.value.dtype]), obj.key, "cpu", obj.value.size)
            return None

    class Kind:
        def __init__(self, name: str):
            self.name = name

        def __reduce__(self) -> Any:
            return (Named("torch", self.name), ())

    class Named:
        """Pickles as a reference to ``module.name``, whatever is or is not installed."""

        def __init__(self, module: str, name: str):
            self.module, self.name = module, name

    import copyreg
    import io

    table = dict(copyreg.dispatch_table)
    buffer = io.BytesIO()
    writer = Writer(buffer, protocol=2)
    writer.dispatch_table = table

    def as_global(obj: Named) -> str:
        return obj.name

    # A name that pickles as "torch <name>" without importing it: written by hand below.
    rows = []
    for number, (name, value) in enumerate(state.items()):
        rows.append((name, str(number), value))
    if pickled is None:
        out = io.BytesIO()
        out.write(b"\x80\x02")  # protocol 2
        out.write(b"ccollections\nOrderedDict\n)R")  # OrderedDict()
        out.write(b"(")  # mark, for the items set below
        for name, key, value in rows:
            shape = "".join(f"I{side}\n" for side in value.shape)
            strides = [int(np.prod(value.shape[i + 1 :])) for i in range(value.ndim)]
            stride = "".join(f"I{step}\n" for step in strides)
            out.write(b"V" + name.encode() + b"\n")
            out.write(b"ctorch._utils\n_rebuild_tensor_v2\n(")
            out.write(
                b"(Vstorage\nctorch\n" + kinds[value.dtype].encode() + b"\nV" + key.encode() + b"\nVcpu\n"
            )
            out.write(f"I{value.size}\n".encode() + b"tQ")  # the storage, by persistent id
            out.write(b"I0\n(" + shape.encode() + b"t(" + stride.encode() + b"t")
            out.write(b"I00\nccollections\nOrderedDict\n)RtR")
        out.write(b"u.")  # set the items, stop
        pickled = out.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickled)
        archive.writestr("archive/version", "3\n")
        for _, key, value in rows:
            archive.writestr(f"archive/data/{key}", value.tobytes())
    return path


def test_a_pytorch_checkpoint_is_read_without_pytorch(tmp_path: Path):
    state = {
        "encoder.0.weight": np.arange(24, dtype=np.float16).reshape(2, 3, 2, 2) / 8,
        "encoder.0.bias": np.array([0.5, -1.25], dtype=np.float32),
    }
    path = checkpoint(tmp_path / "small.pth", state)
    read = taehv.read_checkpoint(path)
    assert list(read) == list(state)
    for name, value in state.items():
        assert read[name].dtype == value.dtype and (read[name] == value).all(), name
    kept = taehv.convert(path, tmp_path / "kept" / "small.safetensors")
    again = mx.load(str(kept))
    assert (
        set(again) == set(state) and (np.array(again["encoder.0.weight"]) == state["encoder.0.weight"]).all()
    )
    assert not list((tmp_path / "kept").glob("*.part*"))


def test_a_checkpoint_made_by_pytorch_itself_reads_the_same(tmp_path: Path):
    torch = pytest.importorskip("torch")
    state = OrderedDict(
        [
            ("decoder.1.weight", torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2).half() / 16),
            ("decoder.1.bias", torch.tensor([1.5, -2.0, 0.25, 8.0])),
        ]
    )
    torch.save(state, tmp_path / "real.pth")
    read = taehv.read_checkpoint(tmp_path / "real.pth")
    assert list(read) == list(state)
    for name, value in state.items():
        assert (read[name] == value.numpy()).all() and read[name].dtype == value.numpy().dtype, name


def test_a_checkpoint_that_names_anything_but_arrays_is_refused_and_runs_nothing(tmp_path: Path):
    marker = tmp_path / "ran"

    class Runs:
        def __reduce__(self) -> Any:
            return (Path.touch, (marker,))

    path = checkpoint(tmp_path / "bad.pth", {}, pickled=pickle.dumps(Runs(), protocol=2))
    with pytest.raises(taehv.CheckpointError, match="no use for"):
        taehv.read_checkpoint(path)
    assert not marker.exists()
    (tmp_path / "junk.pth").write_bytes(b"not a zip at all")
    with pytest.raises(taehv.CheckpointError, match="could not be read"):
        taehv.read_checkpoint(tmp_path / "junk.pth")
    with zipfile.ZipFile(tmp_path / "empty.pth", "w") as archive:
        archive.writestr("readme.txt", "nothing here")
    with pytest.raises(taehv.CheckpointError, match="not a PyTorch checkpoint"):
        taehv.read_checkpoint(tmp_path / "empty.pth")


# --------------------------------------------------------------------------------------
# The text encoder, against the umT5 that transformers has
# --------------------------------------------------------------------------------------


class Words:
    """Stands where the tokenizer does: one token for each word, and the end-of-sequence token."""

    class Encoded:
        def __init__(self, ids: list[int]):
            self.ids = ids

    def encode(self, prompt: str, add_special_tokens: bool = True) -> Words.Encoded:
        return Words.Encoded([2 + len(word) % 60 for word in prompt.split()] + [1])


@pytest.fixture(scope="module")
def small_umt5() -> tuple[Any, TextEncoder]:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.UMT5Config(
        vocab_size=64,
        d_model=32,
        d_kv=8,
        d_ff=48,
        num_layers=3,
        num_heads=4,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        feed_forward_proj="gated-gelu",
    )
    theirs = transformers.UMT5EncoderModel(config).eval()
    with torch.no_grad():
        for name, value in theirs.named_parameters():
            # (Layer norms start as ones, which would hide a wrong one.)
            if "layer_norm" in name:
                value.uniform_(0.5, 1.5)
    arrays = {name: mx.array(value.numpy()) for name, value in theirs.state_dict().items()}
    return theirs, TextEncoder(arrays, config.to_dict(), Words())


def test_the_text_encoder_is_umt5(small_umt5: tuple[Any, TextEncoder]):
    import torch

    theirs, ours = small_umt5
    rng = np.random.default_rng(5)
    # Long enough that tokens are further apart than the furthest distance told apart (128).
    for count in (1, 7, 150):
        ids = rng.integers(2, 64, size=count).tolist()
        with torch.no_grad():
            want = theirs(input_ids=torch.tensor([ids])).last_hidden_state[0].numpy()
        # On the CPU, where MLX multiplies matrices to the last digit of a 32-bit float. On the
        # GPU of the Mac this was written on it is good to three or four digits, which is the
        # precision the model's reference runtime works at and enough for the model, but too
        # coarse to tell a table that is read one place off from one that is read right.
        with mx.stream(mx.cpu):
            got = np.array(ours._run(ids))
        assert got.shape == want.shape == (count, 32)
        assert np.abs(got - want).max() < 2e-4, count
        on_the_gpu = np.array(ours._run(ids))
        assert np.abs(on_the_gpu - want).max() < 0.05, count


def test_a_prompt_is_padded_with_zeros_and_cut_at_the_models_length(small_umt5: tuple[Any, TextEncoder]):
    _, ours = small_umt5
    read = ours.encode(["a red fox", "snow"])
    assert read.shape == (2, LENGTH, 32) and read.dtype == np.float32
    assert (read[0, 4:] == 0).all() and np.abs(read[0, :4]).min() > 0
    assert (read[1, 2:] == 0).all() and not np.allclose(read[0, :2], read[1, :2])
    # A prompt that is too long keeps its end-of-sequence token.
    ids = ours.tokens(" ".join(["word"] * 600))
    assert len(ids) == LENGTH and ids[-1] == 1 and ids[-2] != 1


# --------------------------------------------------------------------------------------
# Keeping a video in the colours of the picture it starts on
# --------------------------------------------------------------------------------------


def scene(seed: int = 6, count: int = 12) -> np.ndarray:
    """Frames of one smooth, coloured picture that drifts a little from frame to frame."""
    rng = np.random.default_rng(seed)
    base = np.asarray(
        Image.fromarray(rng.integers(88, 168, (6, 8, 3), dtype=np.uint8)).resize(
            (64, 48), Image.Resampling.BICUBIC
        )
    )
    return np.stack([np.roll(base, shift, axis=1) for shift in range(count)])


def relit(frames: np.ndarray, contrast: float, lift: float, colour: float) -> np.ndarray:
    """``frames`` in another look: more (or less) contrast, lifted, and with more (or less) colour.

    Worked out by Pillow, as brightness and two colour axes: not by the code that is tested.
    """
    out = []
    for frame in frames:
        y, cb, cr = (
            np.asarray(band, dtype=np.float32) for band in Image.fromarray(frame).convert("YCbCr").split()
        )
        y = (y - 128.0) * contrast + 128.0 + lift
        cb, cr = (cb - 128.0) * colour + 128.0, (cr - 128.0) * colour + 128.0
        bands = [Image.fromarray(np.clip(band + 0.5, 0, 255).astype(np.uint8)) for band in (y, cb, cr)]
        out.append(np.asarray(Image.merge("YCbCr", bands).convert("RGB")))
    return np.stack(out)


def test_frames_after_the_first_are_brought_back_to_its_colours():
    frames = scene()
    made = frames.copy()
    made[1:] = relit(frames[1:], contrast=1.1, lift=8, colour=1.5)
    assert np.abs(made[6].astype(np.int16) - frames[6]).mean() > 6
    gain, offset = look.corrections(made)
    assert gain.shape == offset.shape == (12, 3)
    assert (gain[0] == 1).all() and (offset[0] == 0).all(), "the picture itself is left alone"
    assert gain[1:, 0] == pytest.approx(1 / 1.1, abs=0.03) and gain[1:, 1:] == pytest.approx(
        1 / 1.5, abs=0.05
    )
    assert offset[1:, 1:] == pytest.approx(0, abs=2)
    kept = look.keep(made.copy())
    assert (kept[0] == frames[0]).all()
    assert np.abs(kept[1:].astype(np.int16) - frames[1:]).mean() < 3
    # With five frames given (a video that is continued), it is the frames after those.
    made = frames.copy()
    made[5:] = relit(frames[5:], contrast=0.92, lift=-6, colour=0.7)
    kept = look.keep(made.copy(), held=5)
    assert (kept[:5] == frames[:5]).all() and np.abs(kept[5:].astype(np.int16) - frames[5:]).mean() < 4


def test_colour_and_contrast_that_drift_are_followed_frame_by_frame():
    """The model's colours leave the picture's by degrees, over a second or two."""
    frames = scene(count=40)
    made = frames.copy()
    for number in range(1, 40):
        made[number] = relit(
            frames[number : number + 1], contrast=1.0 + number / 400, lift=0.0, colour=1.0 + number / 40
        )[0]
    gain, _ = look.corrections(made)
    # Less and less of the colour is kept, as more and more was added; and of the contrast.
    assert gain[2, 1] > gain[12, 1] > gain[25, 1] > gain[39, 1] and gain[39, 1] == pytest.approx(0.5, abs=0.1)
    assert gain[25, 1] == pytest.approx(1 / (1 + 25 / 40), abs=0.06)
    assert gain[2, 0] > gain[20, 0] > gain[39, 0] and gain[39, 0] == pytest.approx(1 / 1.0975, abs=0.03)
    kept = look.keep(made.copy())
    there, back = (np.abs(clip[30].astype(np.int16) - frames[30]).mean() for clip in (made, kept))
    assert back < 4 < there


def test_a_scene_that_is_meant_to_get_lighter_or_flatter_mostly_does():
    """Brightness is corrected within narrow bounds: the model's creep is, a sunrise is not."""
    frames = scene(count=40)
    lit = frames.copy()
    for number in range(1, 40):
        lit[number] = relit(frames[number : number + 1], contrast=1.0, lift=number * 1.5, colour=1.0)[0]
    kept = look.keep(lit.copy())
    assert lit[39].mean() - lit[1].mean() > 50
    assert kept[39].mean() - kept[1].mean() > 35, "most of the rise is still there"
    # Fog: half the contrast goes, and all but a little of that stays gone.
    fogged = frames.copy()
    fogged[1:] = relit(frames[1:], contrast=0.5, lift=30, colour=1.0)
    gain, _ = look.corrections(fogged)
    assert gain[1:, 0] == pytest.approx(look.LIGHT_GAIN[1])
    after, before = (
        clip[6].astype(np.float32).mean(axis=-1).std() for clip in (look.keep(fogged.copy()), frames)
    )
    assert after < 0.65 * before


def test_frames_that_are_already_in_its_colours_are_left_as_they_are():
    frames = scene()
    assert (look.keep(frames.copy()) == frames).all()
    # A clip with nothing after what was given has nothing to bring back.
    assert (look.keep(frames[:1].copy()) == frames[:1]).all()
    gain, offset = look.corrections(frames[:5], held=5)
    assert (gain == 1).all() and (offset == 0).all()


def test_a_correction_measured_across_a_cut_is_held_in_bounds():
    frames = scene()
    made = frames.copy()
    made[1:] = 255 - made[1:] // 16  # nearly white, nearly flat: nothing like the picture
    gain, offset = look.corrections(made)
    assert (gain[:, 1:] <= look.GAIN[1]).all() and (gain[:, 1:] >= look.GAIN[0]).all()
    assert (np.abs(offset[:, 1:]) <= look.OFFSET).all()
    assert (gain[:, 0] <= look.LIGHT_GAIN[1] + 1e-5).all() and (gain[:, 0] >= look.LIGHT_GAIN[0] - 1e-5).all()
    assert look.keep(made.copy()).shape == made.shape


# --------------------------------------------------------------------------------------
# A request, with small random parts standing in for the model's
# --------------------------------------------------------------------------------------


class Noise:
    """Stands where the transformer does, for a 4-channel latent: predicts a little of what it sees."""

    channels, patch_size, head_dim = LATENT, (1, 2, 2), 12

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, latents: Any, text: Any, timestep: Any, cos: Any, sin: Any) -> Any:
        self.calls += 1
        return (latents.astype(mx.float32) * 0.5).astype(mx.float16)


class Recorder:
    """A job context that keeps what is reported, and can stop the job at a chosen pass."""

    def __init__(self, stop_at: int | None = None):
        self.events: list[Any] = []
        self._cancel = threading.Event()
        self.stop_at = stop_at
        self.ctx = JobContext("r1", self._emit, self._cancel)

    def _emit(self, event: Any) -> None:
        self.events.append(event)
        if isinstance(event, Progress) and self.stop_at is not None and event.step >= self.stop_at:
            self._cancel.set()

    @property
    def done(self) -> list[ImageDone]:
        return [event for event in self.events if isinstance(event, ImageDone)]


@pytest.fixture
def pipeline(tmp_path: Path) -> wan.Pipeline:
    made = wan.Pipeline.__new__(wan.Pipeline)
    made.folder = tmp_path / "weights"
    arrays = {name: mx.array(value) for name, value in small_autoencoder(seed=9).items()}
    made.small = taehv.TinyAutoencoder(arrays, half=False)
    made.model = Noise()
    made.read = OrderedDict()
    return made


def ask(tmp_path: Path, op: Op, **more: Any) -> GenerateRequest:
    values = {
        "id": "r1",
        "model": WAN.id,
        "op": op,
        "prompt": "a fox in the snow",
        "width": 64,
        "height": 32,
        "frames": 9,
        "fps": 24,
        "seeds": [3],
        "out_dir": str(tmp_path / "out"),
        "basename": "clip",
    }
    return GenerateRequest(**{**values, **more})


def run(pipeline: wan.Pipeline, req: GenerateRequest, recorder: Recorder | None = None) -> Recorder:
    recorder = recorder or Recorder()
    # What the text encoder would have read: nothing has to be loaded for a prompt that is kept.
    pipeline.read.setdefault(req.prompt, np.zeros((1, LENGTH, 8), dtype=np.float16))
    Path(req.out_dir).mkdir(parents=True, exist_ok=True)
    wan.WanVideo(96).generate(pipeline, WAN, req, recorder.ctx)
    return recorder


def frames_of(path: str) -> np.ndarray:
    info = video.probe(path)
    return np.frombuffer(video.pixels(path), dtype=np.uint8).reshape(-1, info.height, info.width, 3)


def blocks(frame: np.ndarray, side: int = 8) -> np.ndarray:
    """A frame as the average of each ``side`` x ``side`` block of it."""
    height, width, _ = frame.shape
    return frame.astype(np.float32).reshape(height // side, side, width // side, side, 3).mean(axis=(1, 3))


def test_a_video_from_words_is_written_with_its_first_frame_and_what_it_was_made_from(
    tmp_path: Path, pipeline: wan.Pipeline
):
    recorder = run(pipeline, ask(tmp_path, Op.TXT2VID, seeds=[3, 4]))
    one, two = recorder.done
    assert (one.index, one.seed, two.index, two.seed) == (0, 3, 1, 4)
    assert (one.width, one.height, one.frames, one.fps, one.sound) == (64, 32, 9, 24.0, False)
    info = video.probe(one.video)
    assert (info.width, info.height, info.frames, info.fps, info.sound) == (64, 32, 9, 24.0, False)
    # Three passes and the writing, for each video.
    steps = [
        (event.index, event.step, event.total) for event in recorder.events if isinstance(event, Progress)
    ]
    assert steps == [(index, step, 4) for index in (0, 1) for step in (1, 2, 3)]
    assert pipeline.model.calls == 6
    with Image.open(one.path) as still:
        assert still.size == (64, 32)
        notes = json.loads(still.text["dreambot"])
        first = np.asarray(still.convert("RGB")).astype(np.int16)
    assert (notes["model"], notes["operation"], notes["prompt"], notes["seed"]) == (
        WAN.id,
        "txt2vid",
        "a fox in the snow",
        3,
    )
    assert (notes["frames"], notes["fps"]) == (9, 24.0)
    # The still is the frame as it was made; in the file it is as the encoder kept it. (These
    # frames are noise, of which an encoder keeps the lie of the land: compared in 8x8 blocks.)
    assert np.abs(blocks(frames_of(one.video)[0]) - blocks(first)).mean() < 6
    assert not (frames_of(one.video) == frames_of(two.video)).all(), "another seed, another video"


def test_a_video_started_on_a_picture_keeps_the_picture_in_view(tmp_path: Path, pipeline: wan.Pipeline):
    picture = tmp_path / "start.png"
    # Twice the size of the video and wider than it: scaled to cover it, and cut about its middle.
    Image.fromarray(scene()[0]).resize((160, 64)).save(picture)
    seen = []
    kept = pipeline.small.picture_latent
    pipeline.small.picture_latent = lambda pixels: seen.append(pixels) or kept(pixels)
    recorder = run(pipeline, ask(tmp_path, Op.IMG2VID, image=str(picture)))
    (shown,) = seen
    assert shown.shape == (32, 64, 3) and shown.dtype == np.float32 and 0 <= shown.min() < shown.max() <= 1
    with Image.open(picture) as source:
        middle = np.asarray(source.convert("RGB").resize((80, 32), Image.Resampling.LANCZOS))[:, 8:72]
    assert np.abs(shown * 255 - middle).mean() < 2
    assert recorder.done[0].frames == 9


def test_low_motion_starts_from_the_picture_held_still_and_high_motion_from_noise(
    tmp_path: Path, pipeline: wan.Pipeline, monkeypatch: pytest.MonkeyPatch
):
    picture = tmp_path / "start.png"
    Image.fromarray(scene()[0]).resize((64, 32)).save(picture)
    asked: list[dict[str, Any]] = []
    sample = sampler.sample

    def watched(model: Any, text: Any, shape: Any, **how: Any) -> np.ndarray:
        asked.append(how)
        return sample(model, text, shape, **how)

    monkeypatch.setattr(sampler, "sample", watched)
    run(pipeline, ask(tmp_path, Op.IMG2VID, image=str(picture), basename="low"))
    run(pipeline, ask(tmp_path, Op.IMG2VID, image=str(picture), extra={"motion": "high"}, basename="high"))
    run(pipeline, ask(tmp_path, Op.TXT2VID, extra={"motion": "low"}, basename="words"))
    low, high, words = asked
    # Low motion, which is what is meant when nothing is said: every latent frame of the clip
    # is the picture, the first as the picture's own and the rest as a frame that stays.
    assert low["noise_share"] == wan.LOW_MOTION_NOISE and low["still"].shape == (1, LATENT, 3, 2, 4)
    assert (low["still"][:, :, :1] == low["start"]).all()
    assert (low["still"][:, :, 1] == low["still"][:, :, 2]).all()
    assert not np.allclose(low["still"][:, :, 1], low["start"][:, :, 0], atol=1e-3)
    assert high["still"] is None and high["noise_share"] == 1.0 and high["start"] is not None
    # With no picture there is nothing to hold still: words are filmed from noise.
    assert words["still"] is None and words["start"] is None
    # A video that is continued at low motion stands on its last frame.
    (first,) = run(pipeline, ask(tmp_path, Op.TXT2VID, basename="first")).done
    run(pipeline, ask(tmp_path, Op.EXTEND, video=first.video, frames=8, basename="longer"))
    longer = asked[-1]
    assert longer["start"].shape[2] == wan.CARRIED and longer["still"].shape[2] == 2 + 8 // 4
    assert (longer["still"][:, :, : wan.CARRIED] == longer["start"]).all()


def test_a_loop_goes_out_and_comes_back_the_way_it_came(tmp_path: Path, pipeline: wan.Pipeline):
    picture = tmp_path / "start.png"
    Image.fromarray(scene()[0]).resize((64, 32)).save(picture)
    recorder = run(pipeline, ask(tmp_path, Op.IMG2VID, image=str(picture), end_image=str(picture), frames=17))
    (clip,) = recorder.done
    # Half is filmed (nine frames), and the frame it turns round on is shown once.
    assert clip.frames == 17
    frames = frames_of(clip.video).astype(np.int16)
    for step in range(8):
        assert np.abs(frames[step] - frames[16 - step]).mean() < 6, step
    assert np.abs(frames[0] - frames[8]).mean() > np.abs(frames[0] - frames[16]).mean()
    assert wan.loops(ask(tmp_path, Op.IMG2VID, image=str(picture), end_image=str(picture)))
    assert not wan.loops(ask(tmp_path, Op.IMG2VID, image=str(picture)))


def test_a_video_is_continued_from_its_last_five_frames_and_keeps_what_it_had(
    tmp_path: Path, pipeline: wan.Pipeline
):
    (first,) = run(pipeline, ask(tmp_path, Op.TXT2VID, basename="first")).done
    held = []
    kept = pipeline.small.latents
    pipeline.small.latents = lambda frames: held.append(frames) or kept(frames)
    recorder = run(pipeline, ask(tmp_path, Op.EXTEND, video=first.video, frames=8, basename="longer"))
    (longer,) = recorder.done
    assert longer.frames == 9 + 8 == video.probe(longer.video).frames
    before, after = frames_of(first.video), frames_of(longer.video)
    # What was there is as it was, to the pixel: it was not encoded again.
    assert (after[:9] == before).all()
    # (Its last five frames, to go on from; and its last frame nine times over, to stand still on.)
    tail, standing = held
    assert tail.shape == (5, 32, 64, 3) and np.abs(tail * 255 - before[-5:]).max() < 1
    assert standing.shape == (9, 32, 64, 3) and (standing == tail[-1]).all()
    assert not (after[9:] == before[-1]).all(), "something new follows"
    with Image.open(longer.path) as still:
        assert np.abs(np.asarray(still.convert("RGB")).astype(np.int16) - before[0]).mean() < 1
        assert json.loads(still.text["dreambot"])["operation"] == "extend"
    # A second time: the parts are put end to end again.
    (longest,) = run(
        pipeline, ask(tmp_path, Op.EXTEND, video=longer.video, frames=4, basename="longest")
    ).done
    assert longest.frames == 21 and (frames_of(longest.video)[:17] == after).all()


def test_what_the_model_cannot_do_is_refused_in_words(tmp_path: Path, pipeline: wan.Pipeline):
    picture = tmp_path / "start.png"
    Image.fromarray(scene()[0]).resize((64, 32)).save(picture)
    other = tmp_path / "end.png"
    Image.fromarray(scene(seed=7)[0]).resize((64, 32)).save(other)
    for req, said in (
        (
            ask(tmp_path, Op.IMG2VID, image=str(picture), end_image=str(other)),
            "cannot end a video on a picture",
        ),
        (ask(tmp_path, Op.TXT2VID, width=72), "steps of 32"),
        (ask(tmp_path, Op.TXT2VID, frames=10), "groups of 4"),
        (ask(tmp_path, Op.TXT2VID, frames=125), "up to 121 frames"),
    ):
        with pytest.raises(BackendError, match=said) as refused:
            run(pipeline, req)
        assert refused.value.kind is ErrorKind.INVALID
    assert not list((tmp_path / "out").glob("*.mp4"))
    (short,) = run(pipeline, ask(tmp_path, Op.TXT2VID, frames=1, basename="short")).done
    with pytest.raises(BackendError, match="too short to continue"):
        run(pipeline, ask(tmp_path, Op.EXTEND, video=short.video, frames=8))
    with pytest.raises(BackendError, match="up to 121 frames"):
        (whole,) = run(pipeline, ask(tmp_path, Op.TXT2VID, basename="whole")).done
        run(pipeline, ask(tmp_path, Op.EXTEND, video=whole.video, frames=120))


def test_a_video_is_stopped_between_passes_and_leaves_nothing_behind(tmp_path: Path, pipeline: wan.Pipeline):
    recorder = Recorder(stop_at=2)
    with pytest.raises(JobCancelled):
        run(pipeline, ask(tmp_path, Op.TXT2VID, seeds=[1, 2]), recorder)
    assert pipeline.model.calls == 2 and not recorder.done
    assert not list((tmp_path / "out").iterdir())


def test_a_model_that_is_not_all_there_says_how_to_get_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    engine = wan.WanVideo(96)
    monkeypatch.setenv("DREAMBOT_ASSETS_DIR", str(tmp_path / "assets"))
    with pytest.raises(BackendError, match=r"dreambot models pull wan-2\.2") as missing:
        engine.load(WAN, ask(tmp_path, Op.TXT2VID, weights=str(tmp_path / "nowhere")))
    assert missing.value.kind is ErrorKind.NOT_INSTALLED
    folder = tmp_path / "weights"
    folder.mkdir()
    # The transformer's files are there and the small autoencoder is not.
    with pytest.raises(BackendError, match=r"dreambot models pull wan-2\.2"):
        engine.load(WAN, ask(tmp_path, Op.TXT2VID, weights=str(folder)))
    with pytest.raises(BackendError, match="LoRAs") as refused:
        engine.load(WAN, ask(tmp_path, Op.TXT2VID, weights=str(folder), loras=[{"path": "x"}]))
    assert refused.value.kind is ErrorKind.INVALID
    (asset,) = WAN.assets
    (tmp_path / "assets").mkdir()
    mx.save_safetensors(
        str(tmp_path / "assets" / asset.name),
        {name: mx.array(value) for name, value in small_autoencoder().items()},
    )
    with pytest.raises(BackendError, match="not a FastMetal checkpoint") as broken:
        engine.load(WAN, ask(tmp_path, Op.TXT2VID, weights=str(folder)))
    assert broken.value.kind is ErrorKind.NOT_INSTALLED


def test_a_prompt_is_read_once_and_a_few_readings_are_kept(
    tmp_path: Path, pipeline: wan.Pipeline, monkeypatch
):
    loaded = []

    class Reader:
        @classmethod
        def load(cls, folder: Path) -> Reader:
            loaded.append(folder)
            return cls()

        def encode(self, prompts: list[str]) -> np.ndarray:
            return np.full((len(prompts), LENGTH, 8), len(prompts[0]), dtype=np.float32)

    monkeypatch.setattr("dreambot.engine.wan.umt5.TextEncoder", Reader)
    engine = wan.WanVideo(96)
    first = engine._text(pipeline, "a fox")
    assert first.dtype == mx.float16 and first.shape == (1, LENGTH, 8) and float(first[0, 0, 0]) == 5
    engine._text(pipeline, "a fox")
    assert loaded == [pipeline.folder], "the text encoder is loaded for a new prompt only"
    for number in range(wan.KEPT_PROMPTS):
        engine._text(pipeline, f"prompt {number}")
    assert len(pipeline.read) == wan.KEPT_PROMPTS and "a fox" not in pipeline.read
    # With little memory, the transformer makes way while a prompt is read.
    assert pipeline.model is not None
    wan.WanVideo(16)._text(pipeline, "another")
    assert pipeline.model is None


# --------------------------------------------------------------------------------------
# The real weights, against FastVideo's runtime
# --------------------------------------------------------------------------------------


def installed() -> weights.Weights:
    found = weights.resolve(WAN)
    if found is None:
        pytest.skip(f"{WAN.id} is not installed")
    return found


@pytest.mark.mlx
def test_the_real_transformer_gives_the_reference_runtimes_numbers():
    reference = np.load(REFERENCE)
    model = Transformer.load(installed().path)
    latents = mx.array(reference["latents"])
    text = mx.zeros((1, LENGTH, 4096), dtype=mx.float16)
    text[0, : len(reference["ids"])] = mx.array(reference["text"])
    tables = (mx.array(reference["positions_cos"]), mx.array(reference["positions_sin"]))
    level = float(reference["level"])
    for name, clean in (("predicted", 0), ("predicted_started", 24)):
        timestep = np.full((1, 120), level, dtype=np.float32)
        timestep[:, :clean] = 0.0
        got = np.array(model(latents, text, mx.array(timestep), *tables).astype(mx.float32))
        want = reference[name].astype(np.float32)
        # On the Mac these numbers were made on they are the same, digit for digit. Thirty
        # blocks of 16-bit arithmetic carry a last digit a long way, so on another chip it is
        # the picture that has to be the same and not the digits.
        print(f"\n[wan] {name}: largest difference {np.abs(got - want).max():.6f}")
        assert np.corrcoef(got.ravel(), want.ravel())[0, 1] > 0.995, name
        assert np.abs(got - want).mean() < 0.1, name


@pytest.mark.mlx
def test_the_real_text_encoder_reads_a_prompt_as_the_reference_runtime_does():
    reference = np.load(REFERENCE)
    encoder = TextEncoder.load(installed().path)
    prompt = str(reference["prompt"])
    ids = reference["ids"].tolist()
    assert encoder.tokens(prompt) == ids
    read = encoder.encode([prompt])[0]
    want = reference["text"].astype(np.float32)
    assert (read[len(ids) :] == 0).all()
    difference = np.abs(read[: len(ids)] - want)
    print(f"\n[wan] text: largest difference {difference.max():.5f}, on average {difference.mean():.6f}")
    # The reference numbers were read with umT5 as 16-bit floats (the copy in FastMetal's
    # smaller model's repo). The copy that comes with this model is the same encoder rounded
    # to bfloat16, which has three bits fewer to a number: its reading is this far from that
    # one however it is run (0.020 at most and 0.0006 on average, on numbers 0.07 in size).
    # With the 16-bit copy this code gives the reference's numbers to the fifth decimal.
    assert difference.max() < 0.05 and difference.mean() < 2e-3
    assert np.corrcoef(read[: len(ids)].ravel(), want.ravel())[0, 1] > 0.999


@pytest.mark.mlx
def test_the_real_small_autoencoder_decodes_and_encodes_as_the_reference_does():
    reference = np.load(REFERENCE)
    installed()
    (asset,) = WAN.assets
    path = weights.asset_path(asset)
    assert path is not None
    small = taehv.TinyAutoencoder.load(path, half=False)
    frames = small.frames(reference["small_latents"].transpose(1, 0, 2, 3), chunk=2)
    assert frames.shape == (13, 96, 160, 3)
    for number, want in zip(
        reference["small_frame_numbers"].tolist(), reference["small_frames"], strict=True
    ):
        assert np.abs(frames[number].astype(np.int16) - want).max() <= 1, number
    latent = small.picture_latent(reference["picture"].astype(np.float32) / 255.0)
    assert np.abs(latent[0, :, 0] - reference["picture_latent"]).max() < 1e-3
    # In half precision, which is how it is run, it is within a level of that.
    half = taehv.TinyAutoencoder.load(path).frames(reference["small_latents"].transpose(1, 0, 2, 3))
    assert np.abs(half.astype(np.int16) - frames).max() <= 2
