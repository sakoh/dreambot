"""The Wan 2.2 TI2V-5B transformer, read from a FastMetal checkpoint.

A FastMetal checkpoint is two files: ``mlx_dit.json`` (the model's shape, and which matrices
are quantized how) and ``mlx_dit.safetensors`` (the weights, with every matrix already packed
to 8 bits the way ``mlx.core.quantize`` packs them). Nothing is converted on loading: a packed
matrix is multiplied as it is (``mx.quantized_matmul``), which is what it was trained for.

The model is thirty blocks of self-attention over the video's tokens, attention to the
prompt, and a feed-forward layer, each modulated by the noise level. What sets Wan 2.2's 5B
model apart from the rest of the family is that the noise level is told *per token*: that is
how the frame a video starts on is kept clean while the other frames are noise.

The arithmetic, including where it steps up to 32-bit floats, follows FastVideo's MLX runtime
(``fastvideo/mlx_runtime/wan22.py``, Apache 2.0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MANIFEST = "mlx_dit.json"
WEIGHTS = "mlx_dit.safetensors"
FORMAT_VERSION = 1
#: How many times over a noise level is embedded. Once would do, but MLX multiplies a packed
#: matrix by a few rows with a different kernel than by many, and the two differ in the last
#: digit of a 16-bit float. FastVideo embeds the level once for every token of the clip; with
#: this many rows the same kernel is used and the numbers are the same as its, digit for digit.
LEVEL_ROWS = 128


class WanError(Exception):
    """The video model's files are not what this code reads."""


class Linear:
    """A weight matrix and its bias. Packed matrices are multiplied without unpacking."""

    def __init__(self, arrays: dict[str, Any], name: str, packed: dict[str, Any] | None):
        self.weight = arrays[f"{name}.weight"]
        self.bias = arrays.get(f"{name}.bias")
        self.packed = packed
        self.scales = arrays[f"{name}.weight.scales"] if packed else None
        self.biases = arrays.get(f"{name}.weight.biases") if packed else None

    def __call__(self, x: Any) -> Any:
        import mlx.core as mx

        if self.packed is None:
            y = x @ self.weight.T
        else:
            y = mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=self.packed["group_size"],
                bits=self.packed["bits"],
                mode=self.packed["mode"],
            ).astype(x.dtype)
        return y if self.bias is None else y + self.bias


def _layer_norm(x: Any, weight: Any = None, bias: Any = None, eps: float = 1e-6) -> Any:
    import mlx.core as mx

    x = x.astype(mx.float32)
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * mx.rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def _rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    import mlx.core as mx

    wide = x.astype(mx.float32)
    y = wide * mx.rsqrt(mx.mean(mx.square(wide), axis=-1, keepdims=True) + eps)
    return y.astype(x.dtype) * weight


def _rotate(x: Any, cos: Any, sin: Any) -> Any:
    """Turn each pair of a head's dimensions by its token's position (rotary embedding)."""
    import mlx.core as mx

    wide = x.astype(mx.float32)
    pairs = wide.reshape(*x.shape[:-1], -1, 2)
    turned = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(*x.shape)
    return (wide * cos[None, :, None, :] + turned * sin[None, :, None, :]).astype(x.dtype)


def _silu(x: Any) -> Any:
    import mlx.core as mx

    return x * mx.sigmoid(x)


def _sinusoid(levels: Any, dim: int) -> Any:
    import mlx.core as mx

    half = dim // 2
    freqs = mx.exp(-math.log(10000) * mx.arange(0, half, dtype=mx.float32) / half)
    args = levels[:, None].astype(mx.float32) * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class Block:
    """One of the thirty: self-attention, attention to the prompt, feed-forward."""

    def __init__(self, arrays: dict[str, Any], prefix: str, packed: dict[str, Any], heads: int, eps: float):
        def linear(name: str) -> Linear:
            return Linear(arrays, f"{prefix}{name}", packed.get(f"{prefix}{name}.weight"))

        self.heads, self.eps = heads, eps
        self.to_q, self.to_k, self.to_v, self.to_out = (linear(n) for n in ("to_q", "to_k", "to_v", "to_out"))
        self.norm_q, self.norm_k = arrays[f"{prefix}norm_q.weight"], arrays[f"{prefix}norm_k.weight"]
        self.cross_q, self.cross_k, self.cross_v, self.cross_out = (
            linear(f"attn2.{n}") for n in ("to_q", "to_k", "to_v", "to_out")
        )
        self.cross_norm_q = arrays[f"{prefix}attn2.norm_q.weight"]
        self.cross_norm_k = arrays[f"{prefix}attn2.norm_k.weight"]
        self.mid_weight = arrays[f"{prefix}self_attn_residual_norm.norm.weight"]
        self.mid_bias = arrays[f"{prefix}self_attn_residual_norm.norm.bias"]
        self.fc_in, self.fc_out = linear("ffn.fc_in"), linear("ffn.fc_out")
        self.table = arrays[f"{prefix}scale_shift_table"]

    def _attend(self, q: Any, k: Any, v: Any, batch: int, dim: int) -> Any:
        import mlx.core as mx

        head_dim = dim // self.heads
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), scale=head_dim**-0.5
        )
        return out.transpose(0, 2, 1, 3).reshape(batch, -1, dim)

    def __call__(self, x: Any, text: Any, modulation: Any, cos: Any, sin: Any) -> Any:
        """``modulation`` is ``[B, L or 1, 6, dim]``: what the noise level asks of each token."""
        import mlx.core as mx
        import mlx.nn as nn

        kind = x.dtype
        batch, _, dim = x.shape
        head_dim = dim // self.heads
        e = self.table[None] + modulation.astype(mx.float32)
        shift, scale, gate, c_shift, c_scale, c_gate = (part.squeeze(2) for part in mx.split(e, 6, axis=2))

        # 1. The video's tokens attend to each other.
        h = (_layer_norm(x, eps=self.eps) * (1.0 + scale) + shift).astype(kind)
        q = _rms_norm(self.to_q(h), self.norm_q, self.eps).reshape(batch, -1, self.heads, head_dim)
        k = _rms_norm(self.to_k(h), self.norm_k, self.eps).reshape(batch, -1, self.heads, head_dim)
        v = self.to_v(h).reshape(batch, -1, self.heads, head_dim)
        attended = self.to_out(self._attend(_rotate(q, cos, sin), _rotate(k, cos, sin), v, batch, dim))
        x = x + (attended * gate).astype(kind)

        # 2. They attend to the prompt.
        h = _layer_norm(x, self.mid_weight, self.mid_bias, self.eps).astype(kind)
        q = _rms_norm(self.cross_q(h), self.cross_norm_q, self.eps).reshape(batch, -1, self.heads, head_dim)
        k = _rms_norm(self.cross_k(text), self.cross_norm_k, self.eps).reshape(
            batch, -1, self.heads, head_dim
        )
        v = self.cross_v(text).reshape(batch, -1, self.heads, head_dim)
        x = x + self.cross_out(self._attend(q, k, v, batch, dim))

        # 3. Feed-forward.
        h = (_layer_norm(x, eps=self.eps) * (1.0 + c_scale) + c_shift).astype(kind)
        h = self.fc_out(nn.gelu_approx(self.fc_in(h)))
        return (x + (h * c_gate).astype(kind)).astype(kind)


class Transformer:
    """The whole model. Call it with noisy latents and get the predicted noise direction."""

    def __init__(self, arrays: dict[str, Any], manifest: dict[str, Any], *, compiled: bool = True):
        config = manifest["config"]
        packed = {key: manifest["quantization"] for key in manifest.get("quantized_keys", {})}
        self.config = config
        #: Every array of the checkpoint, by name.
        self.arrays = arrays
        self.heads = int(config["num_attention_heads"])
        self.head_dim = int(config["attention_head_dim"])
        self.dim = self.heads * self.head_dim
        self.channels = int(config["in_channels"])
        self.out_channels = int(config["out_channels"])
        self.patch_size = tuple(config["patch_size"])
        self.freq_dim = int(config["freq_dim"])
        self.eps = float(config.get("eps", 1e-6))

        def linear(name: str) -> Linear:
            return Linear(arrays, name, packed.get(f"{name}.weight"))

        self.patch = linear("patch_embedding")
        self.time_1 = linear("condition_embedder.time_embedder.linear_1")
        self.time_2 = linear("condition_embedder.time_embedder.linear_2")
        self.time_proj = linear("condition_embedder.time_proj")
        self.text_1 = linear("condition_embedder.text_embedder.linear_1")
        self.text_2 = linear("condition_embedder.text_embedder.linear_2")
        self.proj_out = linear("proj_out")
        self.table = arrays["scale_shift_table"]
        self.blocks = [
            Block(arrays, f"blocks.{index}.", packed, self.heads, self.eps)
            for index in range(int(manifest["num_blocks"]))
        ]
        self._compile = compiled
        self._traced: Any = None
        self._traced_for: tuple | None = None

    @classmethod
    def load(cls, folder: str | Path, *, compiled: bool = True) -> Transformer:
        import mlx.core as mx

        folder = Path(folder)
        manifest_path, weights_path = folder / MANIFEST, folder / WEIGHTS
        if not manifest_path.is_file() or not weights_path.is_file():
            raise WanError(f"{folder} has no {MANIFEST} and {WEIGHTS}: it is not a FastMetal checkpoint.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version") != FORMAT_VERSION:
            raise WanError(
                f"{manifest_path} is format {manifest.get('format_version')}; this reads format {FORMAT_VERSION}."
            )
        quantization = manifest.get("quantization")
        if quantization is not None and quantization.get("mode") != "affine":
            raise WanError(
                f"{manifest_path} is quantized as {quantization.get('mode')}; this reads affine only."
            )
        return cls(mx.load(str(weights_path)), manifest, compiled=compiled)

    # --------------------------------------------------------------------- forward ----

    def _forward(self, latents: Any, text: Any, level: Any, cos: Any, sin: Any, *, clean: int) -> Any:
        """``clean`` leading tokens are told they carry no noise; the rest are at ``level``."""
        import mlx.core as mx
        import mlx.nn as nn

        batch, channels, frames, height, width = latents.shape
        pt, ph, pw = self.patch_size
        x = latents.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
        x = self.patch(x.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, -1, channels * pt * ph * pw))
        tokens = x.shape[1]

        # What the noise level asks of a token: worked out once for each level there is (one, or
        # two when some tokens are clean), then laid out along the tokens.
        levels = mx.broadcast_to(level, (LEVEL_ROWS,))
        if clean:
            levels = mx.concatenate([mx.zeros((LEVEL_ROWS,), dtype=mx.float32), levels])
        timed = self.time_2(_silu(self.time_1(_sinusoid(levels, self.freq_dim).astype(mx.float16))))
        asked = self.time_proj(_silu(timed)).reshape(-1, 6, self.dim)[::LEVEL_ROWS]
        timed = timed[::LEVEL_ROWS]
        if clean:

            def along(values: Any) -> Any:
                """``values`` for (clean, noisy), laid out along the clip's tokens."""
                spans = ((0, clean), (1, tokens - clean))
                rows = [
                    mx.broadcast_to(values[which : which + 1], (count, *values.shape[1:]))
                    for which, count in spans
                ]
                return mx.concatenate(rows)[None]

            timed, asked = along(timed), along(asked)
        else:
            timed, asked = timed[:, None, :], asked[:, None, :, :]

        text = self.text_2(nn.gelu_approx(self.text_1(text)))
        for block in self.blocks:
            x = block(x, text, asked, cos, sin)

        e = self.table[None] + timed[:, :, None, :].astype(mx.float32)
        shift, scale = (part.squeeze(2) for part in mx.split(e, 2, axis=2))
        x = (_layer_norm(x, eps=self.eps) * (1.0 + scale) + shift).astype(mx.float16)
        x = self.proj_out(x).reshape(
            batch, frames // pt, height // ph, width // pw, pt, ph, pw, self.out_channels
        )
        return x.transpose(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch, self.out_channels, frames, height, width)

    def __call__(self, latents: Any, text: Any, timestep: Any, cos: Any, sin: Any) -> Any:
        """``timestep`` is ``[1, tokens]``: zero for the tokens held clean, the noise level for the rest."""
        import mlx.core as mx
        import numpy as np

        told = np.array(timestep).reshape(-1)
        noisy = np.flatnonzero(told)
        if not noisy.size:
            raise WanError("Every token is held clean, which leaves nothing to make.")
        # Clean tokens are the clip's first ones: a video can be held to how it starts. Held to
        # a picture at its end as well, this model does not lead up to it. It cuts to it.
        clean = int(noisy[0])
        if noisy.size != told.size - clean or (told[noisy] != told[noisy[0]]).any():
            raise WanError(
                "Only tokens at the start of a clip can be held clean, and the rest share one level."
            )
        level = mx.array([float(told[noisy[0]])], dtype=mx.float32)
        if not self._compile:
            return self._forward(latents, text, level, cos, sin, clean=clean)
        # One traced graph is kept. Another shape retires it first: each graph holds on to its
        # own working copy of the weights.
        shape = (tuple(latents.shape), tuple(text.shape), clean)
        if self._traced is not None and self._traced_for != shape:
            self._traced, self._traced_for = None, None
            mx.clear_cache()
        if self._traced is None:

            def run(latents: Any, text: Any, level: Any, cos: Any, sin: Any) -> Any:
                return self._forward(latents, text, level, cos, sin, clean=clean)

            self._traced, self._traced_for = mx.compile(run), shape
        return self._traced(latents, text, level, cos, sin)
