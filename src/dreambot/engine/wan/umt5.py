"""umT5-XXL, the encoder that reads Wan's prompts.

Google's multilingual T5 (Apache 2.0): twenty-four layers of self-attention over the prompt's
tokens, 5.7 billion parameters, 11 GB as 16-bit floats. It is the largest file of the video
model and it is needed for a second or two per prompt, so it is loaded when a prompt has to be
read and let go again; the video model itself is half its size.

Read from the files ``transformers`` saves (a ``config.json`` and safetensors shards), without
``transformers`` or PyTorch. Differences from the T5 that mflux has for FLUX: every layer has
a relative-position table of its own, and the feed-forward layer is gated.

The weights stay 16-bit and what flows between the layers is kept in 32-bit floats: T5's
activations grow past what a 16-bit float holds, which is why its PyTorch implementation
clamps them in half precision. (What comes out agrees with the reference runtime's 16-bit
reading to the third decimal, on numbers a tenth in size. The implementation itself is
checked exactly, against the umT5 in ``transformers``, on a small model: ``tests/test_wan.py``.)
Only a prompt's own tokens are run: the padding up to 512 that Wan was trained with is
attended to by nothing, so leaving it out changes nothing but the time it takes, and the rows
for it are zeros either way.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: The length Wan pads (and cuts) every prompt to.
LENGTH = 512


class TextEncoder:
    def __init__(self, arrays: dict[str, Any], config: dict[str, Any], tokenizer: Any):
        self.w = arrays
        self.tokenizer = tokenizer
        self.layers = int(config["num_layers"])
        self.heads = int(config["num_heads"])
        self.dim = int(config["d_model"])
        self.buckets = int(config["relative_attention_num_buckets"])
        self.reach = int(config["relative_attention_max_distance"])
        self.eps = float(config.get("layer_norm_epsilon", 1e-6))

    @classmethod
    def load(cls, folder: str | Path) -> TextEncoder:
        """``folder`` holds ``text_encoder/`` and ``tokenizer/`` as the model's repository does."""
        import mlx.core as mx
        from tokenizers import Tokenizer

        folder = Path(folder)
        encoder = folder / "text_encoder"
        config = json.loads((encoder / "config.json").read_text())
        arrays: dict[str, Any] = {}
        for shard in sorted(encoder.glob("*.safetensors")):
            arrays.update(mx.load(str(shard)))
        if "shared.weight" not in arrays:
            raise FileNotFoundError(f"{encoder} holds no umT5 weights.")
        return cls(arrays, config, Tokenizer.from_file(str(folder / "tokenizer" / "tokenizer.json")))

    # -------------------------------------------------------------------- pieces ----

    def tokens(self, prompt: str) -> list[int]:
        """A prompt's tokens, ending in the end-of-sequence token, cut to :data:`LENGTH`."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=True).ids
        return ids[: LENGTH - 1] + ids[-1:] if len(ids) > LENGTH else ids

    def _norm(self, x: Any, name: str) -> Any:
        import mlx.core as mx

        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps) * self.w[name]

    def _bias(self, count: int, layer: int) -> Any:
        """How much each head favours a token at each distance: ``[1, heads, count, count]``."""
        import mlx.core as mx

        distance = np.arange(count)[None, :] - np.arange(count)[:, None]
        half = self.buckets // 2
        bucket = (distance > 0).astype(np.int64) * half
        away = np.abs(distance)
        exact = half // 2
        far = exact + (
            np.log(np.maximum(away, 1) / exact) / math.log(self.reach / exact) * (half - exact)
        ).astype(np.int64)
        bucket += np.where(away < exact, away, np.minimum(far, half - 1))
        table = self.w[f"encoder.block.{layer}.layer.0.SelfAttention.relative_attention_bias.weight"]
        return table[mx.array(bucket)].transpose(2, 0, 1)[None].astype(mx.float32)

    def _run(self, ids: list[int]) -> Any:
        import mlx.core as mx
        import mlx.nn as nn

        count = len(ids)
        x = self.w["shared.weight"][mx.array(ids)][None].astype(mx.float32)
        for layer in range(self.layers):
            prefix = f"encoder.block.{layer}.layer"
            attention = f"{prefix}.0.SelfAttention"
            h = self._norm(x, f"{prefix}.0.layer_norm.weight")
            q, k, v = (
                (h @ self.w[f"{attention}.{part}.weight"].T)
                .reshape(1, count, self.heads, -1)
                .transpose(0, 2, 1, 3)
                for part in ("q", "k", "v")
            )
            # T5 does not scale its attention scores.
            scores = q @ k.transpose(0, 1, 3, 2) + self._bias(count, layer)
            h = (mx.softmax(scores, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(1, count, -1)
            x = x + h @ self.w[f"{attention}.o.weight"].T
            dense = f"{prefix}.1.DenseReluDense"
            h = self._norm(x, f"{prefix}.1.layer_norm.weight")
            h = nn.gelu_approx(h @ self.w[f"{dense}.wi_0.weight"].T) * (h @ self.w[f"{dense}.wi_1.weight"].T)
            x = x + h @ self.w[f"{dense}.wo.weight"].T
            mx.eval(x)
        return self._norm(x, "encoder.final_layer_norm.weight")[0]

    def encode(self, prompts: Sequence[str]) -> np.ndarray:
        """What the video model is given for each prompt: ``[prompts, 512, 4096]``, zeros after its tokens."""
        import mlx.core as mx

        out = np.zeros((len(prompts), LENGTH, self.dim), dtype=np.float32)
        for row, prompt in enumerate(prompts):
            ids = self.tokens(prompt)
            hidden = self._run(ids)
            mx.eval(hidden)
            out[row, : len(ids)] = np.array(hidden)
        return out
