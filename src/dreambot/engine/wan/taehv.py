"""TAEHV: the small autoencoder that turns Wan 2.2's latents into frames, and a picture into a latent.

Wan 2.2 comes with a 2.8 GB autoencoder of its own. Ollin Boer Bohan's TAEHV (MIT) does the
same job for the same latents in 22 MB of plain convolutions, in a fraction of the time. Its
frames are a little softer than the full autoencoder's; FastMetal was made to be decoded by it.

Two things differ here from how it is usually run:

* **Frames are made a few at a time.** Decoding a whole clip at once holds every frame at full
  size in 64 channels: 75 GB for five seconds of 480p, which is what took the worker's memory
  when it was tried that way. Each of TAEHV's memory blocks looks one frame back, so a clip
  can be decoded in pieces by carrying each block's last frame over to the next piece. The
  frames come out the same, digit for digit, in a few GB.
* **It also encodes the picture a video starts on.** Its decoder gives four frames for every
  latent frame and drops the first three, so a clip's first latent stands for its first frame
  alone; encoding four copies of a picture gives that latent. Started on it, the video model
  did as well as when started on the full autoencoder's (tried on real weights).

The weights are published as a PyTorch checkpoint. That is a zip of raw arrays and a pickle
that says how to put them together; :func:`read_checkpoint` reads it without PyTorch and
without running anything the pickle names, other than building arrays.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

#: Latent frames decoded in one go. Two keeps five seconds of 480p under 5 GB.
CHUNK = 2
#: Frames the decoder makes before a clip's first: each latent frame becomes four, and a clip
#: has one frame for its first latent.
LEAD = 3


class CheckpointError(Exception):
    """A weights file is not what it should be."""


# ------------------------------------------------------------------ reading the file ----

_STORAGES = {"HalfStorage": np.float16, "FloatStorage": np.float32, "DoubleStorage": np.float64}


class _Storage:
    def __init__(self, dtype: Any):
        self.dtype = dtype


def _rebuild(
    storage: np.ndarray, offset: int, size: tuple[int, ...], stride: tuple[int, ...], *_: Any
) -> np.ndarray:
    count = int(np.prod(size)) if size else 1
    flat = storage[offset : offset + count]
    expected = tuple(int(np.prod(size[i + 1 :])) for i in range(len(size)))
    if tuple(stride) != expected:
        raise CheckpointError("The checkpoint stores an array in an order this reader does not handle.")
    return flat.reshape(size)


class _Reader(pickle.Unpickler):
    """Unpickles a PyTorch state dict and nothing else: any other name in the pickle is refused."""

    def __init__(self, data: bytes, archive: zipfile.ZipFile, root: str):
        import io

        super().__init__(io.BytesIO(data))
        self.archive, self.root = archive, root

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("collections", "OrderedDict"):
            import collections

            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild
        if module == "torch" and name in _STORAGES:
            return _Storage(_STORAGES[name])
        raise CheckpointError(
            f"The checkpoint asks for {module}.{name}, which a file of weights has no use for."
        )

    def persistent_load(self, pid: Any) -> np.ndarray:
        kind, storage, key, _location, _count = pid
        if kind != "storage" or not isinstance(storage, _Storage):
            raise CheckpointError("The checkpoint refers to something that is not an array.")
        return np.frombuffer(self.archive.read(f"{self.root}/data/{key}"), dtype=storage.dtype)


def read_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """The arrays of a PyTorch ``.pth`` state dict, by name."""
    try:
        with zipfile.ZipFile(path) as archive:
            pickled = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            if len(pickled) != 1:
                raise CheckpointError(f"{path} is not a PyTorch checkpoint.")
            root = pickled[0].rsplit("/", 1)[0]
            state = _Reader(archive.read(pickled[0]), archive, root).load()
    except (zipfile.BadZipFile, pickle.UnpicklingError, KeyError, ValueError) as exc:
        raise CheckpointError(f"{path} could not be read as a PyTorch checkpoint: {exc}") from exc
    return {name: np.array(value) for name, value in state.items()}


def convert(checkpoint: str | Path, target: str | Path) -> Path:
    """Keep a checkpoint's arrays as safetensors. (Where it is fetched, it is checked first.)"""
    import mlx.core as mx

    arrays = {name: mx.array(value) for name, value in read_checkpoint(checkpoint).items()}
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part.safetensors")
    mx.save_safetensors(str(partial), arrays)
    partial.replace(target)
    return target


# ------------------------------------------------------------------- the autoencoder ----


class TinyAutoencoder:
    """TAEHV for Wan 2.2's 48-channel latents. Pictures are ``[frames, height, width, 3]``."""

    #: Pixels per latent along a side, and frames per latent frame.
    SPACE, TIME = 16, 4
    PATCH = 2

    def __init__(self, arrays: dict[str, Any], *, half: bool = True):
        import mlx.core as mx

        self.kind = mx.float16 if half else mx.float32
        # Convolution weights the way MLX wants them: [out, height, width, in].
        self.w = {
            name: (value.transpose(0, 2, 3, 1) if value.ndim == 4 else value).astype(self.kind)
            for name, value in arrays.items()
        }
        self.channels = int(self.w["decoder.1.weight"].shape[-1])
        self._kept: dict[int, Any] = {}

    @classmethod
    def load(cls, path: str | Path, *, half: bool = True) -> TinyAutoencoder:
        import mlx.core as mx

        if not Path(path).is_file():
            raise CheckpointError(f"The small autoencoder's weights are not at {path}.")
        return cls(mx.load(str(path)), half=half)

    # ----------------------------------------------------------------- pieces ----

    def _conv(self, x: Any, name: str, *, stride: int = 1) -> Any:
        import mlx.core as mx

        weight = self.w[f"{name}.weight"]
        y = mx.conv2d(x, weight, stride=stride, padding=weight.shape[1] // 2)
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    def _memory(self, x: Any, base: str, key: int | None) -> Any:
        """A memory block: each frame is worked on together with the frame before it."""
        import mlx.core as mx

        before = self._kept.get(key) if key is not None else None
        first = mx.zeros_like(x[:1]) if before is None else before
        if key is not None:
            self._kept[key] = x[-1:]
        h = mx.concatenate([x, mx.concatenate([first, x[:-1]], axis=0)], axis=-1)
        h = mx.maximum(self._conv(h, f"{base}.conv.0"), 0.0)
        h = mx.maximum(self._conv(h, f"{base}.conv.2"), 0.0)
        h = self._conv(h, f"{base}.conv.4")
        skip = self._conv(x, f"{base}.skip") if f"{base}.skip.weight" in self.w else x
        return mx.maximum(h + skip, 0.0)

    @staticmethod
    def _twice(x: Any) -> Any:
        import mlx.core as mx

        n, h, w, c = x.shape
        return mx.broadcast_to(x.reshape(n, h, 1, w, 1, c), (n, h, 2, w, 2, c)).reshape(n, h * 2, w * 2, c)

    def _grow(self, x: Any, name: str, stride: int) -> Any:
        """One frame's features into ``stride`` frames'."""
        import mlx.core as mx

        weight = self.w[f"{name}.conv.weight"]
        wanted = int(x.shape[-1]) * stride
        if int(weight.shape[0]) > wanted:
            weight = weight[-wanted:]
        y = mx.conv2d(x, weight)
        if stride == 1:
            return y
        n, h, w, c = y.shape
        return (
            y.reshape(n, h, w, stride, c // stride)
            .transpose(0, 3, 1, 2, 4)
            .reshape(n * stride, h, w, c // stride)
        )

    # ----------------------------------------------------------------- decoding ----

    def _decode_piece(self, x: Any) -> Any:
        import mlx.core as mx

        x = mx.tanh(x / 3.0) * 3.0
        x = mx.maximum(self._conv(x, "decoder.1"), 0.0)
        for index in (3, 4, 5):
            x = self._memory(x, f"decoder.{index}", index)
        x = self._conv(self._grow(self._twice(x), "decoder.7", 1), "decoder.8")
        for index in (9, 10, 11):
            x = self._memory(x, f"decoder.{index}", index)
        x = self._conv(self._grow(self._twice(x), "decoder.13", 2), "decoder.14")
        for index in (15, 16, 17):
            x = self._memory(x, f"decoder.{index}", index)
        x = mx.maximum(self._conv(self._grow(self._twice(x), "decoder.19", 2), "decoder.20"), 0.0)
        x = self._conv(x, "decoder.22")
        n, h, w, _ = x.shape
        p = self.PATCH
        x = x.reshape(n, h, w, 3, p, p).transpose(0, 1, 4, 2, 5, 3).reshape(n, h * p, w * p, 3)
        return mx.clip(x, 0.0, 1.0)

    def frames(self, latents: np.ndarray, *, chunk: int = CHUNK) -> np.ndarray:
        """A clip's latents (``[channels, frames, height, width]``) as frames of 8-bit pixels.

        ``chunk`` latent frames are decoded at a time. A clip of ``n`` latent frames has
        ``4 * (n - 1) + 1`` frames.
        """
        import mlx.core as mx

        self._kept = {}
        made: list[np.ndarray] = []
        try:
            for start in range(0, latents.shape[1], chunk):
                piece = mx.array(latents[:, start : start + chunk].transpose(1, 2, 3, 0)).astype(self.kind)
                out = self._decode_piece(piece)
                pixels = (out.astype(mx.float32) * 255.0 + 0.5).astype(mx.uint8)
                mx.eval(pixels, *self._kept.values())
                made.append(np.array(pixels)[LEAD:] if start == 0 else np.array(pixels))
                del piece, out, pixels
                mx.clear_cache()
        finally:
            self._kept = {}
        return np.concatenate(made, axis=0)

    # ----------------------------------------------------------------- encoding ----

    def picture_latent(self, picture: np.ndarray) -> np.ndarray:
        """A picture (``[height, width, 3]``, 0..1) as the first latent frame of a clip.

        Returns ``[1, channels, 1, height / 16, width / 16]``, as the video model takes it.
        """
        return self.latents(picture[None])

    def latents(self, frames: np.ndarray) -> np.ndarray:
        """Frames (``[n, height, width, 3]``, 0..1) as the latent frames of a clip's start.

        ``n`` is one, or one and whole groups of four: the first frame makes a latent frame
        of its own and every four after it make one more. Returns
        ``[1, channels, 1 + (n - 1) / 4, height / 16, width / 16]``.
        """
        import mlx.core as mx

        count, height, width, _ = frames.shape
        if height % self.SPACE or width % self.SPACE:
            raise ValueError(
                f"A picture to start on has sides in steps of {self.SPACE}, not {width}x{height}."
            )
        if (count - 1) % self.TIME:
            raise ValueError(
                f"A clip is encoded from one frame and groups of {self.TIME}, not from {count} frames."
            )
        p = self.PATCH
        # Frames go in four at a time from the start, and the last one is repeated to fill its
        # group: latent frame j is made from frames 4j to 4j+3 and what is remembered of the
        # four before, and stands for frames 4j-3 to 4j, which is how the decoder reads it
        # back. (With the first frame repeated in front instead, everything comes back three
        # frames late: tried, on a clip with movement in it.)
        padded = np.concatenate([frames, np.broadcast_to(frames[-1:], (LEAD, height, width, 3))], axis=0)
        total = count + LEAD
        x = mx.array(padded).astype(self.kind)
        # Pixels in 2x2 squares become channels.
        x = x.reshape(total, height // p, p, width // p, p, 3).transpose(0, 1, 3, 5, 2, 4)
        x = x.reshape(total, height // p, width // p, 3 * p * p)
        x = mx.maximum(self._conv(x, "encoder.0"), 0.0)
        for pool, down, blocks, stride in (
            (2, 3, (4, 5, 6), 2),
            (7, 8, (9, 10, 11), 2),
            (12, 13, (14, 15, 16), 1),
        ):
            # Frames are pooled in time (in pairs, twice), then halved in size.
            n, h, w, c = x.shape
            x = (
                x.reshape(n // stride, stride, h, w, c)
                .transpose(0, 2, 3, 1, 4)
                .reshape(n // stride, h, w, stride * c)
            )
            x = mx.conv2d(x, self.w[f"encoder.{pool}.conv.weight"])
            x = self._conv(x, f"encoder.{down}", stride=2)
            for index in blocks:
                x = self._memory(x, f"encoder.{index}", None)
        x = self._conv(x, "encoder.17")
        return np.array(x.astype(mx.float32)).transpose(3, 0, 1, 2)[None]
