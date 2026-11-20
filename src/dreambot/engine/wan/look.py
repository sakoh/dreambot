"""Keeping a video in the colours of the picture it starts on.

The three-step build has a look of its own: more colour than most pictures have, more
contrast, and whites that lean to yellow. A video that starts on a picture keeps the picture
for its first frame and then drifts to that look over its first second or two: two white
bears were lemon yellow by the end of five seconds.

So each frame after the given ones is brought back to them. How much colour a frame has and
which way it leans (the mean and the spread of its two colour axes), and how light it is and
how much contrast it has (the same for brightness), are measured for every frame and set back
to the given frame's. What the correction is changes slowly, as the drift does, so it is
smoothed over a second: one frame's own content does not swing it.

Brightness is corrected within narrow bounds only. The model adds a tenth or so of contrast,
and that much is taken back; a scene that is meant to get darker, or to fill with fog, changes
by far more than that, and keeps nearly all of its change. (Corrected once, by what the first
frames after the picture needed, a video that was continued went pale for a moment at the
joint and too hard after it: the drift is gradual in brightness as it is in colour.)

It is a correction of colour only. What the model changes in how things are drawn (a pencil
sketch that turns into a photograph) stays changed. The same is done for a video that is
continued, from the last of the frames it goes on from, so that a video that is continued
four times does not drift four times over.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Frames on either side that a frame's correction is averaged over.
REACH = 12
#: A correction of colour further from none than this was measured on frames that show
#: something else (a cut, a fast move). It is held to these: gain between, offset within (of 255).
GAIN = (0.4, 2.0)
OFFSET = 48.0
#: The same for brightness, which a scene may be meant to change: only the model's own creep
#: is taken back.
LIGHT_GAIN = (0.85, 1.15)
LIGHT_OFFSET = 16.0
#: Frames are measured on about this many pixels a side: colour on the whole is all that is asked.
MEASURED_SIDE = 120
#: Frames corrected in one go.
PIECE = 16

#: Brightness and the two colour axes (as JPEG has them, colour centred on nought).
_TO_YCC = np.array(
    [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=np.float32
)
_FROM_YCC = np.linalg.inv(_TO_YCC).astype(np.float32)


def _smoothed(values: np.ndarray, reach: int = REACH) -> np.ndarray:
    """Each row averaged with the ``reach`` before and after it; past the ends, the end row stands in."""
    if len(values) < 2:
        return values
    padded = np.concatenate(
        [np.repeat(values[:1], reach, axis=0), values, np.repeat(values[-1:], reach, axis=0)]
    )
    window = np.ones(2 * reach + 1, dtype=np.float32) / (2 * reach + 1)
    return np.stack(
        [np.convolve(padded[:, column], window, mode="valid") for column in range(values.shape[1])], axis=1
    ).astype(np.float32)


def corrections(frames: np.ndarray, held: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """For every frame, the gain and offset (brightness, and the two colour axes) that bring it back.

    ``frames`` is ``[n, height, width, 3]`` of 8-bit RGB, and its first ``held`` frames are the
    ones that were given (a picture is one). Those are left as they are: gain one, offset nought.
    """
    count = len(frames)
    gain, offset = np.ones((count, 3), dtype=np.float32), np.zeros((count, 3), dtype=np.float32)
    if count <= held:
        return gain, offset
    step = max(1, min(frames.shape[1:3]) // MEASURED_SIDE)
    seen = frames[:, ::step, ::step].astype(np.float32) @ _TO_YCC.T
    seen = seen.reshape(count, -1, 3)
    mean, spread = seen.mean(axis=1), seen.std(axis=1)
    given = held - 1

    # Frame by frame, and smoothed among the frames that are corrected.
    ratio = spread[given] / np.maximum(spread[held:], 1e-3)
    # The two colour axes are spread about nought, which is grey: a gain and an offset do it.
    gain[held:, 1:] = _smoothed(np.clip(ratio[:, 1:], *GAIN))
    leaning = mean[given, 1:] - mean[held:, 1:] * gain[held:, 1:]
    offset[held:, 1:] = np.clip(_smoothed(leaning), -OFFSET, OFFSET)
    # Brightness is spread about a frame's own mean, and that mean is moved towards the given
    # frame's: as one gain and one offset, that is this.
    contrast = _smoothed(np.clip(ratio[:, :1], *LIGHT_GAIN))[:, 0]
    moved = np.clip(_smoothed(mean[given, :1] - mean[held:, :1]), -LIGHT_OFFSET, LIGHT_OFFSET)[:, 0]
    gain[held:, 0] = contrast
    offset[held:, 0] = mean[held:, 0] * (1.0 - contrast) + moved
    return gain, offset


def keep(frames: np.ndarray, held: int = 1) -> np.ndarray:
    """``frames`` with every frame after the given ones in the given ones' colours. Changes ``frames``."""
    import mlx.core as mx

    gain, offset = corrections(frames, held)
    if np.allclose(gain, 1.0, atol=0.01) and np.allclose(offset, 0.0, atol=0.5):
        return frames
    # To brightness and colour, corrected, and back: one matrix and one offset for each frame.
    matrices = np.einsum("ij,nj,jk->nik", _TO_YCC.T, gain, _FROM_YCC.T).astype(np.float32)
    shifts = (offset @ _FROM_YCC.T).astype(np.float32)
    height, width = frames.shape[1:3]
    for start in range(held, len(frames), PIECE):
        stop = min(start + PIECE, len(frames))
        pixels: Any = mx.array(frames[start:stop]).astype(mx.float32).reshape(stop - start, height * width, 3)
        pixels = pixels @ mx.array(matrices[start:stop]) + mx.array(shifts[start:stop])[:, None, :]
        pixels = mx.clip(pixels + 0.5, 0, 255).astype(mx.uint8).reshape(stop - start, height, width, 3)
        frames[start:stop] = np.array(pixels)
    return frames
