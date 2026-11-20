"""The three steps that turn noise into a video's latents.

FastMetal is distilled (DMD2) to denoise in three passes at fixed noise levels. Each pass
predicts the clean video from the noisy one; between passes the prediction is given fresh
noise at the next, lower level. The levels are flow-matching sigmas under a shift that moves
them towards the noisy end, which is where a video's layout is decided.

A video that starts on a picture is made the way Wan 2.2's own image-to-video does it: the
picture's latent is the first latent frame, it is put back before every pass, and its tokens
are told they carry no noise (timestep 0). FastVideo ships no such path for these weights; it
was tried here on real weights and holds the picture's subject and scene (see the README).
How much moves is decided before the first pass, by what the clip is before it is denoised.
From noise alone the model is free: it moves the camera in on small figures and has two
bears wrestle across the frame, whatever the prompt says about the camera, because three
passes without guidance follow a prompt only so far. From the picture held still under the
noise (``still``, with ``noise_share`` of it noise: 97 parts in a hundred leave a trace of the
picture in every frame), the model keeps the frame where it is and moves what is in it. That
is the difference between ``--motion high`` and ``--motion low``. (Tried on five pictures, two
videos each way: from noise the frame travelled about ten times as far in five seconds as
from the still.)

More than one latent frame can be held that way, which is how a video is continued: held to
the last five frames of the one before (two latent frames), the model goes on in the direction
things were moving, where from the last frame alone it starts again from rest. The same does
not work at the other end. Held to a picture as its last latent frame, the model films what it
would have filmed and cuts to the picture five frames before the end (tried on real weights).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: The noise levels FastMetal was distilled at, as positions on the 1000-step training schedule.
STEPS = (1000, 757, 522)
#: The shift the 5B build was trained under.
SHIFT = 5.0
TRAIN_STEPS = 1000


@dataclass(frozen=True)
class Schedule:
    """The training schedule: every noise level (sigma) and the timestep that names it."""

    sigmas: np.ndarray
    timesteps: np.ndarray

    @classmethod
    def shifted(cls, shift: float = SHIFT) -> Schedule:
        # In 32-bit floats, as the scheduler the model was trained with works them out: its
        # levels are then the very numbers the model has been told, not their neighbours.
        one, shift32, steps = np.float32(1), np.float32(shift), np.float32(TRAIN_STEPS)
        base = np.linspace(1, TRAIN_STEPS, TRAIN_STEPS, dtype=np.float32)[::-1] / steps
        sigmas = (shift32 * base / (one + (shift32 - one) * base)).astype(np.float32)
        return cls(sigmas=sigmas, timesteps=(sigmas * steps).astype(np.float32))

    def levels(self, steps: Sequence[int] = STEPS) -> list[float]:
        """The timesteps the model is given: ``steps`` moved onto the shifted schedule."""
        warped = np.concatenate([self.timesteps, [0.0]])
        return [float(warped[TRAIN_STEPS - step]) for step in steps]

    def sigma(self, timestep: float) -> float:
        return float(self.sigmas[int(np.argmin(np.abs(self.timesteps - timestep)))])


def positions(frames: int, height: int, width: int, head_dim: int, theta: float = 10000.0):
    """Rotary position tables for a clip of ``frames`` x ``height`` x ``width`` tokens.

    Each token's place along time, height and width turns a share of every attention head's
    dimensions: ``head_dim - 4 * (head_dim // 6)`` of them for time and ``2 * (head_dim // 6)``
    for each of the other two. Returns ``(cos, sin)``, each ``[tokens, head_dim]``.

    Worked out in 32-bit floats throughout, as the tables the model was trained with were.
    (They then agree with PyTorch's to the last bit or the one before it; in 64-bit floats
    they would be more exact and ten times further from what the model has seen.)
    """
    sixth = head_dim // 6
    shares = (head_dim - 4 * sixth, 2 * sixth, 2 * sixth)
    axes = [np.arange(count, dtype=np.float32) for count in (frames, height, width)]
    grid = np.meshgrid(*axes, indexing="ij")
    cos, sin = [], []
    for share, axis in zip(shares, grid, strict=True):
        exponents = np.arange(0, share, 2, dtype=np.float32)[: share // 2] / np.float32(share)
        freqs = (np.float32(1.0) / np.float32(theta) ** exponents).astype(np.float32)
        angles = np.outer(axis.reshape(-1), freqs).astype(np.float32)
        cos.append(np.repeat(np.cos(angles), 2, axis=1))
        sin.append(np.repeat(np.sin(angles), 2, axis=1))
    return (
        np.concatenate(cos, axis=1).astype(np.float32),
        np.concatenate(sin, axis=1).astype(np.float32),
    )


def sample(
    model: Any,
    text: Any,
    shape: tuple[int, int, int],
    *,
    seed: int,
    start: np.ndarray | None = None,
    still: np.ndarray | None = None,
    noise_share: float = 1.0,
    shift: float = SHIFT,
    steps: Sequence[int] = STEPS,
    on_step: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Make the latents of one video under ``text``: ``[channels, frames, height, width]``.

    ``shape`` is the clip's size in latents (frames, height, width). ``seed`` decides the noise
    it starts as and the noise it is given between passes. ``start`` holds the clip's first
    latent frames to what is given (``[1, C, n, H, W]``: a picture to start on is one frame,
    the end of a video to go on from is two). ``still`` is a clip of the same shape to start
    from under the noise, in place of noise alone, and ``noise_share`` how much of the two is
    noise (see above). ``on_step`` is called after every pass with (passes done, passes in
    all), and may raise to stop.
    """
    import mlx.core as mx

    schedule = Schedule.shifted(shift)
    levels = schedule.levels(steps)
    rng = np.random.default_rng(seed)
    frames, height, width = shape
    noise = rng.standard_normal((1, model.channels, frames, height, width)).astype(np.float32)
    pt, ph, pw = model.patch_size
    per_frame = (height // ph) * (width // pw)
    tokens = (frames // pt) * per_frame
    cos, sin = positions(frames // pt, height // ph, width // pw, model.head_dim)
    cos, sin = mx.array(cos), mx.array(sin)
    first = mx.array(start.astype(np.float32)) if start is not None else None
    lead = int(first.shape[2]) if first is not None else 0
    if lead >= frames:
        raise ValueError("A clip has to be longer than what it is held to at its start.")

    def held(x: Any) -> Any:
        return x if first is None else mx.concatenate([first.astype(x.dtype), x[:, :, lead:]], axis=2)

    if still is not None:
        noise = (1.0 - noise_share) * still.astype(np.float32) + noise_share * noise
    # Between passes the latents are kept as the 16-bit floats the model reads them as.
    latents = mx.array(noise).astype(mx.float16)
    for index, level in enumerate(levels):
        latents = held(latents)
        timestep = np.full((1, tokens), level, dtype=np.float32)
        timestep[:, : (lead // pt) * per_frame] = 0.0
        predicted = model(latents, text, mx.array(timestep), cos, sin).astype(mx.float32)
        # The model predicts the noise's direction; the clean video is what is left without it.
        clean = latents.astype(mx.float32) - schedule.sigma(level) * predicted
        if index < len(levels) - 1:
            fresh = mx.array(rng.standard_normal(noise.shape).astype(np.float32))
            sigma = schedule.sigma(levels[index + 1])
            clean = (1.0 - sigma) * clean + sigma * fresh
        latents = clean.astype(mx.float16)
        mx.eval(latents)
        if on_step is not None:
            on_step(index + 1, len(levels))
    result = np.array(held(latents).astype(mx.float32))[0]
    del latents, predicted, clean
    return result
