"""Worker backends: the code that actually runs models inside the worker subprocess."""

from __future__ import annotations

from dreambot.engine.backends.base import Backend


def load_backend(name: str) -> Backend:
    """Instantiate a backend by name. Imports are lazy so ``fake`` never pulls in MLX."""
    if name == "fake":
        from dreambot.engine.backends.fake import FakeBackend

        return FakeBackend()
    if name == "mlx":
        from dreambot.engine.backends.mlx_backend import MlxBackend

        return MlxBackend()
    raise ValueError(f"unknown backend '{name}' (expected 'mlx' or 'fake')")
