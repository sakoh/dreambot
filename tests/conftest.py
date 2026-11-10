from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from PIL import Image

from dreambot.engine.client import Engine


@pytest.fixture
async def engine() -> AsyncIterator[Engine]:
    """An engine backed by the fake (Pillow) backend: fast and weight-free."""
    eng = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    try:
        yield eng
    finally:
        await eng.close()


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.new("RGB", (512, 384), (200, 40, 40)).save(path)
    return path


@pytest.fixture
def sample_mask(tmp_path: Path) -> Path:
    path = tmp_path / "mask.png"
    mask = Image.new("L", (512, 384), 0)
    mask.paste(255, (128, 96, 384, 288))
    mask.save(path)
    return path
