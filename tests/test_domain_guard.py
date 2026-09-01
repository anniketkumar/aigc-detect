"""Unit tests for src/data/domain_guard.py -- the non-photographic-content
heuristic. See that module's docstring for why it exists (every training
source, real and AI, is photographic -- src/data/sources.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.domain_guard import check_domain


def _solid(size=(256, 256), color=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", size, color)


def _flat_blocks(size=(256, 256)) -> Image.Image:
    """A handful of flat-filled rectangles on a white background -- a cheap
    stand-in for a diagram/screenshot: few unique colors, majority background."""
    img = Image.new("RGB", size, (255, 255, 255))
    arr = np.asarray(img).copy()
    colors = [(20, 90, 200), (200, 40, 40), (30, 160, 60)]
    h, w, _ = arr.shape
    band = h // (len(colors) + 1)
    for i, color in enumerate(colors):
        y0 = (i + 1) * band - band // 4
        y1 = (i + 1) * band + band // 4
        arr[y0:y1, w // 8 : w - w // 8] = color
    return Image.fromarray(arr, "RGB")


def _noise_photo(size=(256, 256), seed=0) -> Image.Image:
    """A camera-photo stand-in: independent per-pixel noise, thousands of
    unique colors, no dominant color -- the opposite profile of a graphic."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_noise_photo_is_not_flagged():
    result = check_domain(_noise_photo())
    assert result.likely_non_photographic is False
    assert result.unique_ratio > 0.2


def test_flat_diagram_style_image_is_flagged():
    result = check_domain(_flat_blocks())
    assert result.likely_non_photographic is True


def test_solid_color_image_is_flagged_via_dominant_fraction():
    result = check_domain(_solid())
    assert result.likely_non_photographic is True
    assert result.dominant_fraction == pytest.approx(1.0)


def test_accepts_non_rgb_modes_without_erroring():
    gray = Image.new("L", (64, 64), 200)
    result = check_domain(gray)
    assert isinstance(result.likely_non_photographic, bool)

    rgba = Image.new("RGBA", (64, 64), (10, 20, 30, 128))
    result = check_domain(rgba)
    assert isinstance(result.likely_non_photographic, bool)


def test_large_image_is_downsized_but_still_scored():
    big = _noise_photo(size=(1024, 1024))
    result = check_domain(big)
    assert 0.0 <= result.unique_ratio <= 1.0
    assert 0.0 <= result.dominant_fraction <= 1.0


def test_result_is_deterministic():
    img = _flat_blocks()
    a = check_domain(img)
    b = check_domain(img)
    assert a == b
