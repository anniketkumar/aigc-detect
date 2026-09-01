"""Tests for src/features/frequency.py -- the fusion branch's feature vector.

Mirrors tests/test_transforms.py's rigor: a null-ish sanity pass (shape,
determinism, no NaN/Inf on edge-case pixels) plus a planted-signal pass that
proves the radial-FFT-ring construction can actually detect a periodic
up-sampling artifact when one is present, not just "looks plausible" on real
photos (same spirit as NOTES.md Phase 1's dummy_brightness detectability
check for the eval harness).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.features import frequency as F


def _textured_img(w: int = 300, h: int = 220, seed: int = 0) -> Image.Image:
    """A smooth, naturalistic test image -- broad low-frequency structure
    plus noise, no injected periodicity. Same construction as
    tests/test_transforms.py's ``_img`` helper."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        127
        + 90 * np.sin(xx / 37.0)
        + 40 * np.cos(yy / 53.0)
        + rng.normal(0, 12, size=(h, w))
    )
    arr = np.clip(
        np.stack([base, np.roll(base, 7, 1), np.roll(base, 13, 0)], -1), 0, 255
    )
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _periodic_img(w: int = 300, h: int = 220, period: int = 4, amp: float = 30.0) -> Image.Image:
    """A checkerboard-style pattern at a fixed spatial period on top of the
    same base texture -- a crude stand-in for the periodic checkerboard
    artifact GAN/diffusion up-sampling leaves in the Fourier spectrum."""
    base_img = _textured_img(w, h, seed=1)
    base = np.asarray(base_img, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    grid = amp * (
        np.sign(np.sin(2 * np.pi * xx / period)) + np.sign(np.sin(2 * np.pi * yy / period))
    )
    arr = np.clip(base + grid[..., None], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _flat_img(w: int = 64, h: int = 64, value: int = 200) -> Image.Image:
    return Image.new("RGB", (w, h), (value, value, value))


# --------------------------------------------------------------------------- #
# Shape, dtype, determinism
# --------------------------------------------------------------------------- #

def test_output_shape_and_dtype():
    feats = F.extract_frequency_features(_textured_img())
    assert feats.shape == (F.FREQ_DIM,)
    assert feats.dtype == np.float32


def test_feature_names_match_dim():
    assert len(F.FREQ_FEATURE_NAMES) == F.FREQ_DIM
    assert len(set(F.FREQ_FEATURE_NAMES)) == F.FREQ_DIM  # all unique


def test_deterministic():
    img = _textured_img(seed=3)
    a = F.extract_frequency_features(img)
    b = F.extract_frequency_features(img)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("w,h", [(64, 64), (300, 220), (1024, 768), (37, 501)])
def test_any_size_and_aspect(w, h):
    """Real corpus images are non-square and vary widely (PLAN.md's source
    geometry table); the fixed-resolution resize in _to_gray must not care."""
    feats = F.extract_frequency_features(_textured_img(w, h))
    assert feats.shape == (F.FREQ_DIM,)
    assert np.isfinite(feats).all()


# --------------------------------------------------------------------------- #
# Edge cases: must never NaN/Inf, even on degenerate pixels
# --------------------------------------------------------------------------- #

def test_flat_image_no_nan():
    """A constant-colour image: zero gradient everywhere, zero AC energy
    everywhere, ring_std == 0 for the peakiness division. Must not crash or
    produce NaN/Inf -- this is exactly the kind of degenerate input a bad
    decode or an all-white padding frame could hand the model."""
    feats = F.extract_frequency_features(_flat_img())
    assert np.isfinite(feats).all()


def test_single_pixel_wide_image_no_nan():
    feats = F.extract_frequency_features(Image.new("RGB", (1, 40), (128, 128, 128)))
    assert np.isfinite(feats).all()


def test_grayscale_and_rgba_input_no_nan():
    """Scorer contract: images arrive already RGB, but the extractor should
    not fall over if handed something else during ad-hoc testing."""
    assert np.isfinite(F.extract_frequency_features(_textured_img().convert("L").convert("RGB"))).all()
    rgba = _textured_img().convert("RGBA")
    assert np.isfinite(F.extract_frequency_features(rgba.convert("RGB"))).all()


# --------------------------------------------------------------------------- #
# The harness can detect a planted signal (mirrors NOTES.md Phase 1)
# --------------------------------------------------------------------------- #

def test_periodic_pattern_raises_ring_peakiness():
    """A real up-sampling checkerboard artifact concentrates FFT energy in a
    narrow radial band; a natural photo's spectrum falls off smoothly. If
    fft_ring_peakiness cannot tell these apart on a pattern this blatant, it
    will never catch a subtler generator fingerprint."""
    natural = F.extract_frequency_features(_textured_img(seed=5))
    periodic = F.extract_frequency_features(_periodic_img(period=4))

    peak_idx = F.FREQ_FEATURE_NAMES.index("fft_ring_peakiness")
    assert periodic[peak_idx] > natural[peak_idx], (
        f"planted period-4 checkerboard should raise ring peakiness: "
        f"natural={natural[peak_idx]:.3f} periodic={periodic[peak_idx]:.3f}"
    )


def test_periodic_pattern_raises_high_frequency_ring():
    """A period-4-pixel grid's fundamental frequency lands close to Nyquist
    -- one of the outer rings should show a clear jump, not just the
    aggregate peakiness statistic."""
    natural = F.extract_frequency_features(_textured_img(seed=5))
    periodic = F.extract_frequency_features(_periodic_img(period=4))
    outer_rings = slice(8, 12)  # fft_ring_8..11, the highest-frequency rings
    assert periodic[outer_rings].max() > natural[outer_rings].max(), (
        "periodic artifact should raise at least one outer FFT ring above "
        "its natural-image counterpart"
    )


def test_hf_energy_distinguishes_flat_from_textured():
    """Sanity check on the carried-forward gradient features: a flat image
    has ~zero hf_energy and flat_frac == 1; a textured one has neither."""
    flat = F.extract_frequency_features(_flat_img())
    textured = F.extract_frequency_features(_textured_img())
    hf_idx = F.FREQ_FEATURE_NAMES.index("hf_energy")
    flat_idx = F.FREQ_FEATURE_NAMES.index("flat_frac")
    assert flat[hf_idx] < textured[hf_idx]
    assert flat[flat_idx] > textured[flat_idx]
