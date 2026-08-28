"""Tests for the eval transform grid (PLAN.md §3.1)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src import transforms as T


def _img(w: int = 96, h: int = 64, seed: int = 0, mode: str = "RGB") -> Image.Image:
    """A textured test image. Flat images can hide resampling/JPEG effects."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        127
        + 90 * np.sin(xx / 3.0)
        + 40 * np.cos(yy / 5.0)
        + rng.normal(0, 12, size=(h, w))
    )
    arr = np.clip(np.stack([base, np.roll(base, 7, 1), np.roll(base, 13, 0)], -1), 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    return img if mode == "RGB" else img.convert(mode)


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.int16)


# --------------------------------------------------------------------------- #
# Grid shape
# --------------------------------------------------------------------------- #

def test_grid_matches_spec_exactly():
    assert T.TRANSFORM_GRID == {
        "clean": [None],
        "jpeg": [90, 70, 50, 30],
        "blur": [0.5, 1.0, 2.0],
        "resize": [0.5, 0.25],
        "noise": [0.02, 0.05, 0.10],
        "jitter": [0.20],
        "center_crop": [0.80],
    }
    assert T.COMPOSED == [
        ("blur", 1.0, "jpeg", 70),
        ("resize", 0.5, "jpeg", 50),
        ("jitter", 0.20, "jpeg", 30),
        ("resize", 0.25, "blur", 0.5, "jpeg", 30),
    ]


def test_build_cells_counts_and_names():
    cells = T.build_cells()
    # 1 clean + 14 single severities + 4 chains
    assert len(cells) == 19
    assert sum(c.kind == "clean" for c in cells) == 1
    assert sum(c.kind == "single" for c in cells) == 14
    assert sum(c.kind == "composed" for c in cells) == 4
    assert cells[0].name == "clean"
    names = {c.name for c in cells}
    assert {"jpeg_30", "blur_1.0", "resize_0.25", "noise_0.1", "jitter_0.2",
            "center_crop_0.8"} <= names
    assert "composed_resize0.25+blur0.5+jpeg30" in names
    # cache/CSV keys must be legal Windows path components
    assert not any(set(n) & set(':*?"<>|/\\') for n in names)


def test_all_cells_preserve_size_and_mode():
    src = _img()
    for cell in T.build_cells():
        out = T.apply_cell(src, cell, image_id="x", base_seed=0)
        assert out.size == src.size, cell.name
        assert out.mode == "RGB", cell.name


# --------------------------------------------------------------------------- #
# Individual transforms
# --------------------------------------------------------------------------- #

def test_clean_is_identity_on_rgb():
    src = _img()
    assert np.array_equal(_arr(T.t_clean(src)), _arr(src))


def test_jpeg_uses_a_real_encoder():
    """A simulation would not produce a decodable JPEG bitstream, and would not
    show monotonically increasing error as quality drops."""
    src = _img()
    errs = []
    for q in T.TRANSFORM_GRID["jpeg"]:
        out = T.t_jpeg(src, q)
        # the returned image really came out of a JPEG decoder
        assert out.format == "JPEG"
        errs.append(float(np.abs(_arr(out) - _arr(src)).mean()))
    assert errs[0] > 0, "q=90 must still lose information"
    assert errs == sorted(errs), f"error must grow as quality falls: {errs}"
    assert errs[-1] > 1.2 * errs[0], "q=30 must be clearly worse than q=90"


def test_jpeg_is_a_true_roundtrip_at_the_byte_level():
    """t_jpeg must equal encode-then-decode with the same settings."""
    src = _img()
    buf = io.BytesIO()
    src.save(buf, format="JPEG", quality=50, subsampling="4:2:0")
    buf.seek(0)
    ref = Image.open(buf)
    ref.load()
    assert np.array_equal(_arr(T.t_jpeg(src, 50)), _arr(ref))


def test_blur_reduces_high_frequency_energy_monotonically():
    src = _img()

    def hf_energy(img: Image.Image) -> float:
        x = np.asarray(img, dtype=np.float32).mean(-1)
        return float(np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean())

    e = [hf_energy(src)] + [hf_energy(T.t_blur(src, s)) for s in T.TRANSFORM_GRID["blur"]]
    assert e == sorted(e, reverse=True), e


def test_blur_zero_sigma_is_identity():
    src = _img()
    assert np.array_equal(_arr(T.t_blur(src, 0.0)), _arr(src))


def test_resize_returns_original_size_and_loses_detail():
    src = _img()
    prev = 0.0
    for s in T.TRANSFORM_GRID["resize"]:  # 0.5 then 0.25 -> increasing damage
        out = T.t_resize(src, s)
        assert out.size == src.size
        err = float(np.abs(_arr(out) - _arr(src)).mean())
        assert err > prev
        prev = err


def test_resize_handles_tiny_images_without_zero_dimensions():
    tiny = _img(3, 2)
    out = T.t_resize(tiny, 0.25)
    assert out.size == (3, 2)


def test_noise_std_tracks_sigma():
    src = _img()
    x = np.asarray(src, dtype=np.float32) / 255.0
    for sigma in T.TRANSFORM_GRID["noise"]:
        out = T.t_noise(src, sigma, np.random.default_rng(0))
        d = np.asarray(out, dtype=np.float32) / 255.0 - x
        # clipping at the [0,1] boundary biases the estimate slightly low
        assert 0.75 * sigma < d.std() <= 1.1 * sigma, sigma


def test_jitter_changes_the_image_but_stays_in_range():
    src = _img()
    out = T.t_jitter(src, 0.20, np.random.default_rng(0))
    assert not np.array_equal(_arr(out), _arr(src))
    # +/-20% on brightness cannot move the mean by more than ~20%
    assert abs(_arr(out).mean() - _arr(src).mean()) < 0.35 * _arr(src).mean()


def test_center_crop_keeps_linear_fraction_of_each_side():
    """0.80 is read as linear, not area: the centre pixel is preserved and the
    output is a 1/0.8 magnification of the central 80% x 80% region."""
    src = _img(100, 100)
    out = T.t_center_crop(src, 0.80)
    assert out.size == (100, 100)
    ref = src.crop((10, 10, 90, 90)).resize((100, 100), T.RESAMPLE)
    assert np.array_equal(_arr(out), _arr(ref))


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #

def test_composed_chain_applies_ops_in_order():
    src = _img()
    cell = next(c for c in T.build_cells() if c.name == "composed_blur1.0+jpeg70")
    got = T.apply_cell(src, cell, image_id="i", base_seed=0)
    ref = T.t_jpeg(T.t_blur(src, 1.0), 70)
    assert np.array_equal(_arr(got), _arr(ref))
    # order matters, so the reverse chain must differ
    other = T.t_blur(T.t_jpeg(src, 70), 1.0)
    assert not np.array_equal(_arr(got), _arr(other))


def test_worst_case_chain_is_worse_than_its_parts():
    """Only compared against the band-limiting cells. center_crop is a geometric
    op, and pixel-wise MAE says nothing useful about a magnified crop."""
    src = _img(256, 256)
    err = {}
    for cell in T.build_cells():
        out = T.apply_cell(src, cell, image_id="i", base_seed=0)
        err[cell.name] = float(np.abs(_arr(out) - _arr(src)).mean())
    assert err["clean"] == 0.0
    worst = err["composed_resize0.25+blur0.5+jpeg30"]
    for part in ("resize_0.25", "blur_0.5", "jpeg_30"):
        assert worst > err[part], part


# --------------------------------------------------------------------------- #
# Determinism (§3.1: "Seed everything")
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cell_name", ["noise_0.05", "jitter_0.2", "composed_jitter0.2+jpeg30"])
def test_stochastic_cells_are_reproducible(cell_name):
    src = _img()
    cell = next(c for c in T.build_cells() if c.name == cell_name)
    a = T.apply_cell(src, cell, image_id="img_7", base_seed=1234)
    b = T.apply_cell(src, cell, image_id="img_7", base_seed=1234)
    assert np.array_equal(_arr(a), _arr(b))


def test_stochastic_cells_differ_across_images_seeds_and_cells():
    src = _img()
    noise = next(c for c in T.build_cells() if c.name == "noise_0.05")
    base = _arr(T.apply_cell(src, noise, image_id="a", base_seed=0))
    assert not np.array_equal(base, _arr(T.apply_cell(src, noise, image_id="b", base_seed=0)))
    assert not np.array_equal(base, _arr(T.apply_cell(src, noise, image_id="a", base_seed=1)))
    other = next(c for c in T.build_cells() if c.name == "noise_0.02")
    assert not np.array_equal(base, _arr(T.apply_cell(src, other, image_id="a", base_seed=0)))


def test_evaluation_order_does_not_affect_results():
    """Per-(image, cell) seeding, not a shared stream: shuffling the work order
    must not change a single pixel."""
    src = _img()
    cells = [c for c in T.build_cells() if c.is_stochastic]
    forward = {c.name: _arr(T.apply_cell(src, c, image_id="z", base_seed=9)) for c in cells}
    for c in reversed(cells):
        assert np.array_equal(forward[c.name], _arr(T.apply_cell(src, c, image_id="z", base_seed=9)))


def test_derive_seed_is_stable_across_processes():
    """Hard-coded value: if this changes, every cached grid becomes stale."""
    assert T.derive_seed(0, "img_0", "noise_0.05") == 10527689486506219619


def test_deterministic_cells_ignore_the_seed():
    src = _img()
    cell = next(c for c in T.build_cells() if c.name == "jpeg_50")
    a = T.apply_cell(src, cell, image_id="a", base_seed=0)
    b = T.apply_cell(src, cell, image_id="b", base_seed=999)
    assert np.array_equal(_arr(a), _arr(b))


def test_cache_key_ignores_seed_only_for_deterministic_cells():
    cells = {c.name: c for c in T.build_cells()}
    jpeg, noise = cells["jpeg_50"], cells["noise_0.05"]
    assert T.cache_key(jpeg, "a", 0) == T.cache_key(jpeg, "a", 1)
    assert T.cache_key(noise, "a", 0) != T.cache_key(noise, "a", 1)
    assert T.cache_key(jpeg, "a", 0) != T.cache_key(jpeg, "b", 0)


# --------------------------------------------------------------------------- #
# Awkward inputs (§9.1 requires these not to crash; the grid sees them first)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK", "I;16", "1"])
def test_every_cell_survives_unusual_input_modes(mode):
    src = _img(48, 32).convert(mode)
    for cell in T.build_cells():
        out = T.apply_cell(src, cell, image_id="m", base_seed=0)
        assert out.mode == "RGB" and out.size == (48, 32), (mode, cell.name)


def test_alpha_is_composited_not_dropped():
    rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 0))  # fully transparent red
    assert T.to_rgb(rgba).getpixel((0, 0)) == (0, 0, 0)  # -> black, not red


def test_grayscale_jitter_does_not_crash_on_saturation():
    out = T.t_jitter(_img(mode="L"), 0.20, np.random.default_rng(0))
    assert out.mode == "RGB"


def test_unknown_family_and_bad_chain_raise():
    with pytest.raises(KeyError):
        T.apply_op(_img(), "sharpen", 1.0)
    with pytest.raises(ValueError):
        T.build_cells(grid={"clean": [None]}, composed=[("jpeg", 70, "blur")])
    with pytest.raises(ValueError):
        T.apply_op(_img(), "noise", 0.05, rng=None)
