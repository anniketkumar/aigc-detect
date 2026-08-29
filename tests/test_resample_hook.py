"""The Phase 4 randomised resample-filter hook (`src/data/resample.py`).

Two things to prove: the hook actually varies the filter and reports which one
it used, and it stays out of the eval grid. The second matters more -- if
training augmentation and evaluation shared a filter policy, the resampling axis
of the robustness grid would stop being an out-of-sample test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import transforms as T
from src.data import resample as R


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    a = 127 + 100 * np.sin(x / 3.0) * np.cos(y / 5.0)   # high-frequency content,
    a = np.clip(a + rng.normal(0, 8, a.shape), 0, 255)  # so filter choice shows
    return Image.fromarray(np.repeat(a.astype(np.uint8)[:, :, None], 3, 2), "RGB")


# --------------------------------------------------------------------------- #
# Eval must stay pinned
# --------------------------------------------------------------------------- #

def test_eval_grid_is_still_bicubic():
    assert T.RESAMPLE is Image.Resampling.BICUBIC


def test_transforms_module_does_not_import_the_hook():
    """A static check, not a runtime one.

    A runtime probe would pass simply because nothing happened to call the hook
    on that path. Reading the import list is what actually says the eval grid
    cannot reach it.
    """
    tree = ast.parse(Path("src/transforms.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("resample" in m for m in imported), (
        f"src/transforms.py imports {imported}; the eval grid must not see the hook"
    )


def test_eval_resize_is_unaffected_by_the_hooks_rng(img):
    """Eval output is a pure function of (image, cell). No RNG reaches it."""
    a = np.asarray(T.t_resize(img, 0.5))
    for i in range(8):
        R.sample_filter(np.random.default_rng(i))
    b = np.asarray(T.t_resize(img, 0.5))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# The hook itself
# --------------------------------------------------------------------------- #

def test_all_four_filters_are_offered():
    assert set(R.FILTER_NAMES) == {"bilinear", "bicubic", "lanczos", "area"}


def test_area_maps_to_box():
    """PIL calls it BOX, OpenCV calls it INTER_AREA, the spec said area."""
    assert R.FILTERS["area"] is Image.Resampling.BOX


def test_sampling_covers_every_filter():
    rng = np.random.default_rng(0)
    seen = {R.sample_filter(rng)[0] for _ in range(400)}
    assert seen == set(R.FILTER_NAMES)


def test_sampling_is_roughly_uniform():
    rng = np.random.default_rng(1)
    draws = [R.sample_filter(rng)[0] for _ in range(4000)]
    for name in R.FILTER_NAMES:
        assert 0.20 < draws.count(name) / 4000 < 0.30


def test_sampling_is_reproducible_from_the_seed():
    a = [R.sample_filter(np.random.default_rng(7))[0] for _ in range(5)]
    b = [R.sample_filter(np.random.default_rng(7))[0] for _ in range(5)]
    assert a == b


def test_no_global_rng_is_touched():
    """A global stream would make a training run depend on worker count."""
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    R.sample_filter(np.random.default_rng(123))
    assert np.random.random() == before


def test_filter_pool_can_be_restricted():
    rng = np.random.default_rng(0)
    seen = {R.sample_filter(rng, ["area", "lanczos"])[0] for _ in range(100)}
    assert seen == {"area", "lanczos"}


def test_unknown_filter_raises():
    with pytest.raises(KeyError):
        R.sample_filter(np.random.default_rng(0), ["sinc"])


def test_empty_pool_raises():
    with pytest.raises(ValueError):
        R.sample_filter(np.random.default_rng(0), [])


# --------------------------------------------------------------------------- #
# The filters must be distinguishable, or randomising them buys nothing
# --------------------------------------------------------------------------- #

def test_filters_actually_produce_different_pixels(img):
    outs = {}
    for name, filt in R.FILTERS.items():
        small = img.resize((64, 64), filt)
        outs[name] = np.asarray(small.resize(img.size, filt), dtype=np.int16)
    names = list(outs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = np.abs(outs[a] - outs[b]).mean()
            assert d > 0.5, f"{a} and {b} are near-identical (mean|diff|={d:.3f})"


def test_lanczos_lifts_high_frequency_energy_relative_to_area(img):
    """The spectral difference is the reason the hook exists.

    Train only on bicubic and the model learns bicubic's kernel as part of what
    a resized image looks like. This asserts the alternatives really do differ
    in the band the Phase 5 artifact branch reads.
    """
    def hf(im):
        a = np.asarray(im, dtype=np.float32).mean(2)
        return float(np.abs(np.diff(a, axis=1)).mean())

    lan = img.resize((64, 64), R.FILTERS["lanczos"]).resize(img.size, R.FILTERS["lanczos"])
    area = img.resize((64, 64), R.FILTERS["area"]).resize(img.size, R.FILTERS["area"])
    assert hf(lan) > hf(area)


# --------------------------------------------------------------------------- #
# The wrappers Phase 4 will call
# --------------------------------------------------------------------------- #

def test_resize_rand_preserves_size_and_reports_both_filters(img):
    out, meta = R.resize_rand(img, 0.5, np.random.default_rng(0))
    assert out.size == img.size and out.mode == "RGB"
    assert meta["op"] == "resize" and meta["scale"] == 0.5
    assert meta["filter_down"] in R.FILTER_NAMES
    assert meta["filter_up"] in R.FILTER_NAMES


def test_resize_rand_can_use_different_filters_each_way(img):
    """Real chains downscale in a thumbnailer and upscale in a viewer."""
    rng = np.random.default_rng(0)
    metas = [R.resize_rand(img, 0.5, rng)[1] for _ in range(60)]
    assert any(m["filter_down"] != m["filter_up"] for m in metas)


def test_resize_rand_same_filter_both_ways_when_asked(img):
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = R.resize_rand(img, 0.5, rng, same_filter_both_ways=True)[1]
        assert m["filter_down"] == m["filter_up"]


def test_center_crop_rand_matches_the_eval_definition_of_keep(img):
    """Linear fraction, like transforms.t_center_crop.

    The augmentation has to cover the eval cell, so the two must agree on what
    0.8 means even though they disagree on the filter. Checked by pinning the
    hook to bicubic and comparing pixels against the eval transform.
    """
    out, meta = R.center_crop_rand(img, 0.8, np.random.default_rng(0), names=["bicubic"])
    ref = T.t_center_crop(img, 0.8)
    assert out.size == ref.size
    assert np.array_equal(np.asarray(out), np.asarray(ref))
    assert meta["filter"] == "bicubic"


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.5])
def test_resize_rand_rejects_bad_scale(img, scale):
    with pytest.raises(ValueError):
        R.resize_rand(img, scale, np.random.default_rng(0))


@pytest.mark.parametrize("keep", [0.0, -0.1, 1.5])
def test_center_crop_rand_rejects_bad_keep(img, keep):
    with pytest.raises(ValueError):
        R.center_crop_rand(img, keep, np.random.default_rng(0))


def test_extreme_downscale_does_not_produce_a_zero_dimension(img):
    out, _ = R.resize_rand(img, 0.001, np.random.default_rng(0))
    assert out.size == img.size
