"""Tests for the Phase 4 training-time augmentation sampler (PLAN.md §6)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src import transforms as T
from src.data import augment as A


def _img(w: int = 96, h: int = 64, seed: int = 0, mode: str = "RGB") -> Image.Image:
    """Same textured fixture as test_transforms.py -- flat images hide effects."""
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
# Config
# --------------------------------------------------------------------------- #

def test_defaults_match_plan_spec_exactly():
    cfg = A.AugmentConfig()
    assert cfg.p_apply == 0.8
    assert (cfg.k_min, cfg.k_max) == (1, 3)
    assert cfg.families == ("jpeg", "blur", "resize", "noise", "jitter", "center_crop")
    assert cfg.severity_range == {
        "jpeg": (20.0, 95.0),
        "blur": (0.0, 3.0),
        "resize": (0.2, 1.0),
        "noise": (0.0, 0.12),
        "jitter": (0.0, 0.3),
        "center_crop": (0.7, 1.0),
    }


def test_ranges_are_wider_than_and_disjoint_from_the_eval_grid():
    """§6: severities must be "drawn continuously and wider than the eval grid".
    A training range that merely contained the eval points would let the eval
    grid double as validation data for the exact severities trained on."""
    for fam, (lo, hi) in A.SEVERITY_RANGE.items():
        if fam == "center_crop":
            eval_points = T.TRANSFORM_GRID["center_crop"]
        else:
            eval_points = T.TRANSFORM_GRID[fam]
        for pt in eval_points:
            assert lo <= pt <= hi, (fam, pt)
        # strictly wider on at least one side, not merely touching
        assert lo < min(eval_points) or hi > max(eval_points), fam


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        A.AugmentConfig(p_apply=1.5)
    with pytest.raises(ValueError):
        A.AugmentConfig(k_min=0)
    with pytest.raises(ValueError):
        A.AugmentConfig(k_min=3, k_max=1)
    with pytest.raises(ValueError):
        A.AugmentConfig(severity_range={"jpeg": (20.0, 95.0)})  # missing families


# --------------------------------------------------------------------------- #
# sample_plan
# --------------------------------------------------------------------------- #

def test_sample_plan_leaves_roughly_twenty_percent_clean():
    rng = np.random.default_rng(0)
    n = 4000
    n_clean = sum(1 for _ in range(n) if not A.sample_plan(rng))
    frac = n_clean / n
    assert 0.15 < frac < 0.25, frac


def test_sample_plan_families_and_k_in_range():
    rng = np.random.default_rng(1)
    for _ in range(500):
        ops = A.sample_plan(rng)
        if not ops:
            continue
        assert 1 <= len(ops) <= 3
        families = [op.family for op in ops]
        assert len(families) == len(set(families)), "no family twice in one plan"
        assert set(families) <= set(A.FAMILIES)


def test_sample_plan_severities_stay_in_configured_range():
    rng = np.random.default_rng(2)
    seen = {fam: [] for fam in A.FAMILIES}
    for _ in range(3000):
        for op in A.sample_plan(rng):
            seen[op.family].append(op.severity)
    for fam, lo_hi in A.SEVERITY_RANGE.items():
        lo, hi = lo_hi
        vals = seen[fam]
        assert vals, f"family {fam} never sampled in 3000 draws"
        assert min(vals) >= lo and max(vals) <= hi, fam
        # jpeg quality must be an integer-valued float (real JPEG encoders take int)
        if fam == "jpeg":
            assert all(float(v).is_integer() for v in vals)


def test_all_k_values_reachable():
    rng = np.random.default_rng(3)
    seen_k = {len(A.sample_plan(rng)) for _ in range(2000)} - {0}
    assert seen_k == {1, 2, 3}


# --------------------------------------------------------------------------- #
# augment_image: determinism (§3.1's rule, extended to Phase 4)
# --------------------------------------------------------------------------- #

def test_reproducible_for_the_same_keys():
    src = _img()
    a_img, a_lab = A.augment_image(src, image_id="img_7", copy_index=2, base_seed=1234)
    b_img, b_lab = A.augment_image(src, image_id="img_7", copy_index=2, base_seed=1234)
    assert np.array_equal(_arr(a_img), _arr(b_img))
    assert a_lab == b_lab


def test_differs_across_copy_index_image_id_and_seed():
    src = _img()
    base_img, base_lab = A.augment_image(src, image_id="a", copy_index=0, base_seed=0)

    other_copy = A.augment_image(src, image_id="a", copy_index=1, base_seed=0)
    other_id = A.augment_image(src, image_id="b", copy_index=0, base_seed=0)
    other_seed = A.augment_image(src, image_id="a", copy_index=0, base_seed=1)

    for other_img, other_lab in (other_copy, other_id, other_seed):
        same_pixels = np.array_equal(_arr(base_img), _arr(other_img))
        same_label = base_lab == other_lab
        assert not (same_pixels and same_label)


def test_call_order_does_not_affect_result():
    """Per-(image_id, copy_index) seeding, not a shared stream."""
    src = _img()
    forward = [A.augment_image(src, image_id="z", copy_index=i, base_seed=9) for i in range(5)]
    backward = [A.augment_image(src, image_id="z", copy_index=i, base_seed=9) for i in reversed(range(5))]
    backward = list(reversed(backward))
    for (fi, fl), (bi, bl) in zip(forward, backward):
        assert np.array_equal(_arr(fi), _arr(bi))
        assert fl == bl


# --------------------------------------------------------------------------- #
# augment_image: correctness
# --------------------------------------------------------------------------- #

def test_ops_applied_in_the_sampled_order():
    src = _img()
    # find a (image_id, copy_index) that samples a multi-op plan
    plan = ()
    image_id = ""
    for i in range(200):
        rng = np.random.default_rng(T.derive_seed(0, f"probe{i}", "randaug0"))
        candidate = A.sample_plan(rng)
        if len(candidate) >= 2:
            plan, image_id = candidate, f"probe{i}"
            break
    assert len(plan) >= 2, "no multi-op plan found in 200 probes"

    out, label = A.augment_image(src, image_id=image_id, copy_index=0, base_seed=0)
    ref = T.to_rgb(src)
    rng_ref = np.random.default_rng(T.derive_seed(0, image_id, "randaug0"))
    A.sample_plan(rng_ref)  # advance past the plan draw, mirroring augment_image
    for op in plan:
        ref = T.apply_op(ref, op.family, op.severity, rng_ref)
    assert np.array_equal(_arr(out), _arr(ref))
    assert label.is_clean is False


def test_clean_draw_is_identity_apart_from_mode():
    cfg = A.AugmentConfig(p_apply=0.0)
    src = _img()
    out, label = A.augment_image(src, image_id="x", copy_index=0, base_seed=0, config=cfg)
    assert np.array_equal(_arr(out), _arr(T.to_rgb(src)))
    assert label.is_clean
    assert label == A.DegradationLabel.clean()


def test_always_augmented_when_p_apply_is_one():
    cfg = A.AugmentConfig(p_apply=1.0)
    rng = np.random.default_rng(0)
    assert all(A.sample_plan(rng, cfg) for _ in range(200))


def test_output_preserves_size_and_is_rgb():
    src = _img(120, 80)
    for copy_index in range(10):
        out, _ = A.augment_image(src, image_id="sz", copy_index=copy_index, base_seed=0)
        assert out.size == src.size
        assert out.mode == "RGB"


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK", "I;16", "1"])
def test_survives_unusual_input_modes(mode):
    src = _img(48, 32).convert(mode)
    for copy_index in range(6):
        out, _ = A.augment_image(src, image_id="m", copy_index=copy_index, base_seed=0)
        assert out.mode == "RGB" and out.size == (48, 32), (mode, copy_index)


def test_iter_augmented_copies_yields_n_and_matches_augment_image():
    src = _img()
    copies = list(A.iter_augmented_copies(src, image_id="k4", n_copies=4, base_seed=0))
    assert len(copies) == 4
    for i, (img, label) in enumerate(copies):
        ref_img, ref_label = A.augment_image(src, image_id="k4", copy_index=i, base_seed=0)
        assert np.array_equal(_arr(img), _arr(ref_img))
        assert label == ref_label


# --------------------------------------------------------------------------- #
# DegradationLabel
# --------------------------------------------------------------------------- #

def test_clean_label_vector_is_all_zero():
    vec = A.DegradationLabel.clean().to_vector()
    assert vec.shape == (12,)
    assert np.all(vec == 0.0)


def test_label_vector_layout_and_normalization():
    label = A.DegradationLabel(
        applied=(True, False, False, False, False, True),
        severity=(57.5, 0.0, 0.0, 0.0, 0.0, 0.85),
    )
    vec = label.to_vector()
    # jpeg: applied=1, severity (57.5-20)/(95-20) = 0.5
    assert vec[0] == 1.0
    assert vec[1] == pytest.approx(0.5)
    # blur/resize/noise/jitter: untouched -> zero
    assert np.all(vec[2:10] == 0.0)
    # center_crop: applied=1, severity (0.85-0.7)/(1.0-0.7) = 0.5
    assert vec[10] == 1.0
    assert vec[11] == pytest.approx(0.5)


def test_label_rejects_wrong_length():
    with pytest.raises(ValueError):
        A.DegradationLabel(applied=(True,), severity=(1.0,))


def test_label_from_augment_image_matches_the_sampled_plan():
    src = _img()
    for copy_index in range(30):
        _, label = A.augment_image(src, image_id="lab", copy_index=copy_index, base_seed=0)
        rng = np.random.default_rng(T.derive_seed(0, "lab", f"randaug{copy_index}"))
        plan = A.sample_plan(rng)
        expected = {op.family: op.severity for op in plan}
        for i, fam in enumerate(A.FAMILIES):
            assert label.applied[i] == (fam in expected), (copy_index, fam)
            if fam in expected:
                assert label.severity[i] == pytest.approx(expected[fam])
            else:
                assert label.severity[i] == 0.0
