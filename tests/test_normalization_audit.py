"""Task F -- the Phase 2 leakage audit, turned into a permanent regression test.

The audit found that a classifier reading **no pixels at all** scored 0.98 AUROC
on raw SID_Set. This re-runs the same metadata probe against the *normalized*
manifest and asserts every channel is now uninformative.

    channel            raw SID_Set    after normalization
    container_is_png      0.7574              0.5000
    megapixels            0.9814              0.5000
    is_1024sq             0.9814              0.5000
    is_square             0.9806              0.5000
    height                0.8966              0.5000
    has_icc               0.7027              0.5000
    n_bytes               0.6037              0.5017
    n_exif                0.5000              0.5000
    n_text_chunks         0.5000              0.5000

Most of those are 0.5000 *exactly*, because after normalization the feature is
constant across the whole corpus -- every file is a 512x512 RGB q95 JPEG with no
metadata. A constant feature has AUROC 0.5 by construction, which is a stronger
guarantee than "measured below a threshold": there is no sample size at which it
could come out otherwise.

``n_bytes`` is the exception and the interesting one. It cannot be made
constant -- file size at fixed quality is a measure of how compressible the
picture is, which is a genuine property of the pixels rather than of the
container. What it *must not* be is a proxy for source or class. This test
enforces that, and it has already earned its place: it caught Pexels at 0.7120
(see ``src/data/sources.py``), which was a perfect source fingerprint
masquerading as a class signal.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import roc_auc_score

DETAIL = Path("data/manifests/detail.csv")

#: PLAN.md §4 / task F. Direction-agnostic, so 0.60 means "no better than 60/40
#: either way". Deliberately tight: these channels should be at 0.50, and
#: anything drifting toward 0.60 is a bug being caught early rather than a
#: tolerance being used up.
MAX_AUROC = 0.60

#: Channels that must be *constant*, not merely weak. Each was a live leak in
#: the raw data; a constant is the only durable fix.
CONSTANT_CHANNELS = [
    "container_is_png", "width", "height", "megapixels", "is_square",
    "is_1024sq", "has_icc", "n_exif_tags", "n_text_chunks", "mode_is_rgb",
    "has_adobe_marker", "chroma_subsampling",
]

#: Content-derived, so it cannot be constant. Still bounded.
CONTENT_CHANNELS = ["n_bytes", "bytes_per_pixel"]

pytestmark = pytest.mark.skipif(
    not DETAIL.exists(),
    reason="no manifest; run scripts/download_data.py then python -m src.data.manifest",
)


def _probe_one(path: str) -> dict:
    """Every metadata channel obtainable without looking at the picture."""
    from PIL.JpegImagePlugin import get_sampling

    with Image.open(path) as im:
        info = im.info or {}
        try:
            n_exif = len(im.getexif())
        except Exception:
            n_exif = 0
        try:
            sub = get_sampling(im)
        except Exception:
            sub = -1
        w, h = im.size
        fmt = im.format or ""
        mode = im.mode
        # PNG tEXt/iTXt and stray string keys. JFIF density fields are written
        # by every Pillow JPEG encode and are identical for all of them, so
        # they are excluded rather than counted as nine spurious "text chunks".
        boring = {"jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi",
                  "progression", "progressive", "icc_profile", "exif"}
        n_text = sum(1 for k, v in info.items()
                     if k not in boring and isinstance(v, (str, bytes)))
        has_adobe = int("adobe" in info)
        has_icc = int(bool(info.get("icc_profile")))

    size = os.path.getsize(path)
    return {
        "container_is_png": int(fmt == "PNG"),
        "width": w, "height": h,
        "megapixels": w * h / 1e6,
        "is_square": int(w == h),
        "is_1024sq": int(w == 1024 and h == 1024),
        "has_icc": has_icc,
        "n_exif_tags": n_exif,
        "n_text_chunks": n_text,
        "mode_is_rgb": int(mode == "RGB"),
        "has_adobe_marker": has_adobe,
        "chroma_subsampling": sub,
        "n_bytes": size,
        "bytes_per_pixel": size / max(w * h, 1),
    }


@pytest.fixture(scope="module")
def probe() -> pd.DataFrame:
    d = pd.read_csv(DETAIL)
    rows = [_probe_one(p) for p in d["image_path"]]
    f = pd.DataFrame(rows)
    f["label"] = d["label"].values
    f["generator"] = d["generator"].values
    f["split"] = d["split"].values
    return f


def _auroc(y: np.ndarray, x: np.ndarray) -> float:
    """Direction-agnostic. A feature that predicts *real* leaks just as badly."""
    if np.std(x) == 0:
        return 0.5
    a = roc_auc_score(y, x)
    return max(a, 1.0 - a)


# --------------------------------------------------------------------------- #
# The headline assertion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("channel", CONSTANT_CHANNELS + CONTENT_CHANNELS)
def test_channel_is_uninformative(probe, channel):
    y = (probe["label"] == 1).astype(int).values
    a = _auroc(y, probe[channel].astype(float).values)
    assert a < MAX_AUROC, (
        f"metadata channel {channel!r} scores AUROC {a:.4f} >= {MAX_AUROC}. "
        f"A classifier reading no pixels can separate the classes this well. "
        f"Per-generator medians:\n"
        f"{probe.groupby(['label', 'generator'])[channel].median().to_string()}"
    )


@pytest.mark.parametrize("channel", CONSTANT_CHANNELS)
def test_channel_is_actually_constant(probe, channel):
    """Stronger than the AUROC bound, and the reason the bound is safe.

    A constant cannot separate anything at any sample size. Measuring 0.5 on 300
    images only says the leak is smaller than the noise floor; measuring zero
    variance says there is nothing to leak.
    """
    vals = probe[channel].unique()
    assert len(vals) == 1, (
        f"{channel!r} takes {len(vals)} distinct values {sorted(vals)[:6]} after "
        "normalization; it was supposed to be pinned by the canonical decode path"
    )


def test_the_specific_leaks_the_audit_found_are_gone(probe):
    """Named against the audit's own numbers, so a regression is legible."""
    y = (probe["label"] == 1).astype(int).values
    was = {"container_is_png": 0.7574, "megapixels": 0.9814, "is_1024sq": 0.9814,
           "is_square": 0.9806, "height": 0.8966, "has_icc": 0.7027,
           "n_bytes": 0.6037}
    now = {k: _auroc(y, probe[k].astype(float).values) for k in was}
    bad = {k: (was[k], now[k]) for k in was if now[k] >= MAX_AUROC}
    assert not bad, f"leaks that the audit found and normalization did not fix: {bad}"


# --------------------------------------------------------------------------- #
# Source fingerprinting -- the failure mode this test actually caught
# --------------------------------------------------------------------------- #

def test_no_real_source_is_identifiable_by_file_size(probe):
    """Pexels was separable from every other real source at AUROC 1.0000.

    A source signature inside one class is not a class leak on its own, but it
    becomes one the moment that source carries a distinguishing property -- and
    Pexels was the *only* polished real source, so "tiny file" would have meant
    "polished and real", exactly the shortcut task G exists to remove.
    """
    real = probe[probe["label"] == 0]
    sources = sorted(real["generator"].unique())
    if len(sources) < 2:
        pytest.skip("only one real source in this corpus")
    worst = {}
    for s in sources:
        y = (real["generator"] == s).astype(int).values
        worst[s] = _auroc(y, real["n_bytes"].astype(float).values)
    flagged = {k: round(v, 4) for k, v in worst.items() if v >= 0.90}
    assert not flagged, (
        f"real source(s) identifiable by file size alone: {flagged}. "
        f"Medians:\n{real.groupby('generator')['n_bytes'].median().to_string()}"
    )


def test_no_generator_is_identifiable_by_file_size(probe):
    """Same check on the AI side: a per-generator signature would let the model
    memorise generators instead of learning generation."""
    ai = probe[probe["label"] == 1]
    gens = sorted(ai["generator"].unique())
    if len(gens) < 2:
        pytest.skip("only one generator in this corpus")
    flagged = {}
    for g in gens:
        y = (ai["generator"] == g).astype(int).values
        a = _auroc(y, ai["n_bytes"].astype(float).values)
        if a >= 0.95:
            flagged[g] = round(a, 4)
    assert not flagged, f"generator(s) identifiable by file size alone: {flagged}"


# --------------------------------------------------------------------------- #
# The probe must be capable of failing
# --------------------------------------------------------------------------- #

def test_the_probe_detects_a_leak_when_one_exists(probe):
    """A test that only ever passes proves nothing about its own sensitivity.

    Plant a synthetic channel perfectly correlated with the label and confirm
    the probe reports 1.0. If this ever drops, the AUROC plumbing is broken and
    every clean result above is meaningless.
    """
    y = (probe["label"] == 1).astype(int).values
    assert _auroc(y, y.astype(float)) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noisy = y + rng.normal(0, 0.25, len(y))
    assert _auroc(y, noisy) > 0.90


def test_corpus_is_big_enough_for_the_bound_to_mean_something(probe):
    """AUROC has a wide null distribution on small samples.

    At n_real = n_ai = 50 the null SD is ~0.047, so 0.60 is only ~2.1 sigma and
    a clean channel would breach it by chance about 3% of the time. Below ~100
    per class the assertion is weak, and that should be visible rather than
    implied.
    """
    n_real = int((probe["label"] == 0).sum())
    n_ai = int((probe["label"] == 1).sum())
    sd = np.sqrt((n_real + n_ai + 1) / (12.0 * n_real * n_ai))
    margin_sigma = (MAX_AUROC - 0.5) / sd
    assert n_real >= 50 and n_ai >= 50, (
        f"corpus too small to audit: {n_real} real, {n_ai} AI"
    )
    if margin_sigma < 3.0:
        pytest.skip(
            f"corpus is {n_real}+{n_ai}; the {MAX_AUROC} bound is only "
            f"{margin_sigma:.1f} sigma from the null. Assertions above still ran "
            "but are weak until the full corpus is downloaded."
        )
