"""Deterministic frequency-domain features for the fusion branch (Phase 5,
resumed after being cut in Phase 3/4 for budget -- see NOTES.md).

Why this exists, precisely
---------------------------
``results/aesthetic_probe/aesthetic_probe.md`` already answered the question
that decides whether a fusion branch is worth building at all: a two-feature,
no-CLIP texture probe (spatial-gradient magnitude + flat-pixel fraction) has
a per-generator difficulty ordering with Spearman rho = -0.143 (p = 0.760)
against the CLIP head's ordering -- statistically indistinguishable from
zero correlation. CLIP and that probe are not finding the same generators
hard, i.e. they carry different information. That is the actual precondition
for concatenation-fusion to add signal instead of duplicating what the CLIP
branch already reads (DEVPOST.md, Innovation section).

That probe used spatial gradients as a *proxy* for "high-frequency content."
This module is the real thing the workshop names explicitly: "Frequency
artifacts: GAN/diffusion up-sampling leaves periodic patterns in the Fourier
spectrum that cameras don't produce." A periodic up-sampling artifact shows
up as one or two unusually strong *rings* in the radially-averaged FFT
magnitude spectrum -- a real photograph's spectrum falls off smoothly with
frequency and has no such peak. The radial-bin features below are built to
read exactly that shape, not just "how much high-frequency energy is there."

Contract
--------
``extract_frequency_features(img)`` is a pure function of RGB pixels, no
learned parameters, no fit-on-data normalization step (deliberately: a
scaler fit on train-split statistics is one more thing that could leak
train/test coupling into the pipeline this project has otherwise been
paranoid about auditing). Scaling is instead a handful of fixed constants in
``_SCALE``, chosen once by inspecting real values on ``data/corpus_smoke``
(``scripts/calibrate_frequency_scale.py``) and frozen here -- the same spirit
as picking BICUBIC resampling in ``src/transforms.py`` because it is what
real pipelines use, not because it was tuned against a metric.

Called at both feature-caching time (``scripts/cache_features.py
--fuse-freq``) and inference time (``src/models/clip_fusion.py``), always on
post-transform pixels -- same Scorer contract as ``ClipBackbone.embed``: this
module never sees an image before the eval grid's degradations are applied,
which is the entire point of measuring whether a frequency branch survives
JPEG/blur/resize or collapses under it.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

__all__ = ["FREQ_FEATURE_NAMES", "FREQ_DIM", "extract_frequency_features"]

_SIZE = 256          # fixed working resolution -- makes radial-bin geometry
                      # comparable across images of any native size, same
                      # reason CLIP's own preprocess resizes to a fixed input.
_N_RINGS = 12         # radial FFT bins, innermost (DC-dominated) ring excluded
_DCT_BLOCK = 8        # JPEG's own block size -- reads the structure a real
                      # JPEG re-encode is most likely to disturb or reveal

FREQ_FEATURE_NAMES = (
    [f"fft_ring_{i}" for i in range(_N_RINGS)]
    + ["fft_hf_ratio", "fft_ring_peakiness"]
    + ["dct_low", "dct_mid", "dct_high"]
    + ["hf_energy", "flat_frac"]
)
FREQ_DIM = len(FREQ_FEATURE_NAMES)

# Fixed per-feature scale divisors -- chosen once from the *magnitude* of
# each raw feature on an 80-image, label-blind sample of data/corpus_smoke
# (scripts/calibrate_frequency_scale.py; no class information used, so this
# is a units fix, not anything fit against the classification target), and
# frozen here. Purpose: land typical raw values near O(1) so a linear head's
# weight_decay treats this branch comparably to CLIP's L2-normalized
# embedding, without fitting a statistic on the data this model is actually
# trained/evaluated on. See the module docstring for why a data-fit scaler
# (e.g. sklearn StandardScaler on train-split stats) was deliberately avoided.
_SCALE = np.array(
    [8.0] * _N_RINGS        # fft_ring_0..11: log1p FFT magnitude ~6-10, ring-averaged
    + [0.3, 3.0]             # fft_hf_ratio (~0.25-0.31 fraction), fft_ring_peakiness (~1.8-2.4)
    + [15.0, 4.0, 2.0]       # dct_low (~14), dct_mid (~3.5), dct_high (~1.6)
    + [12.0, 1.0],           # hf_energy (~10, gradient magnitude), flat_frac (already 0..1)
    dtype=np.float32,
)
assert _SCALE.shape == (FREQ_DIM,)


def _to_gray(img: Image.Image) -> np.ndarray:
    g = img.convert("L").resize((_SIZE, _SIZE), Image.Resampling.BICUBIC)
    return np.asarray(g, dtype=np.float32)


def _radial_fft_features(gray: np.ndarray) -> np.ndarray:
    spec = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(spec))
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(min(cy, cx))

    # _N_RINGS+2 edges: bin 0 (edges[0]..edges[1]) is the DC-dominated
    # innermost ring and is deliberately skipped -- it tracks average
    # brightness, not a generator fingerprint.
    edges = np.linspace(0.0, r_max, _N_RINGS + 2)
    rings = np.zeros(_N_RINGS, dtype=np.float32)
    for i in range(_N_RINGS):
        lo, hi = edges[i + 1], edges[i + 2]
        mask = (r >= lo) & (r < hi)
        rings[i] = float(mag[mask].mean()) if mask.any() else 0.0

    total = float(mag.sum())
    outer_mask = r >= edges[-2]
    hf_ratio = float(mag[outer_mask].sum() / total) if total > 0 else 0.0

    # peakiness: how far the strongest ring stands out from the ring-to-ring
    # mean, in ring-std units. A smoothly decaying spectrum (real camera
    # photo) has low peakiness; a periodic up-sampling artifact concentrates
    # energy in one narrow ring and spikes this number.
    ring_mean, ring_std = float(rings.mean()), float(rings.std())
    peakiness = float((rings.max() - ring_mean) / ring_std) if ring_std > 1e-6 else 0.0

    return np.concatenate([rings, [hf_ratio, peakiness]]).astype(np.float32)


def _dct_band_features(gray: np.ndarray) -> np.ndarray:
    """Mean |DCT| energy in low/mid/high triangular bands, averaged over
    every non-overlapping 8x8 block -- the same block grid JPEG uses."""
    h, w = gray.shape
    h8, w8 = (h // _DCT_BLOCK) * _DCT_BLOCK, (w // _DCT_BLOCK) * _DCT_BLOCK
    g = gray[:h8, :w8]
    n_by, n_bx = h8 // _DCT_BLOCK, w8 // _DCT_BLOCK
    blocks = (
        g.reshape(n_by, _DCT_BLOCK, n_bx, _DCT_BLOCK)
        .transpose(0, 2, 1, 3)
        .reshape(-1, _DCT_BLOCK, _DCT_BLOCK)
    )
    energies = np.empty((blocks.shape[0], _DCT_BLOCK, _DCT_BLOCK), dtype=np.float32)
    for i in range(blocks.shape[0]):
        energies[i] = np.abs(cv2.dct(blocks[i]))
    mean_energy = energies.mean(axis=0)  # (8, 8), index (0,0) = DC

    ii, jj = np.mgrid[0:_DCT_BLOCK, 0:_DCT_BLOCK]
    band = ii + jj  # 0 = DC; up to 14 = highest-frequency corner
    low = float(mean_energy[(band > 0) & (band <= 4)].mean())
    mid = float(mean_energy[(band > 4) & (band <= 9)].mean())
    high = float(mean_energy[band > 9].mean())
    return np.array([low, mid, high], dtype=np.float32)


def _spatial_gradient_features(gray: np.ndarray) -> np.ndarray:
    """``hf_energy`` / ``flat_frac`` exactly as
    ``scripts/audit_leakage.content_features`` and ``scripts/aesthetic_probe``
    compute them -- kept because the aesthetic probe already measured this
    pair as near-uncorrelated with CLIP's difficulty ordering (module
    docstring above). Recomputed rather than imported so ``src/`` does not
    depend on ``scripts/``."""
    d = np.abs(np.diff(gray, axis=1))
    gx = float(d.mean()) if gray.shape[1] > 1 else 0.0
    gy = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
    hf_energy = gx + gy
    flat_frac = float((d < 1.0).mean())
    return np.array([hf_energy, flat_frac], dtype=np.float32)


def extract_frequency_features(img: Image.Image) -> np.ndarray:
    """RGB PIL image -> ``(FREQ_DIM,)`` float32, pre-scaled by the fixed
    ``_SCALE`` constants (module docstring). Pure function of pixels: no
    learned state, nothing fit on the split being processed."""
    gray = _to_gray(img)
    raw = np.concatenate([
        _radial_fft_features(gray),
        _dct_band_features(gray),
        _spatial_gradient_features(gray),
    ])
    return (raw / _SCALE).astype(np.float32)
