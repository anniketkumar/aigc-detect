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

# --------------------------------------------------------------------------- #
# Precomputed, image-independent constants
# --------------------------------------------------------------------------- #
# Both the radial-ring geometry and the DCT band assignment depend only on
# _SIZE/_N_RINGS/_DCT_BLOCK, all fixed -- so they're built once here instead
# of being recomputed (mgrid, sqrt, per-ring boolean masks, ...) on every
# single image. This is a units-of-work fix, not a numerics change: outputs
# match the original per-image-recomputed version to float32 precision
# (see tests/test_frequency_features.py's equivalence checks and the
# ad-hoc benchmark in NOTES.md -- ~2.4x faster end to end, no GPU needed
# because the bottleneck was Python-loop overhead, not raw FLOPs).

_h = _w = _SIZE
_cy, _cx = _h / 2.0, _w / 2.0
_yy, _xx = np.mgrid[0:_h, 0:_w]
_R = np.sqrt((_yy - _cy) ** 2 + (_xx - _cx) ** 2)
_R_MAX = float(min(_cy, _cx))
# _N_RINGS+2 edges: bin 0 (edges[0]..edges[1]) is the DC-dominated innermost
# ring and is deliberately excluded from _RING_IDX (-1) -- it tracks average
# brightness, not a generator fingerprint.
_EDGES = np.linspace(0.0, _R_MAX, _N_RINGS + 2)
_RING_IDX = np.full(_R.shape, -1, dtype=np.int16)
for _i in range(_N_RINGS):
    _RING_IDX[(_R >= _EDGES[_i + 1]) & (_R < _EDGES[_i + 2])] = _i
_OUTER_MASK = (_R >= _EDGES[-2]).ravel()
_RING_IDX_FLAT = _RING_IDX.ravel()
_RING_VALID = _RING_IDX_FLAT >= 0
_RING_IDX_VALID = _RING_IDX_FLAT[_RING_VALID]
_RING_COUNTS = np.bincount(_RING_IDX_VALID, minlength=_N_RINGS).astype(np.float32)
del _h, _w, _cy, _cx, _yy, _xx, _R, _R_MAX, _EDGES, _i, _RING_IDX, _RING_IDX_FLAT

# Orthonormal DCT-II basis, analytic (matches cv2.dct to ~1e-7, float32) --
# 2D block DCT is separable: dct2d(block) == _DCT_BASIS @ block @ _DCT_BASIS.T
_dct_n = np.arange(_DCT_BLOCK)
_dct_k = _dct_n.reshape(-1, 1)
_DCT_BASIS = (
    np.sqrt(2.0 / _DCT_BLOCK) * np.cos(np.pi * (2 * _dct_n + 1) * _dct_k / (2 * _DCT_BLOCK))
).astype(np.float32)
_DCT_BASIS[0, :] *= 1.0 / np.sqrt(2.0)
del _dct_n, _dct_k

_dct_ii, _dct_jj = np.mgrid[0:_DCT_BLOCK, 0:_DCT_BLOCK]
_dct_band = _dct_ii + _dct_jj  # 0 = DC; up to 14 = highest-frequency corner
_DCT_LOW_MASK = (_dct_band > 0) & (_dct_band <= 4)
_DCT_MID_MASK = (_dct_band > 4) & (_dct_band <= 9)
_DCT_HIGH_MASK = _dct_band > 9
del _dct_ii, _dct_jj, _dct_band


def _to_gray(img: Image.Image) -> np.ndarray:
    g = img.convert("L").resize((_SIZE, _SIZE), Image.Resampling.BICUBIC)
    return np.asarray(g, dtype=np.float32)


def _radial_fft_features(gray: np.ndarray) -> np.ndarray:
    spec = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(spec)).ravel()

    # Ring-averaged magnitude via bincount against the precomputed ring
    # index -- one pass over the flattened spectrum instead of _N_RINGS
    # separate boolean-mask passes.
    sums = np.bincount(_RING_IDX_VALID, weights=mag[_RING_VALID], minlength=_N_RINGS)
    rings = np.divide(
        sums, _RING_COUNTS, out=np.zeros(_N_RINGS, dtype=np.float32),
        where=_RING_COUNTS > 0,
    ).astype(np.float32)

    total = float(mag.sum())
    hf_ratio = float(mag[_OUTER_MASK].sum() / total) if total > 0 else 0.0

    # peakiness: how far the strongest ring stands out from the ring-to-ring
    # mean, in ring-std units. A smoothly decaying spectrum (real camera
    # photo) has low peakiness; a periodic up-sampling artifact concentrates
    # energy in one narrow ring and spikes this number.
    ring_mean, ring_std = float(rings.mean()), float(rings.std())
    peakiness = float((rings.max() - ring_mean) / ring_std) if ring_std > 1e-6 else 0.0

    return np.concatenate([rings, [hf_ratio, peakiness]]).astype(np.float32)


def _dct_band_features(gray: np.ndarray) -> np.ndarray:
    """Mean |DCT| energy in low/mid/high triangular bands, averaged over
    every non-overlapping 8x8 block -- the same block grid JPEG uses.

    Vectorized: one batched matrix multiply against the precomputed DCT-II
    basis for all blocks at once, instead of one cv2.dct() Python call per
    block (~1024 blocks at 256x256 -- that loop's interpreter overhead, not
    FLOPs, was the actual cost; see the module's precomputed-constants note).
    """
    h, w = gray.shape
    h8, w8 = (h // _DCT_BLOCK) * _DCT_BLOCK, (w // _DCT_BLOCK) * _DCT_BLOCK
    g = gray[:h8, :w8]
    n_by, n_bx = h8 // _DCT_BLOCK, w8 // _DCT_BLOCK
    blocks = (
        g.reshape(n_by, _DCT_BLOCK, n_bx, _DCT_BLOCK)
        .transpose(0, 2, 1, 3)
        .reshape(-1, _DCT_BLOCK, _DCT_BLOCK)
    )
    dct = _DCT_BASIS @ blocks @ _DCT_BASIS.T  # (n_blocks, 8, 8), broadcast matmul
    mean_energy = np.abs(dct).mean(axis=0)  # (8, 8), index (0,0) = DC

    low = float(mean_energy[_DCT_LOW_MASK].mean())
    mid = float(mean_energy[_DCT_MID_MASK].mean())
    high = float(mean_energy[_DCT_HIGH_MASK].mean())
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
