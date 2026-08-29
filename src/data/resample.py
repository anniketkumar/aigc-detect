"""Randomised resample-filter hook for Phase 4 training augmentation.

Not imported by the eval grid, and there is a test that says so.

Why randomise the filter at all
-------------------------------
``src/transforms.py`` pins every eval-time rescale to BICUBIC, for the reason
recorded in NOTES.md: a fixed, neutral filter that neither destroys nor
manufactures high-frequency structure. That is right for *measurement* -- one
filter, so the grid measures degradation and not filter choice.

It is wrong for *training*. A model trained only on bicubic rescales learns
bicubic's specific interpolation kernel as part of what "a resized image" looks
like. Each filter leaves a distinguishable spectral fingerprint -- LANCZOS rings
and lifts high frequencies, BOX flattens them, BILINEAR sits between -- and the
Phase 5 artifact branch reads exactly that band. Train on one filter and the
artifact branch overfits to it; the eval grid, which uses that same filter,
would never reveal the overfit. Randomising the filter during training and
holding it fixed at eval is what makes the eval an honest out-of-sample test of
the resampling axis.

So the asymmetry is deliberate and load-bearing:

    training  -> filter sampled per call from FILTERS
    eval      -> transforms.RESAMPLE, always BICUBIC

``area`` maps to PIL ``BOX``. For downscaling these are the same operation --
average over the source footprint -- and BOX is the name PIL uses; OpenCV calls
it INTER_AREA. It is included because it is what most real thumbnailers use.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

__all__ = ["FILTERS", "FILTER_NAMES", "sample_filter", "resize_rand", "center_crop_rand"]

#: The four filters, by the names the spec uses.
FILTERS: dict[str, Image.Resampling] = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "area": Image.Resampling.BOX,
}

FILTER_NAMES: tuple[str, ...] = tuple(FILTERS)


def sample_filter(
    rng: np.random.Generator,
    names: Sequence[str] | None = None,
) -> tuple[str, Image.Resampling]:
    """Draw one filter uniformly. Returns ``(name, enum)``.

    The name comes back too because Phase 4 hands the applied-transform vector
    to the degradation head as a free label, and "which filter" is part of that
    vector. Returning only the enum would mean recovering the name by reverse
    lookup at every call site.

    Takes an explicit ``rng`` rather than touching the global RNG, matching
    ``src/transforms.py``: reproducibility here is per-(image, epoch), and a
    global stream would make a run depend on dataloader worker count.
    """
    pool = FILTER_NAMES if names is None else tuple(names)
    unknown = set(pool) - set(FILTERS)
    if unknown:
        raise KeyError(f"unknown resample filter(s): {sorted(unknown)}")
    if not pool:
        raise ValueError("filter pool is empty")
    name = pool[int(rng.integers(len(pool)))]
    return name, FILTERS[name]


def resize_rand(
    img: Image.Image,
    scale: float,
    rng: np.random.Generator,
    *,
    names: Sequence[str] | None = None,
    same_filter_both_ways: bool = False,
) -> tuple[Image.Image, dict]:
    """Training-time ``resize``: downscale to ``scale x``, upscale back.

    ``same_filter_both_ways=False`` by default -- a real redistribution chain is
    a thumbnailer downscaling and, later, a different viewer or CDN upscaling.
    Forcing one filter for both legs would model a case that rarely occurs and
    would halve the diversity of the augmentation.

    Returns the image and the applied-parameter dict that Phase 4 forwards to
    the degradation head.
    """
    if not 0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    down_name, down = sample_filter(rng, names)
    up_name, up = (down_name, down) if same_filter_both_ways else sample_filter(rng, names)

    w, h = img.size
    dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    out = img.resize((dw, dh), down).resize((w, h), up)
    return out, {"op": "resize", "scale": float(scale),
                 "filter_down": down_name, "filter_up": up_name}


def center_crop_rand(
    img: Image.Image,
    keep: float,
    rng: np.random.Generator,
    *,
    names: Sequence[str] | None = None,
) -> tuple[Image.Image, dict]:
    """Training-time ``center_crop``: keep ``keep`` of each side, resize back.

    Linear fraction, matching the eval definition in ``transforms.t_center_crop``
    -- the augmentation must cover the eval cell, so the two must agree on what
    "0.8" means even though they disagree on the filter.
    """
    if not 0 < keep <= 1.0:
        raise ValueError(f"keep must be in (0, 1], got {keep}")
    name, filt = sample_filter(rng, names)
    w, h = img.size
    cw, ch = max(1, int(round(w * keep))), max(1, int(round(h * keep)))
    left, top = (w - cw) // 2, (h - ch) // 2
    out = img.crop((left, top, left + cw, top + ch)).resize((w, h), filt)
    return out, {"op": "center_crop", "keep": float(keep), "filter": name}
