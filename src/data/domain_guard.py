"""Flags images whose content looks like a rendered graphic rather than a
camera photograph -- diagrams, screenshots, infographics, text-heavy vector
art.

Why this exists
----------------
Every source in ``src/data/sources.py``, both classes, is photographic: the
three real sets (OpenImagesV7, Megalith-Flickr, Unsplash) are camera photos,
and all seven generators (SDXL, Mobius, RealVisXL, Aura, MidJourney, Gemini,
FLUX.1-dev) are photorealistic image generators. Nothing in training, real or
AI, is a diagram, screenshot, or vector graphic. Feeding the CLIP linear probe
one of those is asking a question it was never trained to answer -- and
observed behaviour is that it defaults toward "AI-generated" on this input,
consistent with ``results/error_analysis/note.md``'s finding that the model's
signal correlates with "does this look like a certain kind of photo" rather
than generation artifacts directly: a rendered diagram is the extreme case of
everything that already pushes a *real* photo toward that class (flat color,
no sensor noise, deliberately crisp edges), on top of being completely absent
from the "real" class it's being compared against.

This module does not try to correct the model's score on such input -- there
is no training signal that would make a corrected number meaningful. It flags
the input instead, so a caller (``predict.py``, ``app.py``) can surface "this
score is unreliable for this content type" rather than a confident label.

Method: two cheap global pixel statistics, no ML, no training data of its own.
  - ``unique_ratio``: distinct (quantized) colors / pixel count. A camera
    photo carries sensor noise and continuous gradients, so even a small crop
    has thousands of unique colors. A rendered graphic is built from flat-
    filled regions -- typically a few dozen to a few hundred unique colors.
  - ``dominant_fraction``: the single most common color's share of all
    pixels. Diagrams and screenshots are usually majority background (white,
    or a UI chrome color); a photograph rarely has any one color cover much
    of the frame.

Thresholds are deliberately conservative -- tuned to rarely fire on real
photographs -- since this is an advisory flag, not a gate that suppresses or
overrides a score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = ["DomainCheck", "check_domain"]

#: Downsize before scanning colors -- result is resolution-invariant enough
#: past this, and it keeps the check cheap on large images.
_MAX_SIDE = 256
#: Below this fraction of unique (quantized) colors, content is almost always
#: flat-filled/rendered rather than photographic. Calibrated loosely: a noisy
#: photographic crop lands well above 0.2, a rendered diagram well below 0.05.
_UNIQUE_RATIO_THRESHOLD = 0.06
#: Above this share of pixels being one single color, content is almost
#: always a large flat background (paper white, UI chrome) rather than a photo.
_DOMINANT_FRACTION_THRESHOLD = 0.35


@dataclass(frozen=True)
class DomainCheck:
    """Result of :func:`check_domain`. Advisory only -- see module docstring."""

    unique_ratio: float
    dominant_fraction: float
    likely_non_photographic: bool


def check_domain(image: Image.Image) -> DomainCheck:
    """Cheap heuristic: does ``image`` look like a camera photograph?

    Not a classifier and not trained on anything -- see module docstring for
    why this exists and what it can't do. Accepts any PIL image (any mode);
    converts to RGB itself, so it can run on the same already-decoded image a
    :class:`~src.models.base.Scorer` scores, no extra decode needed.
    """
    img = image.convert("RGB")
    if max(img.size) > _MAX_SIDE:
        img = img.copy()
        img.thumbnail((_MAX_SIDE, _MAX_SIDE))
    arr = np.asarray(img, dtype=np.uint8).reshape(-1, 3)

    # Quantize to 5 bits/channel (32 levels) before counting: a real photo's
    # sensor noise and a JPEG's ringing around otherwise-flat regions
    # shouldn't inflate the unique-color count into looking photographic, but
    # actual photographic gradients/noise should still dwarf this bucket size.
    quantized = (arr >> 3).astype(np.int32)
    packed = (quantized[:, 0] << 10) | (quantized[:, 1] << 5) | quantized[:, 2]
    _, counts = np.unique(packed, return_counts=True)

    n_pixels = packed.shape[0]
    unique_ratio = float(len(counts) / n_pixels)
    dominant_fraction = float(counts.max() / n_pixels)
    flag = bool(
        unique_ratio < _UNIQUE_RATIO_THRESHOLD
        or dominant_fraction > _DOMINANT_FRACTION_THRESHOLD
    )
    return DomainCheck(
        unique_ratio=unique_ratio,
        dominant_fraction=dominant_fraction,
        likely_non_photographic=flag,
    )
