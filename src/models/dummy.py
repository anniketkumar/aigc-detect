"""Reference scorers used to validate the harness itself (PLAN.md §3.3).

Neither of these is a detector. They exist so that the harness can be shown to
be correct before there is anything real to measure.

:class:`RandomScorer`
    The §3.3 acceptance model. Scores from a hash of the *pixels*, not of the
    filename. That matters: a filename-keyed random model would return the same
    score in every cell, so the grid would come out with 19 identical AUROCs and
    a bug that fed the clean image to every cell would be invisible. Hashing
    pixels means each cell gets its own independent draw, and identical AUROCs
    across cells become a signal that something is wrong.

:class:`BrightnessScorer`
    Mean luminance. Also not a detector, but unlike the random model it has a
    real, predictable sensitivity to the grid (jitter and noise move it, JPEG
    barely does), so it produces a non-zero robustness_gap. That makes it the
    end-to-end check that the harness can *detect* a robustness difference
    rather than merely compute a number.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from PIL import Image

__all__ = ["RandomScorer", "BrightnessScorer"]


def _pixel_digest(img: Image.Image) -> bytes:
    """Hash of the exact pixel buffer, plus size and mode."""
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{img.mode}:{img.size}".encode())
    h.update(np.ascontiguousarray(np.asarray(img)).tobytes())
    return h.digest()


class RandomScorer:
    """Uniform random score in [0, 1), deterministic in (pixels, seed).

    Expected behaviour on the grid: AUROC ~ 0.5 in every cell, fluctuating by
    about ``metrics.auroc_null_sd(n_fake, n_real)``; AP ~ class prevalence;
    acc@0.5 ~ 0.5. Same input, same output, run to run and machine to machine.
    """

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.name = f"dummy_random(seed={self.seed})"

    def score(
        self, images: Sequence[Image.Image], image_ids: Sequence[str]
    ) -> list[float | None]:
        out: list[float | None] = []
        for img in images:
            h = hashlib.blake2b(
                self.seed.to_bytes(8, "little") + _pixel_digest(img), digest_size=8
            )
            # 53-bit mantissa -> uniform in [0, 1) without modulo bias
            u = int.from_bytes(h.digest(), "little") >> 11
            out.append(u / float(1 << 53))
        return out


class BrightnessScorer:
    """Score = mean luminance, rescaled to [0, 1]. A deliberate weak baseline."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.name = "dummy_brightness"

    def score(
        self, images: Sequence[Image.Image], image_ids: Sequence[str]
    ) -> list[float | None]:
        out: list[float | None] = []
        for img in images:
            x = np.asarray(img.convert("L"), dtype=np.float32)
            out.append(float(x.mean() / 255.0))
        return out
