"""Reproduces the ``_SCALE`` constants hardcoded in
``src/features/frequency.py``. Not run automatically -- a one-off calibration
utility, kept so those numbers are a regeneratable measurement rather than a
number transcribed once and left unverifiable.

    python -m scripts.calibrate_frequency_scale

Deliberately label-blind: it prints raw per-feature mean/std/min/max over a
random image sample without ever reading ``label``, so nothing about this
script could leak class information into the scale constants -- they are a
units fix (get every feature to roughly the same order of magnitude), not
anything tuned against the classification target.
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.features.frequency import (
    FREQ_FEATURE_NAMES,
    _dct_band_features,
    _radial_fft_features,
    _spatial_gradient_features,
    _to_gray,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glob", default="data/corpus_smoke/images/**/*.jpg")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    paths = glob.glob(a.glob, recursive=True)
    if not paths:
        raise SystemExit(f"no images matched {a.glob!r}")
    rng = random.Random(a.seed)
    rng.shuffle(paths)
    paths = paths[: a.n]

    rows = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        gray = _to_gray(img)
        rows.append(np.concatenate([
            _radial_fft_features(gray),
            _dct_band_features(gray),
            _spatial_gradient_features(gray),
        ]))

    arr = np.stack(rows)
    print(f"n images: {arr.shape[0]} (of {len(paths)} sampled)")
    print(f"{'feature':20s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
    for i, name in enumerate(FREQ_FEATURE_NAMES):
        col = arr[:, i]
        print(f"{name:20s} {col.mean():9.3f} {col.std():9.3f} {col.min():9.3f} {col.max():9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
