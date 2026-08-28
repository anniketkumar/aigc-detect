"""Generate a synthetic labelled image set + manifest, for testing the harness.

This is a harness fixture, not data (Phase 2 owns real data). It exists so the
§3.3 acceptance check can run before a single byte has been downloaded.

Default mode is ``--signal none``: image content is drawn from a seed that is
independent of the label, so the fixture contains **no** real/fake signal at
all. Any model scoring meaningfully above AUROC 0.5 on it is either cheating
(reading filenames or labels) or the harness is leaking labels into the scores.
That makes it a null test for the harness, not just for the dummy model.

``--signal brightness`` plants a small brightness offset in the fake class. A
brightness-reading model then gets a real clean AUROC that degrades under
jitter and noise, which is how you check that the grid can *detect* a
robustness difference rather than merely compute numbers.

Usage:
    python -m scripts.make_dummy_fixture --out data/fixtures/dummy --n 400
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

#: A slice of the fixture is written in awkward formats. The transform grid is
#: the first thing that touches raw files, so it should be proven against these
#: here rather than in Phase 7 when predict.py meets a real directory.
ODD_MODES = ["RGBA", "L", "CMYK"]


def _texture(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Band-limited random texture: smooth structure plus fine grain.

    Flat or purely-white-noise images would make several transforms
    indistinguishable (blur does nothing visible to a flat patch), so the
    fixture deliberately carries energy at both low and high frequencies.
    """
    lo_h, lo_w = max(2, h // 16), max(2, w // 16)
    low = rng.normal(0, 1, size=(lo_h, lo_w, 3))
    low = np.asarray(
        Image.fromarray(((low - low.min()) / np.ptp(low) * 255).astype(np.uint8))
        .resize((w, h), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    edges = np.zeros((h, w, 3), dtype=np.float32)
    for _ in range(6):  # a few hard edges, so JPEG has something to ring on
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = min(w, x0 + rng.integers(4, max(5, w // 3))), min(
            h, y0 + rng.integers(4, max(5, h // 3))
        )
        edges[y0:y1, x0:x1] += rng.uniform(-60, 60, size=3)
    grain = rng.normal(0, 8, size=(h, w, 3))
    return np.clip(low * 0.7 + edges + grain + 40, 0, 255)


def build(
    out_dir: Path,
    n: int = 400,
    size: int = 128,
    seed: int = 0,
    signal: str = "none",
    signal_strength: float = 12.0,
    odd_fraction: float = 0.1,
    split: str = "test",
) -> Path:
    """Write ``n`` images plus ``manifest.csv``; returns the manifest path."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(labels)

    rows = []
    for i, label in enumerate(labels):
        # Content seed is derived from the index only, never the label, so under
        # --signal none the two classes are drawn from the identical distribution.
        content_rng = np.random.default_rng(seed * 1_000_003 + i)
        arr = _texture(content_rng, size, size)
        if signal == "brightness" and label == 1:
            arr = np.clip(arr + signal_strength, 0, 255)
        elif signal not in ("none", "brightness"):
            raise ValueError(f"unknown --signal {signal!r}")

        img = Image.fromarray(arr.astype(np.uint8), "RGB")

        # a deterministic slice gets an awkward mode / format
        odd = ODD_MODES[i % len(ODD_MODES)] if (i % max(1, int(1 / odd_fraction)) == 0
                                               and odd_fraction > 0) else None
        # The filename must NOT encode the label. image_id seeds the stochastic
        # cells and is handed to Scorer.score(), so a label in the path is a leak
        # channel that would quietly invalidate the null test.
        if odd == "RGBA":
            rgba = img.convert("RGBA")
            alpha = np.full((size, size), 255, dtype=np.uint8)
            alpha[: size // 4, : size // 4] = 0  # a genuinely transparent corner
            rgba.putalpha(Image.fromarray(alpha))
            path = img_dir / f"{i:05d}.png"
            rgba.save(path)
        elif odd == "L":
            path = img_dir / f"{i:05d}.png"
            img.convert("L").save(path)
        elif odd == "CMYK":
            path = img_dir / f"{i:05d}.jpg"
            img.convert("CMYK").save(path, format="JPEG", quality=95)
        else:
            path = img_dir / f"{i:05d}.png"
            img.save(path)

        rows.append(
            {
                "image_path": path.as_posix(),
                "label": int(label),
                "generator": "synthetic_fake" if label == 1 else "synthetic_real",
                "source_dataset": f"dummy_fixture(signal={signal})",
                "split": split,
            }
        )

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["image_path", "label", "generator", "source_dataset", "split"]
        )
        w.writeheader()
        w.writerows(rows)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/fixtures/dummy"))
    p.add_argument("--n", type=int, default=400,
                   help="total images, split evenly between classes. 400 gives an "
                        "AUROC null SD of ~0.029, so a random model's cells land "
                        "inside +/-0.09 of 0.5 at 3 sigma.")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--signal", choices=["none", "brightness"], default="none")
    p.add_argument("--signal-strength", type=float, default=12.0)
    p.add_argument("--odd-fraction", type=float, default=0.1,
                   help="fraction written as RGBA / grayscale / CMYK")
    p.add_argument("--split", default="test")
    a = p.parse_args(argv)

    manifest = build(
        a.out, n=a.n, size=a.size, seed=a.seed, signal=a.signal,
        signal_strength=a.signal_strength, odd_fraction=a.odd_fraction, split=a.split,
    )
    print(f"wrote {a.n} images and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
