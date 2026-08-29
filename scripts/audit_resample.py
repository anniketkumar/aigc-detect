"""Re-sample SID_Set *across the whole dataset*, to check the Phase 2 audit.

Why this exists
---------------
``fetch_audit_sample.py sid`` loops ``for shard in range(shards): for rg in
range(row_groups)`` -- shards 0..N-1, row groups 0..M-1. The audit it fed
therefore read **validation shards 0-5, row groups 0-2**: 1800 images out of
~240k, all from the head of one split, with zero train shards touched. Parquet
shards are commonly written in class or source order, so every finding in
``results/audit_sid_set.md`` could be an ordering artifact. This module re-draws
the sample uniformly and reports whether the findings survive.

Two passes, because the two kinds of finding cost very different amounts:

``geometry``
    Uses the HF datasets-server ``/rows`` endpoint, which serves arbitrary row
    offsets as JSON. ``width``/``height``/``label`` for 20k rows spread over the
    entire split costs a few MB and a couple of minutes. This settles the
    ``is_1024sq`` = 0.98 AUROC finding at ~10% coverage rather than 0.75%.

``bytes``
    Container format, ICC and EXIF need the *original* encoded bytes, which only
    exist inside the parquet. Parquet's unit of retrieval is a row-group column
    chunk (~55 MB, ~100 images), so this pass is bandwidth-bound and stays
    small: random shards across the full index range, one random row group in
    each. Breadth over depth -- 12 row groups from 12 scattered shards tells you
    far more about ordering bias than 12 row groups from shard 0.

The images the ``bytes`` pass writes are the exact parquet bytes, never a PIL
round-trip, because a re-encode would strip the very metadata under audit and
return a clean bill of health.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SID_REPO = "saberzl/SID_Set"
SHARDS = {"train": 249, "validation": 34}
ROWS_URL = "https://datasets-server.huggingface.co/rows"
SIZE_URL = "https://datasets-server.huggingface.co/size"
MAX_ROWS_PER_REQ = 100

SID_LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}


# --------------------------------------------------------------------------- #
# Pass 1 -- geometry, via the rows API
# --------------------------------------------------------------------------- #

def split_num_rows(split: str, timeout: int = 60) -> int:
    r = requests.get(SIZE_URL, params={"dataset": SID_REPO}, timeout=timeout)
    r.raise_for_status()
    for s in r.json()["size"]["splits"]:
        if s["split"] == split:
            return int(s["num_rows"])
    raise KeyError(f"split {split!r} not reported by the size endpoint")


def _fetch_rows(split: str, offset: int, length: int, attempts: int = 4) -> list[dict]:
    for a in range(attempts):
        try:
            r = requests.get(
                ROWS_URL,
                params={
                    "dataset": SID_REPO, "config": "default",
                    "split": split, "offset": offset, "length": length,
                },
                timeout=120,
            )
            if r.status_code == 429:          # the endpoint rate-limits; back off
                time.sleep(5 * (a + 1))
                continue
            r.raise_for_status()
            return r.json()["rows"]
        except Exception as exc:
            print(f"    retry {a+1}/{attempts} @off {offset}: {type(exc).__name__}",
                  file=sys.stderr, flush=True)
            time.sleep(3 * (a + 1))
    return []


def scan_geometry(
    split: str, n_batches: int, seed: int, out: Path, batch: int = MAX_ROWS_PER_REQ
) -> pd.DataFrame:
    """Random *blocks* of rows spread uniformly over the split.

    Blocks rather than individual rows: the endpoint charges per request, not
    per row, so 100 contiguous rows cost the same as one. Contiguity within a
    block is harmless for the question being asked -- the question is whether
    ``label`` and ``width`` co-vary with *position in the file*, and 200 blocks
    at 200 independent positions resolve that. Sampling 20k isolated rows would
    cost 20k requests to learn the same thing.
    """
    total = split_num_rows(split)
    rng = random.Random(seed)
    offsets = sorted(rng.sample(range(0, max(1, total - batch)), k=min(n_batches, total // batch)))
    print(f"{split}: {total} rows; {len(offsets)} blocks x {batch} "
          f"= {len(offsets)*batch} rows ({len(offsets)*batch/total:.1%})", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, off in enumerate(offsets):
        for r in _fetch_rows(split, off, batch):
            d = r["row"]
            rows.append({
                "split": split,
                "row_idx": r["row_idx"],
                "frac": r["row_idx"] / total,
                "shard_est": int(r["row_idx"] * SHARDS[split] / total),
                "img_id": d["img_id"],
                "width": int(d["width"]),
                "height": int(d["height"]),
                "label": int(d["label"]),
                "label_name": SID_LABEL_NAMES.get(int(d["label"]), "?"),
            })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(offsets)} blocks, {len(rows)} rows, "
                  f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"-> {out}  ({len(df)} rows in {time.time()-t0:.0f}s)")
    return df


# --------------------------------------------------------------------------- #
# Pass 2 -- original bytes, via random parquet row groups
# --------------------------------------------------------------------------- #

MAGIC = [
    (b"\xff\xd8\xff", "jpeg", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", "png", ".png"),
    (b"RIFF", "webp", ".webp"),
    (b"GIF8", "gif", ".gif"),
    (b"BM", "bmp", ".bmp"),
    (b"II*\x00", "tiff", ".tif"),
    (b"MM\x00*", "tiff", ".tif"),
]


def sniff(data: bytes) -> tuple[str, str]:
    for magic, name, ext in MAGIC:
        if data.startswith(magic):
            if name == "webp" and data[8:12] != b"WEBP":
                continue
            return name, ext
    return "unknown", ".bin"


def scan_bytes(
    out_dir: Path,
    n_shards: int = 12,
    rgs_per_shard: int = 1,
    seed: int = 0,
    splits: tuple[str, ...] = ("train", "validation"),
) -> Path:
    """Pull whole row groups chosen at random over the full shard index range."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    rng = random.Random(seed)
    picks: list[tuple[str, int, int]] = []
    for split in splits:
        n = SHARDS[split]
        share = max(1, round(n_shards * n / sum(SHARDS[s] for s in splits)))
        for shard in rng.sample(range(n), k=min(share, n)):
            picks.append((split, shard, -1))       # row group chosen after footer read
    rng.shuffle(picks)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total_bytes = 0
    t0 = time.time()

    for split, shard, _ in picks:
        path = (f"datasets/{SID_REPO}/data/"
                f"{split}-{shard:05d}-of-{SHARDS[split]:05d}.parquet")
        for attempt in range(3):
            try:
                with HfFileSystem().open(path, "rb") as fh:
                    pf = pq.ParquetFile(fh)
                    n_rg = pf.metadata.num_row_groups
                    chosen = rng.sample(range(n_rg), k=min(rgs_per_shard, n_rg))
                    print(f"  {split}-{shard:05d} rg {chosen} of {n_rg}", flush=True)
                    for rg in chosen:
                        d = pf.read_row_group(
                            rg, columns=["img_id", "width", "height", "label", "image"]
                        ).to_pandas()
                        for _, r in d.iterrows():
                            cell = r["image"]
                            raw = cell["bytes"] if isinstance(cell, dict) else cell
                            if raw is None:
                                continue
                            label = int(r["label"])
                            name = SID_LABEL_NAMES.get(label, str(label))
                            fmt, ext = sniff(raw)
                            p = img_dir / name / f"{r['img_id']}{ext}"
                            p.parent.mkdir(parents=True, exist_ok=True)
                            p.write_bytes(raw)      # exact bytes, no PIL round-trip
                            total_bytes += len(raw)
                            rows.append({
                                "image_path": p.as_posix(),
                                "source_dataset": "SID_Set",
                                "sid_label": label,
                                "sid_label_name": name,
                                "label": 0 if label == 0 else 1,
                                "generator": "OpenImagesV7" if label == 0 else "SID_Set_synth",
                                "declared_width": int(r["width"]),
                                "declared_height": int(r["height"]),
                                "container": fmt,
                                "n_bytes": len(raw),
                                "parquet_split": split,
                                "shard": shard,
                                "row_group": rg,
                            })
                        _flush(out_dir, rows)
                break
            except Exception as exc:
                print(f"    retry {attempt+1}/3 {split}-{shard}: {type(exc).__name__}",
                      file=sys.stderr, flush=True)
                time.sleep(3 * (attempt + 1))

    dt = time.time() - t0
    print(f"bytes pass: {len(rows)} images, {total_bytes/1e6:.0f} MB in {dt:.0f}s "
          f"({total_bytes/1e6/max(dt,1):.2f} MB/s)")
    return _flush(out_dir, rows)


def _flush(out_dir: Path, rows: list[dict]) -> Path:
    m = out_dir / "sample_manifest.csv"
    if rows:
        pd.DataFrame(rows).to_csv(m, index=False)
    return m


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("geometry", help="wide, cheap: width/height/label via rows API")
    g.add_argument("--split", default="train", choices=sorted(SHARDS))
    g.add_argument("--n-batches", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", type=Path, default=None)

    b = sub.add_parser("bytes", help="narrow, costly: original bytes from random row groups")
    b.add_argument("--out", type=Path,
                   default=Path("data/audit_sample/sid_set_resampled"))
    b.add_argument("--n-shards", type=int, default=12)
    b.add_argument("--rgs-per-shard", type=int, default=1)
    b.add_argument("--seed", type=int, default=0)

    a = p.parse_args(argv)
    if a.cmd == "geometry":
        out = a.out or Path(f"results/audit_resample_geometry_{a.split}.csv")
        scan_geometry(a.split, a.n_batches, a.seed, out)
    else:
        scan_bytes(a.out, a.n_shards, a.rgs_per_shard, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
