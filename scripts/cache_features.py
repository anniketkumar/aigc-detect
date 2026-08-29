"""Cache frozen CLIP embeddings for one manifest split (PLAN.md §5).

    python -m scripts.cache_features --manifest data/manifests/train.csv \\
        --out data/features/train --device cuda

Run once per split. Training the head then reads three small ``.npy``/``.json``
files and takes seconds -- the whole point of freezing the backbone (§
Workflow: "once cached, every later head experiment runs in seconds with no
GPU and no download").

Writes into ``--out`` (a directory):

    embeddings.npy   (N, embed_dim) float32, L2-normalized
    labels.npy       (N,) int64, 0=real / 1=ai
    paths.json       image_path per row, same order -- kept for error analysis
    meta.json        backbone, pretrained tag, embed_dim, source manifest, counts

Images that fail to decode are skipped (with a warning), not fatal -- a corrupt
file already happened once in Phase 2 and does not get to cost a whole caching
run here either.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import imageio as IIO


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ("image_path", "label") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: manifest is missing column(s) {missing}")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def cache_split(
    manifest: Path,
    out: Path,
    device: str = "cpu",
    backbone: str | None = None,
    pretrained: str | None = None,
    batch_size: int = 64,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    from src.models.clip_backbone import BACKBONE, PRETRAINED, ClipBackbone

    # Resolved here, not as an argument default, so the import stays deferred:
    # one place decides the backbone (src/models/clip_backbone.py) and no
    # caller can silently pin a stale one.
    backbone = backbone or BACKBONE
    pretrained = pretrained or PRETRAINED

    df = _load_manifest(manifest)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"{manifest}: no rows")

    print(f"loading {backbone} ({pretrained}) on {device} ...", flush=True)
    t0 = time.time()
    clip = ClipBackbone(device=device, backbone=backbone, pretrained=pretrained)
    print(f"  backbone ready in {time.time() - t0:.1f}s, embed_dim={clip.embed_dim}")

    feats: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    n_failed = 0

    bar = tqdm(total=len(df), desc="embed", unit="img", disable=not progress)
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        imgs, ok_labels, ok_paths = [], [], []
        for path, label in zip(chunk["image_path"], chunk["label"]):
            img = IIO.load_rgb(path)
            if img is None:
                print(f"[warn] could not read {path}, skipping", file=sys.stderr)
                n_failed += 1
                continue
            imgs.append(img)
            ok_labels.append(int(label))
            ok_paths.append(str(path))
        if imgs:
            feats.append(clip.embed(imgs))
            labels.extend(ok_labels)
            paths.extend(ok_paths)
        bar.update(len(chunk))
    bar.close()

    if not feats:
        raise SystemExit(f"{manifest}: every image failed to decode")

    embeddings = np.concatenate(feats, axis=0).astype(np.float32)
    labels_arr = np.array(labels, dtype=np.int64)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", embeddings)
    np.save(out / "labels.npy", labels_arr)
    (out / "paths.json").write_text(json.dumps(paths), encoding="utf-8")
    meta = {
        "manifest": str(manifest),
        "backbone": backbone,
        "pretrained": pretrained,
        "embed_dim": clip.embed_dim,
        "n_images": len(paths),
        "n_failed": n_failed,
        "n_real": int((labels_arr == 0).sum()),
        "n_ai": int((labels_arr == 1).sum()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"-> {out}")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                     help="output directory, e.g. data/features/train")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backbone", default=None,
                    help="default: src.models.clip_backbone.BACKBONE")
    ap.add_argument("--pretrained", default=None,
                    help="default: src.models.clip_backbone.PRETRAINED")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                     help="cache only the first N rows (smoke testing)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    cache_split(
        a.manifest, a.out, device=a.device, backbone=a.backbone,
        pretrained=a.pretrained, batch_size=a.batch_size, limit=a.limit,
        progress=not a.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
