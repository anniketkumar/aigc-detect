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

Phase 4 (PLAN.md §6): ``--augment-copies K`` embeds ``K`` randomly augmented
copies of every image instead of the image itself -- option (a) of "either
precompute embeddings for K augmented copies per image, or run CLIP live in
the dataloader". Feature caching and *online* augmentation cannot coexist
(caching only pays off because the backbone never runs again), so this is the
augmented alternative to a plain run, not an addition to it: pass one or the
other. With ``--augment-copies K`` the outputs above hold ``N*K`` rows
(``paths.json`` entries become ``"{image_path}#aug{copy_index}"`` so each
embedding still traces to a source file) and one extra array is written:

    degradation.npy  (N*K, 12) float32 -- src.data.augment.DegradationLabel
                      vectors, one per embedded copy; free supervision for a
                      future degradation head (PLAN.md §7.2), unused today
                      (Phase 5 was cut, HANDOFF.md) but cheap to keep.

Phase 5, resumed: ``--fuse-freq`` appends ``src.features.frequency``'s
deterministic Fourier/DCT feature vector (see that module's docstring) onto
every embedding before it is saved, so ``embeddings.npy`` becomes
``(N, clip_embed_dim + FREQ_DIM)`` instead of ``(N, clip_embed_dim)``.
``src/train.py`` needs no change to consume this -- it already reads
``embed_dim`` off the array's own shape -- but the resulting checkpoint must
be scored with ``clip_freq_fusion``, not ``clip_linear``
(``src/models/clip_fusion.py``). Composable with ``--augment-copies``: each
augmented copy gets its own frequency features computed on its own (already
degraded) pixels, same as the CLIP embedding does, which is the entire point
-- a frequency branch that only ever sees clean pixels would never be
measured under the JPEG/blur/resize conditions it's meant to survive.
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
    augment_copies: int = 0,
    augment_seed: int = 0,
    fuse_freq: bool = False,
) -> dict:
    from src.models.clip_backbone import BACKBONE, PRETRAINED, ClipBackbone

    if augment_copies < 0:
        raise ValueError(f"augment_copies must be >= 0, got {augment_copies}")

    if fuse_freq:
        from src.features.frequency import FREQ_DIM, extract_frequency_features

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
    if augment_copies:
        from src.data.augment import iter_augmented_copies
        print(f"  augmenting: {augment_copies} copies/image, seed={augment_seed}")

    feats: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    degradations: list[np.ndarray] = []
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
            if augment_copies:
                for copy_idx, (aug_img, aug_label) in enumerate(
                    iter_augmented_copies(img, str(path), augment_copies, base_seed=augment_seed)
                ):
                    imgs.append(aug_img)
                    ok_labels.append(int(label))
                    ok_paths.append(f"{path}#aug{copy_idx}")
                    degradations.append(aug_label.to_vector())
            else:
                imgs.append(img)
                ok_labels.append(int(label))
                ok_paths.append(str(path))
        if imgs:
            batch_feats = clip.embed(imgs)
            if fuse_freq:
                freq = np.stack([extract_frequency_features(img) for img in imgs])
                batch_feats = np.concatenate([batch_feats, freq], axis=1)
            feats.append(batch_feats)
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
    if augment_copies:
        degradation_arr = np.stack(degradations, axis=0).astype(np.float32)
        np.save(out / "degradation.npy", degradation_arr)
    meta = {
        "manifest": str(manifest),
        "backbone": backbone,
        "pretrained": pretrained,
        "clip_embed_dim": clip.embed_dim,
        "embed_dim": embeddings.shape[1],  # = clip_embed_dim (+ FREQ_DIM if fuse_freq)
        "n_images": len(paths),
        "n_failed": n_failed,
        "n_real": int((labels_arr == 0).sum()),
        "n_ai": int((labels_arr == 1).sum()),
        "augment_copies": augment_copies,
        "augment_seed": augment_seed if augment_copies else None,
        "fuse_freq": fuse_freq,
        "freq_dim": FREQ_DIM if fuse_freq else 0,
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
    ap.add_argument("--augment-copies", type=int, default=0,
                     help="Phase 4: embed K randomly augmented copies per image "
                          "instead of the image itself (0 = off, plain Phase 3 "
                          "caching). See src.data.augment.")
    ap.add_argument("--augment-seed", type=int, default=0,
                     help="base seed for --augment-copies (default 0)")
    ap.add_argument("--fuse-freq", action="store_true",
                     help="Phase 5: append src.features.frequency's Fourier/DCT "
                          "feature vector onto every embedding (0 = off, plain "
                          "CLIP-only caching). Score the resulting checkpoint "
                          "with --model clip_freq_fusion, not clip_linear.")
    a = ap.parse_args(argv)

    cache_split(
        a.manifest, a.out, device=a.device, backbone=a.backbone,
        pretrained=a.pretrained, batch_size=a.batch_size, limit=a.limit,
        progress=not a.quiet, augment_copies=a.augment_copies,
        augment_seed=a.augment_seed, fuse_freq=a.fuse_freq,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
