"""Run a model over the robustness grid (PLAN.md §3.3).

    python -m src.evaluate --ckpt <path> --split test --out results/<name>/

Emits into ``--out``:

    grid.csv     one row per (transform, severity) cell
    report.md    the markdown table, headline numbers first
    summary.json the §3.2 aggregates plus the full run config
    scores.csv   raw per-(image, cell) scores, for §11 error analysis

The harness owns loading, transforming, batching and metrics; the model owns
only preprocessing and the forward pass (see src/models/base.py). Adding a
Phase 3-5 model means registering it, not editing this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src import metrics as M
from src import transforms as T
from src.models.base import MODEL_REGISTRY, load_model

MANIFEST_COLUMNS = ["image_path", "label"]
DEFAULT_MANIFEST_ROOT = Path("data/manifests")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def resolve_manifest(
    manifest: Path | None, split: str | None, root: Path = DEFAULT_MANIFEST_ROOT
) -> Path:
    if manifest is not None:
        path = manifest
    elif split is not None:
        path = root / f"{split}.csv"
    else:
        raise SystemExit("give either --manifest or --split")
    if not path.exists():
        raise SystemExit(
            f"manifest not found: {path}\n"
            "Phase 2 builds data/manifests/{{train,val,test}}.csv. Until then, "
            "generate a fixture:\n"
            "  python -m scripts.make_dummy_fixture --out data/fixtures/dummy\n"
            "  python -m src.evaluate --manifest data/fixtures/dummy/manifest.csv ..."
        )
    return path


def load_manifest(path: Path, limit: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Read a §4.2 manifest. ``limit`` takes a seeded, class-stratified subset."""
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: manifest is missing column(s) {missing}")
    df["label"] = df["label"].astype(int)
    bad = sorted(set(df["label"].unique()) - {0, 1})
    if bad:
        raise SystemExit(f"{path}: labels must be 0=real / 1=ai, got {bad}")

    if limit is not None and limit < len(df):
        # Stratified so that shrinking the set does not also skew the class
        # balance, which would move AP and TPR@FPR for reasons unrelated to the
        # model. Seeded, so --limit is reproducible.
        rng = np.random.default_rng(seed)
        per_class = limit // 2
        parts = []
        for label, group in df.groupby("label", sort=True):
            take = min(per_class, len(group))
            idx = rng.permutation(len(group))[:take]
            parts.append(group.iloc[np.sort(idx)])
        df = pd.concat(parts).sort_index()
    return df.reset_index(drop=True)


def image_id_for(path: str) -> str:
    """Stable per-image identity used for seeding and caching.

    Path-derived and normalized to forward slashes so the same manifest yields
    the same noise on Windows and Linux.
    """
    return Path(str(path)).as_posix()


# --------------------------------------------------------------------------- #
# Image IO + transformed-image cache
# --------------------------------------------------------------------------- #

def _load_image(path: str) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as exc:  # unreadable, truncated, or not an image
        print(f"[warn] could not read {path}: {exc}", file=sys.stderr)
        return None


class TransformCache:
    """Optional on-disk cache of transformed images (§3.3).

    Stored as lossless PNG -- caching as JPEG would silently add a re-encode to
    every cell and corrupt the whole experiment. The key covers the image, the
    op chain, and (for stochastic cells) the seed, so a cache entry can never
    be served for a different transform than the one requested.

    Off by default. It is the right call once Phase 3 runs a real GPU model over
    high-resolution data, where a blur on a 4K image is expensive and the GPU is
    the thing you are trying to keep busy. With a cheap model and small images
    the PNG encode/decode costs more than the transform it saves, and the disk
    cost is ~19x the dataset.
    """

    def __init__(self, root: Path | None, base_seed: int):
        self.root = Path(root) if root else None
        self.base_seed = base_seed
        self.hits = 0
        self.misses = 0
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, cell: T.Cell, image_id: str) -> Path:
        key = T.cache_key(cell, image_id, self.base_seed)
        # shard, or a big test set puts 100k+ files in one directory
        return self.root / cell.name / key[:2] / f"{key}.png"

    def get(self, cell: T.Cell, image_id: str) -> Image.Image | None:
        if not self.enabled:
            return None
        p = self._path(cell, image_id)
        if not p.exists():
            self.misses += 1
            return None
        try:
            img = Image.open(p)
            img.load()
            self.hits += 1
            return img
        except Exception:
            self.misses += 1
            return None

    def put(self, cell: T.Cell, image_id: str, img: Image.Image) -> None:
        if not self.enabled:
            return
        p = self._path(cell, image_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.png")
        img.save(tmp, format="PNG", compress_level=1)
        tmp.replace(p)  # atomic, so an interrupted run leaves no partial entry

    def has(self, cell: T.Cell, image_id: str) -> bool:
        return self.enabled and self._path(cell, image_id).exists()


# --------------------------------------------------------------------------- #
# The grid run
# --------------------------------------------------------------------------- #

def select_cells(names: Sequence[str] | None) -> list[T.Cell]:
    cells = T.build_cells()
    if not names:
        return cells
    by_name = {c.name: c for c in cells}
    picked: list[T.Cell] = []
    for n in names:
        matches = [c for c in cells if c.name == n or c.family == n]
        if not matches:
            raise SystemExit(
                f"unknown cell or family {n!r}. cells: {sorted(by_name)}"
            )
        picked += [c for c in matches if c not in picked]
    return picked


def run_grid(
    model,
    df: pd.DataFrame,
    cells: Sequence[T.Cell],
    base_seed: int = 0,
    batch_size: int = 32,
    cache: TransformCache | None = None,
    fpr_target: float = M.DEFAULT_FPR_TARGET,
    progress: bool = True,
) -> tuple[list[M.CellMetrics], pd.DataFrame]:
    """Score every image in every cell; return per-cell metrics and raw scores.

    Iterates image-major: each original file is read from disk at most once per
    batch and then transformed into every cell, rather than re-reading it 19
    times. When the cache is enabled and already holds every cell for an image,
    the original is not read at all.
    """
    cache = cache or TransformCache(None, base_seed)
    ids = [image_id_for(p) for p in df["image_path"]]
    labels = df["label"].to_numpy(dtype=int)

    scores: dict[str, list[float | None]] = {c.name: [] for c in cells}
    bar = tqdm(
        total=len(df) * len(cells),
        desc="grid",
        unit="img-cell",
        disable=not progress,
    )

    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        batch_ids = ids[start:stop]

        # Read originals only where at least one cell is not already cached.
        need = [
            not all(cache.has(c, i) for c in cells) for i in batch_ids
        ]
        originals = [
            _load_image(df["image_path"].iloc[k]) if need[k - start] else None
            for k in range(start, stop)
        ]

        for cell in cells:
            imgs: list[Image.Image | None] = []
            for j, image_id in enumerate(batch_ids):
                got = cache.get(cell, image_id)
                if got is None:
                    src = originals[j]
                    if src is None and need[j]:
                        imgs.append(None)  # unreadable file
                        continue
                    if src is None:
                        # cache said it had every cell but this entry vanished
                        src = _load_image(df["image_path"].iloc[start + j])
                        if src is None:
                            imgs.append(None)
                            continue
                    got = T.apply_cell(src, cell, image_id=image_id, base_seed=base_seed)
                    cache.put(cell, image_id, got)
                imgs.append(got)

            ok = [j for j, im in enumerate(imgs) if im is not None]
            out: list[float | None] = [None] * len(imgs)
            if ok:
                got_scores = model.score(
                    [imgs[j] for j in ok], [batch_ids[j] for j in ok]
                )
                if len(got_scores) != len(ok):
                    raise RuntimeError(
                        f"{getattr(model, 'name', model)}.score returned "
                        f"{len(got_scores)} scores for {len(ok)} images"
                    )
                for j, s in zip(ok, got_scores):
                    out[j] = s
            scores[cell.name].extend(out)
            bar.update(len(imgs))
    bar.close()

    per_cell = [
        M.cell_metrics(
            labels,
            scores[c.name],
            cell=c.name,
            kind=c.kind,
            family=c.family,
            severity=c.severity,
            chain=c.chain,
            fpr_target=fpr_target,
        )
        for c in cells
    ]

    raw = pd.DataFrame(
        {
            "image_path": np.repeat(df["image_path"].to_numpy(), len(cells)),
            "label": np.repeat(labels, len(cells)),
            "cell": [c.name for _ in range(len(df)) for c in cells],
            "score": [scores[c.name][i] for i in range(len(df)) for c in cells],
        }
    )
    return per_cell, raw


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_outputs(
    out_dir: Path,
    per_cell: Sequence[M.CellMetrics],
    summary: M.GridSummary,
    raw: pd.DataFrame | None,
    config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [c.as_row() for c in per_cell]
    with (out_dir / "grid.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    subtitle = (
        f"Model `{config['model_name']}` on `{config['manifest']}` "
        f"({config['n_images']} images, {config['n_cells']} cells, "
        f"seed {config['seed']}).\n\n"
        f"Reproduce: `{config['command']}`"
    )
    (out_dir / "report.md").write_text(
        M.markdown_grid(
            list(per_cell), summary,
            title=f"Robustness grid — {config['name']}",
            subtitle=subtitle,
        ),
        encoding="utf-8",
    )

    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary.as_row(), "config": config}, indent=2),
        encoding="utf-8",
    )

    if raw is not None:
        raw.to_csv(out_dir / "scores.csv", index=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate a model over the PLAN.md §3.1 transform grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # §3.3 acceptance check, no data or model needed\n"
            "  python -m scripts.make_dummy_fixture --out data/fixtures/dummy\n"
            "  python -m src.evaluate --model dummy_random \\\n"
            "      --manifest data/fixtures/dummy/manifest.csv --out results/dummy/\n\n"
            "  # once Phase 2 has built manifests\n"
            "  python -m src.evaluate --ckpt runs/baseline.pt --split test \\\n"
            "      --out results/baseline/\n"
        ),
    )
    p.add_argument("--model", default="dummy_random",
                   choices=sorted(MODEL_REGISTRY),
                   help="registered scorer to evaluate (default: %(default)s)")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="checkpoint passed to the model. Optional: the dummy "
                        "models take no weights.")
    p.add_argument("--split", default=None,
                   help="reads --manifest-root/<split>.csv")
    p.add_argument("--manifest", type=Path, default=None,
                   help="explicit manifest CSV; overrides --split")
    p.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    p.add_argument("--out", type=Path, required=True,
                   help="output directory, e.g. results/baseline/")
    p.add_argument("--name", default=None,
                   help="run label for the report title (default: --out's name)")
    p.add_argument("--cells", nargs="*", default=None, metavar="CELL_OR_FAMILY",
                   help="restrict the grid, e.g. --cells clean jpeg. Default: all 19.")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate a seeded, class-stratified subset of N images")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed for the stochastic cells and for --limit")
    p.add_argument("--fpr-target", type=float, default=M.DEFAULT_FPR_TARGET)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="cache transformed images here as lossless PNG. Off by "
                        "default; see TransformCache for when it pays.")
    p.add_argument("--no-scores", action="store_true",
                   help="skip scores.csv (raw per-image scores)")
    p.add_argument("--quiet", action="store_true", help="no progress bar")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    manifest = resolve_manifest(args.manifest, args.split, args.manifest_root)
    df = load_manifest(manifest, limit=args.limit, seed=args.seed)
    if df.empty:
        raise SystemExit(f"{manifest}: no rows to evaluate")
    cells = select_cells(args.cells)
    model = load_model(args.model, ckpt=args.ckpt, device=args.device, seed=args.seed)
    cache = TransformCache(args.cache_dir, args.seed)

    n_real = int((df["label"] == 0).sum())
    n_fake = int((df["label"] == 1).sum())
    print(
        f"model={getattr(model, 'name', args.model)}  images={len(df)} "
        f"(real={n_real}, ai={n_fake})  cells={len(cells)}  seed={args.seed}"
        + ("  cache=on" if cache.enabled else "")
    )
    null_sd = M.auroc_null_sd(n_fake, n_real)
    if np.isfinite(null_sd):
        print(
            f"AUROC null SD at this sample size: {null_sd:.4f} "
            f"(a chance-level model should land within ~{3 * null_sd:.3f} of 0.500)"
        )

    per_cell, raw = run_grid(
        model, df, cells,
        base_seed=args.seed,
        batch_size=args.batch_size,
        cache=cache,
        fpr_target=args.fpr_target,
        progress=not args.quiet,
    )
    summary = M.summarize(per_cell)

    config = {
        "name": args.name or args.out.name or "run",
        "model": args.model,
        "model_name": getattr(model, "name", args.model),
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "manifest": manifest.as_posix(),
        "split": args.split,
        "n_images": len(df),
        "n_real": n_real,
        "n_ai": n_fake,
        "n_cells": len(cells),
        "cells": [c.name for c in cells],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": args.device,
        "fpr_target": args.fpr_target,
        "auroc_null_sd": None if not np.isfinite(null_sd) else round(null_sd, 6),
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "elapsed_s": None,  # filled below
        "command": "python -m src.evaluate " + " ".join(sys.argv[1:]),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    config["elapsed_s"] = round(time.time() - t0, 2)

    write_outputs(args.out, per_cell, summary, None if args.no_scores else raw, config)

    print(
        f"\nclean AUROC        {summary.clean_auroc:.4f}\n"
        f"mean transformed   {summary.mean_transformed_auroc:.4f}"
        f"   (family-balanced over {summary.n_families} families)\n"
        f"robustness_gap     {summary.robustness_gap:+.4f}   (lower is better)\n"
        f"worst_case         {summary.worst_case:.4f}   ({summary.worst_cell})\n"
        f"  flat mean        {summary.mean_transformed_auroc_flat:.4f}"
        f"   gap {summary.robustness_gap_flat:+.4f}   (§3.2 literal, secondary)"
    )
    for w in summary.warnings:
        print(f"[warn] {w}")
    print(f"\nwrote {args.out / 'grid.csv'} and {args.out / 'report.md'} "
          f"in {config['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
