"""Stream, normalize and checksum the training corpus. Resumable.

    python -m scripts.download_data --plan            # footprint, downloads nothing
    python -m scripts.download_data                   # fetch everything
    python -m scripts.download_data --only Pexels SDXL --quota 200

Design notes, each of which is a thing that went wrong in an earlier draft.

**Nothing raw is ever written.** Row groups are decoded in memory, normalized
through ``src/data/normalize.py``, and only the 512x512 q95 JPEG hits disk. That
is the difference between 1.7 GB and 19 GB resident, and it removes the
possibility of a later stage accidentally reading an un-normalized file.

**Shards and row groups are visited in seeded random order.** The Phase 2 audit
read shard 0 row group 0 of every file and drew conclusions from it. Parquet
shards are often written in source or class order; reading the head of one is
not a sample. ``--seed`` makes the draw reproducible without making it
positional.

**Resume is keyed on source content hash, not on row position.** Row position
would break the moment a repo is re-sharded, silently re-downloading everything
or, worse, appending duplicates. The ledger stores ``src_sha256`` and the
downloader skips any hash it has already normalized -- which also deduplicates
within and across sources for free.

**The ledger is flushed after every row group**, so an interrupted run leaves a
consistent, self-describing partial corpus rather than orphan files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests

from src.data import imageio as IIO
from src.data.normalize import NormalizeConfig, QualityPrior, normalize
from src.data.sources import SOURCES, BY_NAME, Source, estimate_footprint

PARQUET_URL = "https://datasets-server.huggingface.co/parquet"


def _hf_headers() -> dict[str, str]:
    """``Authorization`` header for Hugging Face, if a token is available.

    Both the datasets-server listing below and the resolved CDN URLs it
    returns count against HF's per-IP anonymous rate limit, which is what
    makes an unauthenticated Colab download slow. Reads ``HF_TOKEN`` --  the
    name ``huggingface_hub.login()`` and Colab's Secrets pane both use -- so
    setting the one secret speeds up both request paths. Empty dict (i.e. no
    header) when unset, which is exactly the previous unauthenticated behavior.
    """
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}

LEDGER_FIELDS = [
    "image_path", "label", "generator", "family", "source_dataset", "split_hint",
    "src_sha256", "out_sha256", "src_width", "src_height", "src_format",
    "src_jpeg_quality", "crop_x", "crop_y", "first_gen_quality",
    "first_gen_subsampling", "load_status", "pad_frac", "n_bytes",
    "shard", "row_group", "config_hash",
]


# --------------------------------------------------------------------------- #
# Locating shards
# --------------------------------------------------------------------------- #

def list_shards(src: Source, timeout: int = 120) -> list[str]:
    """Parquet paths for one (repo, config, split), via the datasets-server.

    Preferred over guessing ``data/{split}-*.parquet``: the four repos in the
    registry use three different layouts (``data/``, ``chunk_0000/``, and a
    split name that is a date). The endpoint knows; a glob pattern would need
    per-repo special cases and would break silently when one changed.
    """
    r = requests.get(PARQUET_URL, params={"dataset": src.repo}, timeout=timeout,
                      headers=_hf_headers())
    r.raise_for_status()
    files = [
        f for f in r.json()["parquet_files"]
        if f["config"] == src.config and f["split"] == src.split
    ]
    if not files:
        have = sorted({(f["config"], f["split"]) for f in r.json()["parquet_files"]})
        raise KeyError(f"{src.repo}: no {src.config}/{src.split}; has {have[:8]}")
    return [f["url"] for f in sorted(files, key=lambda f: f["filename"])]


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #

#: Errors that mean "the network hiccuped", not "this data is bad". Matched on
#: class name because the concrete types come from aiohttp/fsspec/requests and
#: importing all three here to build an exception tuple would drag async deps
#: into a module that otherwise needs none.
TRANSIENT = (
    "ClientConnectorDNSError", "ClientConnectorError", "ClientPayloadError",
    "ClientOSError", "ServerDisconnectedError", "ConnectionError",
    "ConnectionResetError", "TimeoutError", "ReadTimeout", "ConnectTimeout",
    "ChunkedEncodingError", "IncompleteRead", "ProtocolError", "OSError",
    "FileNotFoundError",   # fsspec raises this when a range GET 404s mid-stream
)


def _is_transient(exc: BaseException) -> bool:
    return type(exc).__name__ in TRANSIENT


def with_retry(fn, what: str, attempts: int = 4, base_sleep: float = 2.0):
    """Run ``fn``, retrying transient network failures with linear backoff.

    Written after a smoke run lost two of three sources to a single DNS blip:
    the loop treated one failed row group as a permanent verdict on the shard
    and moved on, so a five-second outage cost the whole source. Non-transient
    errors are *not* retried -- a schema mismatch will fail identically four
    times and only obscure the message.
    """
    last: BaseException | None = None
    for a in range(attempts):
        try:
            return fn()
        except BaseException as exc:
            last = exc
            if not _is_transient(exc):
                print(f"    {what}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return None
            if a < attempts - 1:
                time.sleep(base_sleep * (a + 1))
    print(f"    {what}: gave up after {attempts} -- {type(last).__name__}: {last}",
          file=sys.stderr)
    return None


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #

class Ledger:
    """Append-only CSV of everything successfully normalized."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.seen_src: set[str] = set()
        self.counts: dict[str, int] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.rows.append(row)
                    self.seen_src.add(row["src_sha256"])
                    g = row["generator"]
                    self.counts[g] = self.counts.get(g, 0) + 1

    def add(self, row: dict) -> None:
        self.rows.append(row)
        self.seen_src.add(row["src_sha256"])
        g = row["generator"]
        self.counts[g] = self.counts.get(g, 0) + 1

    def flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)
        tmp.replace(self.path)   # atomic: an interrupted flush cannot truncate


# --------------------------------------------------------------------------- #
# The per-source loop
# --------------------------------------------------------------------------- #

def _extract_bytes(cell) -> bytes | None:
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def fetch_source(
    src: Source,
    out_root: Path,
    ledger: Ledger,
    cfg: NormalizeConfig,
    prior: QualityPrior,
    quota: int | None = None,
    seed: int = 0,
    max_shards: int | None = None,
) -> dict:
    import pyarrow.parquet as pq

    want = quota if quota is not None else src.quota
    have = ledger.counts.get(src.name, 0)
    if have >= want:
        print(f"  {src.name}: {have}/{want} already done")
        return {"generator": src.name, "kept": have, "skipped": 0, "fetched_mb": 0.0}

    shards = with_retry(lambda: list_shards(src), f"{src.name}: shard listing",
                        attempts=5, base_sleep=3.0)
    if not shards:
        print(f"  {src.name}: SHARD LISTING FAILED", file=sys.stderr)
        return {"generator": src.name, "kept": have, "error": "shard listing failed"}

    rng = random.Random(f"{seed}:{src.name}")
    order = list(range(len(shards)))
    rng.shuffle(order)
    if max_shards:
        order = order[:max_shards]

    img_root = out_root / "images" / src.name
    rejected: dict[str, int] = {}
    fetched_bytes = 0
    t0 = time.time()

    for si in order:
        if ledger.counts.get(src.name, 0) >= want:
            break
        url = shards[si]

        def _open():
            handle = _http_file(url)
            return handle, pq.ParquetFile(handle)

        opened = with_retry(_open, f"shard {si}: open")
        if opened is None:
            continue
        fh, pf = opened

        rgs = list(range(pf.metadata.num_row_groups))
        rng.shuffle(rgs)
        for rg in rgs:
            if ledger.counts.get(src.name, 0) >= want:
                break
            names = pf.schema_arrow.names
            cols = [c for c in (src.image_column, "image") if c in names]                 or [names[0]]
            cols = cols[:1]
            # Sources whose split mixes classes need their label column too, or
            # every row in the shard gets stamped with this source's label.
            if src.label_column and src.label_column in names:
                cols.append(src.label_column)
            tbl = with_retry(lambda: pf.read_row_group(rg, columns=cols),
                             f"shard {si} rg {rg}")
            if tbl is None:
                continue

            col = tbl.column(0).to_pylist()
            src_labels = (tbl.column(1).to_pylist() if len(cols) > 1
                          else [None] * len(col))
            for raw, row_label in zip(col, src_labels):
                if src.label_values and row_label is not None                         and int(row_label) not in src.label_values:
                    rejected["wrong_class"] = rejected.get("wrong_class", 0) + 1
                    continue
                if ledger.counts.get(src.name, 0) >= want:
                    break
                data = _extract_bytes(raw)
                if data is None:
                    rejected["no_bytes"] = rejected.get("no_bytes", 0) + 1
                    continue
                fetched_bytes += len(data)
                src_sha = hashlib.sha256(data).hexdigest()
                if src_sha in ledger.seen_src:
                    rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                    continue

                # normalize() takes a path, so the source bytes land in a
                # scratch file. Deliberately one reused temp file, not a
                # per-image write: the point is that nothing raw accumulates.
                scratch = out_root / f".scratch_{src.name}.bin"
                scratch.parent.mkdir(parents=True, exist_ok=True)
                scratch.write_bytes(data)
                res = normalize(scratch, src.label, cfg, prior, image_id=src_sha)
                if not res.ok:
                    rejected[res.status] = rejected.get(res.status, 0) + 1
                    continue

                out_sha = hashlib.sha256(res.data).hexdigest()
                dest = img_root / out_sha[:2] / f"{out_sha}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(res.data)

                ledger.add({
                    "image_path": dest.as_posix(),
                    "label": src.label,
                    "generator": src.name,
                    "family": src.family,
                    "source_dataset": src.repo,
                    "split_hint": "holdout" if src.holdout else "train_pool",
                    "src_sha256": src_sha,
                    "out_sha256": out_sha,
                    "src_width": res.src_size[0], "src_height": res.src_size[1],
                    "src_format": res.src_format,
                    "src_jpeg_quality": _quality_of(data),
                    "crop_x": res.crop_xy[0], "crop_y": res.crop_xy[1],
                    "first_gen_quality": res.first_gen_quality,
                    "first_gen_subsampling": res.first_gen_subsampling,
                    "load_status": res.load_status,
                    "pad_frac": round(res.pad_frac, 4),
                    "n_bytes": res.n_bytes,
                    "shard": si, "row_group": rg,
                    "config_hash": cfg.config_hash,
                })

            ledger.flush()
            n = ledger.counts.get(src.name, 0)
            print(f"    shard {si:>3d} rg {rg:>2d}: {n}/{want}  "
                  f"{fetched_bytes/1e6:.0f} MB  {time.time()-t0:.0f}s", flush=True)

        try:
            fh.close()
        except Exception:
            pass

    scratch = out_root / f".scratch_{src.name}.bin"
    scratch.unlink(missing_ok=True)
    kept = ledger.counts.get(src.name, 0)
    dt = time.time() - t0
    print(f"  {src.name}: {kept}/{want} kept, {fetched_bytes/1e6:.0f} MB in {dt:.0f}s "
          f"({fetched_bytes/1e6/max(dt,1):.2f} MB/s); rejected {rejected}")
    return {"generator": src.name, "kept": kept, "rejected": rejected,
            "fetched_mb": round(fetched_bytes / 1e6, 1), "seconds": round(dt)}


def _quality_of(data: bytes) -> int:
    from src.data.normalize import estimate_jpeg_quality
    buf = io.BytesIO(data)
    try:
        from PIL import Image
        with Image.open(buf) as im:
            if im.format != "JPEG":
                return 0
    except Exception:
        return 0
    buf.seek(0)
    return estimate_jpeg_quality(buf) or 0


def _http_file(url: str):
    """Seekable file object over an HTTP range-capable URL.

    fsspec rather than ``HfFileSystem``: the parquet endpoint hands back
    fully-resolved CDN URLs, and going through the Hub filesystem again would
    re-resolve them on every read.
    """
    import fsspec
    return fsspec.open(url, "rb", block_size=8 * 1024 * 1024,
                        headers=_hf_headers()).open()


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--only", nargs="+", default=None, metavar="GENERATOR")
    ap.add_argument("--quota", type=int, default=None,
                    help="override the per-source quota (useful for smoke runs)")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--plan", action="store_true",
                    help="print the disk/transfer footprint and exit")
    a = ap.parse_args(argv)

    chosen = SOURCES if not a.only else tuple(BY_NAME[n] for n in a.only)
    if a.quota:
        chosen = tuple(
            Source(**{**asdict(s), "quota": a.quota}) for s in chosen
        )

    est = estimate_footprint(chosen, crop=a.crop, jpeg_quality=a.jpeg_quality)
    print(f"{'source':24s} {'lab':>3s} {'quota':>6s} {'MB/img':>7s} "
          f"{'transfer GB':>11s} {'disk GB':>8s}")
    for r in est["per_source"]:
        print(f"{r['name']:24s} {r['label']:3d} {r['quota']:6d} "
              f"{r['src_mb_per_img']:7.3f} {r['transfer_gb']:11.2f} "
              f"{r['normalized_gb']:8.3f}")
    print(f"\n  images      {est['n_images']:,}")
    print(f"  transfer    {est['transfer_gb']} GB  (network)")
    print(f"  on disk     {est['normalized_gb']} GB  (normalized, kept)")
    print(f"  peak disk   {est['peak_gb']} GB")
    if a.plan:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "footprint.json").write_text(json.dumps(est, indent=2), encoding="utf-8")
        print(f"\n-> {a.out/'footprint.json'}  (nothing downloaded)")
        return 0

    cfg = NormalizeConfig(crop=a.crop, jpeg_quality=a.jpeg_quality, seed=a.seed)
    ledger = Ledger(a.out / "ledger.csv")
    print(f"\nconfig {cfg.config_hash}; ledger has {len(ledger.rows)} rows\n")

    # The real class's JPEG quality prior drives the AI class's synthetic first
    # generation, so real sources must be fetched first. If they were not, the
    # first AI images would use the baked-in prior and later ones the measured
    # one -- a within-class inconsistency that is itself a weak signal.
    ordered = sorted(chosen, key=lambda s: s.label)
    prior = QualityPrior()
    report = []
    for i, s in enumerate(ordered):
        print(f"[{i+1}/{len(ordered)}] {s.name} ({s.repo} {s.config}/{s.split})")
        if s.label == 1 and prior.n_observed == 0:
            real_paths = [Path(r["image_path"]) for r in ledger.rows
                          if r["label"] in (0, "0")]
            prior = QualityPrior.from_paths(real_paths)
            print(f"  quality prior: {prior.source}")
        report.append(fetch_source(s, a.out, ledger, cfg, prior,
                                   quota=a.quota, seed=a.seed,
                                   max_shards=a.max_shards))

    ledger.flush()
    (a.out / "download_report.json").write_text(
        json.dumps({"config": vars(cfg), "prior": prior.source,
                    "sources": report}, indent=2, default=str), encoding="utf-8")
    total = len(ledger.rows)
    disk = sum(f.stat().st_size for f in (a.out / "images").rglob("*.jpg")) / 1e9
    print(f"\n{total:,} images, {disk:.2f} GB on disk -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
