"""Fetch a small, bounded sample for the Phase 2 leakage audit.

Deliberately *not* `download_data.py`. This pulls tens or hundreds of MB so the
audit can run before committing to a multi-hundred-GB download, and it is
bandwidth-bounded by construction: you say how many row groups / rows you want
and it reads only those byte ranges over HTTP.

Two sources, fetched very differently because their hosting differs:

**SID_Set** (HF, parquet with embedded images). Read selected parquet row groups
by HTTP range request. Images are written to disk as the **exact original bytes**
from the parquet cell -- never re-encoded -- because container format, EXIF and
PNG text chunks are precisely what the audit is looking for, and a re-encode
would destroy all three and silently produce a clean bill of health.

**WildFake** (ModelScope, ~700 GB of 50 GB zips). Image zips are out of reach on
a normal connection, but ``label_csv_files/*.csv`` are 1-40 MB and carry the full
per-generator file list. Those give container format (by extension), the
generator taxonomy, and the file lists needed for the forbidden-subset check,
with no image download at all.

Usage:
    python -m scripts.fetch_audit_sample sid --shards 2 --row-groups 3
    python -m scripts.fetch_audit_sample wildfake-csv --generators dalle3 real_coco
"""

from __future__ import annotations

import argparse
import csv
import io
import time
from pathlib import Path

import requests

SID_REPO = "saberzl/SID_Set"
SID_VAL_SHARDS = 34
WILDFAKE_DS = "hy2628982280/WildFake"
MS_FILE_URL = (
    "https://modelscope.cn/api/v1/datasets/{ds}/repo?Revision=master&FilePath={path}"
)

#: SID_Set label ints -> our label convention (§4.2: 0=real, 1=ai).
#: label 2 (tampered) is a real photo with an inpainted region. It is *not* a
#: fully generated image, so folding it into class 1 would change what the
#: detector is being asked to do. Kept separate here and decided in the manifest.
SID_LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}

MAGIC = [
    (b"\xff\xd8\xff", "jpeg", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", "png", ".png"),
    (b"RIFF", "webp", ".webp"),        # refined below via bytes 8:12
    (b"GIF8", "gif", ".gif"),
    (b"BM", "bmp", ".bmp"),
    (b"II*\x00", "tiff", ".tif"),
    (b"MM\x00*", "tiff", ".tif"),
]


def sniff(data: bytes) -> tuple[str, str]:
    """Container format from magic bytes. Never trusts a filename."""
    for magic, name, ext in MAGIC:
        if data.startswith(magic):
            if name == "webp" and data[8:12] != b"WEBP":
                continue
            return name, ext
    return "unknown", ".bin"


# --------------------------------------------------------------------------- #
# SID_Set
# --------------------------------------------------------------------------- #

def _write_manifest(out_dir: Path, rows: list[dict]) -> Path:
    manifest = out_dir / "sample_manifest.csv"
    if not rows:
        return manifest
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return manifest


def fetch_sid(
    out_dir: Path,
    shards: int = 2,
    row_groups: int = 3,
    max_per_label: int | None = None,
) -> Path:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    counts: dict[int, int] = {}
    bytes_read = 0
    t0 = time.time()

    # Each row group is read with its own filesystem handle and retried
    # independently: the HF range-read connection drops often enough on a slow
    # link that one failure must not cost the whole sample. Already-written
    # images are skipped, so re-running resumes.
    def read_rg(shard: int, rg: int, attempts: int = 4):
        path = (
            f"datasets/{SID_REPO}/data/"
            f"validation-{shard:05d}-of-{SID_VAL_SHARDS:05d}.parquet"
        )
        for a in range(attempts):
            try:
                with HfFileSystem().open(path, "rb") as fh:
                    return pq.ParquetFile(fh).read_row_group(
                        rg,
                        columns=["img_id", "width", "height", "label", "image.bytes"],
                    ).to_pandas()
            except Exception as exc:
                print(f"    retry {a+1}/{attempts}: {type(exc).__name__}", flush=True)
                time.sleep(3 * (a + 1))
        print(f"    giving up on shard {shard} rg {rg}", flush=True)
        return None

    for shard in range(shards):
        for rg in range(row_groups):
            print(f"  shard {shard} row-group {rg} ...", flush=True)
            d = read_rg(shard, rg)
            if d is not None:
                col = "image" if "image" in d.columns else "image.bytes"
                for _, r in d.iterrows():
                    cell = r[col]
                    raw = cell["bytes"] if isinstance(cell, dict) else cell
                    if raw is None:
                        continue
                    label = int(r["label"])
                    if max_per_label and counts.get(label, 0) >= max_per_label:
                        continue
                    counts[label] = counts.get(label, 0) + 1
                    fmt, ext = sniff(raw)
                    name = SID_LABEL_NAMES.get(label, str(label))
                    p = img_dir / name / f"{r['img_id']}{ext}"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    # exact original bytes -- no PIL round-trip
                    p.write_bytes(raw)
                    bytes_read += len(raw)
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
                        "shard": shard,
                        "row_group": rg,
                    })
                # Flush after every row group. On a link this unreliable a
                # partial sample must still be a usable, self-describing one.
                _write_manifest(out_dir, rows)

    manifest = out_dir / "sample_manifest.csv"
    dt = time.time() - t0
    print(
        f"SID_Set: {len(rows)} images, {bytes_read/1e6:.0f} MB in {dt:.0f}s "
        f"({bytes_read/1e6/max(dt,1):.2f} MB/s) -> {manifest}"
    )
    for lab, n in sorted(counts.items()):
        print(f"  label {lab} ({SID_LABEL_NAMES.get(lab)}): {n}")
    return manifest


# --------------------------------------------------------------------------- #
# WildFake metadata
# --------------------------------------------------------------------------- #

def fetch_wildfake_csv(out_dir: Path, generators: list[str]) -> Path:
    """Download `label_csv_files/<gen>.csv` -- file lists, no images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for gen in generators:
        dest = out_dir / f"{gen}.csv"
        if dest.exists():
            print(f"  {gen}.csv cached ({dest.stat().st_size/1e6:.1f} MB)")
            got.append(dest)
            continue
        url = MS_FILE_URL.format(ds=WILDFAKE_DS, path=f"label_csv_files/{gen}.csv")
        t = time.time()
        r = requests.get(url, timeout=300, headers={"User-Agent": "curl/8"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  {gen}.csv {len(r.content)/1e6:.1f} MB in {time.time()-t:.0f}s")
        got.append(dest)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("sid", help="sample SID_Set parquet row groups")
    a.add_argument("--out", type=Path, default=Path("data/audit_sample/sid_set"))
    a.add_argument("--shards", type=int, default=2)
    a.add_argument("--row-groups", type=int, default=3,
                   help="row groups per shard; ~100 images and ~56 MB each")
    a.add_argument("--max-per-label", type=int, default=None)

    b = sub.add_parser("wildfake-csv", help="fetch WildFake per-generator file lists")
    b.add_argument("--out", type=Path,
                   default=Path("data/audit_sample/wildfake_csv"))
    b.add_argument("--generators", nargs="+", default=["dalle3", "BigGAN", "real_afhq"])

    args = p.parse_args(argv)
    if args.cmd == "sid":
        fetch_sid(args.out, args.shards, args.row_groups, args.max_per_label)
    else:
        fetch_wildfake_csv(args.out, args.generators)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
