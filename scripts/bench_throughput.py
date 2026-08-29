"""Measure download and disk throughput *on the machine that will do the work*.

The 0.8-1.85 MB/s in ``results/audit_sid_set.md`` was measured on a laptop. The
download runs on Colab. Those differ by one to two orders of magnitude -- Colab
sits inside GCP with a short path to Hugging Face's CDN -- and the whole data
plan (how many generators, how many images each, whether to stage to Drive)
turns on which number is real.

Run this in a Colab cell before running ``download_data.py``:

    !git clone <repo> && cd <repo> && python -m scripts.bench_throughput

Three things get measured, because they fail differently:

``cdn``     Raw HTTPS range read from the HF CDN. The ceiling.
``parquet`` A real row-group read through ``HfFileSystem`` + pyarrow -- what the
            downloader actually does. Slower than ``cdn`` because pyarrow issues
            many small range requests, and that per-request latency is what
            dominates on a high-latency link. The gap between these two numbers
            is the argument for reading whole row groups rather than rows.
``disk``    Sequential write. Colab's local disk is fast; Google Drive mounted
            over FUSE is roughly 100x slower for many small files, which is the
            single most common way a Colab data pipeline silently stalls.

Nothing is cached between runs and every temporary file is removed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import requests

#: Public, no-auth parquet shards used as the probe targets. Chosen small
#: enough that the benchmark costs well under a minute even on a slow link.
PROBES = [
    ("SID_Set", "saberzl/SID_Set", "data/validation-00000-of-00034.parquet"),
    ("GenImage_BigGAN", "bitmind/GenImage_BigGAN", "data/train-00000-of-00008.parquet"),
]

RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _fmt(mbps: float) -> str:
    return f"{mbps:7.2f} MB/s" + (f"  ({mbps*3.6:.0f} GB/h)" if mbps > 0 else "")


# --------------------------------------------------------------------------- #
# 1. Raw CDN range read
# --------------------------------------------------------------------------- #

def bench_cdn(repo: str, path: str, mb: int = 48, reps: int = 3) -> dict:
    """Sequential range GETs. Measures the ceiling, with no parquet overhead.

    Reads a *different* offset each rep so a CDN edge cache does not turn rep 2
    onward into a memory read and flatter the result.
    """
    url = RESOLVE.format(repo=repo, path=path)
    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers.get("content-length", 0))
    if total == 0:
        return {"error": "no content-length", "url": url}

    chunk = min(mb * 1024 * 1024, total // max(reps, 1))
    rates = []
    for i in range(reps):
        start = (i * chunk) % max(total - chunk, 1)
        t = time.perf_counter()
        n = 0
        with requests.get(
            url, headers={"Range": f"bytes={start}-{start+chunk-1}"},
            stream=True, timeout=300,
        ) as r:
            r.raise_for_status()
            for block in r.iter_content(1 << 20):
                n += len(block)
        dt = time.perf_counter() - t
        rates.append(n / 1e6 / max(dt, 1e-6))
    return {
        "file_bytes": total,
        "chunk_mb": round(chunk / 1e6, 1),
        "reps": reps,
        "mbps_median": round(statistics.median(rates), 2),
        "mbps_all": [round(x, 2) for x in rates],
    }


# --------------------------------------------------------------------------- #
# 2. Parquet row-group read (what the downloader really does)
# --------------------------------------------------------------------------- #

def bench_parquet(repo: str, path: str, row_groups: int = 1) -> dict:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        return {"error": f"missing dep: {exc}"}

    full = f"datasets/{repo}/{path}"
    t0 = time.perf_counter()
    try:
        with HfFileSystem().open(full, "rb") as fh:
            pf = pq.ParquetFile(fh)
            meta = pf.metadata
            t_footer = time.perf_counter() - t0
            n = min(row_groups, meta.num_row_groups)
            t1 = time.perf_counter()
            rows = 0
            nbytes = 0
            for rg in range(n):
                tbl = pf.read_row_group(rg)
                rows += tbl.num_rows
                nbytes += meta.row_group(rg).total_byte_size
            dt = time.perf_counter() - t1
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "footer_read_s": round(t_footer, 2),
        "row_groups": n,
        "rows": rows,
        "uncompressed_mb": round(nbytes / 1e6, 1),
        "seconds": round(dt, 2),
        "mbps": round(nbytes / 1e6 / max(dt, 1e-6), 2),
        "images_per_s": round(rows / max(dt, 1e-6), 1),
    }


# --------------------------------------------------------------------------- #
# 3. Disk
# --------------------------------------------------------------------------- #

def bench_disk(target: Path, mb: int = 256, n_small: int = 400) -> dict:
    """Sequential write, then many small writes.

    Both, because they diverge wildly on network filesystems: a Drive mount can
    do 30 MB/s sequential and still take 20 ms per file, which is what actually
    kills a 20k-image normalize loop.
    """
    target.mkdir(parents=True, exist_ok=True)
    blob = os.urandom(1 << 20)

    big = target / "_bench_seq.bin"
    t = time.perf_counter()
    with big.open("wb") as fh:
        for _ in range(mb):
            fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    seq = mb / max(time.perf_counter() - t, 1e-6)
    big.unlink(missing_ok=True)

    small_dir = target / "_bench_small"
    small_dir.mkdir(exist_ok=True)
    payload = blob[:200_000]                      # ~ one normalized 512px JPEG
    t = time.perf_counter()
    for i in range(n_small):
        (small_dir / f"{i:05d}.bin").write_bytes(payload)
    dt = time.perf_counter() - t
    shutil.rmtree(small_dir, ignore_errors=True)

    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "seq_write_mbps": round(seq, 1),
        "small_files_per_s": round(n_small / max(dt, 1e-6), 1),
        "small_file_ms": round(dt / n_small * 1000, 2),
        "free_gb": round(usage.free / 1e9, 1),
        "total_gb": round(usage.total / 1e9, 1),
    }


# --------------------------------------------------------------------------- #

def describe_host() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "in_colab": "google.colab" in sys.modules or os.path.exists("/content"),
        "in_kaggle": os.path.exists("/kaggle"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except Exception:
        info["torch"] = None
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--disk-path", type=Path, default=None,
                    help="where the data will land; defaults to a temp dir")
    ap.add_argument("--cdn-mb", type=int, default=48)
    ap.add_argument("--skip-parquet", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/throughput.json"))
    a = ap.parse_args(argv)

    report = {"host": describe_host(), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
              "cdn": {}, "parquet": {}}
    print(json.dumps(report["host"], indent=2))

    for name, repo, path in PROBES:
        print(f"\n[cdn] {name} ...", flush=True)
        c = bench_cdn(repo, path, mb=a.cdn_mb)
        report["cdn"][name] = c
        print(f"      {_fmt(c.get('mbps_median', 0)):>28s}   {c}")

        if not a.skip_parquet:
            print(f"[parquet] {name} ...", flush=True)
            p = bench_parquet(repo, path)
            report["parquet"][name] = p
            print(f"      {_fmt(p.get('mbps', 0)):>28s}   {p}")

    disk_path = a.disk_path or Path(tempfile.gettempdir()) / "aigc_bench"
    print(f"\n[disk] {disk_path} ...", flush=True)
    report["disk"] = bench_disk(disk_path)
    print(f"      {report['disk']}")

    med = [v["mbps_median"] for v in report["cdn"].values() if "mbps_median" in v]
    if med:
        mbps = statistics.median(med)
        report["headline_cdn_mbps"] = round(mbps, 2)
        print(f"\nHeadline CDN throughput: {_fmt(mbps)}")
        for gb in (5, 10, 20, 40):
            print(f"  {gb:>3d} GB would take {gb*1000/mbps/60:6.1f} min")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
