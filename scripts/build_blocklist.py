"""Build the forbidden-subset content blocklist (PLAN.md §4.3.2).

    python -m scripts.build_blocklist

The forbidden subset is WildFake's reference split: **COCO val2017, 4,998
images** and **DALL-E Advanced, 8,843 images**. Training on either is a
disqualification risk.

Why hashes and not paths
------------------------
`results/audit_wildfake_paths.md` confirmed both prefixes exist and the counts
match exactly. It also found that WildFake has **renamed COCO's files to
`img000000.jpg`**, so the original identifiers are gone. A path rule would
therefore have to encode WildFake's own directory layout, and would silently
stop protecting anything the moment that layout changed -- or the moment someone
obtained the same images from anywhere else, which is the actual risk, because
COCO val2017 is one of the most-redistributed image sets in existence.

So the rule is on content:

* ``sha256`` of the raw file bytes -- exact, zero false positives, catches
  redistributions that did not re-encode.
* ``phash`` at Hamming <= 6 -- survives re-encoding and rescaling, which is what
  WildFake did. Calibrated in `results/audit_sid_set.md` §(e).

Coverage, stated honestly
-------------------------
COCO val2017 is fetched from ``images.cocodataset.org`` and hashed in full.
DALL-E Advanced has no public standalone distribution: it exists only inside
WildFake's ~700 GB of ModelScope zips, which is not a download this project can
make. That gap is recorded in the blocklist's ``gaps`` field, surfaced by
``manifest.build`` and asserted on in the tests, so it is visible rather than
quietly absent. The residual risk is low and bounded -- no source in
``src/data/sources.py`` is DALL-E-derived, and the source registry is asserted
against a DALL-E denylist separately -- but it is a real gap and it is named.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image

from src.data.manifest import Blocklist, phash

COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
#: WildFake keeps 4,998 of COCO's 5,000 val2017 images (audit confirmed the
#: count). We hash all 5,000: a superset is strictly safer, and guessing which
#: two were dropped is not possible from a file list.
COCO_EXPECTED = 5000

#: Repos that must never appear as a training source. A blunt second line of
#: defence for the part of the forbidden set we cannot hash.
DALLE_DENY_SUBSTRINGS = ("dalle", "dall-e", "dall_e", "DALLE", "DALL-E")


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    if dest.exists() and dest.stat().st_size > 700_000_000:
        print(f"  cached {dest} ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    t0, n = time.time(), 0
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for b in r.iter_content(chunk):
                fh.write(b)
                n += len(b)
    tmp.replace(dest)
    print(f"  {dest.name}: {n/1e6:.0f} MB in {time.time()-t0:.0f}s "
          f"({n/1e6/max(time.time()-t0,1):.2f} MB/s)")
    return dest


def hash_zip(zip_path: Path, limit: int | None = None) -> tuple[set[str], list[int], int]:
    """sha256 + phash of every image inside a zip, without extracting it.

    Streaming from the archive keeps the peak disk cost at the zip's own size;
    extracting COCO val2017 would double it for no benefit.
    """
    shas: set[str] = set()
    phs: list[int] = []
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            i.filename for i in zf.infolist()
            if not i.is_dir() and i.filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for i, name in enumerate(names[: limit or len(names)]):
            raw = zf.read(name)
            shas.add(hashlib.sha256(raw).hexdigest())
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    phs.append(phash(im.convert("RGB")))
            except Exception:
                pass          # sha256 still covers it; phash is the bonus layer
            n += 1
            if (i + 1) % 1000 == 0:
                print(f"    hashed {i+1}/{len(names)}", flush=True)
    return shas, phs, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/forbidden/blocklist.json"))
    ap.add_argument("--work", type=Path, default=Path("data/forbidden"))
    ap.add_argument("--limit", type=int, default=None, help="hash only N (testing)")
    ap.add_argument("--keep-zip", action="store_true")
    a = ap.parse_args(argv)

    bl = Blocklist()

    print("COCO val2017 (WildFake reference subset, 4,998 of these)")
    zp = download(COCO_VAL2017_URL, a.work / "val2017.zip")
    shas, phs, n = hash_zip(zp, a.limit)
    bl.sha256 |= shas
    bl.phash += phs
    bl.sources.append(f"COCO val2017: {n} images, {len(shas)} sha256, {len(phs)} phash")
    print(f"  -> {n} images, {len(shas)} unique sha256, {len(phs)} phash")
    if n < COCO_EXPECTED and not a.limit:
        bl.gaps.append(f"COCO val2017: hashed {n}, expected {COCO_EXPECTED}")

    bl.gaps.append(
        "DALL-E Advanced (8,843 imgs): no public standalone distribution -- it "
        "ships only inside WildFake's ModelScope zips (~700 GB), which is not a "
        "feasible download here. Not hashed. Mitigated by the source-registry "
        "denylist asserted in tests/test_manifest.py::"
        "test_no_dalle_derived_source_in_the_registry."
    )

    if not a.keep_zip:
        zp.unlink(missing_ok=True)
        print(f"  removed {zp}")

    bl.save(a.out)
    print(f"\n-> {a.out}: {len(bl.sha256)} sha256, {len(bl.phash)} phash, "
          f"{len(bl.gaps)} recorded gap(s)")
    for g in bl.gaps:
        print(f"   GAP: {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
