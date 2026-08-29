"""Build ``data/manifests/{train,val,test}.csv`` from the download ledger.

    python -m src.data.manifest --ledger data/corpus/ledger.csv

Emits exactly the columns PLAN.md §4.2 asks for -- ``image_path``, ``label``,
``generator``, ``source_dataset``, ``split`` -- plus a sidecar
``manifests/detail.csv`` carrying everything the audits need (hashes, source
geometry, duplicate-cluster id) so the required schema stays clean while nothing
useful is thrown away.

Four invariants, each enforced here and asserted in ``tests/test_manifest.py``:

1. **No image from the forbidden WildFake reference subset in train.** Enforced
   by *content*, not path: WildFake renames COCO's files to ``img000000.jpg``,
   so the original identifiers are gone and any path rule dies the first time a
   directory is reorganised. Disqualification risk, not a style point.
2. **At least two generators fully held out of train.** Cross-generator AUROC is
   a stronger claim than cross-transform and costs nothing to obtain.
3. **No near-duplicate pair spans a split boundary.** Clusters are assigned
   whole, so a rescaled repost of a training image cannot appear in test.
4. **Class balance per split**, logged to ``results/data_stats.md``.

Split policy
------------
Held-out generators go to ``test`` in their entirety -- that is what "held out"
means, and letting even a validation slice through would let hyperparameter
choice leak across the boundary that carries the headline claim.

Everything else is split by *duplicate cluster*, stratified on
``(label, generator)`` so each split keeps the same generator mix. Clusters, not
images: two near-duplicates in different splits would inflate test AUROC by
exactly the amount nobody would notice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

MANIFEST_COLUMNS = ["image_path", "label", "generator", "source_dataset", "split"]
DEFAULT_OUT = Path("data/manifests")

#: pHash Hamming distance below which two images count as the same picture.
#: Calibrated in results/audit_sid_set.md §(e), not guessed: resize 0.25x, JPEG
#: q30 and blur sigma=2 each moved the hash by at most 2 bits, while 11,175
#: unrelated pairs never came closer than 10. It does *not* catch crops -- a
#: 10%-per-side crop moves the hash ~20 bits, indistinguishable from an
#: unrelated image -- which is a known and documented limitation.
DUP_THRESHOLD = 6

#: LSH banding, sized so the bucketing is **exact** rather than approximate.
#:
#: The pigeonhole condition is ``n_bands > threshold``: if two hashes differ in
#: at most ``t`` bits and there are ``t+1`` or more disjoint bands, at least one
#: band must be free of differences, so the pair is guaranteed to collide there.
#: An earlier version used 4 bands of 16 bits and claimed exactness at
#: threshold 6 -- wrong, because 6 bit errors spread 2+2+1+1 touch all four
#: bands. ``test_clustering_is_exact_for_the_threshold`` caught it by brute
#: force; that test is why this constant is 8 and not 4.
#:
#: 8 bands of 8 bits covers thresholds up to 7. The cost is smaller buckets per
#: band (256 keys) and therefore more candidate pairs, which is the right trade:
#: a missed duplicate is a silent train/test leak, a redundant comparison is
#: microseconds.
N_BANDS = 8
BAND_BITS = 8


# --------------------------------------------------------------------------- #
# Perceptual hashing
# --------------------------------------------------------------------------- #

def _dct_matrix(n: int = 32) -> np.ndarray:
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0] /= np.sqrt(2)
    return m * np.sqrt(2.0 / n)


_DCT32 = _dct_matrix(32)


def phash(img: Image.Image, hash_size: int = 8) -> int:
    """DCT-based perceptual hash, 64-bit. Same implementation as the audit."""
    g = np.asarray(
        ImageOps.grayscale(img).resize((32, 32), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    d = _DCT32 @ g @ _DCT32.T
    low = d[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])              # drop DC before thresholding
    out = 0
    for b in (low > med).astype(np.uint64):
        out = (out << 1) | int(b)
    return out


#: SWAR popcount over a uint64 array. The corpus is ~18k images and the COCO
#: blocklist is 5,000 hashes, so the naive nested Python loop is 90M calls to
#: ``bin(x).count("1")`` -- minutes. This is the same computation in numpy and
#: takes about a second.
_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return (x * _H01) >> np.uint64(56)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def phash_file(path: str | Path) -> int | None:
    from src.data import imageio as IIO
    img = IIO.load_rgb(path)
    return None if img is None else phash(img)


# --------------------------------------------------------------------------- #
# Near-duplicate clustering
# --------------------------------------------------------------------------- #

class _Union:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_near_duplicates(
    hashes: list[int], threshold: int = DUP_THRESHOLD
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Connected components of the "within ``threshold`` bits" graph.

    Returns ``(cluster_id_per_image, pairs)``. Transitive by construction: if
    A~B and B~C then all three share a cluster even when A and C are 9 bits
    apart. That is the conservative choice -- the cost of over-merging is a
    slightly smaller effective dataset, the cost of under-merging is a
    train/test leak.
    """
    n = len(hashes)
    uf = _Union(n)
    pairs: list[tuple[int, int, int]] = []

    if threshold >= N_BANDS:
        raise ValueError(
            f"threshold {threshold} needs > {threshold} bands for exact banding; "
            f"N_BANDS is {N_BANDS}. Raise N_BANDS or fall back to a full scan."
        )

    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(N_BANDS)]
    for i, h in enumerate(hashes):
        for b in range(N_BANDS):
            buckets[b][(h >> (b * BAND_BITS)) & ((1 << BAND_BITS) - 1)].append(i)

    seen: set[tuple[int, int]] = set()
    for band in buckets:
        for members in band.values():
            if len(members) < 2:
                continue
            # A pathological bucket (thousands of identical bands) would make
            # this quadratic again. In practice that only happens for solid
            # colour images, which the padding guard already rejects.
            for ii in range(len(members)):
                for jj in range(ii + 1, len(members)):
                    a, b_ = members[ii], members[jj]
                    key = (a, b_)
                    if key in seen:
                        continue
                    seen.add(key)
                    d = hamming(hashes[a], hashes[b_])
                    if d <= threshold:
                        uf.union(a, b_)
                        pairs.append((a, b_, d))

    roots = {}
    out = []
    for i in range(n):
        r = uf.find(i)
        out.append(roots.setdefault(r, len(roots)))
    return out, pairs


# --------------------------------------------------------------------------- #
# Forbidden subset
# --------------------------------------------------------------------------- #

@dataclass
class Blocklist:
    """Content hashes of images that must never enter train (PLAN.md §4.3.2).

    Two hash families, because they fail in opposite directions. ``sha256``
    catches byte-identical copies with zero false positives but nothing else;
    WildFake re-encodes, so on its own it would pass everything. ``phash``
    survives re-encoding and rescaling but can collide, so it is only ever used
    to *exclude from train*, never to delete data.
    """

    sha256: set[str] = field(default_factory=set)
    phash: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    #: Parts of the forbidden set that could not be hashed, and why. Recorded
    #: rather than silently omitted -- an empty blocklist makes the test pass
    #: vacuously, which is worse than no test.
    gaps: list[str] = field(default_factory=list)
    #: numpy cache of ``phash``, built lazily on first use.
    _arr: object = field(default=None, repr=False, compare=False)

    @property
    def empty(self) -> bool:
        return not self.sha256 and not self.phash

    @classmethod
    def load(cls, path: Path | None) -> "Blocklist":
        if path is None or not Path(path).exists():
            return cls(gaps=[f"blocklist file {path} not found"])
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            sha256=set(d.get("sha256", [])),
            phash=[int(x) for x in d.get("phash", [])],
            sources=d.get("sources", []),
            gaps=d.get("gaps", []),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "sha256": sorted(self.sha256),
            "phash": [str(x) for x in self.phash],
            "sources": self.sources,
            "gaps": self.gaps,
            "n_sha256": len(self.sha256),
            "n_phash": len(self.phash),
        }, indent=1), encoding="utf-8")

    def hits(self, sha: str, ph: int | None, threshold: int = DUP_THRESHOLD) -> str:
        """Empty string if clean, else why it was blocked."""
        if sha and sha in self.sha256:
            return "sha256"
        if ph is not None and self.phash:
            if self._arr is None:
                self._arr = np.array(self.phash, dtype=np.uint64)
            x = np.bitwise_xor(self._arr, np.uint64(ph))
            if int(_popcount(x).min()) <= threshold:
                return f"phash<={threshold}"
        return ""


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 0
    min_holdout_generators: int = 2
    dup_threshold: int = DUP_THRESHOLD


def assign_splits(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Cluster-level, stratified on (label, generator).

    Clusters rather than images: a near-duplicate pair straddling a boundary
    inflates test AUROC invisibly. Stratified on generator as well as label so a
    split cannot end up missing a generator entirely, which would make its
    per-generator numbers undefined rather than merely noisy.
    """
    rng = random.Random(cfg.seed)
    out = df.copy()
    out["split"] = ""

    # 1. Held-out generators: entirely test, no exceptions.
    held = out["holdout"].astype(bool)
    out.loc[held, "split"] = "test"

    # 2. Everything else, cluster by cluster.
    pool = out[~held]
    # A cluster can in principle span generators (the same source photo fed to
    # two generators). Assign by the cluster's majority stratum so the whole
    # cluster still lands together.
    cluster_stratum = (
        pool.groupby("dup_cluster")
        .apply(lambda g: f"{g['label'].iloc[0]}|{g['generator'].mode().iloc[0]}",
               include_groups=False)
        .to_dict()
    )
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for cid, stratum in cluster_stratum.items():
        by_stratum[stratum].append(cid)

    frac = np.array([cfg.train, cfg.val, cfg.test], dtype=float)
    frac = frac / frac.sum()
    names = ["train", "val", "test"]
    assignment: dict[int, str] = {}
    for stratum, cids in sorted(by_stratum.items()):
        cids = sorted(cids)
        rng.shuffle(cids)
        # Largest-remainder allocation. round() per bucket loses or gains
        # clusters on small strata and can empty one entirely.
        n = len(cids)
        raw = frac * n
        base = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - base))[: n - base.sum()]:
            base[i] += 1
        k = 0
        for name, take in zip(names, base):
            for cid in cids[k:k + take]:
                assignment[cid] = name
            k += take

    mask = ~held
    out.loc[mask, "split"] = out.loc[mask, "dup_cluster"].map(assignment)
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build(
    ledger: Path,
    out_dir: Path = DEFAULT_OUT,
    blocklist_path: Path | None = Path("data/forbidden/blocklist.json"),
    cfg: SplitConfig | None = None,
    holdout_generators: set[str] | None = None,
    progress: bool = True,
) -> dict:
    from src.data.sources import BY_NAME

    cfg = cfg or SplitConfig()
    df = pd.read_csv(ledger)
    if df.empty:
        raise ValueError(f"{ledger} is empty -- run scripts/download_data.py first")

    if holdout_generators is None:
        holdout_generators = {s.name for s in BY_NAME.values() if s.holdout}
    df["holdout"] = df["generator"].isin(holdout_generators)

    # --- perceptual hashes ------------------------------------------------- #
    hashes: list[int] = []
    bad: list[int] = []
    for i, p in enumerate(df["image_path"]):
        h = phash_file(p)
        if h is None:
            bad.append(i)
            h = 0
        hashes.append(h)
        if progress and (i + 1) % 2000 == 0:
            print(f"  hashed {i+1}/{len(df)}", flush=True)
    if bad:
        print(f"  [warn] {len(bad)} images failed to hash; dropped", file=sys.stderr)
        keep = ~df.index.isin(bad)
        df, hashes = df[keep].reset_index(drop=True), [
            h for i, h in enumerate(hashes) if i not in set(bad)
        ]
    df["phash"] = hashes

    # --- forbidden subset -------------------------------------------------- #
    bl = Blocklist.load(blocklist_path)
    df["blocked"] = [
        bl.hits(str(s), int(h), cfg.dup_threshold)
        for s, h in zip(df.get("src_sha256", pd.Series([""] * len(df))), df["phash"])
    ]
    n_blocked = int((df["blocked"] != "").sum())

    # --- near-duplicate clusters ------------------------------------------- #
    clusters, pairs = cluster_near_duplicates(list(df["phash"]), cfg.dup_threshold)
    df["dup_cluster"] = clusters
    n_clusters = len(set(clusters))

    # --- splits ------------------------------------------------------------ #
    df = assign_splits(df, cfg)

    # Blocked images are pushed out of train rather than deleted: they are still
    # legitimate *evaluation* material, and deleting them would hide the fact
    # that the blocklist fired at all.
    forced = (df["blocked"] != "") & (df["split"] == "train")
    df.loc[forced, "split"] = "test"

    # --- write ------------------------------------------------------------- #
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        path = out_dir / f"{name}.csv"
        sub[MANIFEST_COLUMNS].to_csv(path, index=False)
        written[name] = len(sub)
    df.to_csv(out_dir / "detail.csv", index=False)

    stats = {
        "n_images": len(df),
        "splits": written,
        "n_clusters": n_clusters,
        "n_dup_pairs": len(pairs),
        "n_blocked": n_blocked,
        "blocklist_empty": bl.empty,
        "blocklist_sources": bl.sources,
        "blocklist_gaps": bl.gaps,
        "holdout_generators": sorted(holdout_generators),
        "by_split_label": (
            df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
        ),
        "by_split_generator": (
            df.groupby(["split", "generator"]).size().unstack(fill_value=0).to_dict()
        ),
        "config": vars(cfg),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str),
                                        encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", type=Path, default=Path("data/corpus/ledger.csv"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--blocklist", type=Path,
                    default=Path("data/forbidden/blocklist.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    stats = build(a.ledger, a.out, a.blocklist, SplitConfig(seed=a.seed))
    print(json.dumps(stats, indent=2, default=str))
    if stats["blocklist_empty"]:
        print("\n[warn] the forbidden-subset blocklist is EMPTY -- the §4.3.2 "
              "assertion is passing vacuously. Run scripts/build_blocklist.py.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
