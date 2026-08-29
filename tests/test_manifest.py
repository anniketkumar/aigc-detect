"""The four manifest invariants from PLAN.md §4.3, as tests.

Three of these are cheap to write and expensive to get wrong:

* the forbidden-subset rule is a **disqualification** risk, not a style point;
* a generator that leaks into train destroys the cross-generator claim, which is
  the strongest claim the project makes;
* a near-duplicate spanning a split boundary inflates test AUROC by an amount
  nobody would notice.

Most tests run against a *synthetic* corpus built in ``tmp_path`` -- real
encoded files, but content we control, so we can plant a near-duplicate and a
forbidden image and assert they are actually caught. A trailing group runs
against the real downloaded corpus when one is present, and skips otherwise, so
the suite stays green on a fresh clone but still checks the real artifact on a
machine that has it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import manifest as M
from src.data.manifest import (
    Blocklist, SplitConfig, cluster_near_duplicates, hamming, phash,
)
from src.data.sources import SOURCES, holdout_sources

REAL_LEDGER = Path("data/corpus/ledger.csv")
SMOKE_LEDGER = Path("data/corpus_smoke/ledger.csv")


# --------------------------------------------------------------------------- #
# A synthetic corpus we can plant things in
# --------------------------------------------------------------------------- #

def _img(seed: int, size: int = 512) -> Image.Image:
    """Distinct photograph-*like* content, deterministic per seed.

    Broad spectrum, deliberately. pHash thresholds the top-left 8x8 DCT
    coefficients against their median, so what makes it stable is those 64
    coefficients having *decisive* signs. A photograph has energy spread across
    all of them. An earlier version of this fixture used three low-frequency
    sinusoids: most coefficients then sat near zero, their signs were noise, and
    a plain 2x rescale moved the hash 12 bits -- above the threshold -- which
    looked like the threshold-6 calibration being wrong when it was the fixture
    that was unphotographic.

    Measured on the real normalized corpus (24 images, 276 pairs): unrelated
    pairs are >= 20 bits apart, and JPEG q30 or a 4x rescale move the hash by 0.
    Fourteen components reproduce that profile; three do not.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    a = np.zeros((size, size), np.float32)
    for _ in range(14):
        fx, fy = rng.uniform(0.5, 6), rng.uniform(0.5, 6)
        a += rng.uniform(0.4, 1.0) * np.sin(
            2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 6.283))
    a = (a - a.min()) / (np.ptp(a) + 1e-9) * 230 + 12
    a = a + rng.normal(0, 3, (size, size))
    return Image.fromarray(
        np.clip(np.stack([a, a * 0.93 + seed % 25, a * 1.06 - seed % 20], -1), 0, 255)
        .astype(np.uint8), "RGB",
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Build a ledger + images with a known near-duplicate pair planted.

    Six generators, two flagged holdout, so every invariant has something to
    bite on.

    Module-scoped: building 145 512x512 images and JPEG-encoding them takes a
    couple of seconds, and a function-scoped fixture rebuilt it for each of the
    ~15 tests that need it. Nothing here mutates the corpus, so sharing it is
    safe; the tests that *do* need a variant write their own ledger alongside.
    """
    root = tmp_path_factory.mktemp("corpus")
    (root / "images").mkdir(parents=True)
    rows = []
    gens = [
        ("SDXL", 1, "sdxl", False), ("Aura", 1, "aura", False),
        ("Pexels", 0, "pexels", False), ("Megalith-Flickr", 0, "flickr", False),
        ("MidJourney", 1, "midjourney", True),
        ("Gemini-nano-banana", 1, "gemini", True),
    ]
    seed = 0
    for gname, label, family, hold in gens:
        for i in range(24):
            seed += 1
            img = _img(seed)
            p = root / "images" / f"{gname}_{i:03d}.jpg"
            img.save(p, "JPEG", quality=95)
            raw = p.read_bytes()
            rows.append({
                "image_path": p.as_posix(), "label": label, "generator": gname,
                "family": family, "source_dataset": f"test/{gname}",
                "split_hint": "holdout" if hold else "train_pool",
                "src_sha256": hashlib.sha256(raw).hexdigest(),
                "out_sha256": hashlib.sha256(raw).hexdigest(),
                "src_width": 512, "src_height": 512, "src_format": "JPEG",
                "src_jpeg_quality": 95, "crop_x": 0, "crop_y": 0,
                "first_gen_quality": 0, "first_gen_subsampling": "",
                "load_status": "ok", "pad_frac": 0.0, "n_bytes": len(raw),
                "shard": 0, "row_group": 0, "config_hash": "test",
            })

    # --- plant a near-duplicate: same picture, rescaled and recompressed ---- #
    # Exactly the case the threshold was calibrated for. It must end up in the
    # same split as its twin.
    twin_src = Image.open(rows[0]["image_path"])
    twin = twin_src.resize((256, 256), Image.Resampling.BICUBIC).resize(
        (512, 512), Image.Resampling.BICUBIC)
    tp = root / "images" / "SDXL_dup.jpg"
    twin.save(tp, "JPEG", quality=40)
    raw = tp.read_bytes()
    rows.append({**rows[0], "image_path": tp.as_posix(),
                 "src_sha256": hashlib.sha256(raw).hexdigest(),
                 "out_sha256": hashlib.sha256(raw).hexdigest()})

    ledger = root / "ledger.csv"
    pd.DataFrame(rows).to_csv(ledger, index=False)
    return root, ledger, rows


@pytest.fixture(scope="module")
def blocklist_file(tmp_path_factory, corpus):
    """A blocklist containing one image that IS in the corpus.

    The point of the fixture: a blocklist that matches nothing makes the §4.3.2
    test pass vacuously. This one is guaranteed to fire.
    """
    _, _, rows = corpus
    victim = rows[5]
    bl = Blocklist(
        sha256={victim["src_sha256"]},
        phash=[phash(Image.open(victim["image_path"]))],
        sources=["synthetic test victim"],
    )
    p = tmp_path_factory.mktemp("bl") / "blocklist.json"
    bl.save(p)
    return p, victim


@pytest.fixture(scope="module")
def built(corpus, blocklist_file, tmp_path_factory):
    root, ledger, rows = corpus
    bl_path, victim = blocklist_file
    out = tmp_path_factory.mktemp("manifests")
    stats = M.build(ledger, out, bl_path, SplitConfig(seed=0), progress=False)
    frames = {n: pd.read_csv(out / f"{n}.csv") for n in ("train", "val", "test")}
    detail = pd.read_csv(out / "detail.csv")
    return stats, frames, detail, victim


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_manifests_have_exactly_the_required_columns(built):
    _, frames, _, _ = built
    for name, df in frames.items():
        assert list(df.columns) == M.MANIFEST_COLUMNS, f"{name}.csv schema drift"


def test_every_image_lands_in_exactly_one_split(built):
    stats, frames, detail, _ = built
    paths = [p for df in frames.values() for p in df["image_path"]]
    assert len(paths) == len(set(paths)), "an image appears in two splits"
    assert len(paths) == len(detail) == stats["n_images"]


def test_all_three_splits_are_non_empty(built):
    _, frames, _, _ = built
    for name, df in frames.items():
        assert len(df) > 0, f"{name} split is empty"


def test_every_listed_image_exists_on_disk(built):
    _, frames, _, _ = built
    for name, df in frames.items():
        missing = [p for p in df["image_path"] if not Path(p).exists()]
        assert not missing, f"{name}: {len(missing)} manifest rows point at nothing"


# --------------------------------------------------------------------------- #
# §4.3.2 -- the forbidden WildFake reference subset
# --------------------------------------------------------------------------- #

def test_forbidden_image_is_kept_out_of_train(built):
    stats, frames, detail, victim = built
    assert stats["n_blocked"] >= 1, "the blocklist never fired -- test is vacuous"
    assert victim["image_path"] not in set(frames["train"]["image_path"]), (
        "a blocklisted image reached train.csv -- this is the disqualification risk"
    )


def test_blocking_is_by_content_hash_not_by_path(tmp_path, corpus):
    """WildFake renames COCO files to img000000.jpg, so paths cannot enforce this.

    The victim here is copied to a brand-new, innocuous filename before the
    manifest is built. A path-based rule sees a new file and lets it through; a
    content rule still catches it.
    """
    root, ledger, rows = corpus
    victim = rows[7]
    renamed = root / "images" / "totally_innocent_img000000.jpg"
    renamed.write_bytes(Path(victim["image_path"]).read_bytes())

    df = pd.read_csv(ledger)
    df.loc[df["image_path"] == victim["image_path"], "image_path"] = renamed.as_posix()
    ledger2 = root / "ledger_renamed.csv"
    df.to_csv(ledger2, index=False)

    bl = Blocklist(sha256={victim["src_sha256"]}, sources=["renamed victim"])
    bp = tmp_path / "bl2.json"
    bl.save(bp)

    out = tmp_path / "m2"
    stats = M.build(ledger2, out, bp, SplitConfig(seed=0), progress=False)
    train = pd.read_csv(out / "train.csv")
    assert stats["n_blocked"] >= 1
    assert renamed.as_posix() not in set(train["image_path"])


def test_a_reencoded_forbidden_image_is_still_caught(tmp_path, corpus):
    """The sha256 layer alone would pass this; phash is why both layers exist."""
    root, ledger, rows = corpus
    victim = rows[9]
    src = Image.open(victim["image_path"])
    re_enc = root / "images" / "reencoded.jpg"
    src.resize((400, 400), Image.Resampling.BICUBIC).resize(
        (512, 512), Image.Resampling.BICUBIC).save(re_enc, "JPEG", quality=45)

    assert hashlib.sha256(re_enc.read_bytes()).hexdigest() != victim["src_sha256"], (
        "fixture is not actually re-encoded"
    )

    df = pd.read_csv(ledger)
    df.loc[df["image_path"] == victim["image_path"], "image_path"] = re_enc.as_posix()
    df.loc[df["image_path"] == re_enc.as_posix(), "src_sha256"] = hashlib.sha256(
        re_enc.read_bytes()).hexdigest()
    l2 = root / "ledger_reenc.csv"
    df.to_csv(l2, index=False)

    bl = Blocklist(phash=[phash(src)], sources=["phash-only victim"])
    bp = tmp_path / "bl3.json"
    bl.save(bp)

    out = tmp_path / "m3"
    stats = M.build(l2, out, bp, SplitConfig(seed=0), progress=False)
    assert stats["n_blocked"] >= 1, "phash layer failed to catch a re-encode"
    assert re_enc.as_posix() not in set(pd.read_csv(out / "train.csv")["image_path"])


def test_an_empty_blocklist_is_reported_not_silently_accepted(tmp_path, corpus):
    """An empty blocklist makes every other §4.3.2 assertion pass for free.

    That is a worse failure than having no test at all, so ``build`` reports it
    and this pins that it does.
    """
    root, ledger, _ = corpus
    out = tmp_path / "m4"
    stats = M.build(ledger, out, tmp_path / "does_not_exist.json",
                    SplitConfig(seed=0), progress=False)
    assert stats["blocklist_empty"] is True
    assert stats["blocklist_gaps"], "an absent blocklist must record why"


def test_no_dalle_derived_source_in_the_registry():
    """Second line of defence for the un-hashable half of the forbidden set.

    DALL-E Advanced (8,843 images) ships only inside WildFake's ~700 GB of
    ModelScope zips, so its content hashes cannot be obtained here. What *can*
    be enforced is that no source in the registry is DALL-E-derived at all.
    """
    from scripts.build_blocklist import DALLE_DENY_SUBSTRINGS
    bad = [
        s.name for s in SOURCES
        if any(t.lower() in f"{s.repo} {s.name}".lower() for t in DALLE_DENY_SUBSTRINGS)
    ]
    assert not bad, f"DALL-E-derived source(s) in the registry: {bad}"


# --------------------------------------------------------------------------- #
# §4.3.1 -- held-out generators
# --------------------------------------------------------------------------- #

def test_at_least_two_generators_are_fully_held_out_of_train(built):
    stats, frames, detail, _ = built
    train_gens = set(frames["train"]["generator"])
    all_gens = set(detail["generator"])
    held = all_gens - train_gens
    assert len(held) >= 2, (
        f"only {len(held)} generator(s) held out ({held}); §4.3.1 requires >= 2"
    )


def test_holdout_generators_appear_nowhere_but_test(built):
    _, frames, detail, _ = built
    held = set(detail.loc[detail["holdout"].astype(bool), "generator"])
    assert held, "fixture has no holdout generators -- test is vacuous"
    for name in ("train", "val"):
        leaked = set(frames[name]["generator"]) & held
        assert not leaked, f"held-out generator(s) {leaked} leaked into {name}.csv"
    assert held <= set(frames["test"]["generator"])


def test_the_registry_itself_declares_at_least_two_holdouts():
    """The fixture could satisfy §4.3.1 while the real config does not."""
    assert len(holdout_sources()) >= 2, "src/data/sources.py holds out < 2 generators"


def test_held_out_generators_are_distinct_families():
    """Holding out two variants of one family is not a cross-generator test."""
    fams = {s.family for s in holdout_sources()}
    assert len(fams) >= 2, f"all held-out generators share family/families {fams}"


# --------------------------------------------------------------------------- #
# Near-duplicates
# --------------------------------------------------------------------------- #

def test_the_planted_near_duplicate_is_detected(built):
    _, _, detail, _ = built
    d = detail.set_index("image_path")
    a = [p for p in d.index if p.endswith("SDXL_000.jpg")][0]
    b = [p for p in d.index if p.endswith("SDXL_dup.jpg")][0]
    dist = hamming(int(d.loc[a, "phash"]), int(d.loc[b, "phash"]))
    assert dist <= M.DUP_THRESHOLD, (
        f"planted duplicate is {dist} bits away, above the {M.DUP_THRESHOLD} "
        "threshold -- the fixture or the threshold is wrong"
    )
    assert d.loc[a, "dup_cluster"] == d.loc[b, "dup_cluster"]


def test_no_near_duplicate_pair_spans_a_split_boundary(built):
    """The invariant itself, checked over every pair, not just the planted one."""
    _, _, detail, _ = built
    hashes = [int(h) for h in detail["phash"]]
    _, pairs = cluster_near_duplicates(hashes, M.DUP_THRESHOLD)
    splits = list(detail["split"])
    straddling = [
        (detail["image_path"].iloc[i], detail["image_path"].iloc[j], d)
        for i, j, d in pairs if splits[i] != splits[j]
    ]
    assert not straddling, (
        f"{len(straddling)} near-duplicate pair(s) span a split boundary, "
        f"e.g. {straddling[0]}"
    )


def test_clustering_is_exact_for_the_threshold():
    """LSH banding must not miss pairs that a brute-force scan would find.

    4 bands of 16 bits: by pigeonhole, 6 differing bits cannot touch all four
    bands, so any pair within threshold collides on at least one band. This
    checks that reasoning against brute force on random data with planted pairs.
    """
    rng = np.random.default_rng(0)
    base = [int(rng.integers(0, 1 << 63)) for _ in range(300)]
    for i in range(0, 40, 2):                       # plant near-duplicates
        flip = 0
        for b in rng.choice(64, size=int(rng.integers(0, 7)), replace=False):
            flip |= 1 << int(b)
        base[i + 1] = base[i] ^ flip

    _, pairs = cluster_near_duplicates(base, M.DUP_THRESHOLD)
    found = {(min(i, j), max(i, j)) for i, j, _ in pairs}
    brute = {
        (i, j) for i in range(len(base)) for j in range(i + 1, len(base))
        if hamming(base[i], base[j]) <= M.DUP_THRESHOLD
    }
    assert found == brute, f"LSH missed {brute - found}, invented {found - brute}"


def test_clusters_are_transitive():
    """A~B and B~C puts all three together even if A and C are far apart."""
    a = 0
    b = 0b111 << 10          # 3 bits from a
    c = 0b111111 << 10       # 6 from a... construct explicitly instead
    a, b, c = 0, 0b1111, 0b11111111
    assert hamming(a, b) <= M.DUP_THRESHOLD and hamming(b, c) <= M.DUP_THRESHOLD
    assert hamming(a, c) > M.DUP_THRESHOLD
    clusters, _ = cluster_near_duplicates([a, b, c], M.DUP_THRESHOLD)
    assert clusters[0] == clusters[1] == clusters[2]


# --------------------------------------------------------------------------- #
# §4.3.3 -- class balance
# --------------------------------------------------------------------------- #

def test_class_balance_is_logged_per_split(built):
    stats, _, _, _ = built
    assert "by_split_label" in stats and stats["by_split_label"]
    assert "by_split_generator" in stats


def test_train_and_val_are_not_wildly_imbalanced(built):
    """Not a hard 50/50 -- holdout generators are all AI and all land in test,
    so test is expected to skew. Train and val are the ones that must be sane."""
    _, frames, _, _ = built
    for name in ("train", "val"):
        df = frames[name]
        share = (df["label"] == 1).mean()
        assert 0.25 < share < 0.75, f"{name} is {share:.0%} AI"


def test_splits_are_reproducible(corpus, blocklist_file, tmp_path):
    root, ledger, _ = corpus
    bp, _ = blocklist_file
    a = M.build(ledger, tmp_path / "a", bp, SplitConfig(seed=0), progress=False)
    b = M.build(ledger, tmp_path / "b", bp, SplitConfig(seed=0), progress=False)
    assert a["splits"] == b["splits"]
    assert pd.read_csv(tmp_path / "a" / "train.csv").equals(
        pd.read_csv(tmp_path / "b" / "train.csv"))


def test_a_different_seed_gives_a_different_split(corpus, blocklist_file, tmp_path):
    root, ledger, _ = corpus
    bp, _ = blocklist_file
    M.build(ledger, tmp_path / "a", bp, SplitConfig(seed=0), progress=False)
    M.build(ledger, tmp_path / "b", bp, SplitConfig(seed=99), progress=False)
    a = set(pd.read_csv(tmp_path / "a" / "train.csv")["image_path"])
    b = set(pd.read_csv(tmp_path / "b" / "train.csv")["image_path"])
    assert a != b, "seed has no effect on the split"


# --------------------------------------------------------------------------- #
# Against the real corpus, when there is one
# --------------------------------------------------------------------------- #

def _real_ledger() -> Path | None:
    for p in (REAL_LEDGER, SMOKE_LEDGER):
        if p.exists() and len(pd.read_csv(p)) > 0:
            return p
    return None


@pytest.mark.skipif(_real_ledger() is None,
                    reason="no downloaded corpus; run scripts/download_data.py")
def test_real_corpus_manifest_builds_and_holds_every_invariant(tmp_path):
    ledger = _real_ledger()
    out = tmp_path / "real_manifests"
    stats = M.build(ledger, out, Path("data/forbidden/blocklist.json"),
                    SplitConfig(seed=0), progress=False)

    frames = {n: pd.read_csv(out / f"{n}.csv") for n in ("train", "val", "test")}
    detail = pd.read_csv(out / "detail.csv")

    for name, df in frames.items():
        assert list(df.columns) == M.MANIFEST_COLUMNS

    # every path resolves
    for name, df in frames.items():
        assert all(Path(p).exists() for p in df["image_path"]), f"{name} has dead paths"

    # near-duplicates do not straddle
    hashes = [int(h) for h in detail["phash"]]
    _, pairs = cluster_near_duplicates(hashes, M.DUP_THRESHOLD)
    splits = list(detail["split"])
    assert not [(i, j) for i, j, _ in pairs if splits[i] != splits[j]]

    # nothing blocklisted in train
    train_paths = set(frames["train"]["image_path"])
    blocked = set(detail.loc[detail["blocked"].fillna("") != "", "image_path"])
    assert not (train_paths & blocked)

    print(json.dumps(stats["splits"]), stats["by_split_label"])
