"""Tests for scripts/cache_features.py, including the Phase 4 augmented path
(PLAN.md §6: "precompute embeddings for K augmented copies per image").

The real backbone (network download, ~350 MB) is never exercised here --
``ClipBackbone`` is monkeypatched with a deterministic stand-in so these tests
run offline and in milliseconds. What's under test is the caching script's own
plumbing: chunking, augmentation expansion, path bookkeeping, and the files it
writes -- not CLIP itself.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts import cache_features as CF

EMBED_DIM = 8


class DummyClipBackbone:
    """Stands in for src.models.clip_backbone.ClipBackbone: same interface,
    no network, no torch model. The "embedding" is a deterministic function of
    pixel content, which is enough to check per-copy variation without a real
    encoder."""

    embed_dim = EMBED_DIM

    def __init__(self, device="cpu", backbone="dummy", pretrained="dummy"):
        self.device = device

    def embed(self, images) -> np.ndarray:
        out = np.zeros((len(images), self.embed_dim), dtype=np.float32)
        for i, img in enumerate(images):
            arr = np.asarray(img.resize((4, 4)).convert("L"), dtype=np.float32).reshape(-1)
            out[i] = arr[: self.embed_dim]
        return out


@pytest.fixture(autouse=True)
def _stub_backbone(monkeypatch):
    monkeypatch.setattr("src.models.clip_backbone.ClipBackbone", DummyClipBackbone)


def _make_manifest(tmp_path, n=3, size=48):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(n):
        rng = np.random.default_rng(i)
        arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        path = img_dir / f"img_{i}.png"
        Image.fromarray(arr, "RGB").save(path)
        rows.append({"image_path": str(path), "label": i % 2})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


# --------------------------------------------------------------------------- #
# Plain (non-augmented) path: unchanged Phase 3 behaviour
# --------------------------------------------------------------------------- #

def test_plain_cache_writes_expected_files_and_shapes(tmp_path):
    manifest = _make_manifest(tmp_path, n=3)
    out = tmp_path / "out"
    meta = CF.cache_split(manifest, out, device="cpu")

    assert (out / "embeddings.npy").exists()
    assert (out / "labels.npy").exists()
    assert (out / "paths.json").exists()
    assert not (out / "degradation.npy").exists()

    emb = np.load(out / "embeddings.npy")
    labels = np.load(out / "labels.npy")
    paths = json.loads((out / "paths.json").read_text())
    assert emb.shape == (3, EMBED_DIM)
    assert labels.shape == (3,)
    assert len(paths) == 3
    assert all("#aug" not in p for p in paths)
    assert meta["augment_copies"] == 0
    assert meta["augment_seed"] is None


# --------------------------------------------------------------------------- #
# Augmented path (Phase 4)
# --------------------------------------------------------------------------- #

def test_augmented_cache_expands_rows_by_k(tmp_path):
    manifest = _make_manifest(tmp_path, n=3)
    out = tmp_path / "out_aug"
    meta = CF.cache_split(manifest, out, device="cpu", augment_copies=4, augment_seed=0)

    emb = np.load(out / "embeddings.npy")
    labels = np.load(out / "labels.npy")
    paths = json.loads((out / "paths.json").read_text())
    degradation = np.load(out / "degradation.npy")

    assert emb.shape == (12, EMBED_DIM)          # 3 images * 4 copies
    assert labels.shape == (12,)
    assert len(paths) == 12
    assert degradation.shape == (12, 12)          # 2 * len(FAMILIES)
    assert meta["n_images"] == 12
    assert meta["augment_copies"] == 4
    assert meta["augment_seed"] == 0


def test_augmented_paths_trace_back_to_the_source_image(tmp_path):
    manifest = _make_manifest(tmp_path, n=2)
    out = tmp_path / "out_aug"
    CF.cache_split(manifest, out, device="cpu", augment_copies=3, augment_seed=0)
    paths = json.loads((out / "paths.json").read_text())

    src_paths = pd.read_csv(manifest)["image_path"].tolist()
    for src in src_paths:
        copies = [p for p in paths if p.startswith(src + "#aug")]
        assert sorted(copies) == [f"{src}#aug{i}" for i in range(3)]


def test_augmented_labels_repeat_the_source_class(tmp_path):
    manifest = _make_manifest(tmp_path, n=3)  # labels 0,1,0
    out = tmp_path / "out_aug"
    CF.cache_split(manifest, out, device="cpu", augment_copies=2, augment_seed=0)
    labels = np.load(out / "labels.npy")
    assert list(labels) == [0, 0, 1, 1, 0, 0]


def test_augmentation_is_reproducible_across_runs(tmp_path):
    manifest = _make_manifest(tmp_path, n=2)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    CF.cache_split(manifest, out_a, device="cpu", augment_copies=3, augment_seed=7)
    CF.cache_split(manifest, out_b, device="cpu", augment_copies=3, augment_seed=7)

    assert np.array_equal(np.load(out_a / "embeddings.npy"), np.load(out_b / "embeddings.npy"))
    assert np.array_equal(np.load(out_a / "degradation.npy"), np.load(out_b / "degradation.npy"))


def test_different_augment_seed_changes_degradation_labels(tmp_path):
    manifest = _make_manifest(tmp_path, n=2)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    CF.cache_split(manifest, out_a, device="cpu", augment_copies=3, augment_seed=0)
    CF.cache_split(manifest, out_b, device="cpu", augment_copies=3, augment_seed=1)
    assert not np.array_equal(np.load(out_a / "degradation.npy"), np.load(out_b / "degradation.npy"))


def test_negative_augment_copies_rejected(tmp_path):
    manifest = _make_manifest(tmp_path, n=1)
    with pytest.raises(ValueError):
        CF.cache_split(manifest, tmp_path / "out", device="cpu", augment_copies=-1)


# --------------------------------------------------------------------------- #
# Frequency fusion path (Phase 5, resumed)
# --------------------------------------------------------------------------- #

def test_fuse_freq_widens_embeddings_by_freq_dim(tmp_path):
    from src.features.frequency import FREQ_DIM

    manifest = _make_manifest(tmp_path, n=3)
    out = tmp_path / "out_fused"
    meta = CF.cache_split(manifest, out, device="cpu", fuse_freq=True)

    emb = np.load(out / "embeddings.npy")
    assert emb.shape == (3, EMBED_DIM + FREQ_DIM)
    assert meta["clip_embed_dim"] == EMBED_DIM
    assert meta["embed_dim"] == EMBED_DIM + FREQ_DIM
    assert meta["fuse_freq"] is True
    assert meta["freq_dim"] == FREQ_DIM


def test_fuse_freq_off_by_default_matches_plain_shape(tmp_path):
    """The additive flag must not change anything about the default path --
    same shape and same meta fields Phase 3/4 already depend on."""
    manifest = _make_manifest(tmp_path, n=3)
    out = tmp_path / "out_plain"
    meta = CF.cache_split(manifest, out, device="cpu")

    emb = np.load(out / "embeddings.npy")
    assert emb.shape == (3, EMBED_DIM)
    assert meta["clip_embed_dim"] == EMBED_DIM
    assert meta["embed_dim"] == EMBED_DIM
    assert meta["fuse_freq"] is False
    assert meta["freq_dim"] == 0


def test_fuse_freq_composes_with_augmentation(tmp_path):
    """Each augmented copy gets frequency features computed on its own
    (already-degraded) pixels -- the whole point of the branch."""
    from src.features.frequency import FREQ_DIM

    manifest = _make_manifest(tmp_path, n=2)
    out = tmp_path / "out_fused_aug"
    meta = CF.cache_split(manifest, out, device="cpu", augment_copies=3,
                           augment_seed=0, fuse_freq=True)

    emb = np.load(out / "embeddings.npy")
    assert emb.shape == (6, EMBED_DIM + FREQ_DIM)  # 2 images * 3 copies
    assert meta["fuse_freq"] is True


def test_fuse_freq_reproducible_across_runs(tmp_path):
    manifest = _make_manifest(tmp_path, n=2)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    CF.cache_split(manifest, out_a, device="cpu", fuse_freq=True)
    CF.cache_split(manifest, out_b, device="cpu", fuse_freq=True)
    assert np.array_equal(np.load(out_a / "embeddings.npy"), np.load(out_b / "embeddings.npy"))


def test_cli_accepts_fuse_freq_flag(tmp_path):
    from src.features.frequency import FREQ_DIM

    manifest = _make_manifest(tmp_path, n=2)
    out = tmp_path / "out_cli_fused"
    rc = CF.main([
        "--manifest", str(manifest), "--out", str(out), "--device", "cpu",
        "--fuse-freq", "--quiet",
    ])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["fuse_freq"] is True
    emb = np.load(out / "embeddings.npy")
    assert emb.shape == (2, EMBED_DIM + FREQ_DIM)


def test_cli_accepts_augment_flags(tmp_path):
    manifest = _make_manifest(tmp_path, n=2)
    out = tmp_path / "out_cli"
    rc = CF.main([
        "--manifest", str(manifest), "--out", str(out), "--device", "cpu",
        "--augment-copies", "2", "--augment-seed", "5", "--quiet",
    ])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["augment_copies"] == 2
    assert meta["augment_seed"] == 5
    assert meta["n_images"] == 4
