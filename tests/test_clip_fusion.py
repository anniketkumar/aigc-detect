"""Tests for src/models/clip_fusion.py -- the Phase 5 fusion Scorer.

Same offline-monkeypatch approach as tests/test_cache_features.py and
tests/test_predict_smoke.py: ClipBackbone is stubbed so no network call or
real CLIP weights are needed, and what's under test is the fusion Scorer's
own plumbing (checkpoint validation, concatenation, registry wiring) rather
than CLIP itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.features.frequency import FREQ_DIM
from src.models import base as B
from src.models.clip_fusion import CLIPFreqFusionScorer
from src.models.semantic_head import LinearHead

EMBED_DIM = 8
FUSED_DIM = EMBED_DIM + FREQ_DIM


class DummyClipBackbone:
    """Same stand-in as tests/test_cache_features.py / test_predict_smoke.py."""

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
    # Patch the name where clip_fusion.py imported it (``from ... import
    # ClipBackbone``), not the source module -- same convention as
    # tests/test_predict_smoke.py patching src.models.clip_baseline.ClipBackbone.
    monkeypatch.setattr("src.models.clip_fusion.ClipBackbone", DummyClipBackbone)


def _random_img(seed: int = 0, size: int = 48) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _write_ckpt(path, embed_dim=FUSED_DIM):
    head = LinearHead(embed_dim)
    torch.save({
        "state_dict": head.state_dict(),
        "backbone": "dummy",
        "pretrained": "dummy",
        "embed_dim": embed_dim,
        "config": {},
    }, path)


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #

def test_registered_as_clip_freq_fusion():
    assert "clip_freq_fusion" in B.MODEL_REGISTRY
    assert B.MODEL_REGISTRY["clip_freq_fusion"] == "src.models.clip_fusion:CLIPFreqFusionScorer"


def test_load_model_resolves_clip_freq_fusion(tmp_path):
    ckpt = tmp_path / "fusion.pt"
    _write_ckpt(ckpt)
    scorer = B.load_model("clip_freq_fusion", ckpt=ckpt, device="cpu")
    assert isinstance(scorer, CLIPFreqFusionScorer)


# --------------------------------------------------------------------------- #
# Scoring behaviour
# --------------------------------------------------------------------------- #

def test_requires_ckpt():
    with pytest.raises(ValueError):
        CLIPFreqFusionScorer(ckpt=None)


def test_score_returns_one_probability_per_image(tmp_path):
    ckpt = tmp_path / "fusion.pt"
    _write_ckpt(ckpt)
    scorer = CLIPFreqFusionScorer(ckpt=ckpt, device="cpu")
    imgs = [_random_img(i) for i in range(3)]
    out = scorer.score(imgs, ["a", "b", "c"])
    assert len(out) == 3
    assert all(0.0 <= p <= 1.0 for p in out)


def test_score_is_deterministic(tmp_path):
    ckpt = tmp_path / "fusion.pt"
    _write_ckpt(ckpt)
    scorer = CLIPFreqFusionScorer(ckpt=ckpt, device="cpu")
    img = [_random_img(5)]
    a = scorer.score(img, ["x"])
    b = scorer.score(img, ["x"])
    assert a == b


def test_rejects_checkpoint_with_wrong_embed_dim(tmp_path):
    """A plain clip_linear checkpoint (embed_dim == clip_embed_dim, no
    frequency block) must fail loudly here, not silently misinterpret its
    weights against a differently-shaped fused input."""
    ckpt = tmp_path / "wrong_dim.pt"
    _write_ckpt(ckpt, embed_dim=EMBED_DIM)  # missing the +FREQ_DIM block
    with pytest.raises(ValueError, match="embed_dim"):
        CLIPFreqFusionScorer(ckpt=ckpt, device="cpu")
