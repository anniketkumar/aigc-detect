"""Smoke test for app.py, the Gradio demo (PLAN.md §9.2).

Never launches a real Gradio server (that would block forever waiting for
connections) -- ``app`` module import already builds the ``gr.Blocks`` object
at module scope, so importing it is itself the wiring check; the tests below
also call the callback functions directly, the same way Gradio's own event
dispatch does.

``ClipBackbone`` is stubbed exactly as in ``tests/test_predict_smoke.py``
(patched on ``src.models.clip_baseline``, not ``src.models.clip_backbone`` --
see that file's comment for why). ``app.CHECKPOINTS`` is hardcoded to
``runs/aug.pt``/``runs/baseline.pt``, so tests monkeypatch it to point at
fake, small checkpoints instead -- both to stay offline and because the real
checkpoints' embed_dim (512) doesn't match the stub backbone's (8).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

import app as A

EMBED_DIM = 8


class DummyClipBackbone:
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
    monkeypatch.setattr("src.models.clip_baseline.ClipBackbone", DummyClipBackbone)


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """app._models is a module-level dict keyed by checkpoint path -- reset it
    so one test's (possibly monkeypatched) checkpoint path never serves a
    model instance cached by another."""
    A._models.clear()
    yield
    A._models.clear()


@pytest.fixture()
def fake_checkpoints(tmp_path, monkeypatch):
    from src.models.semantic_head import LinearHead

    paths = {}
    for name, seed in (("baseline", 0), ("aug", 1)):
        torch.manual_seed(seed)
        p = tmp_path / f"{name}.pt"
        torch.save(
            {
                "state_dict": LinearHead(EMBED_DIM).state_dict(),
                "backbone": "dummy", "pretrained": "dummy",
                "embed_dim": EMBED_DIM, "config": {},
            },
            p,
        )
        paths[name] = p

    labels = {
        "aug -- Phase 4 (+augmentation)": paths["aug"],
        "baseline -- Phase 3 (no augmentation)": paths["baseline"],
    }
    monkeypatch.setattr(A, "CHECKPOINTS", labels)
    return labels


def _test_image_file(tmp_path, name="img.png", seed=0, size=48) -> str:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(arr, "RGB").save(path)
    return str(path)


# --------------------------------------------------------------------------- #
# Module import / wiring
# --------------------------------------------------------------------------- #

def test_module_builds_a_gradio_blocks():
    import gradio as gr
    assert isinstance(A.demo, gr.Blocks)


def test_checkpoints_point_at_the_real_run_paths():
    from pathlib import Path
    assert A.CHECKPOINTS["aug -- Phase 4 (+augmentation)"] == Path("runs/aug.pt")
    assert A.CHECKPOINTS["baseline -- Phase 3 (no augmentation)"] == Path("runs/baseline.pt")
    assert A.DEFAULT_CHECKPOINT in A.CHECKPOINTS


# --------------------------------------------------------------------------- #
# _get_model / caching
# --------------------------------------------------------------------------- #

def test_get_model_caches_by_checkpoint_path(fake_checkpoints):
    m1 = A._get_model("aug -- Phase 4 (+augmentation)")
    m2 = A._get_model("aug -- Phase 4 (+augmentation)")
    assert m1 is m2  # same object, not reloaded
    m3 = A._get_model("baseline -- Phase 3 (no augmentation)")
    assert m3 is not m1


# --------------------------------------------------------------------------- #
# _score / _jpeg_kb / _rescore / _status
# --------------------------------------------------------------------------- #

def test_score_returns_a_probability(fake_checkpoints):
    model = A._get_model(A.DEFAULT_CHECKPOINT)
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), "RGB")
    p = A._score(model, img)
    assert p is None or 0.0 <= p <= 1.0


def test_jpeg_kb_shrinks_as_quality_drops():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8), "RGB")
    kb_high = A._jpeg_kb(img, 95)
    kb_low = A._jpeg_kb(img, 30)
    assert kb_low < kb_high


def test_rescore_uses_a_real_jpeg_roundtrip(fake_checkpoints):
    model = A._get_model(A.DEFAULT_CHECKPOINT)
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8), "RGB")
    degraded, score, kb = A._rescore(img, model, 30)
    assert degraded.size == img.size
    assert degraded.format == "JPEG" or degraded.mode == "RGB"
    assert kb > 0
    assert score is None or 0.0 <= score <= 1.0


def test_status_reports_clean_and_live_scores():
    msg = A._status(None, 0.42, 30, 0.55, 12.3)
    assert "0.420" in msg and "0.550" in msg and "q=30" in msg and "12.3" in msg


def test_status_surfaces_a_warning():
    msg = A._status("short read recovered", 0.1, 90, 0.1, 5.0)
    assert "short read recovered" in msg


def test_status_handles_unscoreable_degraded_image():
    msg = A._status(None, 0.3, 50, None, 4.0)
    assert "could not score" in msg


# --------------------------------------------------------------------------- #
# on_upload / on_quality_or_model_change -- the actual Gradio callbacks
# --------------------------------------------------------------------------- #

def test_on_upload_with_no_file_prompts_for_one(fake_checkpoints):
    result = A.on_upload(None, A.DEFAULT_CHECKPOINT, 95)
    assert result == (None, None, "Upload an image to begin.", None)


def test_on_upload_scores_a_real_file(fake_checkpoints, tmp_path):
    path = _test_image_file(tmp_path)
    decoded, degraded, status, state = A.on_upload(path, A.DEFAULT_CHECKPOINT, 95)
    assert decoded is not None and degraded is not None
    assert state is decoded
    assert "P(AI-generated)" in status


def test_on_upload_on_a_corrupt_file_reports_not_crashes(fake_checkpoints, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    decoded, degraded, status, state = A.on_upload(str(bad), A.DEFAULT_CHECKPOINT, 95)
    assert decoded is None and degraded is None and state is None
    assert "could not decode" in status


def test_on_quality_or_model_change_with_no_state_prompts_for_upload(fake_checkpoints):
    degraded, status = A.on_quality_or_model_change(None, A.DEFAULT_CHECKPOINT, 50)
    assert degraded is None
    assert "Upload an image first" in status


def test_on_quality_or_model_change_rescoes_the_cached_image(fake_checkpoints, tmp_path):
    path = _test_image_file(tmp_path)
    _, _, _, state = A.on_upload(path, A.DEFAULT_CHECKPOINT, 95)
    degraded, status = A.on_quality_or_model_change(state, A.DEFAULT_CHECKPOINT, 30)
    assert degraded is not None
    assert "q=30" in status


def test_switching_checkpoint_uses_the_other_model(fake_checkpoints, tmp_path):
    path = _test_image_file(tmp_path)
    _, _, _, state = A.on_upload(path, A.DEFAULT_CHECKPOINT, 95)
    other = [k for k in fake_checkpoints if k != A.DEFAULT_CHECKPOINT][0]
    degraded, status = A.on_quality_or_model_change(state, other, 95)
    assert degraded is not None
    assert "P(AI-generated)" in status


def test_deterministic_same_image_same_scores(fake_checkpoints, tmp_path):
    path = _test_image_file(tmp_path)
    r1 = A.on_upload(path, A.DEFAULT_CHECKPOINT, 60)
    A._models.clear()  # force a fresh model load, not the cached instance
    r2 = A.on_upload(path, A.DEFAULT_CHECKPOINT, 60)
    assert r1[2] == r2[2]  # identical status text -> identical scores
