"""Smoke checks for the thin HTTP adapter behind the React interface."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as A


class DummyModel:
    def score(self, images, image_ids):
        return [0.72 for _ in images]


def _png_bytes() -> bytes:
    image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8), "RGB")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_health_check_is_available():
    assert TestClient(A.app).get("/api/health").json() == {"status": "ok"}


def test_analyze_uses_canonical_pipeline(monkeypatch):
    monkeypatch.setattr(A, "_get_model", lambda checkpoint: DummyModel())
    response = TestClient(A.app).post(
        "/api/analyze",
        data={"checkpoint": "aug", "quality": "30"},
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["clean_score"] == 0.72
    assert body["reencoded_score"] == 0.72
    assert body["quality"] == 30
    assert body["clean_preview"].startswith("data:image/jpeg;base64,")


def test_analyze_validates_controls():
    response = TestClient(A.app).post(
        "/api/analyze",
        data={"checkpoint": "unknown", "quality": "99"},
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 422


def test_analyze_batch_scores_every_file_and_sorts_by_name(monkeypatch):
    monkeypatch.setattr(A, "_get_model", lambda checkpoint: DummyModel())
    response = TestClient(A.app).post(
        "/api/analyze-batch",
        data={"checkpoint": "aug"},
        files=[
            ("images", ("b.png", _png_bytes(), "image/png")),
            ("images", ("a.png", _png_bytes(), "image/png")),
        ],
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["image_path"] for r in results] == ["a.png", "b.png"]
    assert all(r["pred"] == 0.72 for r in results)


def test_analyze_batch_never_crashes_on_a_bad_file(monkeypatch):
    monkeypatch.setattr(A, "_get_model", lambda checkpoint: DummyModel())
    response = TestClient(A.app).post(
        "/api/analyze-batch",
        data={"checkpoint": "aug"},
        files=[
            ("images", ("ok.png", _png_bytes(), "image/png")),
            ("images", ("empty.png", b"", "image/png")),
            ("images", ("garbage.png", b"not an image", "image/png")),
        ],
    )
    assert response.status_code == 200
    results = {r["image_path"]: r for r in response.json()["results"]}
    assert results["ok.png"]["pred"] == 0.72
    assert results["empty.png"]["pred"] is None
    assert results["garbage.png"]["pred"] is None


def test_analyze_batch_validates_controls():
    response = TestClient(A.app).post(
        "/api/analyze-batch",
        data={"checkpoint": "unknown"},
        files=[("images", ("test.png", _png_bytes(), "image/png"))],
    )
    assert response.status_code == 422


def test_analyze_batch_rejects_too_many_files(monkeypatch):
    monkeypatch.setattr(A, "_get_model", lambda checkpoint: DummyModel())
    files = [
        ("images", (f"{i}.png", _png_bytes(), "image/png"))
        for i in range(A.MAX_BATCH_FILES + 1)
    ]
    response = TestClient(A.app).post(
        "/api/analyze-batch", data={"checkpoint": "aug"}, files=files,
    )
    assert response.status_code == 413
