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
