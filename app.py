"""HTTP interface for the React detector UI.

This deliberately stays thin: it uses the canonical decoder from
``predict.py`` and the same JPEG round-trip and checkpoint scorer as the
command-line deliverable. It does not duplicate model or image processing.
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from predict import load_for_scoring
from src.models.base import load_model
from src.transforms import t_jpeg, to_rgb

CHECKPOINTS = {"aug": Path("runs/aug.pt"), "baseline": Path("runs/baseline.pt")}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_models: dict[Path, object] = {}

app = FastAPI(title="AIGC detector API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


def _get_model(checkpoint: str):
    ckpt = CHECKPOINTS[checkpoint]
    if ckpt not in _models:
        _models[ckpt] = load_model("clip_linear", ckpt=ckpt, device="cpu")
    return _models[ckpt]


def _jpeg_kb(img: Image.Image, quality: int) -> float:
    buf = io.BytesIO()
    to_rgb(img).save(buf, format="JPEG", quality=quality, subsampling="4:2:0")
    return len(buf.getvalue()) / 1024


def _image_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    to_rgb(img).save(buf, format="JPEG", quality=92, subsampling="4:2:0")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    checkpoint: str = Form("aug"),
    quality: int = Form(95),
):
    if checkpoint not in CHECKPOINTS:
        raise HTTPException(422, "Choose either the augmented or baseline model.")
    if not 30 <= quality <= 95:
        raise HTTPException(422, "JPEG quality must be between 30 and 95.")
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Images must be 25 MB or smaller.")
    if not payload:
        raise HTTPException(422, "Choose an image file to analyze.")

    suffix = Path(image.filename or "upload").suffix or ".img"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(payload)
        temporary.flush()
        decoded, warning = load_for_scoring(Path(temporary.name))
    if decoded is None:
        raise HTTPException(422, warning or "This file could not be decoded as an image.")

    model = _get_model(checkpoint)
    clean_score = model.score([decoded], ["ui-upload"])[0]
    reencoded = t_jpeg(decoded, quality)
    reencoded_score = model.score([reencoded], ["ui-upload-jpeg"])[0]
    return JSONResponse({
        "checkpoint": checkpoint, "quality": quality,
        "clean_score": clean_score, "reencoded_score": reencoded_score,
        "jpeg_kb": round(_jpeg_kb(decoded, quality), 1), "warning": warning,
        "clean_preview": _image_data_url(decoded),
        "reencoded_preview": _image_data_url(reencoded),
    })
