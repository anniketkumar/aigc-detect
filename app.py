"""HTTP interface for the React detector UI.

This deliberately stays thin: it uses the canonical decoder from
``predict.py`` and the same JPEG round-trip and checkpoint scorer as the
command-line deliverable. It does not duplicate model or image processing.
"""

from __future__ import annotations

import base64
import contextlib
import io
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from predict import load_for_scoring
from src.data.domain_guard import check_domain
from src.models.base import load_model
from src.transforms import t_jpeg, to_rgb

# See src/data/domain_guard.py: every training source, real and AI, is
# photographic, so a rendered graphic (screenshot, diagram, infographic) is
# out-of-domain for both classes -- the model's score on one is not
# meaningful. Surfaced as a warning, not suppressed, same spirit as the
# decode-warning path below.
DOMAIN_WARNING = (
    "Disclaimer: This tool provides probabilistic AI detection scores and "
    "general information, not definitive proof or legal advice. Rules and "
    "accuracy can change, so always verify independently."
)

CHECKPOINTS = {"aug": Path("runs/aug.pt"),
               "baseline": Path("runs/baseline.pt")}
# baseline outscores aug on the organizers' Final Score formula
# (0.5*AUC_clean + 0.5*AUC_robust) -- see README.md "Headline results".
# aug stays selectable for the TPR@FPR comparison story, but baseline is
# the checkpoint we'd actually submit, so it's the default when a caller
# (including the extension) doesn't pick one explicitly.
DEFAULT_CHECKPOINT = "baseline"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_BATCH_FILES = 50
_models: dict[Path, object] = {}

app = FastAPI(title="AIGC detector API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@contextlib.contextmanager
def _spooled_upload(payload: bytes, suffix: str):
    """Write ``payload`` to a real temp file and yield its path for
    ``load_for_scoring`` (PIL) to open by name.

    Plain ``tempfile.NamedTemporaryFile`` opens the handle exclusively on
    Windows, so a second open of the same path -- exactly what
    ``load_for_scoring`` does -- fails with ``PermissionError`` while the
    ``with`` block above it is still holding the file open. ``delete=False``
    plus an explicit close/unlink here works the same way on every OS.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        yield Path(path)
    finally:
        os.remove(path)


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


def _generate_ela_heatmap(img: Image.Image, quality: int) -> str:
    img_rgb = to_rgb(img)
    buf = io.BytesIO()
    # High compression to find differences, like frontend (85)
    img_rgb.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    reencoded_img = Image.open(buf)

    arr1 = np.array(img_rgb).astype(np.float32)
    arr2 = np.array(reencoded_img).astype(np.float32)

    diff = np.abs(arr1 - arr2)
    mag = np.mean(diff, axis=2)

    # Auto-scale the ELA so the 99th percentile hits t=0.8 (Bright Red/Yellow)
    p99 = np.percentile(mag, 99)
    if p99 < 1:
        p99 = 1.0

    amp = np.clip((mag / p99) * 204, 0, 255)
    t = amp / 255.0

    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)

    # 0.0 -> 0.33: Black (0,0,0) to Purple (120,0,120)
    m1 = t < 0.33
    subT1 = t[m1] / 0.33
    r[m1] = 120 * subT1
    b[m1] = 120 * subT1

    # 0.33 -> 0.66: Purple (120,0,120) to Red (255,0,0)
    m2 = (t >= 0.33) & (t < 0.66)
    subT2 = (t[m2] - 0.33) / 0.33
    r[m2] = 120 + (255 - 120) * subT2
    b[m2] = 120 - (120 * subT2)

    # 0.66 -> 1.0: Red (255,0,0) to Yellow (255,255,0)
    m3 = t >= 0.66
    subT3 = (t[m3] - 0.66) / 0.34
    r[m3] = 255
    g[m3] = 255 * subT3

    heatmap = np.stack([r, g, b], axis=2).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap, 'RGB')

    return _image_data_url(heatmap_img)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    checkpoint: str = Form(DEFAULT_CHECKPOINT),
    quality: int = Form(95),
    fast_mode: bool = Form(False),
):
    if checkpoint not in CHECKPOINTS:
        raise HTTPException(
            422, "Choose either the augmented or baseline model.")
    if not 30 <= quality <= 95:
        raise HTTPException(422, "JPEG quality must be between 30 and 95.")
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Images must be 25 MB or smaller.")
    if not payload:
        raise HTTPException(422, "Choose an image file to analyze.")

    suffix = Path(image.filename or "upload").suffix or ".img"
    with _spooled_upload(payload, suffix) as temp_path:
        decoded, warning = load_for_scoring(temp_path)
    if decoded is None:
        raise HTTPException(
            422, warning or "This file could not be decoded as an image.")

    domain_warning = DOMAIN_WARNING if check_domain(decoded).likely_non_photographic else None

    model = _get_model(checkpoint)
    clean_score = model.score([decoded], ["ui-upload"])[0]
    reencoded = t_jpeg(decoded, quality)
    reencoded_score = model.score([reencoded], ["ui-upload-jpeg"])[0]
    if fast_mode:
        return JSONResponse({
            "checkpoint": checkpoint, "quality": quality,
            "clean_score": clean_score, "reencoded_score": reencoded_score,
            "warning": warning, "domain_warning": domain_warning,
        })
    else:
        return JSONResponse({
            "checkpoint": checkpoint, "quality": quality,
            "clean_score": clean_score, "reencoded_score": reencoded_score,
            "jpeg_kb": round(_jpeg_kb(decoded, quality), 1), "warning": warning,
            "domain_warning": domain_warning,
            "clean_preview": _image_data_url(decoded),
            "reencoded_preview": _image_data_url(reencoded),
            "ela_preview": _generate_ela_heatmap(decoded, quality),
        })


@app.post("/api/analyze-batch")
async def analyze_batch(
    images: list[UploadFile] = File(...),
    checkpoint: str = Form(DEFAULT_CHECKPOINT),
):
    """Score many images in one call. Mirrors ``predict.py``'s contract --
    same decoder, same model, same ``{"image_path", "pred"}`` shape per item
    -- so a batch scored here and a batch scored by the CLI deliverable never
    quietly disagree. ``image_path`` is the uploaded filename: browsers don't
    expose a real directory path, so the identifier is the name, sorted for
    the same run-to-run determinism ``predict.py`` guarantees.
    """
    if checkpoint not in CHECKPOINTS:
        raise HTTPException(
            422, "Choose either the augmented or baseline model.")
    if not images:
        raise HTTPException(422, "Choose at least one image file to analyze.")
    if len(images) > MAX_BATCH_FILES:
        raise HTTPException(
            413, f"Batch is limited to {MAX_BATCH_FILES} images at a time.")

    images = sorted(images, key=lambda f: f.filename or "")
    model = _get_model(checkpoint)

    results: list[dict] = [None] * len(images)
    pending_warnings: dict[int, str | None] = {}
    pending_domain_warnings: dict[int, str | None] = {}
    batch_imgs, batch_names, batch_slots = [], [], []

    for i, image in enumerate(images):
        name = image.filename or f"image_{i}"
        payload = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            results[i] = {"image_path": name, "pred": None,
                          "warning": "File exceeds 25 MB limit."}
            continue
        if not payload:
            results[i] = {"image_path": name, "pred": None,
                          "warning": "Empty file."}
            continue

        suffix = Path(name).suffix or ".img"
        with _spooled_upload(payload, suffix) as temp_path:
            decoded, warning = load_for_scoring(temp_path)
        if decoded is None:
            results[i] = {"image_path": name, "pred": None, "warning": warning}
            continue

        batch_imgs.append(decoded)
        batch_names.append(name)
        batch_slots.append(i)
        pending_warnings[i] = warning
        pending_domain_warnings[i] = (
            DOMAIN_WARNING if check_domain(decoded).likely_non_photographic else None
        )

    if batch_imgs:
        scores = model.score(batch_imgs, batch_names)
        for slot, name, score in zip(batch_slots, batch_names, scores):
            results[slot] = {
                "image_path": name,
                "pred": None if score is None else float(score),
                "warning": pending_warnings.get(slot),
                "domain_warning": pending_domain_warnings.get(slot),
            }

    return JSONResponse({"checkpoint": checkpoint, "results": results})
