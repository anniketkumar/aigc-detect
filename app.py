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
import numpy as np

from predict import load_for_scoring
from src.models.base import load_model
from src.transforms import t_jpeg, to_rgb

CHECKPOINTS = {"aug": Path("runs/aug.pt"),
               "baseline": Path("runs/baseline.pt")}
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
    checkpoint: str = Form("aug"),
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
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(payload)
        temporary.flush()
        decoded, warning = load_for_scoring(Path(temporary.name))
    if decoded is None:
        raise HTTPException(
            422, warning or "This file could not be decoded as an image.")

    model = _get_model(checkpoint)
    clean_score = model.score([decoded], ["ui-upload"])[0]
    reencoded = t_jpeg(decoded, quality)
    reencoded_score = model.score([reencoded], ["ui-upload-jpeg"])[0]
    if fast_mode:
        return JSONResponse({
            "checkpoint": checkpoint, "quality": quality,
            "clean_score": clean_score, "reencoded_score": reencoded_score,
            "warning": warning
        })
    else:
        return JSONResponse({
            "checkpoint": checkpoint, "quality": quality,
            "clean_score": clean_score, "reencoded_score": reencoded_score,
            "jpeg_kb": round(_jpeg_kb(decoded, quality), 1), "warning": warning,
            "clean_preview": _image_data_url(decoded),
            "reencoded_preview": _image_data_url(reencoded),
            "ela_preview": _generate_ela_heatmap(decoded, quality),
        })


@app.post("/api/analyze-batch")
async def analyze_batch(
    images: list[UploadFile] = File(...),
    checkpoint: str = Form("aug"),
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
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
            temporary.write(payload)
            temporary.flush()
            decoded, warning = load_for_scoring(Path(temporary.name))
        if decoded is None:
            results[i] = {"image_path": name, "pred": None, "warning": warning}
            continue

        batch_imgs.append(decoded)
        batch_names.append(name)
        batch_slots.append(i)
        pending_warnings[i] = warning

    if batch_imgs:
        scores = model.score(batch_imgs, batch_names)
        for slot, name, score in zip(batch_slots, batch_names, scores):
            results[slot] = {
                "image_path": name,
                "pred": None if score is None else float(score),
                "warning": pending_warnings.get(slot),
            }

    return JSONResponse({"checkpoint": checkpoint, "results": results})
