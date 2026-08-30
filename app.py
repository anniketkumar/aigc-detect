"""Gradio demo (PLAN.md §9.2): upload an image, then a JPEG-quality slider
95 -> 30, re-encoding and re-scoring live.

This file is a thin UI shell. It does not reimplement decoding or inference:

- The canonical decode path is ``predict.load_for_scoring``, imported
  verbatim -- the same function ``predict.py`` calls per image, itself a
  wrapper around ``src/data/imageio.py::load_image`` (§4.4's decoder: EXIF
  orientation applied, all metadata stripped, any mode -> RGB, truncation
  recovered and reported rather than crashing).
- The scoring path is ``src.models.base.load_model("clip_linear", ...)``,
  the exact ``Scorer`` the eval harness and ``predict.py`` both call through
  -- frozen CLIP ViT-B/16 -> trained linear head -> sigmoid.
- The JPEG degradation step is ``src.transforms.t_jpeg``, the same
  real-encoder round-trip (``BytesIO`` + ``Image.save(..., quality=q)``, not
  a simulation) that produced every ``jpeg_*`` cell in ``results/*/grid.csv``.
  So the number this app shows for "quality 30" is the same operation Phase 1
  and Phase 4 measured at population scale, not a demo-only approximation.

Nothing here touches ``src/data/manifest.py``, ``normalize.py`` or any
training-time path -- this is inference-only, on whatever the user uploads.

Framing, deliberately left out of this file: whether the slider should be
narrated as "confidence collapsing" or "confidence holding" depends on
which checkpoint and which metric you're looking at (see
``results/aug/report.md`` and ``results/tpr_analysis_aug/report.md`` --
AUROC barely moves across the JPEG axis for either checkpoint, TPR@FPR=1%
moves more, and augmentation narrows but does not close that drop). This
app shows the live number for both checkpoints and lets you pick the honest
framing after watching it, rather than asserting one in the UI copy.
"""

from __future__ import annotations

import io
from pathlib import Path

import gradio as gr
from PIL import Image

from predict import load_for_scoring
from src.models.base import load_model
from src.transforms import t_jpeg, to_rgb

CHECKPOINTS = {
    "aug -- Phase 4 (+augmentation)": Path("runs/aug.pt"),
    "baseline -- Phase 3 (no augmentation)": Path("runs/baseline.pt"),
}
DEFAULT_CHECKPOINT = "aug -- Phase 4 (+augmentation)"
DEVICE = "cpu"  # demo runs on whatever laptop plays the video; no GPU assumed

# One backbone+head per checkpoint, loaded lazily and kept warm for the
# session -- open_clip construction is the slow part, scoring one image
# after that is fast even on CPU.
_models: dict[Path, object] = {}


def _get_model(label: str):
    ckpt = CHECKPOINTS[label]
    if ckpt not in _models:
        _models[ckpt] = load_model("clip_linear", ckpt=ckpt, device=DEVICE)
    return _models[ckpt]


def _score(model, img: Image.Image) -> float | None:
    return model.score([img], ["ui-upload"])[0]


def _jpeg_kb(img: Image.Image, quality: int) -> float:
    """File size a real encoder produces at this quality -- same params
    ``t_jpeg`` uses internally, just also kept around for display."""
    buf = io.BytesIO()
    to_rgb(img).save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0")
    return len(buf.getvalue()) / 1024


def on_upload(file_path: str | None, model_label: str, quality: int):
    """New image: decode via the canonical path, cache it in State, and
    score both clean and at the current slider position."""
    if file_path is None:
        return None, None, "Upload an image to begin.", None

    img, warning = load_for_scoring(Path(file_path))
    if img is None:
        return None, None, f"⚠ could not decode this file: {warning}", None

    model = _get_model(model_label)
    clean_score = _score(model, img)
    degraded, live_score, kb = _rescore(img, model, quality)
    status = _status(warning, clean_score, quality, live_score, kb)
    return img, degraded, status, img


def on_quality_or_model_change(state_img, model_label: str, quality: int):
    """Slider release, or checkpoint switch: re-encode + re-score against the
    already-decoded image in State -- no re-decode from disk needed."""
    if state_img is None:
        return None, "Upload an image first."
    model = _get_model(model_label)
    clean_score = _score(model, state_img)
    degraded, live_score, kb = _rescore(state_img, model, quality)
    status = _status(None, clean_score, quality, live_score, kb)
    return degraded, status


def _rescore(img: Image.Image, model, quality: int):
    degraded = t_jpeg(img, int(quality))  # real encode/decode, PLAN.md §3.1
    score = _score(model, degraded)
    kb = _jpeg_kb(img, quality)
    return degraded, score, kb


def _status(warning, clean_score, quality, live_score, kb) -> str:
    lines = []
    if warning:
        lines.append(f"⚠ {warning}")
    if clean_score is not None:
        lines.append(f"**clean** — P(AI-generated) = `{clean_score:.3f}`")
    if live_score is not None:
        lines.append(f"**q={quality}** ({kb:.1f} KB) — P(AI-generated) = `{live_score:.3f}`")
    else:
        lines.append(f"**q={quality}** — could not score this image")
    return "\n\n".join(lines)


with gr.Blocks(title="AIGC detector — JPEG-quality demo") as demo:
    gr.Markdown(
        "# AIGC image detector\n"
        "Upload an image, then drag the JPEG-quality slider. Each release "
        "re-encodes the *canonically decoded* image through a real JPEG "
        "encoder at that quality and re-scores it with the same model "
        "`predict.py` uses."
    )

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(CHECKPOINTS), value=DEFAULT_CHECKPOINT,
            label="Checkpoint",
        )
        quality = gr.Slider(
            minimum=30, maximum=95, value=95, step=1,
            label="JPEG quality (95 → 30)",
        )

    upload = gr.Image(type="filepath", label="Upload an image")
    decoded_state = gr.State(None)  # canonically decoded PIL image

    with gr.Row():
        decoded_view = gr.Image(label="Canonical decode (clean)", interactive=False)
        reencoded_view = gr.Image(label="Re-encoded at slider quality", interactive=False)

    status_md = gr.Markdown()

    upload.upload(
        on_upload,
        inputs=[upload, model_dd, quality],
        outputs=[decoded_view, reencoded_view, status_md, decoded_state],
    )
    quality.release(
        on_quality_or_model_change,
        inputs=[decoded_state, model_dd, quality],
        outputs=[reencoded_view, status_md],
    )
    model_dd.change(
        on_quality_or_model_change,
        inputs=[decoded_state, model_dd, quality],
        outputs=[reencoded_view, status_md],
    )


if __name__ == "__main__":
    demo.launch()
