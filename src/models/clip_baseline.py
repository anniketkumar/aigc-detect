"""CLIP linear-probe baseline as a Scorer (PLAN.md §5, src/models/base.py).

Deliberately dumb: frozen CLIP ViT-B/16 + a single linear layer, trained with
plain BCE and no augmentation (``src/train.py``). Its whole job is to produce
the "before" picture -- a healthy clean AUROC that collapses under jpeg=30 /
resize=0.25 is the expected, desired result (HANDOFF.md), not a bug.

Registered as ``clip_linear`` in src/models/base.py so it drops into
src/evaluate.py exactly like the dummy models:

    python -m src.evaluate --model clip_linear --ckpt runs/baseline.pt \\
        --split test --out results/baseline/
"""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image

from src.models.clip_backbone import BACKBONE, ClipBackbone
from src.models.semantic_head import LinearHead

__all__ = ["CLIPLinearScorer"]


class CLIPLinearScorer:
    """Frozen CLIP embed -> trained linear head -> sigmoid."""

    def __init__(self, ckpt=None, device: str = "cpu", seed: int = 0):
        if ckpt is None:
            raise ValueError(
                "clip_linear requires --ckpt, a checkpoint written by src/train.py"
            )
        state = torch.load(ckpt, map_location=device, weights_only=False)
        self.device = device
        self.backbone = ClipBackbone(
            device=device,
            backbone=state.get("backbone") or BACKBONE,
            pretrained=state.get("pretrained", "openai"),
        )
        self.head = LinearHead(int(state["embed_dim"])).to(device)
        self.head.load_state_dict(state["state_dict"])
        self.head.eval()
        self.name = f"clip_linear({ckpt})"

    @torch.no_grad()
    def score(
        self, images: Sequence[Image.Image], image_ids: Sequence[str]
    ) -> list[float | None]:
        feats = self.backbone.embed(images)
        logits = self.head(torch.from_numpy(feats).to(self.device))
        probs = torch.sigmoid(logits).cpu().numpy()
        return [float(p) for p in probs]
