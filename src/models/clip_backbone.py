"""Frozen CLIP feature extractor (PLAN.md §5, HANDOFF.md).

ViT-B/16, not ViT-L/14: three times faster to embed the whole corpus, and at
this deadline embedding speed buys more than the last point of clean AUROC
(HANDOFF.md). Frozen throughout Phase 3 -- only the linear head trains, which
is what makes feature caching valid at all (§ Workflow: once cached, every
later head experiment runs in seconds with no GPU).

L/14 is a later swap if there's slack, not a Phase 3 decision.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

BACKBONE = "ViT-B-16"
PRETRAINED = "openai"


class ClipBackbone:
    """Loads once, embeds many batches. Not a Scorer itself -- see
    ``src/models/clip_baseline.py`` for the thing the eval harness calls.
    """

    def __init__(
        self,
        device: str = "cpu",
        backbone: str = BACKBONE,
        pretrained: str = PRETRAINED,
    ):
        import open_clip

        self.device = device
        self.backbone = backbone
        self.pretrained = pretrained
        model, _, preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained
        )
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(device)
        self.preprocess = preprocess
        self.embed_dim = int(model.visual.output_dim)

    @torch.no_grad()
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """RGB PIL images -> ``(N, embed_dim)`` float32, L2-normalized.

        CLIP's own preprocessing (resize/crop/normalize) is applied here, on
        whatever pixels the caller hands in -- at eval time that means *after*
        the Phase 1 transform grid has already degraded the image, exactly as
        src/models/base.py's Scorer contract requires.
        """
        batch = torch.stack(
            [self.preprocess(img.convert("RGB")) for img in images]
        ).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.float().cpu().numpy()
