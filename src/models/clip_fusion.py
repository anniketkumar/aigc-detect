"""CLIP + frequency fusion probe as a Scorer (Phase 5, resumed; src/models/base.py).

Same shape as ``src/models/clip_baseline.py``'s ``CLIPLinearScorer`` -- frozen
backbone(s) -> one trained linear head -> sigmoid -- except the head's input
is ``concat(clip_embedding, frequency_features)`` instead of the CLIP
embedding alone. See ``src/features/frequency.py``'s module docstring for why
this fusion is expected to add information rather than duplicate it
(``results/aesthetic_probe``'s near-zero Spearman correlation between CLIP's
and a texture probe's per-generator difficulty ordering), and why fusion
here means *concatenate-then-fit-one-linear-head*, not a separate gated
network: the workshop's "go hybrid" insight is about the signal, not about
architecture complexity, and ``LinearHead`` is already dimension-agnostic --
reusing it costs nothing and keeps the trainable surface exactly as small
as the Feasibility section already argues it should be.

Registered as ``clip_freq_fusion`` in src/models/base.py so it drops into
src/evaluate.py exactly like clip_linear:

    python -m src.evaluate --model clip_freq_fusion --ckpt runs/fusion.pt \\
        --split test --out results/fusion/

Checkpoint contract: identical to clip_linear's (``src/train.py`` writes it
unmodified) except ``embed_dim`` is ``clip_embed_dim + FREQ_DIM`` because the
cached training features were built with
``scripts/cache_features.py --fuse-freq`` (that flag's docstring). Nothing
about src/train.py needed to change for this to work -- it already reads
``embed_dim`` off the cached array's shape, not off a hardcoded constant.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from src.features.frequency import FREQ_DIM, extract_frequency_features
from src.models.clip_backbone import BACKBONE, ClipBackbone
from src.models.semantic_head import LinearHead

__all__ = ["CLIPFreqFusionScorer"]


class CLIPFreqFusionScorer:
    """Frozen CLIP embed + frequency features -> trained linear head -> sigmoid."""

    def __init__(self, ckpt=None, device: str = "cpu", seed: int = 0):
        if ckpt is None:
            raise ValueError(
                "clip_freq_fusion requires --ckpt, a checkpoint written by "
                "src/train.py over features cached with --fuse-freq"
            )
        state = torch.load(ckpt, map_location=device, weights_only=False)
        self.device = device
        self.backbone = ClipBackbone(
            device=device,
            backbone=state.get("backbone") or BACKBONE,
            pretrained=state.get("pretrained", "openai"),
        )
        embed_dim = int(state["embed_dim"])
        expected = self.backbone.embed_dim + FREQ_DIM
        if embed_dim != expected:
            raise ValueError(
                f"{ckpt}: checkpoint embed_dim={embed_dim} does not match "
                f"clip_embed_dim({self.backbone.embed_dim}) + FREQ_DIM({FREQ_DIM}) "
                f"= {expected}. Was this checkpoint actually trained on "
                f"--fuse-freq features, or is it a plain clip_linear checkpoint?"
            )
        self.head = LinearHead(embed_dim).to(device)
        self.head.load_state_dict(state["state_dict"])
        self.head.eval()
        self.name = f"clip_freq_fusion({ckpt})"

    @torch.no_grad()
    def score(
        self, images: Sequence[Image.Image], image_ids: Sequence[str]
    ) -> list[float | None]:
        clip_feats = self.backbone.embed(images)  # (N, clip_embed_dim)
        freq_feats = np.stack(
            [extract_frequency_features(img) for img in images]
        )  # (N, FREQ_DIM) -- pure numpy, no batching benefit to chase here
        fused = np.concatenate([clip_feats, freq_feats], axis=1).astype(np.float32)
        logits = self.head(torch.from_numpy(fused).to(self.device))
        probs = torch.sigmoid(logits).cpu().numpy()
        return [float(p) for p in probs]
