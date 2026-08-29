"""Linear probe over frozen CLIP features (PLAN.md §5).

Deliberately just ``nn.Linear`` -- the Phase 3 baseline is a control, not a
serious model. Upgrade to a 2-layer MLP only if this underfits; nothing so far
has shown that it does.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)
