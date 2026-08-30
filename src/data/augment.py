"""Training-time augmentation sampler (PLAN.md §6, Phase 4).

The Phase 1 eval grid (`src/transforms.py`) is a fixed, sparse set of cells —
that is what makes it a fair, reproducible *measurement*. Training on exactly
those cells would let the head memorize them, so this module draws from the
same six families with severities sampled **continuously and wider** than the
eval grid, and never at the eval grid's exact points:

| family        | eval grid (`transforms.py`) | this module          |
|---------------|------------------------------|-----------------------|
| jpeg quality  | {90, 70, 50, 30}             | U[20, 95]             |
| blur sigma    | {0.5, 1.0, 2.0}              | U[0, 3.0]             |
| resize scale  | {0.5, 0.25}                  | U[0.2, 1.0]           |
| noise sigma   | {0.02, 0.05, 0.10}           | U[0, 0.12]            |
| jitter amount | {0.20}                       | U[0, 0.3]             |
| crop keep     | {0.80}                       | U[0.7, 1.0]           |

Per image: with probability ``1 - p_apply`` the image is left clean (§6:
"leave ~20% clean"). Otherwise, ``k`` families (``k_min..k_max``, default 1-3)
are drawn without replacement and applied in that (randomized) order.

Ops are reused verbatim from ``src/transforms.py`` — same encoder, same blur
kernel, same clip-and-requantize noise — only the *sampling* differs, so a
training-time "jpeg" and an eval-time "jpeg" are the same transform at a
different severity, not two implementations that could quietly drift apart.

Determinism follows ``transforms.py``'s rule exactly: a seed derived from
``(base_seed, image_id, copy_index)`` via the same ``derive_seed``, never a
global RNG stream, so which epoch or dataloader worker touches an image first
cannot change its augmented copy (§3.1: "seed everything").

Each call also returns a ``DegradationLabel``: which families fired and at
what severity. §6 calls these labels "free" and says "Phase 5 consumes them"
via a degradation head (PLAN.md §7.2) — HANDOFF.md later cuts Phase 5, so
nothing consumes this today. It is kept anyway: it costs nothing to compute
alongside the image, it is the only record distinguishing an augmented copy
from a plain crop during later error analysis, and it is the one part of the
original Phase 4 spec worth keeping ready in case the fusion gate comes back
in scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image

from src import transforms as T

__all__ = [
    "FAMILIES",
    "SEVERITY_RANGE",
    "AugmentConfig",
    "DegradationLabel",
    "sample_plan",
    "augment_image",
    "iter_augmented_copies",
]

# --------------------------------------------------------------------------- #
# The six families and their training-time ranges
# --------------------------------------------------------------------------- #

#: Fixed order. Determines both sampling-without-replacement order in
#: ``sample_plan`` and the column order of ``DegradationLabel.to_vector``.
FAMILIES: tuple[str, ...] = ("jpeg", "blur", "resize", "noise", "jitter", "center_crop")

#: (lo, hi) for each family, verbatim from §6. Wider than, and never
#: coincident with, `src.transforms.TRANSFORM_GRID`'s eval points.
SEVERITY_RANGE: dict[str, tuple[float, float]] = {
    "jpeg": (20.0, 95.0),
    "blur": (0.0, 3.0),
    "resize": (0.2, 1.0),
    "noise": (0.0, 0.12),
    "jitter": (0.0, 0.3),
    "center_crop": (0.7, 1.0),
}

assert set(SEVERITY_RANGE) == set(FAMILIES)


@dataclass(frozen=True)
class AugmentConfig:
    """Sampling hyperparameters. Defaults are PLAN.md §6, verbatim."""

    p_apply: float = 0.8
    k_min: int = 1
    k_max: int = 3
    families: tuple[str, ...] = FAMILIES
    severity_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(SEVERITY_RANGE)
    )

    def __post_init__(self):
        if not (0.0 <= self.p_apply <= 1.0):
            raise ValueError(f"p_apply must be in [0,1], got {self.p_apply}")
        if not (1 <= self.k_min <= self.k_max <= len(self.families)):
            raise ValueError(
                f"need 1 <= k_min <= k_max <= {len(self.families)}, "
                f"got k_min={self.k_min}, k_max={self.k_max}"
            )
        missing = set(self.families) - set(self.severity_range)
        if missing:
            raise ValueError(f"no severity_range for families {sorted(missing)}")


# --------------------------------------------------------------------------- #
# Degradation label — the free supervision for a future degradation head
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DegradationLabel:
    """Which of ``FAMILIES`` fired on one image, and at what raw severity.

    Both tuples are always length ``len(FAMILIES)`` and share its index order,
    regardless of which subset of families actually applied — a fixed-length
    label is what a downstream head can be trained against.
    """

    applied: tuple[bool, ...]
    severity: tuple[float, ...]

    def __post_init__(self):
        n = len(FAMILIES)
        if len(self.applied) != n or len(self.severity) != n:
            raise ValueError(f"applied/severity must have length {n}")

    @classmethod
    def clean(cls) -> "DegradationLabel":
        n = len(FAMILIES)
        return cls(applied=(False,) * n, severity=(0.0,) * n)

    @property
    def is_clean(self) -> bool:
        return not any(self.applied)

    def to_vector(
        self, severity_range: dict[str, tuple[float, float]] = SEVERITY_RANGE
    ) -> np.ndarray:
        """``[applied_0, sev_0_norm, applied_1, sev_1_norm, ...]``, length
        ``2 * len(FAMILIES)``. Severity is min-max normalized to [0, 1] over
        ``severity_range`` and is exactly 0.0 for a family that did not fire —
        indistinguishable from "fired at the range floor", which is fine: the
        ``applied`` flag alongside it is what disambiguates the two.
        """
        out = np.zeros(2 * len(FAMILIES), dtype=np.float32)
        for i, fam in enumerate(FAMILIES):
            out[2 * i] = float(self.applied[i])
            if self.applied[i]:
                lo, hi = severity_range[fam]
                out[2 * i + 1] = (self.severity[i] - lo) / (hi - lo) if hi > lo else 0.0
        return out


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def sample_plan(rng: np.random.Generator, config: AugmentConfig = AugmentConfig()) -> tuple[T.Op, ...]:
    """Draw an ops chain: empty (clean) with probability ``1 - p_apply``,
    otherwise ``k_min..k_max`` distinct families, in random order, each at a
    severity drawn uniformly from its range.

    Returns ``transforms.Op`` so the chain can go straight into
    ``transforms.apply_op`` — one severity-application implementation, shared
    with eval.
    """
    if rng.random() >= config.p_apply:
        return ()

    k = int(rng.integers(config.k_min, config.k_max + 1))
    families = list(rng.choice(config.families, size=k, replace=False))
    rng.shuffle(families)

    ops = []
    for fam in families:
        lo, hi = config.severity_range[fam]
        sev = float(rng.uniform(lo, hi))
        if fam == "jpeg":
            sev = float(int(round(sev)))  # JPEG quality is an integer
        ops.append(T.Op(fam, sev))
    return tuple(ops)


def _label_from_plan(ops: Sequence[T.Op]) -> DegradationLabel:
    applied = [False] * len(FAMILIES)
    severity = [0.0] * len(FAMILIES)
    index = {fam: i for i, fam in enumerate(FAMILIES)}
    for op in ops:
        i = index[op.family]
        applied[i] = True
        severity[i] = float(op.severity)
    return DegradationLabel(applied=tuple(applied), severity=tuple(severity))


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

def augment_image(
    img: Image.Image,
    image_id: str,
    copy_index: int = 0,
    base_seed: int = 0,
    config: AugmentConfig = AugmentConfig(),
) -> tuple[Image.Image, DegradationLabel]:
    """One reproducible augmented copy of ``img``.

    Same seeding contract as ``transforms.apply_cell``: the RNG is derived
    from ``(base_seed, image_id, "randaug{copy_index}")``, so calling this
    twice with the same three keys reproduces the same image and label byte
    for byte, independent of call order or process.
    """
    rng = np.random.default_rng(T.derive_seed(base_seed, image_id, f"randaug{copy_index}"))
    ops = sample_plan(rng, config)
    out = T.to_rgb(img)
    for op in ops:
        out = T.apply_op(out, op.family, op.severity, rng)
    return out, _label_from_plan(ops)


def iter_augmented_copies(
    img: Image.Image,
    image_id: str,
    n_copies: int,
    base_seed: int = 0,
    config: AugmentConfig = AugmentConfig(),
):
    """Yield ``n_copies`` independent ``(image, label)`` pairs for one source
    image — the ``K`` in §6's "precompute embeddings for K augmented copies
    per image" (``scripts/cache_features.py --augment-copies K``).
    """
    for copy_index in range(n_copies):
        yield augment_image(img, image_id, copy_index=copy_index, base_seed=base_seed, config=config)
