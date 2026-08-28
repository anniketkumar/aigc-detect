"""Evaluation transform grid (PLAN.md §3.1).

Every transform is ``f(PIL.Image) -> PIL.Image`` and is applied to the *raw*
image, before any model preprocessing or normalization.

Determinism
-----------
The stochastic transforms (``noise``, ``jitter``) do not draw from a global RNG.
Each call gets a seed derived by hashing ``(base_seed, image_id, cell_name)``
via :func:`derive_seed`. That makes a cell's output a pure function of those
three things, so results are identical across runs regardless of image order,
batch size, worker count, or which subset of the grid you ran. A global RNG
stream would not give you that.

Decisions the spec leaves open are marked "SPEC:" below and repeated in NOTES.md.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

__all__ = [
    "TRANSFORM_GRID",
    "COMPOSED",
    "Op",
    "Cell",
    "build_cells",
    "apply_cell",
    "apply_op",
    "derive_seed",
    "to_rgb",
    "t_clean",
    "t_jpeg",
    "t_blur",
    "t_resize",
    "t_noise",
    "t_jitter",
    "t_center_crop",
]

# --------------------------------------------------------------------------- #
# The grid, verbatim from PLAN.md §3.1
# --------------------------------------------------------------------------- #

TRANSFORM_GRID: dict[str, list] = {
    "clean":       [None],
    "jpeg":        [90, 70, 50, 30],            # quality
    "blur":        [0.5, 1.0, 2.0],             # gaussian sigma
    "resize":      [0.5, 0.25],                 # downscale then upscale back
    "noise":       [0.02, 0.05, 0.10],          # gaussian sigma, on [0,1] pixels
    "jitter":      [0.20],                      # brightness/contrast/saturation ±20%
    "center_crop": [0.80],                      # keep 80% then resize back
}

COMPOSED: list[tuple] = [                       # realistic redistribution chains
    ("blur", 1.0, "jpeg", 70),                  # phone photo -> messaging app
    ("resize", 0.5, "jpeg", 50),                # thumbnail -> repost
    ("jitter", 0.20, "jpeg", 30),               # filter app -> heavy re-encode
    ("resize", 0.25, "blur", 0.5, "jpeg", 30),  # worst case
]

# Interpolation filter for every down/up-scale in this module.
# SPEC: unspecified. BICUBIC both directions -- it is what image-sharing
# pipelines actually use, and unlike NEAREST/LANCZOS it neither preserves nor
# manufactures high-frequency structure that a detector could latch onto.
RESAMPLE = Image.Resampling.BICUBIC


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #

def derive_seed(base_seed: int, image_id: str, cell_name: str) -> int:
    """Stable 64-bit seed for one (image, cell) pair.

    Uses blake2b rather than ``hash()``: Python's string hash is salted per
    process, so ``hash()`` would silently break reproducibility across runs.
    """
    h = hashlib.blake2b(
        f"{base_seed}\x00{image_id}\x00{cell_name}".encode("utf-8"), digest_size=8
    )
    return int.from_bytes(h.digest(), "little")


def _rng(base_seed: int, image_id: str, cell_name: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, image_id, cell_name))


# --------------------------------------------------------------------------- #
# Individual transforms
# --------------------------------------------------------------------------- #

def to_rgb(img: Image.Image) -> Image.Image:
    """Normalize any input mode to 3-channel RGB.

    Required before JPEG (the encoder rejects alpha) and before saturation
    jitter (undefined on mode ``L``). Alpha is composited onto black rather than
    dropped, so a transparent PNG degrades the same way a flattened one would.
    """
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def t_clean(img: Image.Image) -> Image.Image:
    """Identity, apart from mode normalization."""
    return to_rgb(img)


def t_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip through a real JPEG encoder (§3.1: not a simulation)."""
    buf = io.BytesIO()
    to_rgb(img).save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0")
    buf.seek(0)
    out = Image.open(buf)
    out.load()  # force decode before the buffer goes out of scope
    return out


def t_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur. PIL's ``GaussianBlur(radius)`` takes radius == sigma."""
    if sigma <= 0:
        return to_rgb(img)
    return to_rgb(img).filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def t_resize(img: Image.Image, scale: float) -> Image.Image:
    """Downscale to ``scale x`` then upscale back. The information loss is the point."""
    img = to_rgb(img)
    w, h = img.size
    dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((dw, dh), RESAMPLE).resize((w, h), RESAMPLE)


def t_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Additive i.i.d. Gaussian noise on [0,1] pixels, independent per channel.

    SPEC: clipping is unspecified. We clip to [0,1] and re-quantize to uint8,
    which is what any real 8-bit pipeline does; it makes the noise slightly
    non-Gaussian near black and white, and that is the honest behaviour.
    """
    img = to_rgb(img)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = x + rng.normal(0.0, float(sigma), size=x.shape).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    return Image.fromarray(np.round(x * 255.0).astype(np.uint8), mode="RGB")


def t_jitter(img: Image.Image, amount: float, rng: np.random.Generator) -> Image.Image:
    """Brightness/contrast/saturation jitter, each +/- ``amount``.

    SPEC: "±20%" does not say how many factors are drawn or in what order.
    We draw three *independent* factors from U(1-a, 1+a) and apply them in a
    fixed order (brightness, contrast, saturation). torchvision's ColorJitter
    shuffles the order; we do not, because a random order is a second hidden
    source of variance for no measurement benefit.
    """
    img = to_rgb(img)
    lo, hi = 1.0 - float(amount), 1.0 + float(amount)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(float(rng.uniform(lo, hi)))
    return img


def t_center_crop(img: Image.Image, keep: float) -> Image.Image:
    """Centre-crop to ``keep`` of each side, then resize back to the original size.

    SPEC: "keep 80%" is ambiguous between linear and area fraction. We read it
    as linear (each side x 0.80, i.e. 64% of the area), matching how
    ``center_crop`` is defined everywhere in torchvision.
    """
    img = to_rgb(img)
    w, h = img.size
    cw, ch = max(1, int(round(w * keep))), max(1, int(round(h * keep)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), RESAMPLE)


#: Which families need an RNG. Everything else is a pure function of the image.
_STOCHASTIC = frozenset({"noise", "jitter"})

_DISPATCH: dict[str, Callable] = {
    "clean": lambda img, sev, rng: t_clean(img),
    "jpeg": lambda img, sev, rng: t_jpeg(img, sev),
    "blur": lambda img, sev, rng: t_blur(img, sev),
    "resize": lambda img, sev, rng: t_resize(img, sev),
    "noise": lambda img, sev, rng: t_noise(img, sev, rng),
    "jitter": lambda img, sev, rng: t_jitter(img, sev, rng),
    "center_crop": lambda img, sev, rng: t_center_crop(img, keep=sev),
}


def apply_op(
    img: Image.Image,
    family: str,
    severity: float | None,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Apply one (family, severity) op. ``rng`` is required iff the op is stochastic."""
    if family not in _DISPATCH:
        raise KeyError(f"unknown transform family {family!r}")
    if family in _STOCHASTIC and rng is None:
        raise ValueError(f"{family!r} is stochastic and needs an rng")
    return _DISPATCH[family](img, severity, rng)


# --------------------------------------------------------------------------- #
# Cells: the unit of evaluation
# --------------------------------------------------------------------------- #

def _fmt_sev(sev: float | int | None) -> str:
    """Severity -> label. Ints stay ints (JPEG quality); floats keep a decimal
    point so ``blur`` reads 0.5 / 1.0 / 2.0 rather than 0.5 / 1 / 2."""
    if sev is None:
        return ""
    if isinstance(sev, int):
        return str(sev)
    s = f"{float(sev):g}"
    return s if "." in s else s + ".0"


@dataclass(frozen=True)
class Op:
    family: str
    severity: float | None

    def __str__(self) -> str:
        return self.family if self.severity is None else f"{self.family}{_fmt_sev(self.severity)}"


@dataclass(frozen=True)
class Cell:
    """One column of the robustness grid.

    ``name`` is used as a CSV key and as a cache directory name, so it is
    restricted to characters that are legal in a Windows path (no ``:``).
    """

    name: str
    kind: str                      # "clean" | "single" | "composed"
    ops: tuple[Op, ...] = field(default_factory=tuple)

    @property
    def family(self) -> str:
        """Grid row label: the family for single cells, "composed" otherwise."""
        return self.ops[0].family if self.kind in ("clean", "single") else "composed"

    @property
    def severity(self) -> float | None:
        return self.ops[0].severity if self.kind in ("clean", "single") else None

    @property
    def chain(self) -> str:
        return "+".join(str(op) for op in self.ops) if self.ops else "clean"

    @property
    def is_stochastic(self) -> bool:
        return any(op.family in _STOCHASTIC for op in self.ops)

    def __call__(
        self, img: Image.Image, image_id: str = "", base_seed: int = 0
    ) -> Image.Image:
        return apply_cell(img, self, image_id=image_id, base_seed=base_seed)


def apply_cell(
    img: Image.Image, cell: Cell, image_id: str = "", base_seed: int = 0
) -> Image.Image:
    """Apply a cell's ops in order.

    All stochastic ops within a cell share one RNG derived from
    ``(base_seed, image_id, cell.name)``, so the whole chain is reproducible.
    """
    rng = _rng(base_seed, image_id, cell.name) if cell.is_stochastic else None
    if not cell.ops:
        return t_clean(img)
    out = img
    for op in cell.ops:
        out = apply_op(out, op.family, op.severity, rng)
    return out


def _chain_to_ops(chain: Sequence) -> tuple[Op, ...]:
    """``("blur", 1.0, "jpeg", 70)`` -> ``(Op("blur",1.0), Op("jpeg",70))``."""
    if len(chain) % 2 != 0:
        raise ValueError(f"composed chain must be (family, severity) pairs: {chain!r}")
    return tuple(Op(chain[i], chain[i + 1]) for i in range(0, len(chain), 2))


def build_cells(
    grid: dict[str, list] | None = None,
    composed: Sequence[Sequence] | None = None,
) -> list[Cell]:
    """Enumerate the full grid: clean, then each single cell, then each chain."""
    grid = TRANSFORM_GRID if grid is None else grid
    composed = COMPOSED if composed is None else composed

    cells: list[Cell] = []
    for family, severities in grid.items():
        for sev in severities:
            if family == "clean":
                cells.append(Cell(name="clean", kind="clean", ops=(Op("clean", None),)))
            else:
                cells.append(
                    Cell(
                        name=f"{family}_{_fmt_sev(sev)}",
                        kind="single",
                        ops=(Op(family, sev),),
                    )
                )
    for chain in composed:
        ops = _chain_to_ops(chain)
        cells.append(
            Cell(
                name="composed_" + "+".join(str(op) for op in ops),
                kind="composed",
                ops=ops,
            )
        )

    names = [c.name for c in cells]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate cell names in grid: {dupes}")
    return cells


def iter_cells() -> Iterator[Cell]:
    yield from build_cells()


def cache_key(cell: Cell, image_id: str, base_seed: int) -> str:
    """Filename stem for the on-disk transformed-image cache.

    Includes the seed for stochastic cells only, so deterministic cells stay
    shareable across runs with different seeds.
    """
    payload = f"{image_id}\x00{cell.chain}"
    if cell.is_stochastic:
        payload += f"\x00{base_seed}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
