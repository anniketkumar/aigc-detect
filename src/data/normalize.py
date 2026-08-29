"""The canonical decode path. Every training and eval image goes through this.

The Phase 2 audit found that a classifier reading *no pixels at all* scores
0.98 AUROC on SID_Set: the AI class is 1024x1024 and PNG, the real class is
arbitrary shapes and JPEG. This module's job is to make that impossible, by
putting both classes through byte-for-byte the same pipeline:

    decode  ->  apply EXIF orientation  ->  strip ALL metadata  ->  RGB
            ->  [AI only: synthetic first JPEG generation]
            ->  fixed-size CROP at native resolution, no resampling
            ->  one JPEG pass, constant quality, both classes

Four decisions in there are load-bearing and each has a cheaper-looking
alternative that quietly reintroduces a leak.

**Crop, not resize.** Resizing 1024x1024 and 1024x683 to a common size applies
*different scale factors per class*, and a resampling kernel leaves a signature
in precisely the high-frequency band the Phase 5 artifact branch reads. The
model would learn the scale factor. Cropping touches no frequency content at
all: a crop of a photograph is a photograph.

**Crop offsets are multiples of 16.** A JPEG's 8x8 DCT grid — 16x16 MCU under
4:2:0 chroma subsampling — is anchored at the image origin. Crop at an arbitrary
offset and the inherited grid shifts by ``(x mod 16, y mod 16)``. Real images
arrive already JPEG-compressed and so carry a grid; the synthetic first
generation gives AI images one too. If offsets were arbitrary, grid phase would
be a shared nuisance variable — but any per-class difference in the offset
distribution would turn it into a leak. Snapping to 16 removes the axis.

**Constant quality, not distribution-matched.** Matching the AI class's final
quality to the real class's empirical distribution would leave quality
*correlated* with class through the sampling noise, and would need the real
distribution known in advance. One constant makes quality provably uninformative:
a feature with zero variance has AUROC 0.5 by construction, and the Phase 2
regression test can assert exactly that.

**Double JPEG.** A real photo arrives as JPEG and leaves as JPEG: two
generations. An AI image arrives as PNG; without intervention it would leave
with one. Generation count is trivially detectable — periodic structure in the
DCT coefficient histograms — so AI images get a synthetic first generation at a
quality *and chroma subsampling* drawn from the real class's measured
distribution, before the canonical pass. Both classes end at two.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import imageio as IIO

__all__ = [
    "NormalizeConfig",
    "NormalizeResult",
    "QualityPrior",
    "REAL_JPEG_QUALITY_PRIOR",
    "REAL_SUBSAMPLING_PRIOR",
    "normalize",
    "estimate_jpeg_quality",
    "read_source_jpeg_stats",
]

#: Empirical JPEG quality of the real class, measured over the 619 real JPEGs in
#: ``data/audit_sample/sid_set`` by matching each file's luma quantization table
#: against Pillow's own tables at q=1..100. Median 93, mean 91.2, and a long
#: tail down to 68 -- Flickr-era uploads. Used as the *fallback* prior; the
#: downloader measures the real stream it actually pulls and overrides this.
REAL_JPEG_QUALITY_PRIOR: tuple[int, ...] = (
    68, 69, 69, 70, 72, 72, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 76, 77,
    78, 79, 80, 80, 80, 81, 83, 84, 85, 85, 85, 86, 87, 88, 88, 89, 90, 90, 90,
    90, 90, 91, 91, 92, 92, 93, 93, 93, 94, 94, 95, 95, 95, 96, 96, 97, 97, 98,
    98, 98, 99, 99, 99, 99, 99, 99, 99, 100, 100,
)

#: Chroma subsampling of the same 619 files: 4:2:0 245, 4:2:2 159, 4:4:4 192.
#: A real photo's *first* generation has whatever its camera or uploader chose,
#: so the AI class's synthetic first generation samples from the same mix. The
#: final canonical pass is 4:2:0 for everything.
REAL_SUBSAMPLING_PRIOR: tuple[str, ...] = ("4:2:0",) * 40 + ("4:2:2",) * 26 + ("4:4:4",) * 31


@dataclass(frozen=True)
class NormalizeConfig:
    """Anything that changes the output bytes lives here and is logged.

    Frozen and hashable so ``config_hash`` can be written into the manifest: if
    two images were normalized under different settings, the manifest says so
    rather than the difference showing up as an unexplained AUROC.
    """

    crop: int = 512
    #: Multiple-of-16 crop offsets. See the module docstring.
    offset_multiple: int = 16
    #: Final pass. One value, both classes, no exceptions.
    jpeg_quality: int = 95
    subsampling: str = "4:2:0"
    #: Give class-1 images a synthetic first JPEG generation.
    double_jpeg_ai: bool = True
    #: Crop location. "random" (seeded) rather than "center": AI generators
    #: centre their subject, so a centre crop would systematically frame subject
    #: for one class and background for the other -- a semantic leak introduced
    #: by the very step meant to remove leaks.
    crop_mode: str = "random"
    seed: int = 0

    @property
    def config_hash(self) -> str:
        payload = "|".join(f"{k}={v}" for k, v in sorted(vars(self).items()))
        return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()


@dataclass
class NormalizeResult:
    data: bytes | None
    status: str
    reason: str = ""
    out_size: tuple[int, int] = (0, 0)
    src_size: tuple[int, int] = (0, 0)
    src_format: str = ""
    crop_xy: tuple[int, int] = (0, 0)
    first_gen_quality: int = 0
    first_gen_subsampling: str = ""
    load_status: str = ""
    pad_frac: float = 0.0
    n_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.data is not None


# --------------------------------------------------------------------------- #
# JPEG quality estimation
# --------------------------------------------------------------------------- #

_REF_TABLES: dict[str, dict[int, np.ndarray]] = {}


def _ref_tables(subsampling: str = "4:2:0") -> dict[int, np.ndarray]:
    """Pillow's luma quantization table at each quality, built once and cached."""
    if subsampling not in _REF_TABLES:
        tabs = {}
        probe = Image.new("RGB", (16, 16))
        for q in range(1, 101):
            buf = io.BytesIO()
            probe.save(buf, "JPEG", quality=q, subsampling=subsampling)
            buf.seek(0)
            with Image.open(buf) as r:
                tabs[q] = np.asarray(r.quantization[0], dtype=float)
        _REF_TABLES[subsampling] = tabs
    return _REF_TABLES[subsampling]


def estimate_jpeg_quality(path: str | Path) -> int | None:
    """Approximate the IJG quality of an existing JPEG from its quant table.

    Nearest-table match by L1 distance. Approximate by nature -- encoders other
    than libjpeg scale the tables differently -- but it only has to be good
    enough to give the synthetic first generation a realistic *distribution*,
    not to recover any individual file's true setting.
    """
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return None
            qt = im.quantization.get(0)
            if qt is None:
                return None
            a = np.asarray(qt, dtype=float)
    except Exception:
        return None
    ref = _ref_tables()
    return min(ref, key=lambda q: float(np.abs(a - ref[q]).sum()))


@dataclass
class QualityPrior:
    """The real class's measured (quality, subsampling) distribution.

    Sampled from -- rather than fitted -- because the shape is bimodal (a spike
    at 75 from one era of upload tooling, another at 99) and any parametric fit
    would smooth away exactly the structure that makes it look real.
    """

    qualities: tuple[int, ...] = REAL_JPEG_QUALITY_PRIOR
    subsamplings: tuple[str, ...] = REAL_SUBSAMPLING_PRIOR
    n_observed: int = 0
    source: str = "prior (619 SID_Set real JPEGs)"

    def sample(self, rng: np.random.Generator) -> tuple[int, str]:
        q = int(self.qualities[int(rng.integers(len(self.qualities)))])
        s = str(self.subsamplings[int(rng.integers(len(self.subsamplings)))])
        return q, s

    @classmethod
    def from_paths(cls, paths, min_n: int = 100) -> "QualityPrior":
        """Measure the prior from real files actually downloaded.

        Falls back to the baked-in prior below ``min_n`` observations: a prior
        estimated from 12 images would be noise, and noise here becomes a
        systematic difference between the classes.
        """
        qs, ss = [], []
        for p in paths:
            q = estimate_jpeg_quality(p)
            if q is None:
                continue
            qs.append(q)
            try:
                with Image.open(p) as im:
                    from PIL.JpegImagePlugin import get_sampling
                    ss.append({0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(get_sampling(im)))
            except Exception:
                pass
        ss = [s for s in ss if s]
        if len(qs) < min_n:
            return cls(n_observed=len(qs),
                       source=f"prior (only {len(qs)} < {min_n} observed)")
        return cls(
            qualities=tuple(qs),
            subsamplings=tuple(ss) if len(ss) >= min_n else REAL_SUBSAMPLING_PRIOR,
            n_observed=len(qs),
            source=f"measured from {len(qs)} real JPEGs",
        )


def read_source_jpeg_stats(path: str | Path) -> dict:
    """Quality + subsampling of a source file, for the manifest audit columns."""
    q = estimate_jpeg_quality(path)
    return {"src_jpeg_quality": q or 0}


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

def _crop_box(w: int, h: int, size: int, mult: int, mode: str,
              rng: np.random.Generator) -> tuple[int, int]:
    """Top-left corner, snapped down to a multiple of ``mult``."""
    max_x, max_y = w - size, h - size
    if mode == "center":
        x, y = max_x // 2, max_y // 2
    else:
        x = int(rng.integers(max_x + 1)) if max_x > 0 else 0
        y = int(rng.integers(max_y + 1)) if max_y > 0 else 0
    return (x // mult) * mult, (y // mult) * mult


def normalize(
    path: str | Path,
    label: int,
    cfg: NormalizeConfig | None = None,
    prior: QualityPrior | None = None,
    image_id: str | None = None,
) -> NormalizeResult:
    """Run one file through the canonical path. Returns encoded JPEG bytes.

    Bytes rather than a PIL image because the output is written to disk and
    re-read many times during training; returning an array would mean the final
    encode happened somewhere else, and "somewhere else" is where per-class
    differences creep in.

    ``label`` selects only whether the synthetic first generation is applied.
    Nothing else in this function branches on class -- that is the whole point,
    and ``tests/test_normalize.py`` asserts it byte-for-byte.
    """
    cfg = cfg or NormalizeConfig()
    prior = prior or QualityPrior()
    key = image_id or Path(path).as_posix()
    rng = np.random.default_rng(
        int.from_bytes(hashlib.blake2b(f"{cfg.seed}\x00{key}".encode(),
                                       digest_size=8).digest(), "little")
    )

    res = IIO.load_image(path, min_side=cfg.crop)
    if res.image is None:
        return NormalizeResult(None, res.status, res.reason,
                               src_size=res.orig_size, src_format=res.orig_format,
                               load_status=res.status, pad_frac=res.pad_frac)
    if res.status == IIO.TOO_SMALL:
        return NormalizeResult(None, IIO.TOO_SMALL, res.reason,
                               src_size=res.orig_size, src_format=res.orig_format,
                               load_status=res.status)

    img = res.image
    fg_q, fg_sub = 0, ""

    # Synthetic first generation, on the *full* image so its DCT grid is
    # anchored at the origin exactly as a real camera JPEG's would be. Doing it
    # after the crop would anchor it to the crop corner instead, and the grid
    # phase relative to image content would then differ by class.
    if cfg.double_jpeg_ai and label == 1:
        fg_q, fg_sub = prior.sample(rng)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=fg_q, subsampling=fg_sub)
        buf.seek(0)
        with Image.open(buf) as tmp:
            tmp.load()
            img = IIO.strip_metadata(tmp)

    w, h = img.size
    x, y = _crop_box(w, h, cfg.crop, cfg.offset_multiple, cfg.crop_mode, rng)
    img = img.crop((x, y, x + cfg.crop, y + cfg.crop))

    out = io.BytesIO()
    img.save(out, "JPEG", quality=cfg.jpeg_quality, subsampling=cfg.subsampling,
             optimize=False, progressive=False)
    data = out.getvalue()

    return NormalizeResult(
        data=data,
        status=IIO.OK if res.status == IIO.OK else res.status,
        reason=res.reason,
        out_size=img.size,
        src_size=res.orig_size,
        src_format=res.orig_format,
        crop_xy=(x, y),
        first_gen_quality=fg_q,
        first_gen_subsampling=fg_sub,
        load_status=res.status,
        pad_frac=res.pad_frac,
        n_bytes=len(data),
    )
