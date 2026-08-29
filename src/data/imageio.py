"""The one image-loading path. Used by the eval harness *and* the manifest builder.

Two callers, one decoder, deliberately. If the harness and the manifest builder
each had their own ``Image.open``, they would disagree about what "the image" is
-- most dangerously about EXIF orientation and about whether a truncated file is
usable -- and the manifest would then describe a dataset the harness never sees.

What this module guarantees for every file it accepts:

* EXIF orientation is **applied**, not carried. A rotation flag left in the
  metadata is a rotation the model never sees but a human does.
* Metadata is **stripped**, all of it: EXIF, ICC profile, PNG text chunks, XMP.
  The Phase 2 audit found ICC present in 40.6% of real images and 0% of AI ones
  -- a 0.70-AUROC classifier that reads no pixels. Stripping happens here so it
  cannot be forgotten at one call site.
* Mode is normalized to RGB, with alpha composited rather than dropped.
* Truncation is *recovered explicitly and reported*, never silently tolerated.

Nothing here resizes. Sizing policy belongs to the caller (§ normalization: crop
at native resolution, never resample), and a loader that quietly resized would
make that policy unenforceable.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageFile, ImageOps

__all__ = [
    "LoadResult",
    "LoadStatus",
    "load_image",
    "load_rgb",
    "strip_metadata",
    "to_rgb",
    "MIN_SIDE",
    "MAX_PADDING_FRACTION",
    "padding_fraction",
]

#: Below this, an image cannot fill a 224x224 crop without upscaling, and
#: upscaling is exactly the resampling signature the Phase 5 artifact branch
#: reads. Such files are excluded from the manifest rather than stretched.
MIN_SIDE = 224

#: A truncation recovery above this much constant-row padding is discarded.
#: Pillow's permissive decode fills unread scanlines with a flat colour -- grey
#: for JPEG, black for PNG -- so a badly truncated file comes back as a
#: full-size image that is mostly a solid block. Those are not images; and
#: because broken downloads cluster by source, and source correlates with class,
#: a padded block is a *label leak* dressed as a photo. 0.5 is deliberately
#: permissive: it discards files that are more padding than picture and keeps
#: the rest, flagged, with the fraction recorded.
MAX_PADDING_FRACTION = 0.5

#: Guard against decompression-bomb DoS while still allowing legitimately large
#: source images (SID_Set is 1024^2; some Flickr originals are 50+ MP).
Image.MAX_IMAGE_PIXELS = 200_000_000


class LoadStatus(str):
    """Outcome of a load attempt. Subclasses ``str`` so it round-trips to CSV."""


OK = LoadStatus("ok")
RECOVERED_TRUNCATED = LoadStatus("recovered_truncated")
TOO_SMALL = LoadStatus("too_small")
MOSTLY_PADDING = LoadStatus("mostly_padding")
FAILED = LoadStatus("failed")


@dataclass(frozen=True)
class LoadResult:
    """Why this is not just ``Image | None``.

    The manifest builder needs to *record* that a file was truncated-but-usable
    or below the size floor; the harness only needs the pixels. One return type
    serves both, and the reason string ends up in ``data_stats.md`` instead of
    scrolling past on stderr.
    """

    image: Image.Image | None
    status: str
    reason: str = ""
    orig_mode: str = ""
    orig_size: tuple[int, int] = (0, 0)
    orig_format: str = ""
    had_exif: bool = False
    had_icc: bool = False
    had_text_chunks: bool = False
    exif_orientation: int = 0
    pad_frac: float = 0.0

    @property
    def ok(self) -> bool:
        """True if there are usable pixels, truncation-recovered or not."""
        return self.image is not None


# --------------------------------------------------------------------------- #
# Mode normalization
# --------------------------------------------------------------------------- #

def to_rgb(img: Image.Image) -> Image.Image:
    """Any PIL mode -> 3-channel RGB.

    Alpha is composited onto black rather than discarded, so a transparent PNG
    degrades the way a flattened one would instead of exposing whatever garbage
    happens to sit in the RGB channels under a fully-transparent pixel. Many
    encoders leave those channels uninitialised, which would be a per-encoder --
    and therefore per-generator -- fingerprint.

    ``I;16`` / ``F`` are rescaled by actual range rather than truncated: PIL's
    own ``convert("L")`` on 16-bit clips at 255 and returns a near-black image.
    """
    mode = img.mode
    if mode == "RGB":
        return img

    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")

    if mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        a = np.asarray(img).astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)
        return Image.fromarray(
            np.repeat(np.round(a * 255).astype(np.uint8)[:, :, None], 3, axis=2),
            mode="RGB",
        )

    # CMYK, YCbCr, L, P-without-transparency, 1, LAB, HSV.
    # CMYK JPEGs written by Adobe are stored inverted; PIL reads the APP14
    # marker and handles that itself, so a plain convert is correct here. The
    # fixture test pins it, because getting it wrong yields a photographic
    # negative that is still a perfectly valid RGB image and fails silently.
    return img.convert("RGB")


def strip_metadata(img: Image.Image) -> Image.Image:
    """Return pixel-identical RGB with an empty ``.info``.

    ``copy()`` carries ``.info`` forward, and so does every ``convert`` and
    ``resize``; ICC profiles in particular survive most of the operations you
    would expect to drop them. Rebuilding from raw bytes is the only way to be
    sure, and it is cheap.
    """
    rgb = to_rgb(img)
    out = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    assert not out.info, f"metadata survived strip: {sorted(out.info)}"
    return out


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def _truncation_allowed() -> Iterator[None]:
    """Scope ``LOAD_TRUNCATED_IMAGES`` to a single retry.

    Pillow exposes this as a module-level global. Setting it once at import --
    which is what most code does -- means every short read anywhere in the
    process silently yields a grey-padded image. We would then be training on
    images whose bottom third is flat grey, and grey padding correlates with
    source (broken scrapes cluster by origin), which is a leak. So: strict read
    first, and the permissive read only as an explicit, reported fallback.
    """
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev


def _probe_metadata(img: Image.Image) -> dict:
    """Record what metadata *was* present, before it is thrown away.

    The regression test in task F asserts these channels are uninformative after
    normalization; it needs the pre-strip values to have something to assert on.
    """
    info = img.info or {}
    orientation = 0
    try:
        exif = img.getexif()
        n_exif = len(exif)
        orientation = int(exif.get(274, 0) or 0)
    except Exception:
        n_exif = 0
    text = info.get("text") or {}
    n_text = len(text) if isinstance(text, dict) else 0
    # PNG tEXt/iTXt land directly in .info as str->str alongside known keys.
    _known = {
        "transparency", "gamma", "dpi", "aspect", "icc_profile", "exif",
        "srgb", "chromaticity", "text", "interlace", "compression",
        "adobe", "adobe_transform", "jfif", "jfif_version", "jfif_unit",
        "jfif_density", "progression", "progressive", "signed", "background",
        "loop", "duration", "timestamp", "extension", "version", "comment",
        "photoshop", "XML:com.adobe.xmp", "xmp", "date:create", "date:modify",
    }
    n_text += sum(
        1 for k, v in info.items() if k not in _known and isinstance(v, (str, bytes))
    )
    return {
        "had_exif": n_exif > 0,
        "had_icc": bool(info.get("icc_profile")),
        "had_text_chunks": n_text > 0,
        "exif_orientation": orientation,
    }



def padding_fraction(img: Image.Image) -> float:
    """Fraction of rows that are a single constant colour.

    A proxy for Pillow's truncation padding. Chosen over "fraction of pixels
    equal to the modal colour" because a legitimate photo can be 40% sky of one
    quantised colour, but essentially never has 40% of its rows *exactly*
    constant across the full width. Measured on a row subsample: at 1024 rows
    the full scan costs more than the decode it is checking.
    """
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[0] == 0:
        return 0.0
    step = max(1, a.shape[0] // 256)
    rows = a[::step]
    const = (rows == rows[:, :1, :]).all(axis=(1, 2))
    return float(const.mean())


# --------------------------------------------------------------------------- #
# The loader
# --------------------------------------------------------------------------- #

def load_image(
    path: str | Path,
    *,
    min_side: int | None = MIN_SIDE,
    allow_truncated: bool = True,
    max_padding: float = MAX_PADDING_FRACTION,
    apply_exif: bool = True,
    strip: bool = True,
) -> LoadResult:
    """Decode one file into normalized RGB, or explain why not.

    Never raises on a bad file -- a single corrupt image in a 20k-image manifest
    build must not cost the build. Programmer errors (a bad ``min_side``) still
    raise, because those are bugs, not data.

    ``min_side=None`` disables the size floor; the harness passes that, since it
    must score whatever it is handed, including a 32x32 CIFAKE tile.
    """
    p = Path(path)
    status = OK
    reason = ""

    def fail(msg: str) -> LoadResult:
        return LoadResult(None, FAILED, msg)

    try:
        img = Image.open(p)
    except FileNotFoundError:
        return fail("file not found")
    except Image.UnidentifiedImageError:
        return fail("not a recognised image format")
    except Exception as exc:
        return fail(f"{type(exc).__name__}: {exc}")

    orig_mode, orig_size, orig_format = img.mode, img.size, (img.format or "")
    meta = _probe_metadata(img)

    # Strict decode first. Pillow raises OSError on a short read only when
    # LOAD_TRUNCATED_IMAGES is off, which is the whole reason we keep it off.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img.load()
    except Image.DecompressionBombWarning as exc:
        return fail(f"decompression bomb: {exc}")
    except Exception as exc:
        if not allow_truncated:
            return fail(f"truncated: {type(exc).__name__}: {exc}")
        try:
            with _truncation_allowed():
                img = Image.open(p)
                img.load()
            status = RECOVERED_TRUNCATED
            reason = f"short read recovered ({type(exc).__name__})"
        except Exception as exc2:
            return fail(f"undecodable even permissively: {type(exc2).__name__}: {exc2}")

    if apply_exif:
        try:
            img = ImageOps.exif_transpose(img) or img
        except Exception:
            pass  # a malformed orientation tag is not worth losing the image over

    try:
        out = strip_metadata(img) if strip else to_rgb(img)
    except Exception as exc:
        return fail(f"mode conversion failed from {orig_mode}: {type(exc).__name__}: {exc}")

    pad = padding_fraction(out) if status == RECOVERED_TRUNCATED else 0.0
    if status == RECOVERED_TRUNCATED and pad > max_padding:
        return LoadResult(
            None, MOSTLY_PADDING,
            f"{pad:.0%} of rows are decoder padding after truncation recovery",
            orig_mode, orig_size, orig_format, **meta, pad_frac=pad,
        )

    if min_side is not None:
        if min_side < 1:
            raise ValueError(f"min_side must be >= 1, got {min_side}")
        if min(out.size) < min_side:
            return LoadResult(
                out, TOO_SMALL,
                f"min side {min(out.size)} < {min_side}",
                orig_mode, orig_size, orig_format, **meta, pad_frac=pad,
            )

    return LoadResult(out, status, reason, orig_mode, orig_size, orig_format,
                      **meta, pad_frac=pad)


def load_rgb(path: str | Path, *, min_side: int | None = None) -> Image.Image | None:
    """Thin wrapper for callers that only want pixels (the eval harness).

    Note the default ``min_side=None``: at scoring time a too-small image is
    still an image the model must produce a number for. The floor is a
    *manifest-build* policy, not a decode one.
    """
    return load_image(path, min_side=min_side).image
