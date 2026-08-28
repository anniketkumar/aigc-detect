"""Tests for the audit primitives (Phase 2).

`phash` is homegrown (§2 pins no imagehash), and the §4.3 "no near-duplicate
spans a split boundary" assertion will be built on it. It gets verified here
before anything depends on it.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from scripts.audit_leakage import hamming, phash, png_text_chunks
from scripts.fetch_audit_sample import sniff


def _img(w=256, h=256, seed=0) -> Image.Image:
    """A photograph-like test image: energy concentrated at low frequencies with
    light grain on top.

    The noise level matters. An image dominated by white noise has a flat DCT
    spectrum, so the mid-band coefficients pHash thresholds sit right at the
    median and flip on any perturbation — such an image makes pHash look far
    more fragile than it is on real photographs. Calibration on 150 real
    SID_Set images: resize 0.25x, JPEG q30 and blur sigma=2 all stay within
    Hamming distance 2, while 11,175 unrelated pairs never came closer than 10.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        128
        + 60 * np.sin(xx / 40.0 + seed)
        + 40 * np.cos(yy / 55.0 - seed)
        + 25 * np.sin((xx + yy) / 30.0)
    )
    base = base + rng.normal(0, 2, size=(h, w))
    arr = np.clip(np.stack([base, np.roll(base, 9, 1), np.roll(base, 21, 0)], -1), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# phash
# --------------------------------------------------------------------------- #

def test_phash_is_stable_and_64_bit():
    img = _img()
    assert phash(img) == phash(img)
    assert 0 <= phash(img) < (1 << 64)


def test_phash_is_invariant_to_rescale():
    """The whole point: a thumbnail of an image must match the original, or the
    near-duplicate split check misses resized reposts."""
    img = _img()
    small = img.resize((128, 128), Image.Resampling.BICUBIC)
    assert hamming(phash(img), phash(small)) <= 4


def test_phash_survives_jpeg_recompression():
    img = _img()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    buf.seek(0)
    assert hamming(phash(img), phash(Image.open(buf))) <= 6


def test_phash_separates_unrelated_images():
    a, b = _img(seed=0), _img(seed=99)
    assert hamming(phash(a), phash(b)) > 12


def test_phash_is_not_crop_invariant():
    """Documented limitation, not a bug. A 10%-per-side crop moves the hash by
    ~20 bits — as far as an unrelated image. So the near-duplicate check catches
    rescaled and recompressed reposts but NOT crops, and §11's "small crops of
    large fakes" failure mode cannot be defended by phash alone."""
    img = _img(256, 256)
    w, h = img.size
    cropped = img.crop((int(w * .1), int(h * .1), int(w * .9), int(h * .9)))
    assert hamming(phash(img), phash(cropped)) > 10


def test_phash_distance_is_symmetric_and_zero_on_self():
    a, b = phash(_img(seed=1)), phash(_img(seed=2))
    assert hamming(a, a) == 0
    assert hamming(a, b) == hamming(b, a)


def test_phash_ignores_the_dc_term():
    """A uniform brightness shift must not change the hash: reposts get
    re-exposed, and DC is dropped before thresholding.

    Uses an image with headroom on purpose. Shifting one that already peaks near
    255 clips the highlights, which is a real content change, not a DC shift —
    and it would make this test fail for the wrong reason.
    """
    a = np.asarray(_img(), dtype=np.float32)
    a = 40 + a * (170.0 / 255.0)          # squeeze into [40, 210]
    img = Image.fromarray(a.astype(np.uint8), "RGB")
    brighter = Image.fromarray((a + 30).astype(np.uint8), "RGB")
    assert np.asarray(brighter).max() < 255, "test image must not clip"
    assert hamming(phash(img), phash(brighter)) <= 2


def test_phash_handles_grayscale_and_alpha():
    for mode in ("L", "RGBA", "CMYK"):
        assert isinstance(phash(_img(64, 64).convert(mode)), int)


# --------------------------------------------------------------------------- #
# format sniffing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fmt,expected", [
    ("JPEG", "jpeg"), ("PNG", "png"), ("WEBP", "webp"), ("BMP", "bmp"), ("GIF", "gif"),
])
def test_sniff_identifies_real_encoder_output(fmt, expected):
    buf = io.BytesIO()
    img = _img(64, 64)
    img.convert("P" if fmt == "GIF" else "RGB").save(buf, format=fmt)
    assert sniff(buf.getvalue())[0] == expected


def test_sniff_never_trusts_a_filename_and_fails_soft():
    assert sniff(b"not an image at all")[0] == "unknown"
    assert sniff(b"")[0] == "unknown"
    # RIFF that is not WEBP must not be reported as webp
    assert sniff(b"RIFF\x00\x00\x00\x00WAVE")[0] == "unknown"


# --------------------------------------------------------------------------- #
# PNG text chunks (item c)
# --------------------------------------------------------------------------- #

def test_png_text_chunks_reads_a1111_style_parameters(tmp_path):
    from PIL import PngImagePlugin

    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "a photo, Negative prompt: blur, Steps: 30, "
                                "Sampler: DPM++ 2M, CFG scale: 7, Model: sdxl")
    p = tmp_path / "gen.png"
    _img(32, 32).save(p, pnginfo=meta)
    got = png_text_chunks(p)
    assert "parameters" in got
    assert "Sampler" in got["parameters"]


def test_png_text_chunks_reads_compressed_ztxt(tmp_path):
    from PIL import PngImagePlugin

    meta = PngImagePlugin.PngInfo()
    meta.add_text("workflow", '{"nodes": "comfyui graph"}' * 40, zip=True)
    p = tmp_path / "comfy.png"
    _img(32, 32).save(p, pnginfo=meta)
    got = png_text_chunks(p)
    assert "workflow" in got and "comfyui" in got["workflow"]


def test_png_text_chunks_empty_for_clean_png_and_for_jpeg(tmp_path):
    clean = tmp_path / "clean.png"
    _img(32, 32).save(clean)
    assert png_text_chunks(clean) == {}
    j = tmp_path / "x.jpg"
    _img(32, 32).save(j)
    assert png_text_chunks(j) == {}


def test_png_text_chunks_survives_a_truncated_file(tmp_path):
    p = tmp_path / "trunc.png"
    _img(32, 32).save(p)
    p.write_bytes(p.read_bytes()[:120])
    assert isinstance(png_text_chunks(p), dict)  # must not raise
