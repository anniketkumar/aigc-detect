"""Loading hardening for the shared decode path (`src/data/imageio.py`).

Every case runs against a real encoded file from ``tests/fixtures/loading``.
Nothing here is mocked: the point is to find out what Pillow *actually* does
with a CMYK JPEG or a short read on this machine, and a monkeypatch would
answer a different question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFile

from src.data import imageio as iio
from src.data.imageio import (
    MIN_SIDE, FAILED, MOSTLY_PADDING, OK, RECOVERED_TRUNCATED, TOO_SMALL,
)

FIX = Path(__file__).parent / "fixtures" / "loading"

#: Everything that must come back as usable RGB pixels.
LOADABLE = [
    "cmyk.jpg", "alpha.png", "palette_transparency.png", "gray.png", "gray.jpg",
    "gray16.png", "exif_orient6.jpg", "with_icc.jpg", "with_text_chunks.png",
    "truncated.jpg", "tiny_64.png", "thin_1000x100.jpg", "exact_224.png",
    "animated.gif", "image.webp",
]


def test_fixtures_present():
    """A missing fixture must fail loudly, not silently skip the suite."""
    missing = [n for n in LOADABLE if not (FIX / n).exists()]
    assert not missing, f"run `python -m scripts.make_load_fixtures`; missing {missing}"


# --------------------------------------------------------------------------- #
# The contract that holds for every loadable file
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", LOADABLE)
def test_always_rgb_uint8_and_metadata_free(name):
    r = iio.load_image(FIX / name, min_side=None)
    assert r.ok, f"{name}: {r.status} {r.reason}"
    assert r.image.mode == "RGB"
    assert not r.image.info, f"{name} kept metadata: {sorted(r.image.info)}"
    a = np.asarray(r.image)
    assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3


@pytest.mark.parametrize("name", LOADABLE)
def test_no_exception_escapes(name):
    """The loader's job is to never take down a 20k-image manifest build."""
    iio.load_image(FIX / name)


def test_load_is_deterministic():
    a = np.asarray(iio.load_image(FIX / "cmyk.jpg", min_side=None).image)
    b = np.asarray(iio.load_image(FIX / "cmyk.jpg", min_side=None).image)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# CMYK
# --------------------------------------------------------------------------- #

def test_cmyk_jpeg_opens_as_cmyk_so_the_fixture_is_actually_testing_something():
    with Image.open(FIX / "cmyk.jpg") as im:
        assert im.mode == "CMYK"


def test_cmyk_is_not_inverted():
    """An inverted CMYK decode is still valid RGB, so only content can catch it.

    The fixture is a gradient: dark at x=0, bright at x=max. If the Adobe
    inversion is mishandled the image comes back as a negative and the
    comparison flips. Compared against the same gradient decoded from the
    RGB-native PNG, not against a hardcoded number.
    """
    got = np.asarray(iio.load_image(FIX / "cmyk.jpg", min_side=None).image, dtype=float)
    ref = np.asarray(iio.load_image(FIX / "gray.png", min_side=None).image, dtype=float)
    left, right = got[:, :20].mean(), got[:, -20:].mean()
    assert right > left, "CMYK decode looks inverted (bright end came out dark)"
    # and it must correlate positively with the same content decoded natively
    c = np.corrcoef(got.mean(2).ravel(), ref.mean(2).ravel())[0, 1]
    assert c > 0.8, f"CMYK content does not match the RGB reference (r={c:.2f})"


# --------------------------------------------------------------------------- #
# Alpha
# --------------------------------------------------------------------------- #

def test_alpha_is_composited_on_black_not_dropped():
    """The fixture hides saturated magenta under a fully-transparent block.

    Dropping the alpha channel yields magenta there. Compositing yields black.
    Only that distinction proves which one happened.
    """
    img = iio.load_image(FIX / "alpha.png", min_side=None).image
    corner = np.asarray(img)[:64, :64]
    assert corner.max() == 0, (
        f"transparent region came through as {corner.reshape(-1,3).max(0)} "
        "-- alpha was dropped, not composited"
    )
    assert np.asarray(img)[100:, 100:].max() > 0, "whole image went black"


def test_palette_with_transparency_survives():
    r = iio.load_image(FIX / "palette_transparency.png", min_side=None)
    assert r.ok and r.image.mode == "RGB"


# --------------------------------------------------------------------------- #
# Grayscale
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["gray.png", "gray.jpg"])
def test_grayscale_becomes_three_equal_channels(name):
    a = np.asarray(iio.load_image(FIX / name, min_side=None).image)
    assert np.array_equal(a[..., 0], a[..., 1])
    assert np.array_equal(a[..., 1], a[..., 2])
    assert a.std() > 5, "grayscale image came back flat"


def test_16bit_grayscale_is_rescaled_not_clipped():
    """PIL opens I;16 PNGs as mode "I"; ``.convert("L")`` clips at 255.

    A clipping loader returns a near-white or near-black image with almost no
    variance. Rescaling by actual range preserves the gradient.
    """
    with Image.open(FIX / "gray16.png") as im:
        assert im.mode in ("I", "I;16"), f"fixture opened as {im.mode}"
    a = np.asarray(iio.load_image(FIX / "gray16.png", min_side=None).image, dtype=float)
    assert a.std() > 20, f"16-bit image lost its dynamic range (std={a.std():.1f})"
    assert 20 < a.mean() < 235


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #

def test_truncated_jpeg_is_recovered_and_reported():
    r = iio.load_image(FIX / "truncated.jpg", min_side=None)
    assert r.ok
    assert r.status == RECOVERED_TRUNCATED, (
        f"expected the truncation to be reported, got {r.status!r}. "
        "A silent recovery means grey-padded images enter training unlabelled."
    )
    assert r.image.size == (512, 512)


def test_truncated_jpeg_rejected_when_not_allowed():
    r = iio.load_image(FIX / "truncated.jpg", min_side=None, allow_truncated=False)
    assert not r.ok and r.status == FAILED


def test_a_recovery_that_is_all_padding_is_rejected():
    """The case that motivated the padding guard.

    ``truncated_hard.png`` is cut at the start of the first IDAT. Pillow's
    permissive path does *not* fail on it -- it returns a perfectly well-formed
    512x512 image that is 100% black. Nothing about the return value says
    "this is not a photograph", so without an explicit content check a solid
    black rectangle enters training carrying a real label.
    """
    r = iio.load_image(FIX / "truncated_hard.png", min_side=None)
    assert r.status == MOSTLY_PADDING, f"got {r.status!r}"
    assert r.image is None and r.pad_frac == pytest.approx(1.0)


def test_partial_padding_is_kept_but_measured():
    """62% real content: usable, and the padding fraction is on the record."""
    r = iio.load_image(FIX / "truncated.jpg", min_side=None)
    assert r.status == RECOVERED_TRUNCATED
    assert 0.2 < r.pad_frac < 0.5


def test_padding_threshold_is_tunable():
    strict = iio.load_image(FIX / "truncated.jpg", min_side=None, max_padding=0.1)
    assert strict.status == MOSTLY_PADDING and strict.image is None
    loose = iio.load_image(FIX / "truncated_hard.png", min_side=None, max_padding=1.1)
    assert loose.ok


def test_padding_fraction_is_zero_for_a_healthy_image():
    """The guard must not fire on files that were never truncated.

    A legitimately flat image is only suspicious *in combination with* a short
    read, so the fraction is measured on the recovery path alone.
    """
    r = iio.load_image(FIX / "exact_224.png")
    assert r.status == OK and r.pad_frac == 0.0


def test_corrupt_header_fails_cleanly(tmp_path):
    """The hard-failure path still exists: a file with a broken IHDR CRC.

    Written here rather than committed because it is a byte-level mutation of
    another fixture, and keeping the mutation visible in the test is clearer
    than a 45-byte blob in the repo.
    """
    raw = bytearray((FIX / "exact_224.png").read_bytes())
    raw[12:16] = b"XXXX"                        # clobber the IHDR chunk type
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(bytes(raw))
    r = iio.load_image(bad, min_side=None)
    assert not r.ok and r.status == FAILED and r.reason


def test_truncation_flag_is_restored_after_a_recovery():
    """The whole reason for the context manager.

    ``LOAD_TRUNCATED_IMAGES`` is a Pillow global. If a recovery leaked it on,
    every later short read anywhere in the process would silently grey-pad, and
    the ``RECOVERED_TRUNCATED`` status -- the only thing that records it --
    would never fire again.
    """
    before = ImageFile.LOAD_TRUNCATED_IMAGES
    iio.load_image(FIX / "truncated.jpg", min_side=None)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is before

    # and the next strict read still detects truncation
    assert iio.load_image(FIX / "truncated.jpg", min_side=None,
                          allow_truncated=False).status == FAILED


# --------------------------------------------------------------------------- #
# Size floor
# --------------------------------------------------------------------------- #

def test_tiny_image_is_flagged_not_upscaled():
    r = iio.load_image(FIX / "tiny_64.png")
    assert r.status == TOO_SMALL
    assert r.image is not None and r.image.size == (64, 64), (
        "the loader must not resize; sizing policy belongs to the caller"
    )


def test_min_side_uses_the_shorter_side():
    """1000x100: one side clears the floor, the other does not."""
    assert iio.load_image(FIX / "thin_1000x100.jpg").status == TOO_SMALL


def test_exactly_at_the_floor_is_accepted():
    r = iio.load_image(FIX / "exact_224.png")
    assert r.status == OK and min(r.image.size) == MIN_SIDE


def test_min_side_none_disables_the_floor():
    assert iio.load_image(FIX / "tiny_64.png", min_side=None).status == OK


def test_bad_min_side_is_a_programmer_error():
    with pytest.raises(ValueError):
        iio.load_image(FIX / "exact_224.png", min_side=0)


# --------------------------------------------------------------------------- #
# Metadata stripping and EXIF orientation
# --------------------------------------------------------------------------- #

def test_icc_profile_is_present_in_the_fixture_and_gone_after_loading():
    with Image.open(FIX / "with_icc.jpg") as im:
        assert im.info.get("icc_profile"), "fixture lost its ICC profile"
    r = iio.load_image(FIX / "with_icc.jpg", min_side=None)
    assert r.had_icc is True
    assert "icc_profile" not in r.image.info and not r.image.info


def test_png_text_chunks_are_stripped():
    with Image.open(FIX / "with_text_chunks.png") as im:
        assert im.info.get("Software") == "Stable Diffusion"
    r = iio.load_image(FIX / "with_text_chunks.png", min_side=None)
    assert r.had_text_chunks is True
    assert not r.image.info


def test_exif_orientation_is_applied_not_carried():
    """Fixture is 200x400 pixels tagged orientation 6 (display rotated 90 CW).

    Applied correctly the loaded image is 400x200. Left unapplied it stays
    200x400 and the manifest records a shape the viewer never shows.
    """
    with Image.open(FIX / "exif_orient6.jpg") as im:
        assert im.size == (200, 400) and im.getexif().get(274) == 6
    r = iio.load_image(FIX / "exif_orient6.jpg", min_side=None)
    assert r.exif_orientation == 6
    assert r.image.size == (400, 200), "EXIF orientation was not applied"
    assert not r.image.info


def test_exif_can_be_left_unapplied_on_request():
    r = iio.load_image(FIX / "exif_orient6.jpg", min_side=None, apply_exif=False)
    assert r.image.size == (200, 400)


def test_strip_metadata_is_pixel_identical():
    with Image.open(FIX / "with_icc.jpg") as im:
        im.load()
        ref = np.asarray(iio.to_rgb(im))
        got = np.asarray(iio.strip_metadata(im))
    assert np.array_equal(ref, got), "stripping metadata changed pixels"


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["not_an_image.jpg", "empty.png"])
def test_garbage_fails_without_raising(name):
    r = iio.load_image(FIX / name)
    assert not r.ok and r.status == FAILED and r.reason


def test_missing_file_fails_without_raising():
    r = iio.load_image(FIX / "does_not_exist_at_all.png")
    assert not r.ok and "not found" in r.reason


def test_directory_path_fails_without_raising():
    assert not iio.load_image(FIX).ok


# --------------------------------------------------------------------------- #
# The harness wrapper
# --------------------------------------------------------------------------- #

def test_load_rgb_has_no_size_floor_by_default():
    """At scoring time a small image is still an image needing a number."""
    img = iio.load_rgb(FIX / "tiny_64.png")
    assert img is not None and img.size == (64, 64)


def test_load_rgb_returns_none_on_failure():
    assert iio.load_rgb(FIX / "empty.png") is None


def test_evaluate_uses_the_shared_loader():
    """Guards against the harness drifting back to its own ``Image.open``.

    If these two ever diverge, the manifest describes images the harness never
    sees -- different orientation, different truncation policy.
    """
    from src import evaluate

    img = evaluate._load_image(str(FIX / "exif_orient6.jpg"))
    assert img is not None
    assert img.size == (400, 200), "harness is not applying EXIF orientation"
    assert not img.info, "harness is not stripping metadata"


@pytest.mark.parametrize("name", LOADABLE)
def test_transform_grid_runs_on_every_awkward_input(name):
    """End-to-end: whatever the loader accepts, the Phase 1 grid must survive.

    A loader that returns something the grid then crashes on has not hardened
    anything.
    """
    from src import transforms as T

    img = iio.load_rgb(FIX / name)
    assert img is not None
    for cell in T.build_cells():
        out = T.apply_cell(img, cell, image_id=name, base_seed=0)
        assert out.mode == "RGB" and out.size == img.size
