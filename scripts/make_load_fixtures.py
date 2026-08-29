"""Write the awkward-image fixtures that ``tests/test_imageio.py`` loads.

Real encoded files on disk, not mocks. A monkeypatched ``Image.open`` proves the
error handling calls the right branch; it proves nothing about whether Pillow
actually raises on *this* CMYK JPEG, and that is the part that breaks in
production. Every file here is written by a real encoder and read back by the
real loader.

The outputs are small (a few KB each) and committed, so the tests do not depend
on this script having been run, and so a Pillow upgrade that changes decoding
behaviour shows up as a test failure rather than as regenerated fixtures.

    python -m scripts.make_load_fixtures
"""

from __future__ import annotations

import argparse
import io
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

DEFAULT_OUT = Path("tests/fixtures/loading")


def _gradient(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Deterministic non-flat RGB content.

    Non-flat matters: a solid colour survives every conversion bug, so a flat
    fixture would pass a test that an inverted CMYK decode should fail.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 255 * x / max(w - 1, 1)
    g = 255 * y / max(h - 1, 1)
    b = 255 * ((x + y) % 64) / 63.0
    a = np.stack([r, g, b], -1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def build(out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    base = Image.fromarray(_gradient(320, 256))

    # --- CMYK JPEG -------------------------------------------------------- #
    # The classic loader-killer. PIL will open it in mode "CMYK"; naive code
    # either crashes on save-to-JPEG or, worse, produces a photographic
    # negative because Adobe writes CMYK JPEGs inverted.
    p = out / "cmyk.jpg"
    base.convert("CMYK").save(p, format="JPEG", quality=92)
    made["cmyk_jpeg"] = p

    # --- alpha PNG, with a genuinely transparent region -------------------- #
    # The transparent block holds saturated magenta underneath. Compositing on
    # black must yield black there; a loader that merely drops the alpha channel
    # yields magenta, and the test can tell the two apart.
    rgba = base.convert("RGBA")
    px = np.array(rgba)
    px[:64, :64] = [255, 0, 255, 0]
    p = out / "alpha.png"
    Image.fromarray(px, "RGBA").save(p, format="PNG")
    made["alpha_png"] = p

    # --- palette PNG with a transparency index ----------------------------- #
    p = out / "palette_transparency.png"
    base.convert("P", palette=Image.Palette.ADAPTIVE, colors=64).save(
        p, format="PNG", transparency=0
    )
    made["palette_png"] = p

    # --- grayscale ---------------------------------------------------------- #
    p = out / "gray.png"
    base.convert("L").save(p, format="PNG")
    made["gray_png"] = p

    p = out / "gray.jpg"
    base.convert("L").save(p, format="JPEG", quality=92)
    made["gray_jpeg"] = p

    # --- 16-bit grayscale PNG ----------------------------------------------- #
    # PIL opens this as mode "I". A plain .convert("L") clips at 255 and returns
    # near-black; to_rgb rescales by actual range instead.
    p = out / "gray16.png"
    a16 = (np.asarray(base.convert("L"), dtype=np.uint16) * 257)
    Image.fromarray(a16, mode="I;16").save(p, format="PNG")
    made["gray16_png"] = p

    # --- EXIF orientation 6 (rotate 90 CW on display) ----------------------- #
    # Portrait pixels tagged to display as landscape. If orientation is carried
    # rather than applied, the manifest records one shape and the model sees the
    # other -- and orientation-tag prevalence differs by camera, hence by source.
    p = out / "exif_orient6.jpg"
    portrait = Image.fromarray(_gradient(200, 400, seed=3))
    exif = Image.Exif()
    exif[274] = 6
    portrait.save(p, format="JPEG", quality=92, exif=exif)
    made["exif_orient6_jpeg"] = p

    # --- JPEG carrying an ICC profile --------------------------------------- #
    # A minimal but structurally valid ICC header. The audit's 0.70-AUROC leak
    # was ICC presence; the test asserts it does not survive the loader.
    icc = (
        struct.pack(">I", 128) + b"ADBE" + b"\x02\x10\x00\x00" + b"mntr" + b"RGB "
        + b"XYZ " + b"\x00" * 12 + b"acsp" + b"APPL" + b"\x00" * 76
    )
    p = out / "with_icc.jpg"
    base.save(p, format="JPEG", quality=92, icc_profile=icc)
    made["icc_jpeg"] = p

    # --- PNG with tEXt chunks ------------------------------------------------ #
    # Generator tools stamp prompts and model names here. Free labels for a
    # cheating classifier.
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Software", "Stable Diffusion")
    meta.add_text("parameters", "a photo of a cat, steps: 30")
    p = out / "with_text_chunks.png"
    base.save(p, format="PNG", pnginfo=meta)
    made["text_chunk_png"] = p

    # --- truncated JPEG (recoverable) ---------------------------------------- #
    # 62% of the scan data. Pillow raises on a short read with
    # LOAD_TRUNCATED_IMAGES off and grey-pads with it on -- the loader must do
    # the second only as an explicit, reported fallback.
    buf = io.BytesIO()
    Image.fromarray(_gradient(512, 512, seed=7)).save(buf, format="JPEG", quality=95)
    full = buf.getvalue()
    p = out / "truncated.jpg"
    p.write_bytes(full[: int(len(full) * 0.62)])
    made["truncated_jpeg"] = p

    # --- truncated PNG (unrecoverable) ---------------------------------------- #
    # Cut inside the first IDAT: zlib cannot produce a single full row, so even
    # the permissive path fails. Distinguishes "recovered" from "failed".
    buf = io.BytesIO()
    Image.fromarray(_gradient(512, 512, seed=11)).save(buf, format="PNG")
    raw = buf.getvalue()
    idat = raw.find(b"IDAT")
    p = out / "truncated_hard.png"
    p.write_bytes(raw[: idat + 8])
    made["truncated_png"] = p

    # --- below the 224px floor -------------------------------------------------- #
    p = out / "tiny_64.png"
    Image.fromarray(_gradient(64, 64, seed=13)).save(p, format="PNG")
    made["tiny_png"] = p

    p = out / "thin_1000x100.jpg"          # one side over, one under
    Image.fromarray(_gradient(1000, 100, seed=17)).save(p, format="JPEG", quality=90)
    made["thin_jpeg"] = p

    # --- exactly at the floor --------------------------------------------------- #
    p = out / "exact_224.png"
    Image.fromarray(_gradient(224, 224, seed=19)).save(p, format="PNG")
    made["exact_224_png"] = p

    # --- not an image ----------------------------------------------------------- #
    p = out / "not_an_image.jpg"
    p.write_bytes(b"this file has a .jpg extension and no image in it\n" * 8)
    made["not_an_image"] = p

    # --- zero bytes --------------------------------------------------------------- #
    p = out / "empty.png"
    p.write_bytes(b"")
    made["empty"] = p

    # --- animated GIF (P mode, multiple frames) ------------------------------------ #
    p = out / "animated.gif"
    frames = [Image.fromarray(_gradient(256, 256, seed=s)).convert("P") for s in (1, 2, 3)]
    frames[0].save(p, format="GIF", save_all=True, append_images=frames[1:], duration=80)
    made["animated_gif"] = p

    # --- WebP ------------------------------------------------------------------------ #
    p = out / "image.webp"
    base.save(p, format="WEBP", quality=90)
    made["webp"] = p

    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    made = build(a.out)
    for k, v in sorted(made.items()):
        print(f"  {k:24s} {v.stat().st_size:>8,d} B  {v}")
    print(f"{len(made)} fixtures -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
