"""The deliverable (PLAN.md §9.1): an image directory in, ``preds.json`` out.

    python predict.py --image_dir path/to/images --out preds.json

Recurses ``--image_dir`` for jpg/jpeg/png/webp/bmp files (case-insensitive)
and scores each with the frozen-CLIP linear probe
(``src/models/clip_baseline.py``), writing one JSON array:

    [{"image_path": "path/to/images/a.jpg", "pred": 0.873, "domain_flag": null}, ...]

Every image goes through the same hardened decode path the eval harness and
the manifest builder use (``src/data/imageio.py``): PNGs with alpha, grayscale
images, and CMYK JPEGs are all converted to RGB without special-casing here,
and a recoverably-truncated file is still scored (with a warning). Only a
*genuine* decode failure gets ``"pred": null`` -- this must never crash on a
directory the model has never seen, no matter what garbage is in it (§9.1:
"the most common way a good project scores badly").

``domain_flag`` is ``"non_photographic"`` when ``src/data/domain_guard.py``'s
cheap pixel-statistics heuristic thinks the image is a rendered graphic
(screenshot, diagram, infographic) rather than a camera photo, else ``null``.
Every training source, real and AI, is photographic (``src/data/sources.py``),
so the model has no basis for scoring this content type and ``pred`` there is
not meaningful -- the flag exists so a caller can say so instead of presenting
a confident label. It's advisory: ``pred`` is still the model's real output
either way, flagged or not.

Deterministic: the file list is sorted, so the same ``--image_dir`` produces
byte-identical output on every run, on every OS, regardless of the
filesystem's own directory-listing order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from src.data import imageio as IIO
from src.data.domain_guard import check_domain
from src.models.base import load_model

__all__ = ["IMAGE_EXTENSIONS", "find_images", "load_for_scoring", "predict", "main"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CKPT = Path("runs/baseline.pt")


def find_images(image_dir: Path) -> list[Path]:
    """Every file under ``image_dir`` (recursive) with a supported extension,
    sorted so output order never depends on the OS's own listing order."""
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_for_scoring(path: Path) -> tuple[Image.Image | None, str | None]:
    """Decode via the same hardened path the eval harness and manifest builder
    use (``src/data/imageio.py``) -- one decoder, so a file this script scores
    and a file the training pipeline scored never quietly disagree.

    ``min_side=None``: a too-small image is still an image that needs a
    number, same reasoning as ``src.evaluate._load_image``.

    Returns ``(image, warning)``. ``image`` is ``None`` only on a genuine
    decode failure -- the caller emits ``pred: null`` for that, never raises.
    """
    res = IIO.load_image(str(path), min_side=None)
    if res.image is None:
        return None, f"could not read {path}: {res.reason}"
    if res.status == IIO.RECOVERED_TRUNCATED:
        return res.image, f"{path}: {res.reason}"
    return res.image, None


def predict(
    image_dir: Path,
    out: Path,
    ckpt: Path = DEFAULT_CKPT,
    device: str = "cpu",
    batch_size: int = 64,
    quiet: bool = False,
) -> list[dict]:
    """Score every supported image under ``image_dir`` and write ``out``.

    Batches across *readable* images only -- an unreadable file never reaches
    the model and never shrinks a batch the model actually sees, so a
    directory with a few corrupt files scores identically to the same
    directory with those files removed.
    """
    paths = find_images(image_dir)
    if not paths:
        raise SystemExit(
            f"{image_dir}: no images found (looked for {sorted(IMAGE_EXTENSIONS)})"
        )

    model = load_model("clip_linear", ckpt=ckpt, device=device)
    if not quiet:
        print(f"model={model.name}  images={len(paths)}  device={device}", file=sys.stderr)

    results: list[dict | None] = [None] * len(paths)
    batch_imgs: list = []
    batch_ids: list[str] = []
    batch_slots: list[int] = []
    batch_flags: list[str | None] = []

    def flush() -> None:
        if not batch_imgs:
            return
        scores = model.score(batch_imgs, batch_ids)
        if len(scores) != len(batch_imgs):
            raise RuntimeError(
                f"{model.name}.score returned {len(scores)} scores for "
                f"{len(batch_imgs)} images"
            )
        for slot, image_path, score, flag in zip(batch_slots, batch_ids, scores, batch_flags):
            results[slot] = {
                "image_path": image_path,
                "pred": None if score is None else float(score),
                "domain_flag": flag,
            }
        batch_imgs.clear()
        batch_ids.clear()
        batch_slots.clear()
        batch_flags.clear()

    for i, path in enumerate(paths):
        image_path = path.as_posix()
        img, warning = load_for_scoring(path)
        if warning:
            print(f"[warn] {warning}", file=sys.stderr)
        if img is None:
            results[i] = {"image_path": image_path, "pred": None, "domain_flag": None}
            continue
        # Cheap heuristic on the canonically-decoded image, same convention
        # a Scorer uses -- see src/data/domain_guard.py for what this can
        # and can't tell us.
        flag = "non_photographic" if check_domain(img).likely_non_photographic else None
        batch_imgs.append(img)
        batch_ids.append(image_path)
        batch_slots.append(i)
        batch_flags.append(flag)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_null = sum(1 for r in results if r["pred"] is None)
    if not quiet:
        print(f"wrote {len(results)} predictions ({n_null} null) -> {out}", file=sys.stderr)
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="predict.py", description=__doc__.splitlines()[0]
    )
    ap.add_argument("--image_dir", type=Path, required=True,
                     help="directory to recurse for images")
    ap.add_argument("--out", type=Path, required=True,
                     help="output JSON path, e.g. preds.json")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                     help=f"checkpoint written by src/train.py (default: {DEFAULT_CKPT})")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image_dir.is_dir():
        raise SystemExit(f"--image_dir {args.image_dir} is not a directory")

    predict(
        args.image_dir, args.out, ckpt=args.ckpt, device=args.device,
        batch_size=args.batch_size, quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
