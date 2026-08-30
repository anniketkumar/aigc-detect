"""Error analysis note (PLAN.md Sec11, deliverable #5): the highest-confidence
mistakes the aug checkpoint makes on the clean cell of the test set, grouped
as false positives on real images and false negatives by generator.

    python -m scripts.error_analysis --scores results/aug/scores.csv --out results/error_analysis

Why the clean cell only: this is about what the model gets wrong at the
operating point anyone would actually deploy at, not about degradation
robustness (that is scripts/tpr_gap_analysis.py's job). "Highest-confidence
mistake" means furthest past 0.5 on the wrong side: a real image scored near 1
(confident false positive) or an AI image scored near 0 (confident false
negative) -- not merely misclassified, but misclassified with conviction,
which is the failure mode worth spending fix effort on first.

FLUX.1-dev gets its own montage, not just a row in the false-negative table:
it is the floor generator (clean TPR@1% = 0.5375 on aug, the lowest of all
seven -- results/tpr_analysis_aug/report.md) and the most photorealistic of
the seven (SID_Set full_synthetic, saberzl/SID_Set -- src/data/sources.py), so
its misses are the ones informative about what a photorealistic generator gets
past the detector, as opposed to a stylized one nobody would mistake for a
photo anyway.

Writes into ``--out``:

    false_positives.csv           every real image, sorted by score desc
    false_negatives_by_generator.csv   every AI image, sorted by score asc, generator column
    montage_false_positives.jpg   top-N real images scored highest (grid + score/source captions)
    montage_<focus_generator>_false_negatives.jpg   top-N --focus-generator images scored lowest
    note.md                       written findings -- fill in by hand after viewing the montages
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

THUMB = 220
PAD = 6
CAPTION_H = 34


def load_clean(scores_path: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_path)
    clean = df[df["cell"] == "clean"].copy()
    clean["source"] = clean["image_path"].str.split("/").str[3]
    return clean


def build_montage(rows: pd.DataFrame, out_path: Path, caption_fn, cols: int = 4) -> int:
    """Grid of thumbnails, each with a one-line caption drawn below it.
    Skips (and counts) any image_path not found on disk -- expected if the
    corpus subset needed for this montage hasn't been re-fetched locally
    (data/corpus is not checked in; see scripts/download_data.py).
    Returns the number of images actually placed.
    """
    font = ImageFont.load_default(size=14)
    found = []
    for _, row in rows.iterrows():
        p = Path(row["image_path"])
        if not p.exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        img = img.resize((THUMB, THUMB))
        found.append((img, caption_fn(row)))

    if not found:
        return 0

    n = len(found)
    rows_n = (n + cols - 1) // cols
    cell_w, cell_h = THUMB + PAD, THUMB + CAPTION_H + PAD
    canvas = Image.new("RGB", (cols * cell_w + PAD, rows_n * cell_h + PAD), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, caption) in enumerate(found):
        r, c = divmod(i, cols)
        x, y = PAD + c * cell_w, PAD + r * cell_h
        canvas.paste(img, (x, y))
        draw.text((x, y + THUMB + 2), caption, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", type=Path, default=Path("results/aug/scores.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/error_analysis"))
    ap.add_argument("--top-n", type=int, default=12, help="montage size, each grid")
    ap.add_argument("--focus-generator", default="FLUX.1-dev")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    clean = load_clean(a.scores)

    fp = clean[clean["label"] == 0].sort_values("score", ascending=False).reset_index(drop=True)
    fp.to_csv(a.out / "false_positives.csv", index=False)

    fn = clean[clean["label"] == 1].sort_values("score", ascending=True).reset_index(drop=True)
    fn.to_csv(a.out / "false_negatives_by_generator.csv", index=False)

    n_fp = build_montage(
        fp.head(a.top_n), a.out / "montage_false_positives.jpg",
        lambda r: f"{r['source']}  {r['score']:.3f}",
    )
    print(f"-> {a.out / 'montage_false_positives.jpg'} ({n_fp}/{a.top_n} found on disk)")

    fn_focus = fn[fn["source"] == a.focus_generator]
    focus_stem = a.focus_generator.lower().replace(".", "").replace("-", "_")
    fn_montage_path = a.out / f"montage_{focus_stem}_false_negatives.jpg"
    n_fn = build_montage(fn_focus.head(a.top_n), fn_montage_path, lambda r: f"score {r['score']:.3f}")
    print(f"-> {fn_montage_path} ({n_fn}/{a.top_n} found on disk)")

    print("\nFalse-negative rate by generator (clean, threshold 0.5):")
    summary = (
        clean[clean["label"] == 1]
        .groupby("source")["score"]
        .apply(lambda s: (s < 0.5).mean())
        .sort_values(ascending=False)
    )
    print(summary.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
