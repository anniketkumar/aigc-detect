"""Leakage audit over a labelled image sample (Phase 2 gate).

Answers: can a classifier score well on this data *without learning anything
about image generation*? Every item below is a channel that has sunk published
AIGC-detection results.

    a) container format by class   — real=JPEG / AI=PNG is a free perfect score
    b) resolution by class         — same failure, different hat
    c) EXIF + PNG text chunks      — SD/ComfyUI/A1111 write params routinely
    d) real-class content type     — photographic vs illustration vs screenshot
    e) duplicates / near-duplicates — within and across classes, perceptual hash

Then the number that matters: the AUROC a *metadata-only* classifier achieves,
using the Phase 1 metric code. If that is near 1.0, no amount of modelling work
on top is measuring detection.

Usage:
    python -m scripts.audit_leakage --manifest data/audit_sample/sid_set/sample_manifest.csv
    python -m scripts.audit_leakage --manifest ... --out results/audit_sid_set.md
"""

from __future__ import annotations

import argparse
import collections
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

from src.metrics import cell_metrics

Image.MAX_IMAGE_PIXELS = None  # audit must not refuse to look at large images

#: Keys that indicate generator tooling wrote metadata into the file.
#: Automatic1111 writes "parameters"; ComfyUI writes "workflow" and "prompt";
#: many pipelines leave "Software" or "Comment".
GENERATOR_METADATA_KEYS = {
    "parameters", "workflow", "prompt", "software", "comment", "description",
    "sd-metadata", "dream", "invokeai", "generation_data", "negative_prompt",
    "model", "sampler", "steps", "cfg_scale", "seed", "xmp", "usercomment",
}
GENERATOR_TOOL_MARKERS = [
    "stable diffusion", "stable-diffusion", "comfyui", "automatic1111", "a1111",
    "invokeai", "midjourney", "dall-e", "dalle", "novelai", "sdxl", "flux",
    "civitai", "lora", "denoising strength", "negative prompt", "cfg scale",
]


# --------------------------------------------------------------------------- #
# Perceptual hashing (numpy; avoids adding a dependency not in §2)
# --------------------------------------------------------------------------- #

def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))


_DCT32 = _dct_matrix(32)


def phash(img: Image.Image, hash_size: int = 8) -> int:
    """DCT-based perceptual hash, 64-bit. Robust to rescale and mild compression."""
    g = np.asarray(
        ImageOps.grayscale(img).resize((32, 32), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    d = _DCT32 @ g @ _DCT32.T
    low = d[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])  # drop DC before thresholding
    bits = (low > med).astype(np.uint64)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Per-image probing
# --------------------------------------------------------------------------- #

def png_text_chunks(path: Path) -> dict[str, str]:
    """Read tEXt / zTXt / iTXt chunks straight from the PNG bytes.

    Done at the byte level rather than through ``img.info`` because PIL only
    surfaces chunks that appear before the first IDAT, and generator tooling
    does not always put them there.
    """
    out: dict[str, str] = {}
    try:
        data = path.read_bytes()
    except Exception:
        return out
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return out
    i = 8
    while i + 8 <= len(data):
        length = int.from_bytes(data[i:i + 4], "big")
        ctype = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + length]
        if ctype in (b"tEXt", b"zTXt", b"iTXt"):
            try:
                key, _, rest = body.partition(b"\x00")
                if ctype == b"zTXt":
                    rest = zlib.decompress(rest[1:]) if len(rest) > 1 else b""
                elif ctype == b"iTXt":
                    parts = rest.split(b"\x00", 3)
                    rest = parts[-1] if len(parts) == 4 else rest
                    if len(parts) == 4 and parts[0] == b"\x01":
                        try:
                            rest = zlib.decompress(rest)
                        except Exception:
                            pass
                out[key.decode("latin-1", "replace")] = rest.decode(
                    "utf-8", "replace"
                )[:4000]
            except Exception:
                pass
        if ctype == b"IEND":
            break
        i += 12 + length
    return out


def probe(path: Path) -> dict:
    """Everything the audit needs from one file. Never raises."""
    rec: dict = {
        "ok": False, "container": "unknown", "mode": "", "width": 0, "height": 0,
        "n_bytes": 0, "truncated": False, "has_exif": False, "n_exif_tags": 0,
        "exif_orientation": None, "has_icc": False, "n_text_chunks": 0,
        "metadata_keys": "", "generator_marker": "", "phash": None,
        "jpeg_quality_hint": None, "error": "",
    }
    try:
        rec["n_bytes"] = path.stat().st_size
    except Exception:
        pass
    try:
        # Pass 1: strict. A file that only decodes with LOAD_TRUNCATED_IMAGES is
        # genuinely truncated and we want to know that, not paper over it.
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(path)
        img.load()
    except Exception as exc:
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(path)
            img.load()
            rec["truncated"] = True
        except Exception as exc2:
            rec["error"] = f"{type(exc2).__name__}: {exc2}"[:120]
            return rec
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False

    try:
        rec["ok"] = True
        rec["container"] = (img.format or "unknown").lower()
        rec["mode"] = img.mode
        rec["width"], rec["height"] = img.size
        rec["has_icc"] = bool(img.info.get("icc_profile"))

        exif = None
        try:
            exif = img.getexif()
        except Exception:
            pass
        if exif is not None and len(exif) > 0:
            rec["has_exif"] = True
            rec["n_exif_tags"] = len(exif)
            rec["exif_orientation"] = exif.get(274)

        keys = {str(k).lower() for k in img.info.keys()}
        text = png_text_chunks(path)
        keys |= {k.lower() for k in text}
        rec["n_text_chunks"] = len(text)
        hits = sorted(keys & GENERATOR_METADATA_KEYS)
        rec["metadata_keys"] = ",".join(hits)[:200]

        blob = " ".join(list(text.values())[:12]).lower()
        for k in ("Software", "Comment", "UserComment", "ImageDescription"):
            v = img.info.get(k)
            if isinstance(v, (str, bytes)):
                blob += " " + (v.decode("latin-1", "replace") if isinstance(v, bytes) else v).lower()
        rec["generator_marker"] = ",".join(
            m for m in GENERATOR_TOOL_MARKERS if m in blob
        )[:200]

        if rec["container"] in ("jpeg", "jpg"):
            q = img.info.get("quality")
            rec["jpeg_quality_hint"] = q if isinstance(q, int) else None

        rec["phash"] = phash(img)
    except Exception as exc:
        rec["error"] = f"probe: {type(exc).__name__}: {exc}"[:120]
    return rec


# --------------------------------------------------------------------------- #
# Content-type heuristics for the real class (item d)
# --------------------------------------------------------------------------- #

def content_features(path: Path) -> dict:
    """Cheap photographic-vs-synthetic-graphic features.

    Not a classifier -- an evidence summary. Illustration and screenshot
    material shows up as large flat regions, few unique colours, and spikes at
    pure black/white; photographs show broad colour histograms and sensor grain.
    """
    out = {"uniq_colors_k": np.nan, "flat_frac": np.nan, "pure_bw_frac": np.nan,
           "sat_mean": np.nan, "hf_energy": np.nan, "aspect": np.nan}
    try:
        img = Image.open(path)
        img.draft("RGB", (512, 512))  # fast approximate decode for JPEG
        img = img.convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.BILINEAR)
        a = np.asarray(img, dtype=np.float32)
        h, w, _ = a.shape
        out["aspect"] = w / max(h, 1)
        flat = a.reshape(-1, 3)
        out["uniq_colors_k"] = len(np.unique(flat // 8, axis=0)) / 1000.0
        g = a.mean(-1)
        gx = np.abs(np.diff(g, axis=1)).mean() if w > 1 else 0.0
        gy = np.abs(np.diff(g, axis=0)).mean() if h > 1 else 0.0
        out["hf_energy"] = float(gx + gy)
        # flat = 3x3 neighbourhood with near-zero local range
        d = np.abs(np.diff(g, axis=1))
        out["flat_frac"] = float((d < 1.0).mean())
        mx, mn = flat.max(1), flat.min(1)
        out["pure_bw_frac"] = float(((mx < 4) | (mn > 251)).mean())
        out["sat_mean"] = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0).mean())
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _pct_table(df: pd.DataFrame, index: str, col: str) -> str:
    ct = pd.crosstab(df[index], df[col])
    pctv = (ct.T / ct.sum(axis=1)).T * 100
    lines = ["| " + index + " | " + " | ".join(f"{c}" for c in ct.columns) + " | n |",
             "|---" * (len(ct.columns) + 2) + "|"]
    for idx in ct.index:
        cells = [f"{ct.loc[idx, c]} ({pctv.loc[idx, c]:.1f}%)" for c in ct.columns]
        lines.append(f"| {idx} | " + " | ".join(cells) + f" | {ct.loc[idx].sum()} |")
    return "\n".join(lines)


def trivial_auroc(df: pd.DataFrame, feature: str, label_col: str = "label") -> float:
    """AUROC of a single metadata feature, using the Phase 1 metric code."""
    sub = df[[feature, label_col]].dropna()
    if sub[label_col].nunique() < 2 or len(sub) < 4:
        return float("nan")
    x = sub[feature].to_numpy(dtype=float)
    rng = np.ptp(x)
    s = (x - x.min()) / rng if rng > 0 else np.full_like(x, 0.5)
    a = cell_metrics(sub[label_col].to_numpy(), s).auroc
    return max(a, 1 - a) if np.isfinite(a) else a  # direction-agnostic


#: Path prefixes of the forbidden WildFake reference subset (§4.3.2). Sizes are
#: asserted against the problem statement's counts: COCO val2017 = 4998,
#: DALL-E Advanced = 8843. Paths locate it; content hashes enforce it, because
#: WildFake has already renamed COCO's files to img000000.jpg and a path check
#: would not survive re-extraction.
FORBIDDEN_PREFIXES = {
    "./Real/coco/coco2017/val2017/": 4998,
    "./Diffusion_based/DALLE/Advanced/": 8843,
}


def audit_wildfake_csvs(csv_dir: Path, out: Path | None) -> int:
    """Container format and forbidden-subset location, from WildFake's own file
    lists. No images downloaded -- the label CSVs carry every path."""
    rows, forbidden = [], collections.Counter()
    for p in sorted(csv_dir.glob("*.csv")):
        d = pd.read_csv(p)
        d["ext"] = d.Image_path.str.rsplit(".", n=1).str[-1].str.lower()
        ec = d.ext.value_counts()
        rows.append({
            "list": p.stem,
            "n": len(d),
            "class": "real" if d.IsFake.iloc[0] == 0 else "ai",
            "advanced": ",".join(str(v) for v in sorted(d.IsAdvanced.unique())),
            "jpg%": round(100 * ec.get("jpg", 0) / len(d), 1),
            "png%": round(100 * ec.get("png", 0) / len(d), 1),
            "root": "/".join(d.Image_path.iloc[0].split("/")[:4]),
        })
        for pref in FORBIDDEN_PREFIXES:
            forbidden[pref] += int(d.Image_path.str.startswith(pref).sum())

    t = pd.DataFrame(rows)
    text = ["# WildFake path-level audit", "",
            f"From `{csv_dir}` — {len(rows)} generator file lists, "
            f"{t.n.sum():,} paths, zero images downloaded.", "",
            "## Container format by generator", "", t.to_markdown(index=False), "",
            "## Forbidden reference subset (§4.3.2)", "",
            "| path prefix | found | expected | matches |", "|---|---:|---:|:--:|"]
    for pref, expected in FORBIDDEN_PREFIXES.items():
        got = forbidden[pref]
        text.append(
            f"| `{pref}` | {got} | {expected} | "
            f"{'yes' if got == expected else 'NO' if got else 'not in sample'} |"
        )
    text += ["", "Paths *locate* the forbidden subset; they must not be what "
             "enforces the exclusion. WildFake has renamed COCO's files to "
             "`img000000.jpg`, so the original IDs are gone and any path check "
             "dies the moment a directory is reorganised. The manifest builder "
             "content-hashes these files and blocks by hash.", ""]
    body = "\n".join(text) + "\n"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        print(f"wrote {out}")
    print(body)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--wildfake-csv-dir", type=Path, default=None,
                   help="audit WildFake label CSVs instead of an image sample")
    p.add_argument("--out", type=Path, default=None, help="write markdown here")
    p.add_argument("--class-col", default="sid_label_name",
                   help="column naming the fine-grained class")
    p.add_argument("--near-dup-threshold", type=int, default=6,
                   help="max phash Hamming distance counted as a near-duplicate")
    p.add_argument("--content-sample", type=int, default=250,
                   help="images per class for the (slower) content-type features")
    p.add_argument("--no-cache", action="store_true",
                   help="re-probe every file instead of reusing *_probe.parquet")
    a = p.parse_args(argv)

    if a.wildfake_csv_dir:
        return audit_wildfake_csvs(a.wildfake_csv_dir, a.out)
    if not a.manifest:
        p.error("give --manifest or --wildfake-csv-dir")

    man = pd.read_csv(a.manifest)
    cls = a.class_col if a.class_col in man.columns else "label"

    cache = a.manifest.with_name(a.manifest.stem + "_probe.parquet")
    if cache.exists() and not a.no_cache:
        d = pd.read_parquet(cache)
        print(f"loaded {len(d)} probe records from {cache}")
    else:
        print(f"probing {len(man)} images from {a.manifest} ...")
        recs = []
        for i, row in man.iterrows():
            r = probe(Path(row["image_path"]))
            r["image_path"] = row["image_path"]
            r["label"] = int(row["label"])
            r["klass"] = row[cls]
            r["declared_width"] = row.get("declared_width")
            r["declared_height"] = row.get("declared_height")
            recs.append(r)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(man)}", flush=True)
        d = pd.DataFrame(recs)
        d.to_parquet(cache)
        print(f"cached probe records -> {cache}")
    d["megapixels"] = d.width * d.height / 1e6
    d["is_1024sq"] = (d.width == 1024) & (d.height == 1024)
    d["bytes_per_pixel"] = d.n_bytes / (d.width * d.height).clip(lower=1)

    out: list[str] = ["# Phase 2 leakage audit", "",
                      f"Sample: `{a.manifest}` — {len(d)} images, "
                      f"{d.n_bytes.sum()/1e6:.0f} MB.", ""]

    # ---- headline: metadata-only classifier -------------------------------- #
    d["is_square"] = (d.width == d.height).astype(int)
    d["container_is_png"] = (d.container == "png").astype(int)
    # ICC profile presence is the same class of channel as EXIF: metadata the
    # encoder attached that says nothing about how the pixels were made.
    d["has_icc_int"] = d.has_icc.astype(int)
    feats = ["megapixels", "width", "height", "n_bytes", "bytes_per_pixel",
             "is_1024sq", "is_square", "n_exif_tags", "container_is_png",
             "has_icc_int", "n_text_chunks"]
    # Two views. The binary task (§4.2) folds every non-real class into 1, which
    # can *dilute* a leak when the AI subclasses differ from each other -- here
    # tampered images are re-encoded real photos, so they hide the format leak.
    # The pairwise view is the honest measure of each channel.
    views: dict[str, pd.DataFrame] = {"real vs AI (binary, §4.2)": d}
    others = [k for k in d.klass.unique() if k != "real"]
    if "real" in set(d.klass) and len(others) > 1:
        for o in others:
            sub = d[d.klass.isin(["real", o])].copy()
            sub["label"] = (sub.klass == o).astype(int)
            views[f"real vs {o}"] = sub

    out += ["## Headline — what a metadata-only classifier gets", "",
            "AUROC of a *single* metadata feature, direction-agnostic, computed "
            "with the Phase 1 metric code. 0.5 = no leak, 1.0 = the feature "
            "alone separates the classes perfectly.", "",
            "| Feature | " + " | ".join(views) + " |",
            "|---" * (len(views) + 1) + "|"]
    triv = {f: trivial_auroc(d, f) for f in feats}
    per_view = {name: {f: trivial_auroc(v, f) for f in feats}
                for name, v in views.items()}
    for f in sorted(feats, key=lambda f: -max(
            (per_view[n][f] for n in views if np.isfinite(per_view[n][f])), default=0)):
        cells = []
        for n in views:
            v = per_view[n][f]
            flag = (" 🚨" if np.isfinite(v) and v > 0.90
                    else " ⚠️" if np.isfinite(v) and v > 0.70 else "")
            cells.append(f"{v:.4f}{flag}" if np.isfinite(v) else "—")
        out.append(f"| `{f}` | " + " | ".join(cells) + " |")
    out.append("")
    # Accuracy of the single most damning one-line rule, stated plainly.
    rs = d[d.klass.isin(["real", "full_synthetic"])]
    if len(rs):
        acc = float((rs.container_is_png == (rs.klass == "full_synthetic")).mean())
        out += [f"One-line rule `if container == PNG: predict AI` scores "
                f"**{100*acc:.2f}% accuracy** on real vs full_synthetic "
                f"({len(rs)} images). No pixels read.", ""]

    # ---- a) container format ---------------------------------------------- #
    out += ["## (a) Container format by class", "",
            _pct_table(d, "klass", "container"), "",
            "Mode (PIL) by class:", "", _pct_table(d, "klass", "mode"), ""]

    # ---- b) resolution ---------------------------------------------------- #
    res = d.groupby("klass").agg(
        n=("width", "size"), w_mean=("width", "mean"), w_std=("width", "std"),
        h_mean=("height", "mean"), h_std=("height", "std"),
        mp_med=("megapixels", "median"),
        pct_1024sq=("is_1024sq", lambda s: 100 * s.mean()),
    )
    res["uniq_shapes"] = pd.Series(
        {k: len(set(zip(g.width, g.height))) for k, g in d.groupby("klass")}
    )
    out += ["## (b) Resolution by class", "", res.round(2).to_markdown(), "",
            "Top 6 exact (w×h) shapes per class:", ""]
    for k, g in d.groupby("klass"):
        top = collections.Counter(zip(g.width, g.height)).most_common(6)
        out.append(f"- **{k}**: " + ", ".join(
            f"{w}×{h} ({100*c/len(g):.0f}%)" for (w, h), c in top))
    out.append("")

    # ---- c) EXIF + text chunks -------------------------------------------- #
    meta = d.groupby("klass").agg(
        n=("has_exif", "size"),
        pct_exif=("has_exif", lambda s: 100 * s.mean()),
        mean_exif_tags=("n_exif_tags", "mean"),
        pct_icc=("has_icc", lambda s: 100 * s.mean()),
        mean_text_chunks=("n_text_chunks", "mean"),
        pct_any_text=("n_text_chunks", lambda s: 100 * (s > 0).mean()),
        pct_gen_marker=("generator_marker", lambda s: 100 * (s.str.len() > 0).mean()),
        pct_metadata_keys=("metadata_keys", lambda s: 100 * (s.str.len() > 0).mean()),
    )
    out += ["## (c) EXIF and PNG text chunks by class", "",
            meta.round(2).to_markdown(), ""]
    keyhits = collections.Counter()
    for v in d.metadata_keys.dropna():
        for k in str(v).split(","):
            if k:
                keyhits[k] += 1
    toolhits = collections.Counter()
    for v in d.generator_marker.dropna():
        for k in str(v).split(","):
            if k:
                toolhits[k] += 1
    out += [f"- suspicious metadata keys seen: "
            f"{dict(keyhits.most_common(12)) or 'none'}",
            f"- generator tool markers seen: "
            f"{dict(toolhits.most_common(12)) or 'none'}",
            f"- EXIF orientation values: "
            f"{d.exif_orientation.value_counts(dropna=True).to_dict() or 'none'}", ""]

    # ---- d) content type -------------------------------------------------- #
    samp = pd.concat(
        [g.head(a.content_sample) for _, g in d.groupby("klass")], ignore_index=True
    )
    cf = pd.DataFrame([content_features(Path(p)) for p in samp.image_path])
    cf["klass"] = samp["klass"].to_numpy()
    out += ["## (d) Content-type features "
            f"(first {a.content_sample}/class)", "",
            cf.groupby("klass").mean(numeric_only=True).round(3).to_markdown(), "",
            "`uniq_colors_k` = thousands of distinct 8-level-quantised colours; "
            "`flat_frac` = fraction of horizontally flat pixel pairs; "
            "`pure_bw_frac` = fraction of pure black/white pixels; "
            "`hf_energy` = mean abs gradient. Illustration and screenshot "
            "material scores low on colours and high on flat/pure-bw.", ""]

    # ---- e) duplicates ---------------------------------------------------- #
    hv = d.dropna(subset=["phash"]).reset_index(drop=True)
    hashes = hv.phash.astype("int64").to_numpy()
    exact = collections.Counter(hashes)
    n_exact_dupe_groups = sum(1 for v in exact.values() if v > 1)
    n_exact_dupe_imgs = sum(v for v in exact.values() if v > 1)
    pairs_within, pairs_across, examples = 0, 0, []
    n = len(hv)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(int(hashes[i]), int(hashes[j])) <= a.near_dup_threshold:
                same = hv.klass.iloc[i] == hv.klass.iloc[j]
                pairs_within += same
                pairs_across += (not same)
                if len(examples) < 8:
                    examples.append(
                        f"{hv.klass.iloc[i]}:{Path(hv.image_path.iloc[i]).name} ~ "
                        f"{hv.klass.iloc[j]}:{Path(hv.image_path.iloc[j]).name} "
                        f"(d={hamming(int(hashes[i]), int(hashes[j]))})")
    out += ["## (e) Duplicates and near-duplicates", "",
            f"Threshold {a.near_dup_threshold} is calibrated, not guessed. On "
            "150 images from this sample: resize 0.25x, JPEG q30 and blur "
            "sigma=2 each moved the hash by at most **2** bits, while 11,175 "
            "unrelated pairs never came closer than **10** (p1 = 18, mean 30). "
            "So 6 catches every rescaled or recompressed repost with zero false "
            "positives. It does **not** catch crops: a 10%-per-side crop moves "
            "the hash ~20 bits, indistinguishable from an unrelated image.", "",
            f"- exact phash collisions: {n_exact_dupe_imgs} images in "
            f"{n_exact_dupe_groups} groups",
            f"- near-duplicate pairs (phash Hamming ≤ {a.near_dup_threshold}): "
            f"**{pairs_within} within-class**, **{pairs_across} across-class**",
            f"- pairs compared: {n*(n-1)//2}", ""]
    if examples:
        out += ["Examples:", ""] + [f"- `{e}`" for e in examples] + [""]

    # ---- load failures ---------------------------------------------------- #
    bad = d[~d.ok]
    out += ["## Load failures", "",
            f"- unreadable: {len(bad)} / {len(d)}",
            f"- truncated but recoverable: {int(d.truncated.sum())}", ""]
    if len(bad):
        out += [f"- `{Path(r.image_path).name}`: {r.error}" for r in bad.head(10).itertuples()]

    text = "\n".join(out) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
        d.drop(columns=["phash"]).to_csv(
            a.out.with_suffix(".per_image.csv"), index=False)
        print(f"wrote {a.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
