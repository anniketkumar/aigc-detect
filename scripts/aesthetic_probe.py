"""How much of the baseline's headline number is "detects artistic-looking images"?

    python -m scripts.aesthetic_probe --manifests data/manifests --out results/baseline

A logistic regression on **two** hand-made pixel statistics and nothing else:

    hf_energy   mean absolute gradient -- sensor grain and fine texture
    flat_frac   fraction of near-flat horizontal pixel pairs -- posterized,
                illustration-like regions

Both are the Phase 2 definitions from ``scripts.audit_leakage.content_features``
-- same decode path, same formulas -- rather than a second invented measure.
They are recomputed here instead of calling that function because it also
computes ``uniq_colors_k`` via ``np.unique(..., axis=0)`` over 262k rows, about
a second per image: fine for a 300-image audit sample, five hours for 18,200.
``--verify-against-audit`` checks the two paths agree exactly on a sample, so
the equivalence is tested rather than claimed.

Why it exists. The first Phase 3 grid showed unseen MidJourney (0.9894 AUROC,
0.843 TPR@1%) beating every generator the head was trained on, while unseen
FLUX.1-dev sat at 0.433 TPR@1%. That is the signature of a *style* detector --
stylized images easy, photorealistic ones hard -- not of a generator-artifact
detector. If two crude texture features reproduce the same per-generator
ordering, then a large part of the CLIP number is aesthetics rather than
anything forensic, and the ranking has to be read that way.

Trained on the train split and evaluated on test, the same splits the real head
uses, so the AUROC is directly comparable to `results/baseline/summary.json`.
Where `results/baseline/scores.csv` exists, the report puts the CLIP baseline's
per-generator AUROC beside the probe's and reports Spearman agreement between
the two orderings.

Writes into ``--out``:

    aesthetic_probe.md     the report
    aesthetic_probe.json   the same numbers, machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

FEATURES = ["hf_energy", "flat_frac"]


def two_features(path: Path) -> dict:
    """``hf_energy`` and ``flat_frac`` exactly as content_features computes them.

    Mirrors that function's decode path line for line -- draft-decode, RGB,
    thumbnail to 512 bilinear, mean over channels -- and omits only the
    measures this probe does not use. Failures return NaN, like the original.
    """
    from PIL import Image

    out = {"hf_energy": np.nan, "flat_frac": np.nan}
    try:
        img = Image.open(path)
        img.draft("RGB", (512, 512))  # fast approximate decode for JPEG
        img = img.convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.BILINEAR)
        a = np.asarray(img, dtype=np.float32)
        h, w, _ = a.shape
        g = a.mean(-1)
        gx = np.abs(np.diff(g, axis=1)).mean() if w > 1 else 0.0
        gy = np.abs(np.diff(g, axis=0)).mean() if h > 1 else 0.0
        out["hf_energy"] = float(gx + gy)
        d = np.abs(np.diff(g, axis=1))
        out["flat_frac"] = float((d < 1.0).mean())
    except Exception:
        pass
    return out


def verify_against_audit(paths, progress: bool = True) -> int:
    """Assert two_features == content_features on `paths`. Returns count checked."""
    from scripts.audit_leakage import content_features

    n = 0
    for p in tqdm(list(paths), desc="verify", unit="img", disable=not progress):
        mine, theirs = two_features(Path(p)), content_features(Path(p))
        for k in FEATURES:
            a, b = mine[k], theirs[k]
            if not (np.isnan(a) and np.isnan(b)):
                assert a == b, f"{p}: {k} {a!r} != content_features {b!r}"
        n += 1
    print(f"verified {n} image(s): both feature paths agree exactly")
    return n


def _measure(df: pd.DataFrame, progress: bool = True,
             workers: int = 4) -> pd.DataFrame:
    """Features over every row; failures come back NaN and are dropped.

    Threaded: JPEG decode and the numpy reductions both release the GIL, and
    18,200 images single-threaded is ~10 minutes of a Colab session. ``map``
    preserves input order, so the result does not depend on `workers`.
    """
    paths = [Path(p) for p in df["image_path"]]
    bar = tqdm(total=len(paths), desc="measure", unit="img", disable=not progress)

    def measure_one(p):
        try:
            return two_features(p)
        finally:
            bar.update(1)

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(measure_one, paths))
    else:
        rows = [measure_one(p) for p in paths]
    bar.close()
    feats = pd.DataFrame(rows, index=df.index)[FEATURES]
    out = pd.concat([df, feats], axis=1)
    bad = out[FEATURES].isna().any(axis=1)
    if bad.any():
        print(f"[warn] {int(bad.sum())} image(s) failed to measure; dropping them")
    return out[~bad].reset_index(drop=True)


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def _tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.01) -> float:
    """Threshold at the (1-fpr) quantile of the negatives, then TPR above it."""
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0:
        return float("nan")
    k = max(int(np.ceil(fpr * len(neg))) - 1, 0)
    return float((s[y == 1] > neg[k]).mean())


def _clip_per_generator(scores_csv: Path, test: pd.DataFrame) -> dict[str, float]:
    """Per-generator clean AUROC for the CLIP baseline, from its saved scores."""
    sc = pd.read_csv(scores_csv)
    sc = sc[sc["cell"] == "clean"]
    gen = test.set_index("image_path")["generator"]
    sc = sc.join(gen, on="image_path")
    if sc["generator"].isna().all():
        return {}
    real = sc[sc["label"] == 0]
    out = {}
    for g, grp in sc[sc["label"] == 1].groupby("generator"):
        y = np.r_[np.zeros(len(real)), np.ones(len(grp))]
        out[str(g)] = _auroc(y, np.r_[real["score"].values, grp["score"].values])
    return out


def run(manifests: Path, out: Path, limit: int | None = None,
        progress: bool = True, seed: int = 0, verify: int = 0,
        workers: int = 4) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = pd.read_csv(manifests / "train.csv")
    test = pd.read_csv(manifests / "test.csv")
    if limit is not None:
        train = train.iloc[:limit].reset_index(drop=True)
        test = test.iloc[:limit].reset_index(drop=True)

    if verify:
        verify_against_audit(train["image_path"].iloc[:verify], progress)

    t0 = time.time()
    print(f"measuring {len(train)} train + {len(test)} test images "
          f"on {', '.join(FEATURES)} ...", flush=True)
    train = _measure(train, progress, workers)
    test = _measure(test, progress, workers)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, random_state=seed))
    clf.fit(train[FEATURES].values, train["label"].values)
    test_score = clf.predict_proba(test[FEATURES].values)[:, 1]

    y = test["label"].values
    overall = {"auroc": _auroc(y, test_score), "tpr_at_fpr1": _tpr_at_fpr(y, test_score)}

    real = test[test["label"] == 0]
    per_gen = {}
    for g, grp in test[test["label"] == 1].groupby("generator"):
        yy = np.r_[np.zeros(len(real)), np.ones(len(grp))]
        ss = np.r_[test_score[real.index], test_score[grp.index]]
        per_gen[str(g)] = {"n": int(len(grp)), "auroc": _auroc(yy, ss),
                           "tpr_at_fpr1": _tpr_at_fpr(yy, ss)}

    # side-by-side with the real model, when its scores are on disk
    clip_per_gen = {}
    scores_csv = out / "scores.csv"
    if scores_csv.exists():
        clip_per_gen = _clip_per_generator(scores_csv, test)

    spearman = None
    shared = sorted(set(per_gen) & set(clip_per_gen))
    if len(shared) >= 3:
        from scipy.stats import spearmanr  # scipy rides along with scikit-learn

        rho, p = spearmanr([per_gen[g]["auroc"] for g in shared],
                           [clip_per_gen[g] for g in shared])
        spearman = {"rho": float(rho), "p": float(p), "n_generators": len(shared)}

    coef = dict(zip(FEATURES, clf[-1].coef_[0].tolist()))
    result = {
        "features": FEATURES,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "coef_standardized": coef,
        "overall": overall,
        "per_generator": per_gen,
        "clip_per_generator_clean_auroc": clip_per_gen,
        "spearman_vs_clip": spearman,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "aesthetic_probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (out / "aesthetic_probe.md").write_text(
        _report(result, manifests), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("overall", "spearman_vs_clip", "elapsed_s")}, indent=2))
    print(f"-> {out / 'aesthetic_probe.md'}")
    return result


def _report(r: dict, manifests: Path) -> str:
    L = ["# Aesthetic probe — two pixel features, no CLIP", "",
         f"Logistic regression on `{'`, `'.join(FEATURES)}` alone, fit on "
         f"`{manifests.as_posix()}/train.csv` ({r['n_train']} images) and "
         f"evaluated on `{manifests.as_posix()}/test.csv` ({r['n_test']} "
         f"images) — the same splits as "
         "the CLIP baseline, so the AUROC below is directly comparable.", "",
         "Reproduce: `python -m scripts.aesthetic_probe`", "",
         "## Headline", "",
         "| Metric | Value |", "|---|---:|",
         f"| Test AUROC | {r['overall']['auroc']:.4f} |",
         f"| Test TPR@FPR=1% | {r['overall']['tpr_at_fpr1']:.4f} |", ""]

    L += ["Standardized coefficients: "
          + ", ".join(f"`{k}` {v:+.3f}" for k, v in r["coef_standardized"].items())
          + ".", ""]

    clip = r["clip_per_generator_clean_auroc"]
    L += ["## Per generator", "",
          "Each generator against all test reals. `clean` AUROC in both columns."
          if clip else "Each generator against all test reals.", ""]
    header = "| generator | n | probe AUROC | probe TPR@1% |"
    rule = "|---|---:|---:|---:|"
    if clip:
        header += " CLIP AUROC | gap |"
        rule += "---:|---:|"
    L += [header, rule]
    for g, d in sorted(r["per_generator"].items(),
                       key=lambda kv: -kv[1]["auroc"]):
        row = f"| `{g}` | {d['n']} | {d['auroc']:.4f} | {d['tpr_at_fpr1']:.4f} |"
        if clip:
            c = clip.get(g)
            row += (f" {c:.4f} | {c - d['auroc']:+.4f} |" if c is not None
                    else " — | — |")
        L.append(row)
    L.append("")

    sp = r["spearman_vs_clip"]
    if sp:
        L += ["## Do the two orderings agree?", "",
              f"Spearman rho **{sp['rho']:+.3f}** (p = {sp['p']:.3f}) over "
              f"{sp['n_generators']} generators, between this probe's "
              "per-generator AUROC and the CLIP baseline's.", "",
              "A high positive rho means the CLIP head finds the same "
              "generators easy and hard that two texture statistics do — i.e. "
              "much of its ranking is style, not forensics. A rho near zero "
              "means CLIP's difficulty ordering is its own.", ""]

    L += ["Generated by `python -m scripts.aesthetic_probe`. Do not hand-edit.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifests", type=Path, default=Path("data/manifests"))
    ap.add_argument("--out", type=Path, default=Path("results/baseline"))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows of each split; for smoke-testing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-against-audit", type=int, default=0, metavar="N",
                    help="first check two_features matches "
                         "audit_leakage.content_features on N images")
    ap.add_argument("--workers", type=int, default=4,
                    help="decode threads; order-independent, 1 to disable")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(a.manifests, a.out, limit=a.limit, progress=not a.quiet, seed=a.seed,
        verify=a.verify_against_audit, workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
