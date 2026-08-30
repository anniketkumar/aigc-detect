"""TPR@FPR robustness gap -- the operating-point companion to AUROC (HANDOFF.md).

    python -m scripts.tpr_gap_analysis \\
        --run results/baseline/scores.csv \\
        --run results/_prefix_gelu_run/scores.csv \\
        --out results/tpr_analysis

Why this exists: on the post-fix baseline, the AUROC robustness_gap is 0.0099
against an auroc_null_sd of 0.0108 -- statistically indistinguishable from
zero. AUROC is threshold-free and rank-based; a degradation that reshuffles
scores within the real-class tail without swapping ranks globally can leave
AUROC nearly untouched while collapsing the only operating point anyone would
actually deploy at (a fixed false-positive budget). TPR@FPR=1%/5% measures
that directly, at the cost of being much noisier: with 810 reals, the 1%
threshold rests on ~8 images, so every number here needs a bootstrap CI
alongside it, not instead of it.

Bootstrap: image-level, stratified by class (real images and fake images each
resampled with replacement independently), one draw per replicate reused
across every cell of that replicate -- a real image's score at jpeg_30 is
correlated with its score at clean, and resampling cells independently would
throw that away and understate the true CI width. B replicates, seed fixed for
reproducibility.

Per-generator breakdown uses the shared real pool against each AI generator's
own scores (same convention as scripts/aesthetic_probe.py's per-generator
table), with the generator name read directly off ``image_path`` (the corpus
layout is ``data/corpus/images/{generator}/xx/hash.jpg``) since scores.csv
does not itself carry a generator column.

Writes into ``--out``:

    report.md           full per-cell and per-generator tables, both runs
    tpr_grid_<label>.csv   per-cell numbers for each ``--run``, machine-readable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src import transforms as T

FPR_TARGETS = [0.01, 0.05]
HELD_OUT = {"FLUX.1-dev", "Gemini-nano-banana", "MidJourney"}

CELLS = T.build_cells()
CELL_FAMILY = {c.name: (c.family if c.kind in ("clean", "single") else "composed") for c in CELLS}
CELL_ORDER = [c.name for c in CELLS]


def tpr_both(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """TPR@FPR=1% and TPR@FPR=5% from one shared roc_curve call.

    Same conservative (non-interpolating) reading as src.metrics.tpr_at_fpr:
    the highest TPR reachable at an FPR at or below the target.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    out = []
    for target in FPR_TARGETS:
        ok = fpr <= target + 1e-12
        out.append(float(tpr[ok].max()) if ok.any() else float("nan"))
    return tuple(out)


def family_gap(per_cell: dict[str, tuple[float, float]], metric_idx: int) -> tuple[float, dict[str, float]]:
    """robustness_gap = clean - mean(family means); same shape as the AUROC gap."""
    clean = per_cell["clean"][metric_idx]
    fam: dict[str, list[float]] = {}
    for cell, vals in per_cell.items():
        if cell == "clean":
            continue
        fam.setdefault(CELL_FAMILY[cell], []).append(vals[metric_idx])
    fam_means = {k: float(np.mean(v)) for k, v in fam.items()}
    grand = float(np.mean(list(fam_means.values())))
    return clean - grand, fam_means


def load_wide(scores_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(scores_path)
    wide = df.pivot(index="image_path", columns="cell", values="score")[CELL_ORDER]
    labels = df.drop_duplicates("image_path").set_index("image_path")["label"].reindex(wide.index).to_numpy()
    generator = wide.index.to_series().str.split("/").str[3].to_numpy()
    return wide.to_numpy(), labels, generator


def bootstrap_gap(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator, B: int) -> dict:
    real_idx, fake_idx = np.where(labels == 0)[0], np.where(labels == 1)[0]
    n_real, n_fake, n_cells = len(real_idx), len(fake_idx), scores.shape[1]

    cell_tpr1 = np.empty((B, n_cells))
    cell_tpr5 = np.empty((B, n_cells))
    for b in range(B):
        idx = np.concatenate([real_idx[rng.integers(0, n_real, n_real)],
                               fake_idx[rng.integers(0, n_fake, n_fake)]])
        y = np.concatenate([np.zeros(n_real), np.ones(n_fake)])
        for ci in range(n_cells):
            t1, t5 = tpr_both(y, scores[idx, ci])
            cell_tpr1[b, ci], cell_tpr5[b, ci] = t1, t5

    clean_i = CELL_ORDER.index("clean")
    trans_i = [i for i, c in enumerate(CELL_ORDER) if c != "clean"]
    fam_of = [CELL_FAMILY[CELL_ORDER[i]] for i in trans_i]
    families = sorted(set(fam_of))

    def gap_dist(cell_tpr: np.ndarray) -> np.ndarray:
        fam_means = np.stack(
            [cell_tpr[:, [trans_i[j] for j, f in enumerate(fam_of) if f == fam]].mean(axis=1)
             for fam in families], axis=1)
        return cell_tpr[:, clean_i] - fam_means.mean(axis=1)

    return {"cell_tpr1": cell_tpr1, "cell_tpr5": cell_tpr5,
            "gap1_dist": gap_dist(cell_tpr1), "gap5_dist": gap_dist(cell_tpr5)}


def ci95(dist: np.ndarray) -> tuple[float, float, float]:
    return float(np.median(dist)), float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def paired_bootstrap_diff(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
    rng: np.random.Generator, B: int,
) -> dict[str, np.ndarray]:
    """Paired bootstrap of the *difference* in family-balanced TPR gap between
    two runs scored on the same images.

    Overlapping marginal CIs on two separate estimates do not test whether
    their difference is nonzero -- each marginal bootstrap in
    :func:`bootstrap_gap` resamples independently, throwing away the
    correlation between the two runs' scores on the same image. Here one
    resample of image indices per replicate is applied to *both* score
    matrices, so replicate ``b`` compares the two runs on the same simulated
    sample and the resulting ``diff`` distribution is the thing to read a CI
    off of, not the two marginals side by side. Valid only when ``scores_a``
    and ``scores_b`` index the same images in the same row order (assert this
    upstream via matching ``labels``/generator arrays from :func:`load_wide`).
    """
    real_idx, fake_idx = np.where(labels == 0)[0], np.where(labels == 1)[0]
    n_real, n_fake, n_cells = len(real_idx), len(fake_idx), scores_a.shape[1]

    diff1, diff5 = np.empty(B), np.empty(B)
    for b in range(B):
        idx = np.concatenate([real_idx[rng.integers(0, n_real, n_real)],
                               fake_idx[rng.integers(0, n_fake, n_fake)]])
        y = np.concatenate([np.zeros(n_real), np.ones(n_fake)])
        pc_a = {CELL_ORDER[i]: tpr_both(y, scores_a[idx, i]) for i in range(n_cells)}
        pc_b = {CELL_ORDER[i]: tpr_both(y, scores_b[idx, i]) for i in range(n_cells)}
        g1a, _ = family_gap(pc_a, 0)
        g5a, _ = family_gap(pc_a, 1)
        g1b, _ = family_gap(pc_b, 0)
        g5b, _ = family_gap(pc_b, 1)
        diff1[b] = g1b - g1a
        diff5[b] = g5b - g5a
    return {"diff1": diff1, "diff5": diff5}


def paired_spread_bootstrap(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray, generator: np.ndarray,
    rng: np.random.Generator, B: int,
) -> dict[str, np.ndarray]:
    """Paired bootstrap of the *difference* in per-generator clean-TPR spread
    (max-min across generators) between two runs scored on the same images.

    Same rationale as :func:`paired_bootstrap_diff`, applied to the secondary
    ablation target (NOTES.md Sec"Phase 4") instead of the primary
    family-balanced gap: the per-generator spread is what the aggregate gap
    can hide (one generator improving while another, e.g. FLUX.1-dev, stays
    the floor), so it needs its own paired significance test rather than
    reading the two runs' marginal per-generator CIs side by side.

    Each replicate draws one shared real-image resample (reused across
    generators, matching the shared-real-pool convention in
    :func:`analyze_run`) and, independently per generator, one resample of
    that generator's fake images -- the same index draws applied to both
    ``scores_a`` and ``scores_b`` so replicate ``b`` compares the two runs'
    spreads on the same simulated sample.
    """
    real_idx = np.where(labels == 0)[0]
    n_real = len(real_idx)
    generators = sorted(set(generator[labels == 1]))
    fake_idx_by_gen = {g: np.where((labels == 1) & (generator == g))[0] for g in generators}
    clean_i = CELL_ORDER.index("clean")

    diff1, diff5 = np.empty(B), np.empty(B)
    for b in range(B):
        real_samp = real_idx[rng.integers(0, n_real, n_real)]
        t1_a, t5_a, t1_b, t5_b = {}, {}, {}, {}
        for g in generators:
            fidx = fake_idx_by_gen[g]
            n_g = len(fidx)
            fake_samp = fidx[rng.integers(0, n_g, n_g)]
            idx = np.concatenate([real_samp, fake_samp])
            y = np.concatenate([np.zeros(n_real), np.ones(n_g)])
            t1_a[g], t5_a[g] = tpr_both(y, scores_a[idx, clean_i])
            t1_b[g], t5_b[g] = tpr_both(y, scores_b[idx, clean_i])
        spread1_a = max(t1_a.values()) - min(t1_a.values())
        spread1_b = max(t1_b.values()) - min(t1_b.values())
        spread5_a = max(t5_a.values()) - min(t5_a.values())
        spread5_b = max(t5_b.values()) - min(t5_b.values())
        diff1[b] = spread1_b - spread1_a
        diff5[b] = spread5_b - spread5_a
    return {"diff1": diff1, "diff5": diff5}


def per_generator_clean(scores: np.ndarray, labels: np.ndarray, generator: np.ndarray) -> dict[str, tuple[float, float]]:
    """Point-estimate clean TPR@1%/TPR@5% per generator, shared real pool."""
    real_mask = labels == 0
    clean_i = CELL_ORDER.index("clean")
    out = {}
    for gen in sorted(set(generator[labels == 1])):
        keep = real_mask | ((labels == 1) & (generator == gen))
        out[gen] = tpr_both(labels[keep].astype(float), scores[keep, clean_i])
    return out


def paired_section(
    label_a: str, path_a: Path, label_b: str, path_b: Path, B: int, seed: int,
) -> str:
    """``label_b`` minus ``label_a`` (e.g. aug minus baseline): negative means
    ``label_b``'s gap/spread is smaller, i.e. more robust."""
    scores_a, labels_a, gen_a = load_wide(path_a)
    scores_b, labels_b, gen_b = load_wide(path_b)
    if not (np.array_equal(labels_a, labels_b) and np.array_equal(gen_a, gen_b)):
        raise ValueError(
            f"{path_a} and {path_b} do not score the same images in the same "
            "order -- paired bootstrap requires identical rows to pair on."
        )

    per_cell_a = {c: tpr_both(labels_a.astype(float), scores_a[:, i]) for i, c in enumerate(CELL_ORDER)}
    per_cell_b = {c: tpr_both(labels_b.astype(float), scores_b[:, i]) for i, c in enumerate(CELL_ORDER)}
    gap1_a, _ = family_gap(per_cell_a, 0)
    gap5_a, _ = family_gap(per_cell_a, 1)
    gap1_b, _ = family_gap(per_cell_b, 0)
    gap5_b, _ = family_gap(per_cell_b, 1)

    dist = paired_bootstrap_diff(scores_a, scores_b, labels_a, np.random.default_rng(seed), B)
    med1, lo1, hi1 = ci95(dist["diff1"])
    med5, lo5, hi5 = ci95(dist["diff5"])
    sig1 = "excludes zero" if lo1 * hi1 > 0 else "includes zero"
    sig5 = "excludes zero" if lo5 * hi5 > 0 else "includes zero"

    clean_a, clean_b = per_generator_clean(scores_a, labels_a, gen_a), per_generator_clean(scores_b, labels_b, gen_b)
    spread1_a = max(v[0] for v in clean_a.values()) - min(v[0] for v in clean_a.values())
    spread1_b = max(v[0] for v in clean_b.values()) - min(v[0] for v in clean_b.values())
    spread5_a = max(v[1] for v in clean_a.values()) - min(v[1] for v in clean_a.values())
    spread5_b = max(v[1] for v in clean_b.values()) - min(v[1] for v in clean_b.values())
    sdist = paired_spread_bootstrap(scores_a, scores_b, labels_a, gen_a, np.random.default_rng(seed), B)
    smed1, slo1, shi1 = ci95(sdist["diff1"])
    smed5, slo5, shi5 = ci95(sdist["diff5"])
    ssig1 = "excludes zero" if slo1 * shi1 > 0 else "includes zero"
    ssig5 = "excludes zero" if slo5 * shi5 > 0 else "includes zero"

    return (
        f"## Paired bootstrap: {label_b} vs {label_a}\n\n"
        f"Same {len(labels_a)} images scored by both runs; one resampled image set per "
        f"replicate applied to both, B={B}, seed={seed}. `diff = {label_b} - {label_a}`, "
        "negative means smaller (more robust) under "
        f"`{label_b}`.\n\n"
        "**Family-balanced gap (primary target):**\n\n"
        "| Metric | " + f"{label_a} gap" + " | " + f"{label_b} gap" + " | point diff | "
        "bootstrap median diff | 95% CI | 95% CI |\n"
        "|---|---:|---:|---:|---:|---|---|\n"
        f"| TPR@1% gap | {gap1_a:.4f} | {gap1_b:.4f} | {gap1_b - gap1_a:+.4f} | "
        f"{med1:+.4f} | [{lo1:+.4f}, {hi1:+.4f}] | {sig1} |\n"
        f"| TPR@5% gap | {gap5_a:.4f} | {gap5_b:.4f} | {gap5_b - gap5_a:+.4f} | "
        f"{med5:+.4f} | [{lo5:+.4f}, {hi5:+.4f}] | {sig5} |\n\n"
        "**Per-generator clean-TPR spread, max-min (secondary target):**\n\n"
        "| Metric | " + f"{label_a} spread" + " | " + f"{label_b} spread" + " | point diff | "
        "bootstrap median diff | 95% CI | 95% CI |\n"
        "|---|---:|---:|---:|---:|---|---|\n"
        f"| clean TPR@1% spread | {spread1_a:.4f} | {spread1_b:.4f} | {spread1_b - spread1_a:+.4f} | "
        f"{smed1:+.4f} | [{slo1:+.4f}, {shi1:+.4f}] | {ssig1} |\n"
        f"| clean TPR@5% spread | {spread5_a:.4f} | {spread5_b:.4f} | {spread5_b - spread5_a:+.4f} | "
        f"{smed5:+.4f} | [{slo5:+.4f}, {shi5:+.4f}] | {ssig5} |\n"
    )


def analyze_run(label: str, path: Path, out: Path, b_main: int, b_gen: int, seed: int) -> str:
    scores, labels, generator = load_wide(path)
    n_real, n_fake = int((labels == 0).sum()), int((labels == 1).sum())

    per_cell = {c: tpr_both(labels.astype(float), scores[:, i]) for i, c in enumerate(CELL_ORDER)}
    gap1, fam1 = family_gap(per_cell, 0)
    gap5, fam5 = family_gap(per_cell, 1)
    boot = bootstrap_gap(scores, labels, np.random.default_rng(seed), b_main)

    rows = []
    lines = [f"## {label}", "", f"`{path}` -- n={len(labels)} (real={n_real}, ai={n_fake})", "",
             f"| Cell | Family | TPR@1% | 95% CI | TPR@5% | 95% CI |",
             "|---|---|---:|---|---:|---|"]
    for i, cell in enumerate(CELL_ORDER):
        t1, t5 = per_cell[cell]
        _, lo1, hi1 = ci95(boot["cell_tpr1"][:, i])
        _, lo5, hi5 = ci95(boot["cell_tpr5"][:, i])
        lines.append(f"| `{cell}` | {CELL_FAMILY[cell]} | {t1:.4f} | [{lo1:.4f}, {hi1:.4f}] "
                      f"| {t5:.4f} | [{lo5:.4f}, {hi5:.4f}] |")
        rows.append({"cell": cell, "family": CELL_FAMILY[cell], "tpr1": t1, "tpr1_ci_lo": lo1,
                     "tpr1_ci_hi": hi1, "tpr5": t5, "tpr5_ci_lo": lo5, "tpr5_ci_hi": hi5})

    med1, glo1, ghi1 = ci95(boot["gap1_dist"])
    med5, glo5, ghi5 = ci95(boot["gap5_dist"])
    lines += ["", "**Family-balanced gap** (`clean - mean(family means)`, same shape as the AUROC gap):",
              "", f"- TPR@1% gap = **{gap1:.4f}**, bootstrap median {med1:.4f}, 95% CI [{glo1:.4f}, {ghi1:.4f}]",
              f"- TPR@5% gap = **{gap5:.4f}**, bootstrap median {med5:.4f}, 95% CI [{glo5:.4f}, {ghi5:.4f}]", ""]

    lines += ["### Per generator (real pool shared)", "",
              "| Generator | n | Held out | clean TPR@1% | clean TPR@5% | TPR@1% gap | 95% CI | TPR@5% gap | 95% CI |",
              "|---|---:|---|---:|---:|---:|---|---:|---|"]
    gen_rows = []
    real_mask = labels == 0
    for gen in sorted(set(generator[labels == 1])):
        gen_mask = (labels == 1) & (generator == gen)
        keep = real_mask | gen_mask
        sub_scores, sub_labels = scores[keep], labels[keep]
        n_g = int(gen_mask.sum())
        per_cell_g = {c: tpr_both(sub_labels.astype(float), sub_scores[:, i]) for i, c in enumerate(CELL_ORDER)}
        g1, _ = family_gap(per_cell_g, 0)
        g5, _ = family_gap(per_cell_g, 1)
        clean_t1, clean_t5 = per_cell_g["clean"]
        boot_g = bootstrap_gap(sub_scores, sub_labels, np.random.default_rng(seed), b_gen)
        _, glo1g, ghi1g = ci95(boot_g["gap1_dist"])
        _, glo5g, ghi5g = ci95(boot_g["gap5_dist"])
        held = "yes" if gen in HELD_OUT else ""
        lines.append(f"| {gen} | {n_g} | {held} | {clean_t1:.4f} | {clean_t5:.4f} "
                      f"| {g1:.4f} | [{glo1g:.4f}, {ghi1g:.4f}] | {g5:.4f} | [{glo5g:.4f}, {ghi5g:.4f}] |")
        gen_rows.append({"generator": gen, "n": n_g, "held_out": gen in HELD_OUT,
                          "clean_tpr1": clean_t1, "clean_tpr5": clean_t5,
                          "gap_tpr1": g1, "gap_tpr1_ci_lo": glo1g, "gap_tpr1_ci_hi": ghi1g,
                          "gap_tpr5": g5, "gap_tpr5_ci_lo": glo5g, "gap_tpr5_ci_hi": ghi5g})

    stem = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "-")
    pd.DataFrame(rows).to_csv(out / f"tpr_grid_{stem}.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(out / f"tpr_per_generator_{stem}.csv", index=False)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "SCORES_CSV"),
                     help="repeatable: a run label and its scores.csv path")
    ap.add_argument("--pair", action="append", nargs=4,
                     metavar=("LABEL_A", "SCORES_CSV_A", "LABEL_B", "SCORES_CSV_B"),
                     help="repeatable: paired bootstrap of the gap difference between two "
                          "runs scored on the same images (label_b minus label_a)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--b-main", type=int, default=800, help="bootstrap reps, 19-cell grid")
    ap.add_argument("--b-gen", type=int, default=300, help="bootstrap reps, per-generator gap")
    ap.add_argument("--b-pair", type=int, default=2000, help="bootstrap reps, paired gap diff")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    runs = a.run or [["post-fix", "results/baseline/scores.csv"],
                      ["pre-fix (QuickGELU-mismatched)", "results/_prefix_gelu_run/scores.csv"]]

    sections = [
        "# TPR@FPR robustness gap\n",
        "Companion to `results/baseline/report.md`'s AUROC-based `robustness_gap`. "
        "See module docstring (`scripts/tpr_gap_analysis.py`) for why this metric and "
        "how the bootstrap CI is constructed.\n",
        f"Reproduce: `python -m scripts.tpr_gap_analysis --out {a.out}` "
        f"(B={a.b_main} main, B={a.b_gen} per-generator, seed={a.seed})\n",
    ]
    for label, path in runs:
        sections.append(analyze_run(label, Path(path), a.out, a.b_main, a.b_gen, a.seed))
    for label_a, path_a, label_b, path_b in (a.pair or []):
        sections.append(paired_section(label_a, Path(path_a), label_b, Path(path_b), a.b_pair, a.seed))

    (a.out / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"-> {a.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
