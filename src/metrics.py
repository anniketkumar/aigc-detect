"""Per-cell metrics and the two headline robustness numbers (PLAN.md §3.2).

Per (transform, severity) cell: AUROC, AP, acc@0.5, TPR@FPR=1%.

    robustness_gap = AUROC(clean) - mean(AUROC(all transformed cells))
    worst_case     = min(AUROC over all cells)

Those two are the objective. §13 forbids reporting clean accuracy without the
gap beside it, so :func:`markdown_grid` always prints them together.

Conventions the spec leaves open are marked "SPEC:".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

__all__ = [
    "CellMetrics",
    "cell_metrics",
    "tpr_at_fpr",
    "auroc_null_sd",
    "min_negatives_for_fpr",
    "GridSummary",
    "summarize",
    "markdown_grid",
]

#: label convention from §4.2: 0 = real, 1 = AI-generated.
POSITIVE_LABEL = 1

#: acc@0.5 threshold. SPEC: 0.5 is only meaningful for a calibrated score; a raw
#: logit model will score ~chance here even with a perfect AUROC. Kept because
#: §3.2 asks for it, but AUROC/AP are the numbers to read before Phase 6.
DECISION_THRESHOLD = 0.5

DEFAULT_FPR_TARGET = 0.01


# --------------------------------------------------------------------------- #
# Per-cell
# --------------------------------------------------------------------------- #

def min_negatives_for_fpr(fpr_target: float = DEFAULT_FPR_TARGET) -> int:
    """Negatives needed for a non-degenerate TPR@FPR estimate.

    With ``n`` negatives the only achievable FPRs are ``k/n``. Below ``1/target``
    the largest FPR at or under the target is 0, so "TPR@FPR=1%" silently
    becomes "TPR at zero false positives" -- a much harsher and much noisier
    quantity. Callers should surface this rather than quote the number.
    """
    return int(math.ceil(1.0 / fpr_target))


def tpr_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = DEFAULT_FPR_TARGET
) -> float:
    """Highest TPR reachable at an FPR at or below ``fpr_target``.

    SPEC: the interpolation convention is unspecified. We take the conservative
    (non-interpolating) reading -- the best operating point that genuinely does
    not exceed the FPR budget -- rather than interpolating between ROC vertices,
    which reports a TPR no achievable threshold delivers.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= fpr_target + 1e-12
    return float(tpr[ok].max()) if ok.any() else float("nan")


def auroc_null_sd(n_pos: int, n_neg: int) -> float:
    """SD of AUROC under H0 (scores independent of labels).

    ``sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))``, the Mann-Whitney U
    null. Used to turn "AUROC ~ 0.5" into a checkable tolerance: a random
    model's cells should sit inside a few of these of 0.5.
    """
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    return math.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))


@dataclass
class CellMetrics:
    """Metrics for one grid cell."""

    cell: str
    kind: str = "single"          # clean | single | composed
    family: str = ""
    severity: float | None = None
    chain: str = ""
    n: int = 0
    n_real: int = 0
    n_fake: int = 0
    n_failed: int = 0             # images the model could not score
    auroc: float = float("nan")
    ap: float = float("nan")
    acc: float = float("nan")     # acc@0.5
    tpr_at_fpr1: float = float("nan")
    notes: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def cell_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    cell: str = "",
    fpr_target: float = DEFAULT_FPR_TARGET,
    **meta,
) -> CellMetrics:
    """Compute all four metrics for one cell.

    ``None``/NaN scores are treated as unscorable and excluded (§9.1 allows a
    model to emit ``pred: null``); their count is reported in ``n_failed`` so a
    model cannot improve its numbers by declining to answer without it showing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray([np.nan if s is None else s for s in y_score], dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")

    m = CellMetrics(cell=cell, **{k: v for k, v in meta.items() if k in CellMetrics.__annotations__})
    m.n_failed = int(np.isnan(y_score).sum())

    keep = ~np.isnan(y_score)
    y_true, y_score = y_true[keep], y_score[keep]
    m.n = int(y_true.size)
    m.n_fake = int((y_true == POSITIVE_LABEL).sum())
    m.n_real = int(m.n - m.n_fake)

    notes: list[str] = []
    if m.n == 0:
        notes.append("no scorable images")
    elif m.n_fake == 0 or m.n_real == 0:
        notes.append("single-class cell: AUROC/AP undefined")
        m.acc = float(((y_score >= DECISION_THRESHOLD).astype(int) == y_true).mean())
    else:
        m.auroc = float(roc_auc_score(y_true, y_score))
        m.ap = float(average_precision_score(y_true, y_score))
        m.acc = float(((y_score >= DECISION_THRESHOLD).astype(int) == y_true).mean())
        m.tpr_at_fpr1 = tpr_at_fpr(y_true, y_score, fpr_target)
        need = min_negatives_for_fpr(fpr_target)
        if m.n_real < need:
            notes.append(
                f"TPR@FPR={fpr_target:.0%} degenerate: {m.n_real} reals < {need}"
            )
    if m.n_failed:
        notes.append(f"{m.n_failed} unscorable")
    m.notes = "; ".join(notes)
    return m


# --------------------------------------------------------------------------- #
# Grid aggregates
# --------------------------------------------------------------------------- #

@dataclass
class GridSummary:
    clean_auroc: float = float("nan")
    mean_transformed_auroc: float = float("nan")
    robustness_gap: float = float("nan")
    worst_case: float = float("nan")
    worst_cell: str = ""
    mean_single_auroc: float = float("nan")
    mean_composed_auroc: float = float("nan")
    #: Flat (cell-weighted) counterparts of the two headline numbers, kept as
    #: secondary columns so the §3.2 literal definition stays visible.
    mean_transformed_auroc_flat: float = float("nan")
    robustness_gap_flat: float = float("nan")
    family_means: dict[str, float] = field(default_factory=dict)
    n_cells: int = 0
    n_transformed_cells: int = 0
    n_families: int = 0
    n_undefined_cells: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        d = asdict(self)
        d["warnings"] = "; ".join(self.warnings)
        return d


def summarize(
    cells: Iterable[CellMetrics | Mapping],
    clean_cell: str = "clean",
) -> GridSummary:
    """Reduce per-cell metrics to the §3.2 headline numbers.

    ``robustness_gap`` is **family-balanced**: the transformed cells are grouped
    by degradation family, each family's AUROCs are averaged, and the headline
    averages those seven family means. §3.2 as written specifies a flat mean over
    cells, which silently weights the grid by severity count -- JPEG contributes
    4 of the 14 single cells and ``jitter`` contributes 1, so a detector could
    cut the flat gap by a third by being good at JPEG alone while staying
    fragile to everything else. That is the opposite of what the metric is for.
    Composed chains count as one family, matching :attr:`transforms.Cell.family`.

    The flat mean is kept as ``mean_transformed_auroc_flat`` /
    ``robustness_gap_flat`` so the literal §3.2 number stays on the table and
    the two can be compared.

    ``worst_case`` is the min over *all* cells including clean, as §3.2 writes
    it. Clean should never be the argmin; if it is, that is worth seeing.
    """
    rows = [c if isinstance(c, CellMetrics) else CellMetrics(**dict(c)) for c in cells]
    if not rows:
        raise ValueError("no cells to summarize")

    s = GridSummary(n_cells=len(rows))
    by_name = {r.cell: r for r in rows}

    if clean_cell in by_name:
        s.clean_auroc = by_name[clean_cell].auroc
    else:
        s.warnings.append(f"no {clean_cell!r} cell: robustness_gap undefined")

    transformed = [r for r in rows if r.cell != clean_cell]
    s.n_transformed_cells = len(transformed)
    s.n_undefined_cells = sum(1 for r in rows if not np.isfinite(r.auroc))
    if s.n_undefined_cells:
        s.warnings.append(f"{s.n_undefined_cells} cell(s) with undefined AUROC")

    def _mean(sel: Sequence[CellMetrics]) -> float:
        vals = [r.auroc for r in sel if np.isfinite(r.auroc)]
        return float(np.mean(vals)) if vals else float("nan")

    # Family-balanced headline: mean of family means, so a family with four
    # severities does not outvote one with a single severity.
    fam_groups: dict[str, list[CellMetrics]] = {}
    for r in transformed:
        fam_groups.setdefault(r.family or r.kind, []).append(r)
    s.family_means = {
        name: _mean(group) for name, group in sorted(fam_groups.items())
    }
    usable = [v for v in s.family_means.values() if np.isfinite(v)]
    s.n_families = len(s.family_means)
    s.mean_transformed_auroc = float(np.mean(usable)) if usable else float("nan")
    s.robustness_gap = s.clean_auroc - s.mean_transformed_auroc

    # Secondary: the literal §3.2 flat mean over cells.
    s.mean_transformed_auroc_flat = _mean(transformed)
    s.robustness_gap_flat = s.clean_auroc - s.mean_transformed_auroc_flat

    s.mean_single_auroc = _mean([r for r in transformed if r.kind == "single"])
    s.mean_composed_auroc = _mean([r for r in transformed if r.kind == "composed"])

    finite = [r for r in rows if np.isfinite(r.auroc)]
    if finite:
        worst = min(finite, key=lambda r: r.auroc)
        s.worst_case, s.worst_cell = worst.auroc, worst.cell
    return s


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _f(x: float, nd: int = 4) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def markdown_grid(
    cells: Sequence[CellMetrics],
    summary: GridSummary | None = None,
    title: str = "Robustness grid",
    subtitle: str = "",
) -> str:
    """Render the grid as markdown: headline numbers first, then every cell."""
    summary = summarize(cells) if summary is None else summary
    out: list[str] = [f"# {title}", ""]
    if subtitle:
        out += [subtitle, ""]

    out += [
        "## Headline (§3.2)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Clean AUROC | {_f(summary.clean_auroc)} |",
        f"| Mean transformed AUROC (family-balanced) | "
        f"{_f(summary.mean_transformed_auroc)} |",
        f"| **Robustness gap** ↓ | **{_f(summary.robustness_gap)}** |",
        f"| **Worst cell AUROC** ↑ | **{_f(summary.worst_case)}** "
        f"(`{summary.worst_cell}`) |",
        f"| Mean transformed AUROC (flat, §3.2 literal) | "
        f"{_f(summary.mean_transformed_auroc_flat)} |",
        f"| Robustness gap (flat) | {_f(summary.robustness_gap_flat)} |",
        f"| Mean AUROC, single transforms | {_f(summary.mean_single_auroc)} |",
        f"| Mean AUROC, composed chains | {_f(summary.mean_composed_auroc)} |",
        f"| Cells | {summary.n_cells} "
        f"({summary.n_transformed_cells} transformed, "
        f"{summary.n_families} families) |",
        "",
        "`robustness_gap = AUROC(clean) − mean(family mean AUROC)`, lower is "
        "better. The headline weights each degradation family equally rather "
        "than each cell, so being good at JPEG alone (4 of 14 single cells) "
        "cannot mask fragility elsewhere; the flat cell-weighted mean §3.2 "
        "specifies is reported beside it. `worst_case = min(AUROC)` over all "
        "cells, higher is better. Clean AUROC is never to be read on its own "
        "(§13).",
        "",
        "## Per-cell",
        "",
        "| Cell | Chain | n | AUROC | AP | acc@0.5 | TPR@FPR=1% | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in cells:
        out.append(
            f"| `{c.cell}` | `{c.chain or '—'}` | {c.n} | {_f(c.auroc)} | {_f(c.ap)} "
            f"| {_f(c.acc)} | {_f(c.tpr_at_fpr1)} | {c.notes or ''} |"
        )

    fam = _family_means(cells)
    if fam:
        out += ["", "## By family", "", "| Family | Cells | Mean AUROC | Min AUROC |",
                "|---|---:|---:|---:|"]
        for name, (k, mean_a, min_a) in fam.items():
            out.append(f"| `{name}` | {k} | {_f(mean_a)} | {_f(min_a)} |")

    if summary.warnings:
        out += ["", "## Warnings", ""] + [f"- {w}" for w in summary.warnings]
    out.append("")
    return "\n".join(out)


def _family_means(cells: Sequence[CellMetrics]) -> dict[str, tuple[int, float, float]]:
    groups: dict[str, list[float]] = {}
    for c in cells:
        groups.setdefault(c.family or c.kind, []).append(c.auroc)
    out: dict[str, tuple[int, float, float]] = {}
    for name, vals in groups.items():
        finite = [v for v in vals if np.isfinite(v)]
        out[name] = (
            len(vals),
            float(np.mean(finite)) if finite else float("nan"),
            float(np.min(finite)) if finite else float("nan"),
        )
    return out
