"""Tests for the metric layer (PLAN.md §3.2)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import metrics as M


def _labels(n_per_class: int = 50) -> np.ndarray:
    return np.array([0] * n_per_class + [1] * n_per_class)


# --------------------------------------------------------------------------- #
# Per-cell metrics
# --------------------------------------------------------------------------- #

def test_perfect_separation():
    y = _labels(20)
    s = np.where(y == 1, 0.9, 0.1)
    m = M.cell_metrics(y, s, cell="clean")
    assert m.auroc == 1.0
    assert m.ap == 1.0
    assert m.acc == 1.0
    assert m.tpr_at_fpr1 == 1.0
    assert (m.n, m.n_real, m.n_fake) == (40, 20, 20)


def test_inverted_scores_give_auroc_zero():
    y = _labels(20)
    m = M.cell_metrics(y, np.where(y == 1, 0.1, 0.9))
    assert m.auroc == 0.0
    assert m.acc == 0.0


def test_auroc_matches_sklearn_on_random_scores():
    rng = np.random.default_rng(0)
    y = _labels(200)
    s = rng.random(y.size)
    m = M.cell_metrics(y, s)
    assert m.auroc == pytest.approx(roc_auc_score(y, s))


def test_acc_uses_a_ge_half_threshold():
    y = np.array([0, 1])
    # exactly 0.5 counts as positive
    assert M.cell_metrics(y, np.array([0.5, 0.5])).acc == 0.5
    assert M.cell_metrics(y, np.array([0.49, 0.5])).acc == 1.0


def test_ap_of_a_random_model_is_about_prevalence():
    rng = np.random.default_rng(1)
    y = np.array([0] * 900 + [1] * 100)          # 10% prevalence
    m = M.cell_metrics(y, rng.random(y.size))
    assert m.ap == pytest.approx(0.10, abs=0.05)


# --------------------------------------------------------------------------- #
# TPR@FPR
# --------------------------------------------------------------------------- #

def test_tpr_at_fpr_is_conservative_not_interpolated():
    """200 negatives -> achievable FPRs are k/200. At a 1% budget we may accept
    at most 2 false positives, and must report the TPR actually reachable there,
    not a value interpolated between ROC vertices."""
    rng = np.random.default_rng(2)
    n = 200
    y = np.array([0] * n + [1] * n)
    # negatives ~ U(0,1); positives all just above the 3rd-highest negative
    neg = np.sort(rng.random(n))
    thresh = neg[-3]
    s = np.concatenate([neg, np.full(n, thresh + 1e-6)])
    got = M.tpr_at_fpr(y, s, 0.01)
    assert got == pytest.approx(1.0)
    # and pushing the positives below that point must drop TPR to 0
    s2 = np.concatenate([neg, np.full(n, neg[-3] - 1e-6)])
    assert M.tpr_at_fpr(y, s2, 0.01) == pytest.approx(0.0)


def test_tpr_at_fpr_flags_degenerate_cells():
    """Fewer than 100 negatives cannot resolve a 1% FPR; say so rather than
    quoting a number that is really TPR at zero false positives."""
    y = _labels(20)                              # 20 negatives
    m = M.cell_metrics(y, np.where(y == 1, 0.9, 0.1))
    assert "degenerate" in m.notes
    assert M.min_negatives_for_fpr(0.01) == 100
    y2 = _labels(120)                            # 120 negatives
    m2 = M.cell_metrics(y2, np.where(y2 == 1, 0.9, 0.1))
    assert "degenerate" not in m2.notes


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #

def test_single_class_cell_is_nan_not_a_crash():
    m = M.cell_metrics(np.ones(10), np.linspace(0, 1, 10), cell="fakes_only")
    assert not np.isfinite(m.auroc) and not np.isfinite(m.ap)
    assert "single-class" in m.notes
    assert np.isfinite(m.acc)


def test_unscorable_predictions_are_excluded_and_counted():
    y = _labels(60)
    s = np.where(y == 1, 0.9, 0.1)
    s = s.astype(object)
    s[0] = None                                  # a real the model refused
    s[-1] = np.nan                               # a fake it refused
    m = M.cell_metrics(y, s)
    assert m.n_failed == 2
    assert (m.n, m.n_real, m.n_fake) == (118, 59, 59)
    assert m.auroc == 1.0
    assert "2 unscorable" in m.notes


def test_empty_cell():
    m = M.cell_metrics(np.array([]), np.array([]), cell="empty")
    assert m.n == 0 and "no scorable images" in m.notes


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.cell_metrics(np.zeros(3), np.zeros(4))


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #

def _cells(aurocs: dict[str, float]) -> list[M.CellMetrics]:
    out = []
    for name, a in aurocs.items():
        kind = "clean" if name == "clean" else ("composed" if name.startswith("composed") else "single")
        out.append(M.CellMetrics(cell=name, kind=kind, family=name.split("_")[0], auroc=a, n=200))
    return out


def test_robustness_gap_and_worst_case_definitions():
    cells = _cells({"clean": 0.98, "jpeg_30": 0.60, "blur_2.0": 0.80, "composed_x": 0.50})
    s = M.summarize(cells)
    assert s.clean_auroc == 0.98
    assert s.mean_transformed_auroc == pytest.approx((0.60 + 0.80 + 0.50) / 3)
    assert s.robustness_gap == pytest.approx(0.98 - (0.60 + 0.80 + 0.50) / 3)
    assert s.worst_case == 0.50 and s.worst_cell == "composed_x"
    assert s.n_transformed_cells == 3


def test_clean_is_excluded_from_the_transformed_mean():
    s = M.summarize(_cells({"clean": 0.9, "jpeg_30": 0.9}))
    assert s.mean_transformed_auroc == 0.9
    assert s.robustness_gap == pytest.approx(0.0)


def test_worst_case_spans_all_cells_including_clean():
    """§3.2 says min over all cells. If clean really is the worst, show it."""
    s = M.summarize(_cells({"clean": 0.40, "jpeg_30": 0.70}))
    assert s.worst_cell == "clean" and s.worst_case == 0.40


def test_single_and_composed_means_are_reported_separately():
    s = M.summarize(_cells(
        {"clean": 1.0, "jpeg_30": 0.8, "blur_2.0": 0.6, "composed_a": 0.4, "composed_b": 0.2}
    ))
    assert s.mean_single_auroc == pytest.approx(0.7)
    assert s.mean_composed_auroc == pytest.approx(0.3)


def test_undefined_cells_are_skipped_and_warned_about():
    s = M.summarize(_cells({"clean": 0.9, "jpeg_30": float("nan"), "blur_2.0": 0.7}))
    assert s.mean_transformed_auroc == pytest.approx(0.7)
    assert s.n_undefined_cells == 1
    assert any("undefined AUROC" in w for w in s.warnings)


def test_missing_clean_cell_warns_and_leaves_gap_undefined():
    s = M.summarize(_cells({"jpeg_30": 0.7}))
    assert not np.isfinite(s.robustness_gap)
    assert any("robustness_gap undefined" in w for w in s.warnings)


def test_summarize_rejects_an_empty_grid():
    with pytest.raises(ValueError):
        M.summarize([])


def test_summarize_accepts_plain_dicts_from_csv():
    rows = [c.as_row() for c in _cells({"clean": 0.9, "jpeg_30": 0.5})]
    assert M.summarize(rows).robustness_gap == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Null distribution helper
# --------------------------------------------------------------------------- #

def test_auroc_null_sd_matches_simulation():
    n = 100
    y = _labels(n)
    rng = np.random.default_rng(3)
    sim = np.std([roc_auc_score(y, rng.random(y.size)) for _ in range(500)])
    assert M.auroc_null_sd(n, n) == pytest.approx(sim, rel=0.20)


def test_auroc_null_sd_is_nan_for_a_single_class():
    assert not np.isfinite(M.auroc_null_sd(0, 100))


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #

def test_markdown_grid_is_complete_and_pairs_clean_with_the_gap():
    cells = _cells({"clean": 0.98, "jpeg_30": 0.6, "composed_x": 0.5})
    md = M.markdown_grid(cells, title="Test grid")
    assert md.startswith("# Test grid")
    for name in ("clean", "jpeg_30", "composed_x"):
        assert f"`{name}`" in md
    # §13: never clean alone
    assert "Clean AUROC" in md and "Robustness gap" in md and "Worst cell" in md
    assert "By family" in md
    # the per-cell table is well formed: one row per cell, 8 columns each
    per_cell = md.split("## Per-cell")[1].split("## By family")[0]
    rows = [ln for ln in per_cell.splitlines() if ln.startswith("| `")]
    assert len(rows) == 3
    assert all(r.count("|") == 9 for r in rows), rows


def test_markdown_renders_nan_as_a_dash():
    md = M.markdown_grid(_cells({"clean": 0.9, "jpeg_30": float("nan")}))
    assert "—" in md
    assert "nan" not in md.lower()
