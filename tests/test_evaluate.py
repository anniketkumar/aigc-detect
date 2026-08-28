"""Harness tests, including the §3.3 acceptance criterion.

The acceptance criterion is "AUROC ≈ 0.5 in every cell". "≈" needs a number, so
the tolerance here is derived from :func:`metrics.auroc_null_sd` -- the SD of
AUROC when scores are independent of labels -- rather than picked by feel. At
this fixture size that SD is ~0.041, and the bound is 4.5 of them.

Correctness note: the fixture is generated with ``--signal none``, so its two
classes are drawn from an identical content distribution. A cell landing far
from 0.5 therefore means either the harness is leaking labels into the scores or
the model is cheating -- it cannot mean "the model found something".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.make_dummy_fixture import build as build_fixture
from src import evaluate as E
from src import metrics as M
from src import transforms as T

N_IMAGES = 200          # 100 per class -> AUROC null SD ~0.0409
IMG_SIZE = 64
SIGMA_BOUND = 4.5       # per-cell tolerance, in null SDs
NULL_SD = M.auroc_null_sd(N_IMAGES // 2, N_IMAGES // 2)
TOL = SIGMA_BOUND * NULL_SD


@pytest.fixture(scope="module")
def fixture_manifest(tmp_path_factory):
    """A null fixture: content independent of label."""
    d = tmp_path_factory.mktemp("fixture")
    return build_fixture(d, n=N_IMAGES, size=IMG_SIZE, seed=0, signal="none")


@pytest.fixture(scope="module")
def acceptance_run(fixture_manifest, tmp_path_factory):
    out = tmp_path_factory.mktemp("results") / "dummy"
    rc = E.main([
        "--model", "dummy_random",
        "--manifest", str(fixture_manifest),
        "--out", str(out),
        "--name", "dummy_random",
        "--quiet",
    ])
    assert rc == 0
    return out


# --------------------------------------------------------------------------- #
# §3.3 acceptance
# --------------------------------------------------------------------------- #

def test_acceptance_grid_csv_covers_every_cell(acceptance_run):
    df = pd.read_csv(acceptance_run / "grid.csv")
    expected = [c.name for c in T.build_cells()]
    assert list(df["cell"]) == expected
    assert len(df) == 19
    assert (df["n"] == N_IMAGES).all()
    assert (df["n_real"] == N_IMAGES // 2).all()
    assert (df["n_fake"] == N_IMAGES // 2).all()
    assert (df["n_failed"] == 0).all()
    assert df[["auroc", "ap", "acc", "tpr_at_fpr1"]].notna().all().all()


def test_acceptance_auroc_is_chance_in_every_cell(acceptance_run):
    """The §3.3 criterion, with an explicit tolerance."""
    df = pd.read_csv(acceptance_run / "grid.csv")
    z = (df["auroc"] - 0.5).abs() / NULL_SD
    offenders = df.loc[z > SIGMA_BOUND, ["cell", "auroc"]]
    assert offenders.empty, (
        f"cells outside {SIGMA_BOUND} null SDs ({TOL:.3f}) of 0.5:\n{offenders}"
    )
    # and the grid as a whole must be centred on chance, not merely bounded
    assert df["auroc"].mean() == pytest.approx(0.5, abs=2 * NULL_SD)


def test_acceptance_ap_and_acc_are_also_chance(acceptance_run):
    """AP ~ prevalence (0.5 on a balanced set), acc@0.5 ~ 0.5."""
    df = pd.read_csv(acceptance_run / "grid.csv")
    assert df["ap"].mean() == pytest.approx(0.5, abs=0.06)
    assert df["acc"].mean() == pytest.approx(0.5, abs=0.06)


def test_acceptance_tpr_at_fpr_is_near_the_fpr_budget(acceptance_run):
    """For a chance model, TPR at a 1% FPR budget should be ~1%."""
    df = pd.read_csv(acceptance_run / "grid.csv")
    assert df["tpr_at_fpr1"].mean() < 0.10
    assert "degenerate" not in " ".join(df["notes"].fillna(""))


def test_acceptance_robustness_gap_is_near_zero(acceptance_run):
    """A model with no signal has no signal to lose."""
    s = json.loads((acceptance_run / "summary.json").read_text())["summary"]
    assert abs(s["robustness_gap"]) < 2 * TOL
    assert abs(s["clean_auroc"] - 0.5) < TOL
    assert abs(s["worst_case"] - 0.5) < TOL
    assert s["n_cells"] == 19 and s["n_transformed_cells"] == 18
    assert s["n_undefined_cells"] == 0
    assert s["warnings"] == ""


def test_acceptance_markdown_table_is_complete(acceptance_run):
    md = (acceptance_run / "report.md").read_text(encoding="utf-8")
    for cell in T.build_cells():
        assert f"`{cell.name}`" in md, cell.name
    # headline block, per §13 clean is never alone
    for key in ("Clean AUROC", "Mean transformed AUROC", "Robustness gap",
                "Worst cell AUROC", "Per-cell", "By family"):
        assert key in md, key
    # the per-cell table is a well-formed 8-column markdown table
    body = md.split("## Per-cell")[1].split("## By family")[0]
    rows = [ln for ln in body.splitlines() if ln.startswith("| `")]
    assert len(rows) == 19
    assert all(r.count("|") == 9 for r in rows)
    # and the run is reproducible from the report alone
    assert "Reproduce:" in md and "python -m src.evaluate" in md


def test_acceptance_writes_raw_scores_for_error_analysis(acceptance_run):
    df = pd.read_csv(acceptance_run / "scores.csv")
    assert len(df) == N_IMAGES * 19
    assert set(df.columns) == {"image_path", "label", "cell", "score"}
    assert df["score"].between(0, 1).all()
    # every image scored in every cell
    assert (df.groupby("cell").size() == N_IMAGES).all()


def test_acceptance_summary_records_the_full_run_config(acceptance_run):
    cfg = json.loads((acceptance_run / "summary.json").read_text())["config"]
    for key in ("model", "manifest", "seed", "n_images", "cells", "command",
                "auroc_null_sd", "python", "platform", "elapsed_s"):
        assert key in cfg, key
    assert cfg["n_images"] == N_IMAGES
    assert len(cfg["cells"]) == 19


# --------------------------------------------------------------------------- #
# The fixture is a genuine null
# --------------------------------------------------------------------------- #

def test_fixture_has_no_label_signal_in_brightness_or_size(fixture_manifest):
    """If content correlated with the label, the acceptance test would be
    measuring the fixture rather than the harness."""
    from PIL import Image

    df = pd.read_csv(fixture_manifest)
    means = {0: [], 1: []}
    for _, row in df.iterrows():
        img = Image.open(row["image_path"])
        means[int(row["label"])].append(np.asarray(img.convert("L"), float).mean())
    a, b = np.array(means[0]), np.array(means[1])
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    assert abs(t) < 3.0, f"brightness differs by class (t={t:.2f})"


def test_fixture_filenames_carry_no_label_information(fixture_manifest):
    """image_id seeds the stochastic transforms and is passed to
    Scorer.score(), so a label-bearing filename is a leak channel. Stems must be
    a pure function of the index: exactly 0..n-1, in any label order."""
    df = pd.read_csv(fixture_manifest)
    stems = {str(p).rsplit("/", 1)[-1].split(".")[0] for p in df["image_path"]}
    assert stems == {f"{i:05d}" for i in range(len(df))}


def test_fixture_includes_awkward_image_modes(fixture_manifest):
    from PIL import Image

    df = pd.read_csv(fixture_manifest)
    modes = {Image.open(p).mode for p in df["image_path"]}
    assert {"RGBA", "L", "CMYK"} <= modes, modes


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def test_two_runs_with_the_same_seed_are_identical(fixture_manifest, tmp_path):
    args = ["--model", "dummy_random", "--manifest", str(fixture_manifest),
            "--quiet", "--no-scores", "--limit", "60"]
    E.main(args + ["--out", str(tmp_path / "a")])
    E.main(args + ["--out", str(tmp_path / "b")])
    a = (tmp_path / "a" / "grid.csv").read_text()
    b = (tmp_path / "b" / "grid.csv").read_text()
    assert a == b


def test_batch_size_does_not_change_results(fixture_manifest, tmp_path):
    args = ["--model", "dummy_random", "--manifest", str(fixture_manifest),
            "--quiet", "--no-scores", "--limit", "60"]
    E.main(args + ["--out", str(tmp_path / "b1"), "--batch-size", "1"])
    E.main(args + ["--out", str(tmp_path / "b64"), "--batch-size", "64"])
    assert (tmp_path / "b1" / "grid.csv").read_text() == \
           (tmp_path / "b64" / "grid.csv").read_text()


def test_the_transform_cache_does_not_change_results(fixture_manifest, tmp_path):
    """A cache that alters a single pixel would silently corrupt every later
    phase, so this is the test that makes the cache safe to turn on."""
    args = ["--model", "dummy_random", "--manifest", str(fixture_manifest),
            "--quiet", "--no-scores", "--limit", "40"]
    E.main(args + ["--out", str(tmp_path / "nocache")])
    cache = tmp_path / "cache"
    E.main(args + ["--out", str(tmp_path / "cold"), "--cache-dir", str(cache)])
    E.main(args + ["--out", str(tmp_path / "warm"), "--cache-dir", str(cache)])

    ref = (tmp_path / "nocache" / "grid.csv").read_text()
    assert (tmp_path / "cold" / "grid.csv").read_text() == ref
    assert (tmp_path / "warm" / "grid.csv").read_text() == ref

    cold = json.loads((tmp_path / "cold" / "summary.json").read_text())["config"]
    warm = json.loads((tmp_path / "warm" / "summary.json").read_text())["config"]
    assert cold["cache_hits"] == 0
    assert warm["cache_hits"] == 40 * 19 and warm["cache_misses"] == 0


def test_cache_is_keyed_by_transform_so_entries_cannot_cross_cells(tmp_path):
    cache = E.TransformCache(tmp_path / "c", base_seed=0)
    cells = {c.name: c for c in T.build_cells()}
    from PIL import Image
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    cache.put(cells["jpeg_30"], "img_a", img)
    assert cache.get(cells["jpeg_30"], "img_a") is not None
    assert cache.get(cells["jpeg_90"], "img_a") is None      # different severity
    assert cache.get(cells["blur_1.0"], "img_a") is None     # different family
    assert cache.get(cells["jpeg_30"], "img_b") is None      # different image


# --------------------------------------------------------------------------- #
# CLI behaviour
# --------------------------------------------------------------------------- #

def test_cells_flag_accepts_names_and_families(fixture_manifest, tmp_path):
    E.main(["--model", "dummy_random", "--manifest", str(fixture_manifest),
            "--out", str(tmp_path / "o"), "--quiet", "--no-scores",
            "--limit", "40", "--cells", "clean", "jpeg", "blur_2.0"])
    df = pd.read_csv(tmp_path / "o" / "grid.csv")
    assert list(df["cell"]) == ["clean", "jpeg_90", "jpeg_70", "jpeg_50",
                                "jpeg_30", "blur_2.0"]


def test_unknown_cell_is_rejected(fixture_manifest, tmp_path):
    with pytest.raises(SystemExit):
        E.main(["--manifest", str(fixture_manifest), "--out", str(tmp_path / "o"),
                "--quiet", "--cells", "sharpen"])


def test_limit_is_class_stratified(fixture_manifest):
    df = E.load_manifest(fixture_manifest, limit=50, seed=0)
    assert len(df) == 50
    assert df["label"].value_counts().to_dict() == {0: 25, 1: 25}


def test_limit_is_reproducible_and_seed_dependent(fixture_manifest):
    a = E.load_manifest(fixture_manifest, limit=50, seed=0)["image_path"].tolist()
    b = E.load_manifest(fixture_manifest, limit=50, seed=0)["image_path"].tolist()
    c = E.load_manifest(fixture_manifest, limit=50, seed=1)["image_path"].tolist()
    assert a == b and a != c


def test_missing_manifest_gives_a_useful_error(tmp_path):
    with pytest.raises(SystemExit, match="make_dummy_fixture"):
        E.resolve_manifest(None, "test", root=tmp_path)


def test_manifest_without_a_label_column_is_rejected(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("image_path\nfoo.png\n")
    with pytest.raises(SystemExit, match="missing column"):
        E.load_manifest(p)


def test_manifest_with_out_of_range_labels_is_rejected(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("image_path,label\nfoo.png,2\n")
    with pytest.raises(SystemExit, match="0=real"):
        E.load_manifest(p)


def test_neither_split_nor_manifest_is_rejected():
    with pytest.raises(SystemExit, match="--manifest or --split"):
        E.resolve_manifest(None, None)


def test_split_resolves_against_the_manifest_root(tmp_path):
    (tmp_path / "test.csv").write_text("image_path,label\nfoo.png,0\n")
    assert E.resolve_manifest(None, "test", root=tmp_path).name == "test.csv"


# --------------------------------------------------------------------------- #
# Robustness of the harness itself
# --------------------------------------------------------------------------- #

def test_unreadable_images_are_counted_not_fatal(fixture_manifest, tmp_path, capsys):
    """A single corrupt file must not lose the other 199 images' worth of work."""
    df = pd.read_csv(fixture_manifest).head(40)
    broken = tmp_path / "truncated.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage")
    df.loc[len(df)] = {**df.iloc[0].to_dict(), "image_path": broken.as_posix(),
                       "label": 1}
    m = tmp_path / "m.csv"
    df.to_csv(m, index=False)

    E.main(["--model", "dummy_random", "--manifest", str(m),
            "--out", str(tmp_path / "o"), "--quiet", "--no-scores",
            "--cells", "clean", "jpeg_30"])
    grid = pd.read_csv(tmp_path / "o" / "grid.csv")
    assert (grid["n_failed"] == 1).all()
    assert (grid["n"] == 40).all()
    assert grid["notes"].str.contains("1 unscorable").all()


def test_a_model_returning_the_wrong_number_of_scores_is_caught(fixture_manifest):
    class Broken:
        name = "broken"

        def score(self, images, image_ids):
            return [0.5] * (len(images) - 1)

    df = E.load_manifest(fixture_manifest, limit=8)
    with pytest.raises(RuntimeError, match="returned"):
        E.run_grid(Broken(), df, T.build_cells()[:1], progress=False)


def test_every_registered_model_runs_over_the_whole_grid(fixture_manifest, tmp_path):
    from src.models.base import MODEL_REGISTRY

    for name in MODEL_REGISTRY:
        E.main(["--model", name, "--manifest", str(fixture_manifest),
                "--out", str(tmp_path / name), "--quiet", "--no-scores",
                "--limit", "20"])
        df = pd.read_csv(tmp_path / name / "grid.csv")
        assert len(df) == 19 and df["auroc"].notna().all(), name


def test_load_model_rejects_unknown_names():
    from src.models.base import load_model

    with pytest.raises(KeyError, match="unknown model"):
        load_model("clip_vit_l14")


def test_dummy_random_is_deterministic_in_pixels_not_filenames():
    """Pixel-keyed, so each cell gets an independent draw and a harness bug that
    fed the clean image to every cell would show up as identical AUROCs."""
    from PIL import Image
    from src.models.dummy import RandomScorer

    m = RandomScorer(seed=0)
    a = Image.new("RGB", (8, 8), (1, 2, 3))
    b = Image.new("RGB", (8, 8), (1, 2, 4))
    assert m.score([a], ["x"]) == m.score([a], ["totally-different-id"])
    assert m.score([a], ["x"]) != m.score([b], ["x"])
    assert m.score([a], ["x"]) != RandomScorer(seed=1).score([a], ["x"])
    assert 0.0 <= m.score([a], ["x"])[0] < 1.0
