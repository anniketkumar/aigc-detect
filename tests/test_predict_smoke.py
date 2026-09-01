"""Smoke test for predict.py, the deliverable (PLAN.md §9.1).

A 10-image fixture directory mixing plain images with the awkward inputs
§9.1 names by name (an alpha PNG, grayscale images, a CMYK JPEG, a
recoverably-truncated file) plus a genuinely corrupt one and a non-image
file. Reuses the fixtures already vetted by ``tests/test_imageio.py`` rather
than inventing new ones, so predict.py is checked against the exact same
edge cases the rest of the codebase already trusts.

The real backbone (network download) is never exercised -- ``ClipBackbone``
is monkeypatched with a deterministic stand-in, same approach as
``tests/test_cache_features.py``, so this test runs offline in milliseconds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import predict as P

FIXTURES = Path(__file__).parent / "fixtures" / "loading"
EMBED_DIM = 8


class DummyClipBackbone:
    """Stands in for src.models.clip_backbone.ClipBackbone -- same interface,
    no network, no torch model. Deterministic function of pixel content."""

    embed_dim = EMBED_DIM

    def __init__(self, device="cpu", backbone="dummy", pretrained="dummy"):
        self.device = device

    def embed(self, images) -> np.ndarray:
        out = np.zeros((len(images), self.embed_dim), dtype=np.float32)
        for i, img in enumerate(images):
            arr = np.asarray(img.resize((4, 4)).convert("L"), dtype=np.float32).reshape(-1)
            out[i] = arr[: self.embed_dim]
        return out


@pytest.fixture(autouse=True)
def _stub_backbone(monkeypatch):
    # src/models/clip_baseline.py does `from src.models.clip_backbone import
    # ClipBackbone` at module top level -- a one-time name binding. Patching
    # the *defining* module (src.models.clip_backbone.ClipBackbone) only
    # works if clip_baseline hasn't been imported yet anywhere in the process;
    # in the full suite another test typically imports it first, so the name
    # actually consulted at call time is the *consuming* module's own
    # binding. Patch that one instead -- it works regardless of import order.
    monkeypatch.setattr("src.models.clip_baseline.ClipBackbone", DummyClipBackbone)


@pytest.fixture()
def fake_ckpt(tmp_path) -> Path:
    """A checkpoint shaped exactly like one src/train.py would write, but with
    a randomly-initialized head over the dummy backbone's embed_dim."""
    from src.models.semantic_head import LinearHead

    ckpt = tmp_path / "fake.pt"
    torch.save(
        {
            "state_dict": LinearHead(EMBED_DIM).state_dict(),
            "backbone": "dummy",
            "pretrained": "dummy",
            "embed_dim": EMBED_DIM,
            "config": {},
        },
        ckpt,
    )
    return ckpt


@pytest.fixture()
def fixture_dir(tmp_path) -> Path:
    """10 images: 3 plain RGB, plus the awkward inputs §9.1 names, plus one
    genuinely corrupt file and one non-image file that must be ignored."""
    d = tmp_path / "images"
    (d / "nested").mkdir(parents=True)

    # 3 plain, freshly generated RGB images (one nested, to check recursion).
    for i, sub in enumerate(["a.jpg", "b.png", "nested/c.jpg"]):
        rng = np.random.default_rng(i)
        arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(d / sub)

    # The awkward inputs PLAN.md §9.1 names explicitly, borrowed from the
    # already-vetted loading fixtures.
    borrowed = {
        "alpha.png": "alpha.png",              # PNG with alpha
        "gray.jpg": "gray.jpg",                # grayscale JPEG
        "gray.png": "gray.png",                # grayscale PNG
        "cmyk.jpg": "cmyk.jpg",                # CMYK JPEG
        "image.webp": "image.webp",            # a webp, for extension coverage
        "truncated.jpg": "truncated.jpg",      # recoverably truncated (§9.1)
        "not_an_image.jpg": "not_an_image.jpg",  # genuinely corrupt -> pred: null
    }
    for src_name, dst_name in borrowed.items():
        shutil.copy(FIXTURES / src_name, d / dst_name)

    # A non-image file with a look-alike name; must never appear in the output.
    (d / "readme.txt").write_text("not an image", encoding="utf-8")

    return d


def _run(image_dir: Path, out: Path, ckpt: Path) -> list[dict]:
    P.main(["--image_dir", str(image_dir), "--out", str(out), "--ckpt", str(ckpt),
            "--device", "cpu", "--quiet"])
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# find_images
# --------------------------------------------------------------------------- #

def test_find_images_recurses_and_filters_by_extension(fixture_dir):
    found = P.find_images(fixture_dir)
    assert len(found) == 10  # everything above except readme.txt
    assert all(p.suffix.lower() in P.IMAGE_EXTENSIONS for p in found)
    assert not any(p.name == "readme.txt" for p in found)
    assert any("nested" in p.parts for p in found)


def test_find_images_is_sorted(fixture_dir):
    found = P.find_images(fixture_dir)
    assert found == sorted(found)


# --------------------------------------------------------------------------- #
# End-to-end smoke
# --------------------------------------------------------------------------- #

def test_smoke_scores_every_supported_image_without_crashing(fixture_dir, fake_ckpt, tmp_path):
    out = tmp_path / "preds.json"
    results = _run(fixture_dir, out, fake_ckpt)

    assert len(results) == 10
    for row in results:
        assert set(row) == {"image_path", "pred", "domain_flag"}
        assert isinstance(row["image_path"], str)
        assert row["pred"] is None or (isinstance(row["pred"], float) and 0.0 <= row["pred"] <= 1.0)
        assert row["domain_flag"] in (None, "non_photographic")


def test_readable_awkward_inputs_get_a_real_prediction(fixture_dir, fake_ckpt, tmp_path):
    """Alpha PNG, grayscale, CMYK, and recoverably-truncated all score --
    exactly the §9.1 list, minus the one file that's genuinely unreadable."""
    out = tmp_path / "preds.json"
    results = _run(fixture_dir, out, fake_ckpt)
    by_name = {Path(r["image_path"]).name: r for r in results}

    for name in ("alpha.png", "gray.jpg", "gray.png", "cmyk.jpg", "image.webp", "truncated.jpg"):
        assert by_name[name]["pred"] is not None, name


def test_genuinely_corrupt_file_gets_null_not_a_crash(fixture_dir, fake_ckpt, tmp_path):
    out = tmp_path / "preds.json"
    results = _run(fixture_dir, out, fake_ckpt)
    by_name = {Path(r["image_path"]).name: r for r in results}
    assert by_name["not_an_image.jpg"]["pred"] is None


def test_corrupt_file_does_not_prevent_other_images_from_scoring(fixture_dir, fake_ckpt, tmp_path, capsys):
    """A batch containing an unreadable file must not shrink or corrupt the
    scores for the readable images alongside it (src/evaluate.py's own
    run_grid has the identical contract)."""
    out = tmp_path / "preds.json"
    results = _run(fixture_dir, out, fake_ckpt)
    n_null = sum(1 for r in results if r["pred"] is None)
    assert n_null == 1  # only not_an_image.jpg
    err = capsys.readouterr().err
    assert "not_an_image.jpg" in err  # warned, not silent


def test_deterministic_same_dir_same_output(fixture_dir, fake_ckpt, tmp_path):
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    results_a = _run(fixture_dir, out_a, fake_ckpt)
    results_b = _run(fixture_dir, out_b, fake_ckpt)
    assert results_a == results_b


def test_predict_function_matches_cli(fixture_dir, fake_ckpt, tmp_path):
    out = tmp_path / "preds.json"
    direct = P.predict(fixture_dir, out, ckpt=fake_ckpt, device="cpu", quiet=True)
    from_disk = json.loads(out.read_text(encoding="utf-8"))
    assert direct == from_disk


def test_batching_does_not_change_results(fixture_dir, fake_ckpt, tmp_path):
    """Batch size must not change which image gets which path/flag, and must
    not move a prediction beyond float32 rounding noise.

    Not exact ``==`` on the floats: a (1, N) matmul and a (64, N) batched
    matmul inside the real LinearHead can legitimately pick different
    internal BLAS code paths, and float addition is not associative, so a
    difference at the last bit or two (~1e-7, float32 machine epsilon) is
    expected and environment-dependent -- it showed up on Colab's Linux BLAS
    and not on a Windows/CPU run, which is exactly the kind of thing an
    exact-equality assertion should never have depended on. 1e-4 is well
    above that noise floor and would still catch a real batching bug (wrong
    slot, stale state carried across batches, etc.), which would not produce
    a difference this small.
    """
    out_1 = tmp_path / "bs1.json"
    out_big = tmp_path / "bsbig.json"
    r1 = P.predict(fixture_dir, out_1, ckpt=fake_ckpt, device="cpu", batch_size=1, quiet=True)
    r2 = P.predict(fixture_dir, out_big, ckpt=fake_ckpt, device="cpu", batch_size=64, quiet=True)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a["image_path"] == b["image_path"]
        assert a["domain_flag"] == b["domain_flag"]
        if a["pred"] is None or b["pred"] is None:
            assert a["pred"] == b["pred"]
        else:
            assert a["pred"] == pytest.approx(b["pred"], abs=1e-4)


def test_empty_directory_raises_a_clear_error(tmp_path, fake_ckpt):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        P.predict(empty, tmp_path / "out.json", ckpt=fake_ckpt, device="cpu", quiet=True)


def test_nonexistent_image_dir_rejected_by_cli(tmp_path, fake_ckpt):
    with pytest.raises(SystemExit):
        P.main(["--image_dir", str(tmp_path / "does_not_exist"),
                "--out", str(tmp_path / "out.json"), "--ckpt", str(fake_ckpt)])


def test_output_is_a_plain_json_array_matching_the_spec_shape(fixture_dir, fake_ckpt, tmp_path):
    out = tmp_path / "preds.json"
    _run(fixture_dir, out, fake_ckpt)
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert all(
        isinstance(r, dict) and "image_path" in r and "pred" in r and "domain_flag" in r
        for r in raw
    )


def test_unreadable_file_gets_a_null_domain_flag_too(fixture_dir, fake_ckpt, tmp_path):
    out = tmp_path / "preds.json"
    results = _run(fixture_dir, out, fake_ckpt)
    by_name = {Path(r["image_path"]).name: r for r in results}
    assert by_name["not_an_image.jpg"]["domain_flag"] is None


def test_flat_graphic_image_is_flagged_non_photographic(fake_ckpt, tmp_path):
    """A rendered-graphic stand-in (a handful of flat color blocks, no
    photographic noise) should trip the domain guard; the 3 random-noise
    fixtures from ``fixture_dir`` (real camera-photo stand-ins) should not."""
    d = tmp_path / "images"
    d.mkdir()

    graphic = Image.new("RGB", (256, 256), (255, 255, 255))
    for i, color in enumerate([(20, 90, 200), (200, 40, 40), (30, 160, 60)]):
        box = (10, 10 + i * 70, 200, 70 + i * 70)
        Image.new("RGB", (box[2] - box[0], box[3] - box[1]), color).convert("RGB")
        for y in range(box[1], box[3]):
            for x in range(box[0], box[2]):
                graphic.putpixel((x, y), color)
    graphic.save(d / "diagram.png")

    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8), "RGB")
    photo.save(d / "noise_photo.jpg", quality=95)

    out = tmp_path / "preds.json"
    results = _run(d, out, fake_ckpt)
    by_name = {Path(r["image_path"]).name: r for r in results}

    assert by_name["diagram.png"]["domain_flag"] == "non_photographic"
    assert by_name["noise_photo.jpg"]["domain_flag"] is None


# --------------------------------------------------------------------------- #
# Real CLI invocation (subprocess), to catch argv/import-time issues the
# in-process calls above can't
# --------------------------------------------------------------------------- #

def test_cli_runs_as_a_subprocess(fixture_dir, fake_ckpt, tmp_path, monkeypatch):
    """Runs the actual ``python predict.py ...`` entry point in a subprocess.
    Still offline: the stub is installed via a sitecustomize-free trick --
    monkeypatching in-process doesn't reach a subprocess, so this run uses the
    real ClipBackbone import path but never calls .embed on a real model; it
    only checks the process starts, parses argv, and reads/writes files
    correctly by exercising --help, which needs no model at all."""
    result = subprocess.run(
        [sys.executable, "predict.py", "--help"],
        cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--image_dir" in result.stdout
    assert "--out" in result.stdout
