# Robust AIGC Image Detection

> Detectors don't fail because AI images are hard to spot. They fail because
> JPEG re-encoding, resizing and blur destroy the low-level artifacts they were
> trained to read. So: train on the degraded distribution, and give the model an
> explicit estimate of *how* degraded its input is.

Every architectural choice traces back to that sentence. The full build plan is
in [PLAN.md](PLAN.md); the running decision log is in [NOTES.md](NOTES.md).

## Status

| Phase | State |
|---|---|
| 1 — evaluation harness | **done** |
| 2 — data | not started |
| 3 — CLIP linear-probe baseline | not started |
| 4 — augmentation | not started |
| 5 — full model (artifact branch + degradation head + gate) | not started |
| 6 — calibration | not started |
| 7 — `predict.py`, demo | not started |

The measurement layer is built before the thing it measures, on purpose: you
cannot iterate on robustness you cannot measure.

## Install

```bash
pip install -r requirements.txt
```

Phase 1 needs only numpy, pandas, pillow, scikit-learn, tqdm and pyyaml. The
torch/CLIP pins are for Phase 3 onward.

## Phase 1 — the robustness harness

A model is evaluated over a grid of 19 cells: clean, 14 single
(transform, severity) pairs across six degradation families, and four composed
chains standing in for real redistribution paths (`resize 0.25 → blur 0.5 →
jpeg 30` is "worst case"). See [PLAN.md §3.1](PLAN.md).

Per cell it reports AUROC, AP, acc@0.5 and TPR@FPR=1%. The two numbers that
matter:

```
robustness_gap = AUROC(clean) − mean(AUROC(all transformed cells))    ↓ better
worst_case     = min(AUROC over all cells)                           ↑ better
```

Clean AUROC is never to be reported on its own.

### Run it

```bash
# once Phase 2 has built manifests
python -m src.evaluate --ckpt runs/baseline.pt --split test --out results/baseline/
```

Writes `grid.csv`, `report.md` (the markdown table), `summary.json` (aggregates
plus the full run config) and `scores.csv` (raw per-image scores, for the §11
error analysis) into the output directory.

Useful flags: `--cells clean jpeg` to restrict the grid, `--limit N` for a
seeded class-stratified subset, `--cache-dir` to cache transformed images as
lossless PNG, `--seed` for the stochastic cells.

### Acceptance check (no data or model required)

```bash
python -m scripts.make_dummy_fixture --out data/fixtures/dummy --n 400 --seed 0
python -m src.evaluate --model dummy_random \
    --manifest data/fixtures/dummy/manifest.csv --out results/dummy/
```

A random-scoring model must land at chance in all 19 cells. Committed result in
[results/dummy/report.md](results/dummy/report.md): AUROC ∈ [0.471, 0.596],
mean 0.502, against a null SD of 0.0289 at this sample size — i.e. the whole grid
is inside ±3.3σ of 0.5. Over 20 seeds the empirical SD across cells is 0.0288
against a theoretical 0.0289, so the harness reproduces the Mann–Whitney null,
not merely a number near a half.

The fixture is generated with no real/fake signal in it (content seeded from the
image index, never the label; filenames carry no label either), which makes this
a null test of the harness rather than of the model: a cell far from 0.5 can only
mean labels leaked into the scores.

### Reproducibility

Stochastic cells (`noise`, `jitter`) seed from
`blake2b(base_seed, image_id, cell_name)` rather than a global RNG stream, so a
cell's output is a pure function of those three things. Evaluation order, batch
size, and whether the transform cache is warm cannot change a single pixel —
each is pinned by a test.

```bash
python -m pytest        # 84 tests
```
