# NOTES

Running log of decisions, spec gaps, and things that failed. §13 asks for this;
it becomes the README's limitations section for free.

---

## Phase 1 — evaluation harness (done)

Built: `src/transforms.py`, `src/metrics.py`, `src/evaluate.py`,
`src/models/{base,dummy}.py`, `scripts/make_dummy_fixture.py`, 84 tests.
No model, training, or dataset-download code — deliberately.

### §3.3 acceptance

```bash
python -m scripts.make_dummy_fixture --out data/fixtures/dummy --n 400 --seed 0
python -m src.evaluate --model dummy_random \
    --manifest data/fixtures/dummy/manifest.csv --out results/dummy/
```

Result in `results/dummy/`: 19/19 cells present, AUROC ∈ [0.471, 0.596], mean
0.502, robustness_gap +0.043. Null SD at 200 v 200 is 0.0289, so that whole
range is inside ±3.3σ of 0.5.

Verified further, because "≈0.5" invites self-deception:

- 20 seeds × 19 cells: mean AUROC 0.5045, **empirical SD 0.0288 vs theoretical
  0.0289**, zero cells beyond 3σ. The grid reproduces the Mann-Whitney null
  distribution, not just a number near 0.5.
- One cell (`composed_jitter0.2+jpeg30`) sat above 0.529 in the first five
  seeds. Re-run at 20 seeds it came back to 0.5062 (z = 0.95). Chance, from
  eyeballing the max of 19 cells.

### The harness can also *detect* a gap

A null result alone doesn't prove the grid measures anything. `dummy_brightness`
(score = mean luminance) on a fixture with a planted brightness offset:

| cell | AUROC |
|---|---|
| clean | 0.800 |
| jpeg 90/70/50/30 | 0.800 |
| blur, resize, noise (all) | 0.800 |
| center_crop 0.8 | 0.792 |
| **jitter 0.2** | **0.631** |
| **composed jitter+jpeg30** | **0.663** |

Mean-preserving transforms leave a brightness feature untouched; brightness
jitter destroys it; cropping perturbs it slightly. The grid localised the
failure to the exactly correct cell. That is the end-to-end check.

### Spec gaps found in §3, and what was decided

| Gap | Decision |
|---|---|
| Resample filter for `resize` / `center_crop` unspecified | Bicubic both directions. NEAREST/LANCZOS would respectively destroy or manufacture high-frequency structure — exactly the signal an artifact branch reads. |
| `noise` clipping unspecified | Clip to [0,1] and re-quantize to uint8. Makes the noise slightly non-Gaussian near black/white; that is what a real 8-bit pipeline does. |
| `jitter: [0.20]` — how many factors, in what order? | Three independent draws from U(0.8, 1.2), applied brightness → contrast → saturation in fixed order. torchvision's ColorJitter shuffles the order; a random order is a second hidden variance source for no measurement benefit. |
| `center_crop: [0.80]` — linear or area? | Linear (each side ×0.8 = 64% of area), matching torchvision. |
| Composed chains aren't in `TRANSFORM_GRID`, so their CSV naming is undefined | `composed_<op><sev>+<op><sev>`, e.g. `composed_resize0.25+blur0.5+jpeg30`. No `:` — these double as cache directory names on Windows. |
| `robustness_gap` — do composed chains count as "transformed cells"? | Yes. Noted in code that the mean is unweighted, so JPEG dominates it with 4 of 14 single cells. Per-family means go in the CSV so it can be reweighted without a re-run. |
| `worst_case = min(AUROC over all cells)` includes clean | Implemented literally. Clean should never be the argmin; if it is, that is worth seeing rather than hiding. |
| `TPR@FPR=1%` — interpolation convention | Conservative: best TPR at an FPR that genuinely does not exceed the budget. No ROC interpolation, which would report a TPR no threshold delivers. |
| `TPR@FPR=1%` needs ≥100 negatives to resolve at all | Below that it silently becomes "TPR at zero false positives". The cell self-flags in a `notes` column instead of quoting the number bare. |
| `acc@0.5` assumes a calibrated score | 0.5 is meaningless for a raw-logit model, and §8 calibration is Phase 6. Reported as asked, but AUROC/AP are the numbers to read before then. |
| §3.3 CLI has `--ckpt` but the acceptance model has no weights | `--ckpt` made optional; added `--model` to select from a registry. |
| §3.3 needs `--split test`, but manifests are Phase 2 | `--split` resolves `data/manifests/<split>.csv` and `--manifest` takes an explicit path. Missing manifest prints the fixture command rather than a traceback. |

### Deliberate deviations

- **Transformed-image cache is off by default**, though §3.3 says to cache. The
  mechanism is fully implemented (`--cache-dir`, lossless PNG, sharded, atomic
  writes, keyed on image + op chain + seed) and a test asserts cached and
  uncached runs produce identical grids. But the harness iterates image-major:
  each original file is read once and transformed into all 19 cells, so caching
  saves only the transform, not a load. For a cheap model on small images the
  PNG encode/decode costs more than the blur it avoids, and the disk cost is
  ~19× the dataset. It becomes the right call in Phase 3 — real GPU model,
  high-resolution inputs — and can be switched on with one flag then.
- **Reproducibility is per-(image, cell), not a global RNG stream.** Each
  stochastic cell seeds from `blake2b(base_seed, image_id, cell_name)`. A global
  stream would make results depend on evaluation order, worker count, and
  whether a run was resumed. `test_evaluation_order_does_not_affect_results`
  and `test_batch_size_does_not_change_results` pin this.
- **The dummy model hashes pixels, not filenames.** A filename-keyed random
  model returns the same score in every cell, so the grid would show 19
  identical AUROCs — and a bug that fed the clean image to every cell would be
  invisible. Pixel-keyed means identical AUROCs across cells are themselves a
  bug signal.
- **The fixture contains no real/fake signal** (`--signal none`: content seeded
  from the index, never the label; filenames carry no label either). So the
  acceptance run is a null test of the *harness*, not of the model. A cell far
  from 0.5 can only mean the harness leaked labels into the scores.

### Repo notes

- The spec file was named `Robust AIGC Image Detection — Build Plan.md`;
  renamed to `PLAN.md` to match the §1 layout it defines.
- `git rev-parse --show-toplevel` pointed at `C:/Users/Aniket Kumar` — an empty,
  zero-commit git repo at the home directory, almost certainly accidental.
  Committing from the project would have staged the entire home directory.
  Initialised a repo at the project root instead; the stray `~/.git` is
  untouched and can be deleted.
- §1 puts the project in `aigc-detect/`; it lives in `TikTok TechJam/`.
- `configs/` is empty. Writing config files for models that do not exist yet
  would be speculative, and §13 forbids unmeasured complexity.

### Open, for later phases

- §3.2's unweighted mean over transformed cells is JPEG-heavy (4 of 14 single
  cells). If the ablation table ends up hinging on the gap, consider reporting a
  family-balanced mean alongside it. Not changing the headline definition — the
  spec fixed it, and consistency across phases matters more than elegance.
- 128×128 fixture images make `resize 0.25` (→32px) very aggressive. Real
  SID_Set images are high-res, so the eval grid will behave differently there.
  Do not tune anything against fixture numbers.
- `pandas 3.0.3` and `opencv 5.0` are both very recent majors. If Colab pins
  older ones, `requirements.txt` will need a compatible lower bound.
