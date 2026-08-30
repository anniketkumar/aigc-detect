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

---

## Phase 2 — data (done)

Built: `src/data/{imageio,normalize,resample,sources,manifest}.py`,
`scripts/{download_data,build_blocklist,make_data_stats,audit_resample,bench_throughput,make_load_fixtures}.py`,
18 real image fixtures, 156 new tests. **261 passing, 1 self-documenting skip.**

### Was the audit's sampling valid? No. Were its findings? Yes.

`fetch_audit_sample.py` loops `range(shards) × range(row_groups)` — it read
**validation shards 0–5, row groups 0–2**, and none of the 249 `train` shards.
0.75% of one split, all from the head.

Re-drawn: 20,000 rows at 189 independent positions across the whole train split
(HF datasets-server `/rows`, which serves arbitrary offsets for ~nothing).

| | audit | re-sample |
|---|---|---|
| `is_1024sq` | 0.9814 | **0.9807** |
| `megapixels` | 0.9814 | **0.9802** |
| `is_square` | 0.9806 | **0.9787** |

Class share varies by at most **1.6 pp** across deciles of file position, so the
parquet is not class-ordered — the specific failure mode that would have
invalidated everything. Container format checked separately by counting magic
bytes in random 1.5 MB windows (Snappy declines to compress already-compressed
image data, so the files sit verbatim in the parquet): PNG share 0.326 against a
predicted ~0.35, at a cost of 27 MB. Full write-up in
`results/audit_sampling_verification.md`.

### Things that did not survive contact with the data

- **GenImage is unusable.** Every mirrored generator is fixed low-res: BigGAN
  128², ADM/glide/VQDM all 256², 100% of sampled rows. Real sources are
  768–1152. A 512 crop yields zero images; a 224 crop makes resolution a perfect
  class signal again, inverted. Replaced with six 1024px generators, each
  verified by sampling real row dimensions.
- **Pexels reintroduced the leak it was hired to remove.** Added as the
  polished-photography control for the semantic confound. Its sources are 18 KB
  at 768p, so normalized files came out at a median 16 KB against 66–103 KB
  elsewhere: separable from every other real source by **file size alone at
  AUROC 1.0000**, and real-vs-AI `n_bytes` went to 0.7120. Swapped for Unsplash
  (130 KB/image): `n_bytes` 0.5017, source signature 0.5108. The task-F
  regression test caught this, which is the entire argument for writing it.
- **Both SID_Set sources were reading the same rows.** `saberzl/SID_Set` puts
  real, full_synthetic and tampered in one split. The downloader stamped each
  source's own label on every row it read, so "OpenImagesV7" came back with a
  median geometry of 1024×1024 when SID_Set's real class is 3.85% square — about
  two thirds of the supposedly-real images were synthetic. Caught by the source
  geometry table in `data_stats.md`; fixed with a per-row `label_values` filter
  (`rejected: {'wrong_class': 83}` confirms it fires).

### Bugs the tests caught in my own code

- **LSH banding was not exact.** 4 bands × 16 bits, claimed exact at threshold 6
  by pigeonhole. Wrong: 6 bit errors spread 2+2+1+1 touch all four bands. The
  condition is `n_bands > threshold`, so it needs ≥ 7. Now 8 × 8. The
  brute-force comparison test found three missed pairs out of 20 planted.
- **Pillow "recovers" a badly truncated PNG as a 100% black image.** No
  exception, correct dimensions, valid RGB — a solid black rectangle would have
  entered training carrying a real label. Added a constant-row padding measure;
  recoveries above 50% padding are rejected, and the fraction is recorded either
  way.
- **The near-duplicate fixture was unphotographic.** Three low-frequency
  sinusoids leave most of pHash's 64 DCT coefficients near zero, so their signs
  are noise and a plain 2× rescale moved the hash 12 bits. Looked like the
  threshold-6 calibration failing; was the fixture. Measured on the real
  corpus — unrelated pairs ≥ 20 bits, JPEG q30 and 4× rescale both move it 0 —
  and rebuilt the fixture with 14 components to match.

### Decisions

| Question | Decision |
|---|---|
| Crop size | 512. Every source clears it; 224 would make a real-source crop 6% of the frame against 87% for a 256px generator — a scene-scale confound as bad as the original. |
| Crop offset | Multiple of 16 (the 4:2:0 MCU), so inherited DCT grid phase cannot differ by class. |
| Crop location | Random (seeded), not centre. Generators centre their subject; a centre crop frames subject for one class and background for the other. |
| Final JPEG quality | Constant q95 4:2:0. A constant has AUROC 0.5 *by construction* — provable rather than measured. |
| AI first generation | Sampled from the real class's measured (quality, subsampling) joint — median q93, spread 68–100, subsampling 40/26/31% across 4:2:0 / 4:2:2 / 4:4:4. Bimodal, so sampled rather than fitted. |
| Resume key | Source content hash, not row position. Re-sharding a repo would otherwise silently duplicate everything. |
| Raw files | Never written. Row groups are normalized in memory; only the 512² JPEG lands. 19 GB transferred → 1.7 GB resident. |

### Known gaps

- **DALL·E Advanced (8,843 images) is not hashed.** It exists only inside
  WildFake's ~700 GB of ModelScope zips. COCO val2017 is hashed in full (5,000
  sha256 + 5,000 pHash). The gap is recorded in the blocklist's `gaps` field,
  printed by `manifest.build`, and mitigated by a registry-level DALL·E denylist
  test. Residual risk is low — no source in the registry is DALL·E-derived — but
  it is a gap, not a solved problem.
- **pHash does not catch crops.** A 10%-per-side crop moves the hash ~20 bits,
  indistinguishable from an unrelated image. Inherited from the audit's own
  calibration and unchanged.
- **The n_bytes bound is weak at the current corpus size.** At 140 real + 160 AI
  the null SD is 0.033, so 0.60 is 3.0σ. The test skips with that message rather
  than implying more confidence than the sample supports.

## Phase 4 — augmentation (sampler + Colab cells built; run pending)

Phase 3 landed at `e7dced2`: `results/baseline/` (clean AUROC 0.9810,
AUROC `robustness_gap` 0.0099 — under its own null SD, i.e. noise) and
`results/tpr_analysis/report.md` (the metric that actually resolves: TPR@FPR=5%
gap 0.0530, 95% CI [0.0369, 0.0632]). Cached features live on Drive at
`{DRIVE_ROOT}/features/{train,val,test}`; the baseline checkpoint at
`{DRIVE_ROOT}/checkpoints/baseline.pt`.

Built while Phase 3 was still running, so needed neither its checkpoint nor
the downloaded corpus: `src/data/augment.py` (25 tests,
`tests/test_augment.py`); the `--augment-copies`/`--augment-seed` extension to
`scripts/cache_features.py` (8 tests, `tests/test_cache_features.py`, CLIP
backbone stubbed — no network). Now added, once Phase 3's real paths were
known: thirteen cells appended to `scripts/colab_setup.ipynb` — cache
`train` with `--augment-copies 4`, reuse the clean `val` cache as-is, retrain
into `{DRIVE_ROOT}/checkpoints/aug.pt`, evaluate into `results/aug/` on the
same held-out `test` split, then `scripts/tpr_gap_analysis.py --run aug
--run baseline` into `results/tpr_analysis_aug/`.

Two targets, both from the Phase 3 TPR analysis, both checked in the notebook:

- **TPR@FPR=5% robustness gap** — currently 0.0530, 95% CI [0.0369, 0.0632],
  tight enough to be a real signal (unlike the AUROC gap, 0.0099, which sits
  under its own null SD of 0.0108). This is the primary target — the
  family-averaged operating-point cost of degradation, and the number
  augmentation exists to move.
- **Per-generator TPR spread** — currently 0.511 (FLUX.1-dev, held out) to
  0.893 (Aura) clean TPR@1%, a spread AUROC's tight 0.9697–0.9949 per-generator
  band completely hides (`results/baseline/per_generator.md`). Checked
  separately so a Phase 4 win can't just mean "the easy generators got easier
  while FLUX.1-dev stayed put" — narrowing the average gap while widening this
  spread would not be the win it looks like.

The ablation reports both, not just the family-averaged gap.

### What it is

A RandAugment-style sampler over the same six families as the Phase 1 eval
grid (`src/transforms.py`), reusing its op implementations verbatim so
training-time "jpeg" and eval-time "jpeg" cannot drift into two encoders.
Severities are drawn continuously and strictly wider than the eval grid's
fixed points (§6) — jpeg q ∈ [20,95], blur σ ∈ [0,3.0], resize scale ∈
[0.2,1.0], noise σ ∈ [0,0.12], jitter ∈ [0,0.3], crop keep ∈ [0.7,1.0] — so
the eval grid stays a genuine held-out measurement rather than a distribution
the head has already seen exact points of. 1-3 families per image, random
order, ~20% left clean, one seed derived from `(base_seed, image_id,
copy_index)` via `transforms.derive_seed` — same reproducibility contract as
the eval harness, extended with a copy index instead of a cell name.

Each call also returns a `DegradationLabel` (which families fired, at what
severity, plus a fixed 12-dim normalized vector). §6/§7.2 call this "free"
supervision for a degradation head; HANDOFF.md cuts Phase 5 (artifact branch +
fusion gate), so nothing consumes it yet. Kept anyway — it costs nothing
alongside the image, and it is the one piece of the original Phase 4 spec
worth not having to redo if the gate comes back in scope.

`scripts/cache_features.py --augment-copies K` embeds K augmented copies per
image instead of the image itself (§6 option (a): "precompute embeddings for
K augmented copies... much faster and nearly as good" than running CLIP live
in the dataloader). Writes `degradation.npy` (N·K, 12) alongside the usual
three files; `paths.json` entries become `"{path}#aug{i}"` so every embedding
still traces to a source file for error analysis.

### Still pending

- The Colab cells are drafted but not yet executed — `results/aug/` and
  `results/tpr_analysis_aug/` don't exist yet.
- `K=4` and `--augment-seed 0` are the plan's defaults, untuned — nothing to
  tune against until the first augmented run is in.
- The Phase 4 cells assume the *same* Colab session as Phase 3 (they reuse
  `data/corpus`, ephemeral on `/content`); starting fresh means re-running
  Cell 2's download first.
