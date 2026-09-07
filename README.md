# Image Signal

*Robust AI-generated image detection: measuring where a detector actually
breaks, then trying two standard fixes and reporting what they were worth.*

> Detectors don't fail because AI images are hard to spot. They fail because
> aggregate metrics hide where they actually break, and JPEG re-encoding,
> resizing and blur destroy the low-level artifacts they were trained to read.

This repo does two things, in this order: **measure** where an AI-image
detector actually fails (per generator, under real-world post-processing),
then try two standard fixes — training-time augmentation, and a CLIP +
frequency fusion branch — and report honestly how much of the problem each
one closes. The answer, for both, is "less than you'd hope," and that result
is the point rather than an embarrassment to bury.

Full build log and every design decision: [PLAN.md](PLAN.md) (the spec),
[HANDOFF.md](HANDOFF.md) (the phase-3 brief), [NOTES.md](NOTES.md) (the
running decision log — what was tried, what failed, and why),
[WRITEUP.md](WRITEUP.md) (the long-form engineering writeup). This file is
the short version.

## What's actually here

A frozen CLIP ViT-B/16 backbone + a trained linear head — deliberately the
simplest model that could work. Two ablations were run against it at full
scale, and a third phase (post-hoc calibration) was cut on budget.

| Phase | State |
|---|---|
| 1 — 19-cell robustness eval harness | done |
| 2 — leak-audited data pipeline | done |
| 3 — CLIP linear-probe baseline | done |
| 4 — training-time augmentation | done — direction-correct on one target, not significant at this N |
| 5 — CLIP + frequency fusion branch | done — **trained and evaluated at full scale; no measurable gain** |
| 6 — calibration | **cut** — not enough budget to earn its row in the ablation table |
| 7 — `predict.py`, `app.py` | done |

## Headline results

**Composite score.** `Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust`,
where `AUC_robust` is the family-balanced mean transformed AUROC over the
19-cell grid — the same headline aggregate as `robustness_gap` in
`results/*/report.md` §3.2. A single number that refuses to let clean
accuracy and robustness be traded against each other silently:

| Checkpoint | Model | AUC_clean | AUC_robust | Robustness gap ↓ | Worst cell ↑ | Final Score |
|---|---|---:|---:|---:|---:|---:|
| `runs/baseline.pt` | `clip_linear` | 0.9810 | 0.9710 | 0.0099 | 0.9484 | **0.9760** |
| `runs/aug.pt` | `clip_linear` | 0.9779 | 0.9702 | **0.0077** | **0.9526** | 0.9740 |
| `runs/fusion.pt` | `clip_freq_fusion` | 0.9786 | 0.9692 | 0.0095 | 0.9474 | 0.9739 |

**Neither ablation beats the plain baseline, and the whole table is a
statistical tie.** The full spread across all three checkpoints is 0.0021 on
Final Score, against an AUROC null SD of **0.0108** at this sample size
(`results/*/summary.json`). Nothing here separates. Augmentation buys a real
robustness-gap improvement (0.0099 → 0.0077) and the best worst-case cell,
and pays for it in clean AUROC (0.9810 → 0.9779) — a trade the 50/50 weighting
declines by 0.0020. Fusion moves nothing at all. **`runs/baseline.pt` is what
ships** — it is `predict.py`'s and `app.py`'s default — because no ablation
earned the right to replace it. Numbers: `results/baseline/report.md`,
`results/aug/report.md`, `results/fusion/report.md`.

**1. The standard corpus separates perfectly without looking at a pixel.**
SID_Set's real images are 100% JPEG, its AI images 100% PNG — a one-line rule
(`if container == PNG: predict AI`) scores 100.00% accuracy on real-vs-fake
with zero pixels read (`results/audit_sid_set.md`). Same story for geometry
(AI images are all exactly 1024²) and ICC profiles. The whole data pipeline
(`src/data/normalize.py`, `scripts/download_data.py`) exists to force every
image through one canonical decode — strip metadata, crop natively, single
JPEG pass at a matched quality — before a model ever sees it, closing 12 of
13 leak channels the Phase 2 audit found.

**2. Aggregate AUROC hides a 0.35 operating-point spread across generators.**
Per-generator AUROC on the aug checkpoint sits in a tight 0.97–0.99 band —
reads as "uniformly strong." At a 1% false-positive budget (the operating
point that matters for moderation triage, not the threshold-free rank
statistic), clean TPR ranges from **0.5375 on FLUX.1-dev** to **0.8865 on
MidJourney** — the model catches barely half of the hardest generator's
images at the same false-positive budget where it catches ~89% of the
easiest. See `results/tpr_analysis_aug/report.md` and
`results/baseline/per_generator.md`.

**3. Training-time augmentation is a partial, unproven fix.** RandAugment-
style degradation during training (`src/data/augment.py`) moves the primary
target — the family-averaged TPR@5% robustness gap — from 0.0530 to 0.0402,
about a 24% reduction, in the right direction. A **paired bootstrap** (same
6810 test images, B=2000, resampled jointly so the two runs are compared on
the same simulated sample rather than via overlapping marginal CIs) puts
that difference at **-0.0128, 95% CI [-0.0252, +0.0019] — not significant at
this N.** It does not touch the per-generator spread at all: **+0.0021, 95%
CI [-0.0508, +0.0429]**, FLUX.1-dev stays the floor. Full numbers and method:
`results/tpr_analysis_aug/report.md`, `NOTES.md` §"Phase 4".

**4. Frequency fusion had a measured precondition and still didn't pay off.**
Before building Phase 5, a 2-feature, no-CLIP texture probe
(`scripts/aesthetic_probe.py`) established the thing that would have to be
true for a fusion branch to add information rather than duplicate it: the
probe's per-generator difficulty ordering correlates with CLIP's at
**Spearman ρ = -0.143 (p = 0.760)** — statistically indistinguishable from
zero, i.e. the two read different signal
(`results/aesthetic_probe/aesthetic_probe.md`). On that evidence the branch
was built: `src/models/clip_fusion.py` (`clip_freq_fusion`) concatenates 12
FFT radial-energy rings + 3 block-DCT band energies (`src/features/frequency.py`,
531 dims total) onto the frozen CLIP embedding, scored by the same linear
head. Trained and evaluated at full scale on the same 9,380-image train and
6,810-image held-out test split, it lands at Final Score **0.9739 vs the
baseline's 0.9760** — no gain, and the difference is far inside the noise
floor. A necessary precondition turned out not to be a sufficient one. The
branch ships as working, tested, evaluated code with a null result attached,
not as an improvement.

Read together: the metric you pick determines whether this project looks
solved (AUROC), partially fixed (TPR@5% gap), or untouched (per-generator
spread) — which is the point of measuring all three instead of reporting the
one that looks best. And two architecture-level fixes, both with a plausible
story behind them, moved none of it.

## Setup

```bash
pip install -r requirements.txt
```

Phase 1 (the eval harness) needs only numpy, pandas, pillow, scikit-learn,
tqdm, pyyaml. Everything else (torch, open_clip, gradio, pyarrow/fsspec for
data acquisition) is pinned but only imported from Phase 3 onward.

All three trained checkpoints are committed (`runs/baseline.pt`,
`runs/aug.pt`, `runs/fusion.pt` — the linear head only, ~4 KB each; the CLIP
backbone itself downloads from open_clip's pretrained registry on first
use), so `predict.py` and `app.py` below work on a fresh clone with no
training or data download step.

## Run it

**`predict.py` — the deliverable.** Image directory in, `preds.json` out:

```bash
python predict.py --image_dir path/to/images --out preds.json --ckpt runs/aug.pt
```

Writes `[{"image_path": ..., "pred": 0.873}, ...]`, one entry per image,
sorted so output is byte-identical across runs and OSes. Recurses for
jpg/jpeg/png/webp/bmp, converts anything (grayscale, CMYK, alpha) through the
same decode path the eval harness uses, and never crashes on a bad file —
a genuine decode failure gets `"pred": null` with a warning on stderr rather
than stopping the run. Defaults to `runs/baseline.pt`; pass `--ckpt
runs/aug.pt` for the augmented checkpoint, `--device cuda` if you have a GPU.

**React UI — the interactive interface.** A Vite + React client with a thin
FastAPI adapter: upload an image, choose a checkpoint, drag the JPEG-quality
slider from 95 down to 30, and compare the canonical decode with the actual
JPEG re-encoding used for the displayed result. The UI offers light and dark
mode and translates the score into a careful review cue without presenting it
as a certainty.

```bash
uvicorn app:app --port 8000
# in another terminal
npm --prefix frontend run dev
```

The browser is served at `http://localhost:5173` and proxies its API calls to
the local adapter. Every step goes through the same production code paths —
`predict.py`'s decoder, the same scorer, `src/transforms.py`'s real JPEG
encode/decode (not a simulated one) — so the number the UI shows is the same
operation the eval grid measured at population scale, not a demo-only
approximation. The adapter keeps one model instance warm per checkpoint so
changing quality does not reload the backbone.

**Reproduce the measurement layer:**

```bash
python -m pytest                                          # 355 tests
python -m src.evaluate --ckpt runs/aug.pt --split test --out results/aug/
python -m src.evaluate --model clip_freq_fusion --ckpt runs/fusion.pt \
    --split test --out results/fusion/
python -m scripts.tpr_gap_analysis --run baseline results/baseline/scores.csv \
    --run aug results/aug/scores.csv \
    --pair baseline results/baseline/scores.csv aug results/aug/scores.csv \
    --out results/tpr_analysis_aug
python -m scripts.error_analysis   # results/error_analysis/
```

The full cache → train → evaluate pipeline for all three checkpoints is
staged cell-by-cell in `scripts/colab_setup.ipynb`; the executed runs are
archived beside their results (`results/*/colab_run.ipynb`).

## Limitations

- **810 real test images is the binding constraint**, not the model. Every
  non-significant result above (the TPR@5% gap improvement, the
  per-generator spread, the fusion comparison) is non-significant because
  the paired bootstrap CIs are wide at this N, not because the effect is
  known to be zero. A larger held-out real pool is the single
  highest-leverage next step for resolving any of them.
- **FLUX.1-dev is the floor and neither ablation moved it.** Same
  generator, same rank, before and after Phases 4 and 5. `NOTES.md` and
  `results/error_analysis/` look at what it's actually missing; a next step
  (untested) would bias the augmentation sampler toward FLUX.1-dev-hard
  cases specifically rather than degradation families uniformly.
- **No calibration layer.** Scores are a trained sigmoid output, not
  calibrated probabilities — `acc@0.5` in the eval reports should be read
  with that in mind; AUROC/TPR@FPR are threshold-free and don't have this
  problem.
- **`predict.py`/`app.py` run the frozen-backbone linear probe, not the
  fusion branch.** `runs/fusion.pt` is committed and
  `results/fusion/report.md` is a full 19-cell grid on the same held-out
  test split, so the branch is evaluated rather than merely built — but it
  scores below the baseline it would replace, so wiring it into either entry
  point would cost inference time (a per-image FFT + block-DCT pass) for no
  measured accuracy. It stays available via `--model clip_freq_fusion` in
  the eval harness. Fusion was also only ever trained on *clean* features:
  the aug × fusion cell was never run, and that combination is the one place
  the branch might still earn its cost. The augmentation sampler separately
  computes a `DegradationLabel` per training image (which families fired, at
  what severity) that nothing currently consumes — kept because it was free,
  in case a degradation head comes back in scope.
- **Three held-out generators, not an open set.** MidJourney, Gemini
  (nano-banana) and FLUX.1-dev are held fully out of training, but "unseen
  generator" here means these three specifically, not a guarantee about
  generators not represented in the eval at all.
- **Non-photographic input (diagrams, screenshots, infographics) is out of
  domain for both classes, not just one.** Every source in
  `src/data/sources.py` — all three real sets and all seven generators — is
  photographic; nothing rendered (flat-filled vector art, UI chrome, dense
  text) appears as real *or* fake in training. The model reliably scores this
  content "AI-generated," consistent with `results/error_analysis/note.md`'s
  finding that its signal tracks "does this look like a certain kind of
  photo" rather than generation artifacts directly — a rendered diagram is
  the extreme case of everything that already pushes a real photo toward
  that class. `predict.py`/`app.py` flag this (`domain_flag:
  "non_photographic"`, `src/data/domain_guard.py`) via a cheap, untrained
  pixel-statistic heuristic (unique-color ratio, dominant-color fraction) so
  a caller can present the score as unreliable rather than confident — the
  flag doesn't change or suppress `pred` itself, since there's no training
  signal that would make a "corrected" number meaningful either.
- **No number on the DALL·E Advanced reference subset.** COCO val2017 (5,000
  images) was hashed for the exclusion blocklist, but the DALL·E Advanced
  half has no standalone distribution and ships only inside WildFake's
  ~700 GB ModelScope archives, so it could not be evaluated within bandwidth
  constraints.

## What's next

Three extensions, in order of ROI — full reasoning in
[WRITEUP.md](WRITEUP.md#whats-next):

1. **More held-out real images.** 810 is the binding constraint behind
   *every* "not significant at this N" result in this repo — the
   augmentation gap, the per-generator spread, and now the fusion null
   result. It sharpens every existing number rather than adding a new one,
   and it is the only cheap way to find out whether the Phase 5 null result
   is a real null or just an underpowered one.
2. **Train fusion on augmented features (aug × fusion).** Phase 5 was
   trained on clean features only, so the one combination with a mechanism
   behind it — frequency features are exactly what degradation destroys, so
   they should be *most* helped by seeing degraded training data — was never
   run. One cache pass plus one 17-second head retrain, using cells that
   already exist in `scripts/colab_setup.ipynb`.
3. **Bias the augmentation sampler toward FLUX.1-dev.** The per-generator
   spread is the one finding nothing has moved. Uniform degradation-family
   sampling was never targeted at the actual floor.

## Contributions

Commit history: `git shortlog -sne`.

**[Aniket Kumar](https://github.com/anniketkumar)** — project spec
([PLAN.md](PLAN.md)) and the decision log; Phase 1 evaluation harness
(`src/transforms.py`, `src/metrics.py`, `src/evaluate.py` — the 19-cell grid,
family-balanced robustness aggregates, acceptance criteria); Phase 2 leakage
audit and data pipeline (`scripts/audit_leakage.py`, `src/data/`, the
container/geometry/ICC leak findings and the canonical-decode normalizer);
Phase 4 augmentation (`src/data/augment.py`) and the TPR@FPR analysis with
paired bootstrap (`scripts/tpr_gap_analysis.py`); the aesthetic probe and
error analysis; Phase 5 frequency fusion (`src/features/frequency.py`,
`src/models/clip_fusion.py`); `predict.py`, the FastAPI adapter in `app.py`,
and the documentation.

**[Madhavan](https://github.com/Madhavan1333)** — Phase 3 baseline: the CLIP
ViT-B/16 linear probe (`src/models/clip_baseline.py`,
`src/models/clip_backbone.py`), the training loop, and the first Colab
pipeline that made full-scale runs reproducible.

**[Aryash Mullick](https://github.com/aryashmullick)** — the entire frontend
surface: the Vite + React client (`frontend/`) including batch processing,
the robustness-slider view, sharing and the API documentation page; the
browser extension (`extension/`); and the project-brief page
(`project-brief.html`).

## Credits and licensing

Data sources, licences and attribution obligations are documented in
[NOTES.md](NOTES.md) §"Licensing". SID_Set (CC-BY-4.0) inherits
COCO/OpenImages/Flickr30k attribution requirements, carried through here.
