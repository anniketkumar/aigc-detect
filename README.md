# Image Signal

*Robust AIGC Image Detection — TikTok TechJam 2026*

> Detectors don't fail because AI images are hard to spot. They fail because
> aggregate metrics hide where they actually break, and JPEG re-encoding,
> resizing and blur destroy the low-level artifacts they were trained to read.

This repo does two things, in this order: **measure** where an AI-image
detector actually fails (per generator, under real-world post-processing),
then try one standard fix (training-time augmentation) and report honestly
how much of the problem it closes.

Full build log and every design decision: [PLAN.md](PLAN.md) (the spec),
[HANDOFF.md](HANDOFF.md) (phase-3 brief), [NOTES.md](NOTES.md) (the running
decision log — what was tried, what failed, and why). This file is the
short version.

## What's actually here

A frozen CLIP ViT-B/16 backbone + a trained linear head — deliberately the
simplest model that could work, not the original full design. The plan
called for an artifact branch, a degradation head and a fusion gate (Phase
5) plus post-hoc calibration (Phase 6); both were cut once measurement
showed the eval harness and the augmentation ablation mattered more than
architecture at this budget (PLAN.md §13's effort ranking). Phase 5 was
later reopened once a cheap measurement (the aesthetic-probe check, below)
gave a concrete reason to expect fusion would pay off — the branch is now
built and unit-tested, but a real trained-and-evaluated result needs a
full-scale Colab run we don't expect to land before submission, so it ships
as scoped, ready-to-run future work rather than a finished ablation row (see
"What's next"). What shipped:

| Phase | State |
|---|---|
| 1 — 19-cell robustness eval harness | done |
| 2 — leak-audited data pipeline | done |
| 3 — CLIP linear-probe baseline | done |
| 4 — training-time augmentation | done |
| 5 — artifact branch + fusion gate | **built, not yet evaluated at scale** — reopened after the aesthetic-probe measurement, code + tests done, full-scale run is future work |
| 6 — calibration | **cut** — not enough budget to earn its row in the ablation table |
| 7 — `predict.py`, `app.py` | done |

## Headline results

**Organizers' scoring formula.** `Final Score = 0.50 × AUC_clean + 0.50 ×
AUC_robust` (AUC_robust = family-balanced mean transformed AUROC over the
19-cell grid, matching the "robustness_gap" headline in
`results/baseline/report.md` §3.2):

| Checkpoint | AUC_clean | AUC_robust | Final Score |
|---|---:|---:|---:|
| `runs/baseline.pt` | 0.9810 | 0.9710 | **0.9760** |
| `runs/aug.pt` | 0.9779 | 0.9702 | 0.9741 |

**Baseline scores 0.0019 higher than aug on the organizers' own formula.**
Augmentation trades clean AUROC away (0.9810 → 0.9779) for a robustness-gap
improvement (0.0099 → 0.0077) too small to cover that trade once both terms
are weighted equally — we're not hiding this because it's inconvenient for
the augmentation story below. **We would submit `runs/baseline.pt`** — it is
already `predict.py`'s default checkpoint — **not `runs/aug.pt`, if this
formula is what's being scored.** The finer-grained case for augmentation
(TPR@FPR=5% gap, paired bootstrap, headline #3 below) is real but doesn't
survive being collapsed into a single clean/robust average at this sample
size; both views are reported so the metric choice is visible rather than
picked for us. Numbers: `results/baseline/report.md`,
`results/aug/report.md`.

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

Read together: the metric you pick determines whether this project looks
solved (AUROC), partially fixed (TPR@5% gap), or untouched (per-generator
spread) — which is the point of measuring all three instead of reporting the
one that looks best.

## Setup

```bash
pip install -r requirements.txt
```

Phase 1 (the eval harness) needs only numpy, pandas, pillow, scikit-learn,
tqdm, pyyaml. Everything else (torch, open_clip, gradio, pyarrow/fsspec for
data acquisition) is pinned but only imported from Phase 3 onward.

Both trained checkpoints are committed (`runs/baseline.pt`, `runs/aug.pt` —
the linear head only, ~4 KB each; the CLIP backbone itself downloads from
open_clip's pretrained registry on first use), so `predict.py` and `app.py`
below work on a fresh clone with no training or data download step.

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
python -m scripts.tpr_gap_analysis --run baseline results/baseline/scores.csv \
    --run aug results/aug/scores.csv \
    --pair baseline results/baseline/scores.csv aug results/aug/scores.csv \
    --out results/tpr_analysis_aug
python -m scripts.error_analysis   # results/error_analysis/
```

## Limitations

- **810 real test images is the binding constraint**, not the model. Every
  non-significant result above (the TPR@5% gap improvement, the
  per-generator spread) is non-significant because the paired bootstrap CIs
  are wide at this N, not because the effect is known to be zero. A larger
  held-out real pool is the single highest-leverage next step for resolving
  either question.
- **FLUX.1-dev is the floor and augmentation didn't move it.** Same
  generator, same rank, before and after Phase 4. `NOTES.md` and
  `results/error_analysis/` look at what it's actually missing; a next step
  (untested) would bias the augmentation sampler toward FLUX.1-dev-hard
  cases specifically rather than degradation families uniformly.
- **No calibration layer.** Scores are a trained sigmoid output, not
  calibrated probabilities — `acc@0.5` in the eval reports should be read
  with that in mind; AUROC/TPR@FPR are threshold-free and don't have this
  problem.
- **`predict.py`/`app.py` still run the single frozen-backbone linear
  probe, not the fusion branch.** `src/models/clip_fusion.py`
  (`clip_freq_fusion`) concatenates 12 FFT radial-energy rings + 3 block-DCT
  band energies onto the CLIP embedding and is built, unit-tested, and
  validated end-to-end on a 113-image local smoke manifest — but that's
  pipeline correctness only, not a real signal on whether fusion helps (the
  smoke fusion checkpoint actually scored *below* smoke baseline on that
  tiny slice, read as noise from the toy sample size, not a result). No
  trained `clip_freq_fusion` checkpoint is shipped, so it's not wired into
  either deliverable. A full ~18.2k-image Colab run — already staged in
  `scripts/colab_setup.ipynb` — is the next step and is scoped as future
  work, not expected before submission. The augmentation sampler separately
  computes a `DegradationLabel` per training image (which families fired, at
  what severity) that nothing currently consumes — kept because it was
  free, in case a degradation head comes back in scope alongside fusion.
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
  that class. `predict.py`/`app.py` now flag this (`domain_flag:
  "non_photographic"`, `src/data/domain_guard.py`) via a cheap, untrained
  pixel-statistic heuristic (unique-color ratio, dominant-color fraction) so
  a caller can present the score as unreliable rather than confident — the
  flag doesn't change or suppress `pred` itself, since there's no training
  signal that would make a "corrected" number meaningful either.
- **No number on the organizers' reference validation subset.** We hashed
  COCO val2017 (5,000 images) for the exclusion blocklist, but the DALL-E
  Advanced half has no standalone distribution and ships only inside
  WildFake's ~700GB ModelScope archives, so we could not report a number on
  it within bandwidth constraints.

## What's next

Two extensions, in order of ROI, if this continues past submission — full
writeup in [DEVPOST.md](DEVPOST.md#whats-next):

1. **Run the Phase 5 fusion branch (`clip_freq_fusion`) at full scale.**
   Code, tests, and the Colab notebook cells are done; only the ~18.2k-image
   training run is outstanding, and that run is the actual result.
2. **More held-out real images.** 810 is the binding constraint behind both
   "not significant at this N" findings above (the augmentation gap, the
   per-generator spread) — a larger real pool sharpens existing numbers
   rather than adding new ones.
