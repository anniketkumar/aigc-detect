# Image Signal — engineering writeup

*The long-form version of [README.md](README.md): what was measured, what was
built on top of the measurements, what the two ablations were actually worth,
and what's left open.*

---

## The finding, up front

Aggregate metrics said this problem was basically solved. Per-generator
metrics said otherwise, and the gap between the two is the actual result.

The CLIP-linear-probe detector scores AUROC 0.97–0.99 on every one of seven
image generators, including three held fully out of training. Read as a
single number, that's a strong, uniform detector. But AUROC is threshold-free
and rank-based — it doesn't describe what happens at the one operating point
a real moderation system would actually use. At a fixed 1% false-positive
budget, the same model's true-positive rate ranges from **53.75% on
FLUX.1-dev** to **88.65% on MidJourney** — a 0.35 spread the aggregate number
completely hides. The model isn't uniformly strong; it's uniformly confident,
which is a different and more dangerous property.

That finding sits on top of a second one: the standard benchmark corpus for
this task (SID_Set) separates real from AI-generated **perfectly on container
format alone** — real images are 100% JPEG, AI images 100% PNG, so
`if container == PNG: predict AI` scores 100.00% accuracy having read zero
pixels. A leaderboard built on this corpus without a leak audit is measuring
file-format detection, not generation detection.

Two standard fixes were then tried against the first problem, and **neither
one worked.** Training-time degradation augmentation (Phase 4) improves the
family-averaged robustness gap ~24% in the right direction, but a
paired-bootstrap test says that improvement **is not statistically
significant at this sample size** (Δ = -0.0128, 95% CI [-0.0252, +0.0019]),
and it **does not move the per-generator spread at all** (Δ = +0.0021, 95%
CI [-0.0508, +0.0429]). A CLIP + frequency fusion branch (Phase 5), built
only after a measurement said its precondition held, lands **below** the
plain baseline once trained and evaluated at full scale. Both are reported
as results rather than hedged around: the standard fix helps the metric
everyone reports and does nothing for the metric that actually explains
where the model fails, and the architectural fix does nothing at all.

---

## Composite score

`Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust`, computed for all three
checkpoints (AUC_robust = family-balanced mean transformed AUROC over the
19-cell grid). One number that refuses to let clean accuracy and robustness
be traded against each other silently:

| Checkpoint | Model | AUC_clean | AUC_robust | Final Score |
|---|---|---:|---:|---:|
| baseline | `clip_linear` | 0.9810 | 0.9710 | **0.9760** |
| aug | `clip_linear` | 0.9779 | 0.9702 | 0.9740 |
| fusion | `clip_freq_fusion` | 0.9786 | 0.9692 | 0.9739 |

Augmentation costs 0.0031 of clean AUROC (0.9810 → 0.9779) and buys back
0.0022 of robustness gap (0.0099 → 0.0077) — on this formula, baseline comes
out 0.0020 ahead. Fusion comes out 0.0021 behind. The honest reading is that
**the entire table is one statistical tie**: the full spread is 0.0021
against an AUROC null SD of 0.0108 at this sample size. Nothing separates.
`runs/baseline.pt` is what ships, and it is already `predict.py`'s default —
not because it won, but because neither ablation earned the right to replace
it.

The deeper case for augmentation lives in the TPR@FPR=1%/5% analysis below —
a paired-bootstrap comparison on the metric that actually tracks the
moderation operating point, not the threshold-free average this formula
uses. Both are real; they answer different questions, and showing both beats
letting one quietly stand in for the other.

---

## The measurement layer

**Built before the model it measures.** Phase 1 shipped a 19-cell robustness
eval grid (clean + 14 single degradations across six families: JPEG, blur,
resize, noise, jitter, crop + 4 composed chains standing in for real
redistribution paths) before any model existed, and it's validated against a
null model (random scores land inside ±3.3σ of chance across 20 seeds) and a
planted-weakness model (fails in exactly the injected cell) — so a robustness
number from this harness is trustworthy on its own terms, not just
plausible-looking.

**A leak audit that changed the pipeline, not just a footnote.** The Phase 2
audit found 13 metadata channels that separate the classes without touching
pixels (container format, geometry, ICC profiles, byte size). Every image now
passes one canonical decode — strip all metadata, crop at native resolution,
one JPEG pass at a matched quality distribution — closing 12 of 13. The one
open channel is enforced by a blocking test, not a comment.

**Two metrics, deliberately kept apart.** AUROC's `robustness_gap` (clean −
mean transformed AUROC) sits at 0.0099, under its own null SD of 0.0108 —
statistically indistinguishable from zero. TPR@FPR=5%'s gap is 0.0530, 95% CI
[0.0369, 0.0632] — clearly real. Same model, same test set; the metric
choice is the difference between "no robustness problem" and "a real one."
Both get reported because a threshold-free metric and an operating-point
metric answer different questions, and picking only the one that looks better
would be the easiest way to lie with statistics without writing a false
sentence.

**A paired bootstrap, not overlapping confidence intervals.** Two runs'
marginal 95% CIs overlapping does not mean their difference is
non-significant — that comparison throws away the correlation between the
two runs' scores on the same images. Images are resampled once per replicate
and that same resample applied to both runs (`scripts/tpr_gap_analysis.py
--pair`), so the reported CI is on the actual difference distribution. This
is what made it possible to report the augmentation result honestly instead
of eyeballing "the intervals are close, call it a tie."

**Scale:** 355 tests, 6810-image held-out test set, 3 generators held fully
out of training (two closed-commercial, one adversarially chosen because its
raw form is the leak-confounded set the Phase 2 audit was written about).

## The approach: refusing the easy number

The idea here isn't a new architecture — it's declining to let the
convenient number stand in for the hard question. Three places that shows up
concretely:

- **Per-generator TPR@FPR as the headline metric, not AUROC.** Every prior
  pass at this reported AUROC per generator and called a 0.97–0.99 band
  "robust." Re-reading the same scores at a fixed false-positive budget is
  what surfaces the 0.35 spread — the metric change, not a model change,
  is what found the real failure mode.
- **A dataset audit treated as a regression test, not a one-time finding.**
  The container-format leak, the geometry leak, and a second leak
  *introduced by an earlier leak fix* (a "professional" real-photo source
  that was itself perfectly separable by file size) are all now blocking
  tests, so a future data-pipeline change can't silently reopen them.
- **A null result reported as a result.** Phase 5 below is the clearest
  case: a measured precondition, a built branch, a full-scale evaluation,
  and no gain. The alternative — shipping it as "built, evaluation pending"
  — was available and would have read better. It would also have been the
  same failure mode as reporting AUROC and calling it robust.

## Phase 5: a measured bet that didn't pay off

The standard hybrid-detector argument is to fuse high-level CLIP semantics
with low-level frequency signal, because each survives transforms the other
doesn't. Rather than take that on faith, the precondition got measured
first: **is a frequency-domain signal actually non-redundant with what CLIP
already reads?**

A 2-feature, no-CLIP texture probe (`scripts/aesthetic_probe.py`) —
`hf_energy` (mean gradient magnitude, a high-frequency/sensor-artifact
statistic) and `flat_frac` (near-flat pixel fraction, a
compression/posterization tell) — reaches **0.6085 AUROC jointly**, with
near-equal standardized weight (`hf_energy` -0.624, `flat_frac` -0.533).
Weak alone, as expected of two hand-built scalars, but that was never the
question. The question was whether it's *redundant* with CLIP. Spearman rho
between this probe's per-generator difficulty ordering and CLIP's:
**-0.143 (p = 0.760)** — statistically indistinguishable from zero. CLIP and
the texture probe don't find the same generators hard; they're reading
different signal. The error-analysis false positives corroborate it
independently: the highest-confidence real-image mistakes are all
high-production-value professional photography, i.e. CLIP is reading
composition at least in part, not purely a forensic signal a frequency
branch would read redundantly. (`results/aesthetic_probe/aesthetic_probe.md`)

On that evidence the branch was built. `src/features/frequency.py` extracts
12 FFT radial-energy rings (tuned for periodic up-sampling artifacts, not
just generic high-frequency content) plus 3 block-DCT low/mid/high band
energies, concatenated with the frozen CLIP embedding (531 dims total) and
scored by the same linear head as the baseline (`src/models/clip_fusion.py`,
registered as `clip_freq_fusion`) — no architecture change to
`src/train.py`, since it already reads embedding dimension off the cached
array's shape. Scale constants are fixed and label-blind
(`scripts/calibrate_frequency_scale.py`), so no scaler is fit on train/test
data and there's no new leakage surface to audit. 24 tests cover it.

**Then it was trained and evaluated at full scale, and it didn't work.**
Same 9,380-image train split, same 2,010-image val split, same 6,810-image
held-out test set, same 19-cell grid:

| | baseline | fusion |
|---|---:|---:|
| Clean AUROC | 0.9810 | 0.9786 |
| Robust AUROC (family-balanced) | 0.9710 | 0.9692 |
| Robustness gap ↓ | 0.0099 | 0.0095 |
| Worst cell ↑ | 0.9484 | 0.9474 |
| **Final Score** | **0.9760** | 0.9739 |

Every column is a wash or slightly worse, and every difference is far inside
the 0.0108 noise floor. The correct statement is not "fusion hurt" — it is
**"fusion changed nothing measurable, and a necessary precondition turned
out not to be a sufficient one."** Non-redundant signal existing is not the
same as a linear head being able to use it.

Two caveats worth keeping attached to that conclusion, because they bound
how strong a null it is. First, fusion was trained on **clean** fused
features only — the aug × fusion cell was never run, and that is exactly the
combination with a mechanism behind it, since frequency features are what
degradation destroys and should therefore benefit most from degraded
training data. Second, 810 real test images is the same binding constraint
that makes every other comparison here non-significant; this null is
underpowered for the same reason the augmentation result is. Full grid:
`results/fusion/report.md`. Executed run: `results/fusion/colab_run.ipynb`.

## Why it matters: moderation triage

The pitch isn't "detect fake images." It's **moderation triage**: a score
that knows its own reliability changes what a platform can safely automate.
A pristine upload and a fourth-generation repost of the same fake deserve
different confidence thresholds, not the same hard cutoff; the per-generator
spread measured here is exactly the kind of information a hard-block policy
throws away and a triage policy can act on (route
FLUX.1-dev-confidence-range cases to human review, auto-action
MidJourney-confidence-range cases). That only works if the operator knows
the spread exists — which is what this project delivers, ahead of a fully
closed detector.

## Cost and footprint

- **149.6M total parameters** — a frozen CLIP ViT-B/16 backbone plus a
  513-parameter trained linear head. The entire trainable surface is 513
  numbers; the fusion head is 532.
- **Runs on a single Colab T4.** No backbone fine-tuning, ever — the frozen-
  probe pipeline was a deliberate scope decision to fit the compute budget,
  made explicit in the plan rather than discovered under deadline pressure
  (`PLAN.md` §13: "do not fine-tune the CLIP backbone before the
  frozen-probe pipeline is fully working end to end"). Head training is ~17
  seconds; the expensive step is caching CLIP features (~4 min/split) and
  the 19-cell evaluation (~55–75 min/checkpoint).
- **Both entry points run standalone.** `predict.py` (image directory →
  `preds.json`) and `app.py` (the FastAPI adapter behind the React UI) ship
  with their trained checkpoints committed — clone, `pip install`, run. No
  data download, no retraining step.
- **Scope decisions are tracked and named, not silently dropped.** Post-hoc
  calibration (Phase 6) is cut for budget and named as such in the README.
  Phase 5 was cut on the same reasoning initially, then reopened once the
  aesthetic-probe measurement gave a concrete reason to expect it would pay
  off, then run to completion and reported as a null. Each of those three
  states is in the git history with the reasoning attached.

---

## Limitations (stated, not buried)

**810 real test images is the binding constraint on every open question
here**, not the model. The augmentation improvement, the per-generator
spread, and the Phase 5 null are each "not significant at this N" — that is
a statement about statistical power, not a claim that any effect is zero.
More held-out real images is the single highest-leverage next step for
resolving any of them. FLUX.1-dev is the floor generator before and after
both ablations; `results/error_analysis/` looks at what it's actually
missing. No calibration layer exists, so scores are a trained sigmoid
output, not a calibrated probability — the threshold-free metrics (AUROC,
TPR@FPR) don't have this problem, `acc@0.5` does.

**Non-photographic input is out of domain for both classes.** Every training
source is photographic, so diagrams, screenshots and infographics score
"AI-generated" reliably and wrongly. `src/data/domain_guard.py` flags this
without altering the score; see README's Limitations for the full reasoning.

**No number on the DALL·E Advanced reference subset.** COCO val2017 (5,000
images) was hashed for the exclusion blocklist, but the DALL·E Advanced half
has no standalone distribution and ships only inside WildFake's ~700 GB
ModelScope archives, so it could not be evaluated within bandwidth
constraints.

## What's next

Three extensions, in order of ROI:

1. **More held-out real images.** 810 is the binding constraint behind every
   non-significant result in this repo. It sharpens every existing number
   rather than adding a new one, and it is the only cheap way to find out
   whether the Phase 5 null is a real null or an underpowered one. Lowest
   effort, highest information.
2. **Train fusion on augmented features (aug × fusion).** Phase 5 trained on
   clean features only, so the one combination with a mechanism behind it
   was never run — frequency features are precisely what degradation
   destroys, so they should be most helped by seeing degraded training data.
   One cache pass plus one 17-second head retrain, reusing cells that
   already exist in `scripts/colab_setup.ipynb`. This is the cheapest
   remaining shot at making Phase 5 earn its inference cost.
3. **Bias the augmentation sampler toward FLUX.1-dev.** The per-generator
   spread is the one finding nothing has moved. Uniform degradation-family
   sampling was never targeted at the actual floor; targeting it directly is
   untested and is the only idea here that could plausibly move the metric
   that matters most.

All three are scoped rather than speculative — each is either a direct
extension of the existing eval harness or a re-run of committed code.

---

## Built with

Python, PyTorch, open_clip (CLIP ViT-B/16, frozen), scikit-learn, FastAPI +
Vite/React, Google Colab (T4). Data: SID_Set, JourneyDB, nano-banana, and
five more generator/real sources via HuggingFace parquet, streamed and
normalized in-memory (nothing raw ever written to disk).

*[GitHub repo](https://github.com/anniketkumar/aigc-detect) ·
[Results](https://github.com/anniketkumar/aigc-detect/tree/main/results)*
