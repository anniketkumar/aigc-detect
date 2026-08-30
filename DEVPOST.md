# Robust AIGC Image Detection — Devpost Draft

*Paste-ready draft, structured around the stated judging criteria (Technical
Execution 35%, Innovation 20%, Impact 20%, Feasibility 15%, Presentation
10%). Replace bracketed placeholders before submitting.*

---

## The finding, up front

Aggregate metrics said this problem was basically solved. Per-generator
metrics said otherwise, and the gap between the two is the actual submission.

Our CLIP-linear-probe detector scores AUROC 0.97–0.99 on every one of seven
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

We tried the standard fix — training-time degradation augmentation — for the
first problem. It works, partially: the family-averaged robustness gap
improves ~24% in the right direction, but a paired-bootstrap significance
test says that improvement **is not statistically significant at this
sample size** (Δ = -0.0128, 95% CI [-0.0252, +0.0019]), and it **does not
move the per-generator spread at all** (Δ = +0.0021, 95% CI [-0.0508,
+0.0429]). We're reporting that as the result, not hedging around it: the
fix helps the metric everyone reports and does nothing for the metric that
actually explains where the model fails.

---

## Technical Execution — 35%

**A measurement layer built before the model it measures.** Phase 1 shipped
a 19-cell robustness eval grid (clean + 14 single degradations across six
families: JPEG, blur, resize, noise, jitter, crop + 4 composed chains
standing in for real redistribution paths) before any model existed, and it's
validated against a null model (random scores land inside ±3.3σ of chance
across 20 seeds) and a planted-weakness model (fails in exactly the injected
cell) — so a robustness number from this harness is trustworthy on its own
terms, not just plausible-looking.

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
We report both because a threshold-free metric and an operating-point metric
answer different questions, and picking only the one that looks better would
be the easiest way to lie with statistics without writing a false sentence.

**A paired bootstrap, not overlapping confidence intervals.** Two runs'
marginal 95% CIs overlapping does not mean their difference is
non-significant — that comparison throws away the correlation between the
two runs' scores on the same images. We resample images once per replicate
and apply that same resample to both runs (`scripts/tpr_gap_analysis.py
--pair`), so the reported CI is on the actual difference distribution. This
is what let us report the augmentation result honestly instead of eyeballing
"the intervals are close, call it a tie."

**Scale:** 324 tests, 6810-image held-out test set, 3 generators held fully
out of training (two closed-commercial, one adversarially chosen because its
raw form is the leak-confounded set the Phase 2 audit was written about).

## Innovation — 20%

The innovation here isn't a new architecture — it's refusing to let the
easy number stand in for the hard question. Three places that shows up
concretely:

- **Per-generator TPR@FPR as the headline metric, not AUROC.** Every prior
  pass at this reported AUROC per generator and called a 0.97–0.99 band
  "robust." Re-reading the same scores at a fixed false-positive budget is
  what surfaces the 0.35 spread — the metric change, not a model change,
  is what found the real failure mode.
- **An aesthetic-probe control.** A two-feature texture classifier (gradient
  energy + flat-pixel fraction — no CLIP, no learned features) was run on
  the same split to check whether the CLIP head's per-generator ordering was
  secretly "detects stylized images" rather than "detects generation
  artifacts." Spearman correlation between the two orderings: -0.14 —
  effectively uncorrelated, so the CLIP signal is not aesthetics in disguise.
- **A dataset audit treated as a regression test, not a one-time finding.**
  The container-format leak, the geometry leak, and a second leak
  *introduced by an earlier leak fix* (a "professional" real-photo source
  that was itself perfectly separable by file size) are all now blocking
  tests, so a future data-pipeline change can't silently reopen them.

## Impact — 20%

We're not pitching "detect fake images" — every submission to this problem
pitches that. We're pitching **moderation triage**: a score that knows its
own reliability changes what a platform can safely automate. A pristine
upload and a fourth-generation repost of the same fake deserve different
confidence thresholds, not the same hard cutoff; the per-generator spread we
measured is exactly the kind of information a hard-block policy throws away
and a triage policy can act on (route FLUX.1-dev-confidence-range cases to
human review, auto-action MidJourney-confidence-range cases). That only
works if the platform knows the spread exists — which is what this
submission actually delivers, ahead of a fully closed detector.

## Feasibility — 15%

- **149.6M total parameters** (frozen CLIP ViT-B/16 backbone + a 513-parameter
  trained linear head) against a 2B budget — roughly 13x under, with the
  entire trainable surface being 513 numbers.
- **Runs on a single Colab T4.** No backbone fine-tuning, ever — the frozen-
  probe pipeline was a deliberate scope cut to fit the compute and time
  budget, made explicit in the plan rather than discovered under deadline
  pressure (`PLAN.md` §13: "do not fine-tune the CLIP backbone before the
  frozen-probe pipeline is fully working end to end").
- **Both deliverables run standalone.** `predict.py` (image directory →
  `preds.json`) and `app.py` (the live demo) both ship with their trained
  checkpoints committed — clone, `pip install`, run. No data download, no
  retraining step, for a judge to see either working.
- **Two phases were cut on purpose, not silently dropped.** The artifact
  branch / fusion gate (Phase 5) and post-hoc calibration (Phase 6) are
  named as cut in the README, with the reasoning (effort-to-payoff ranking
  said the eval harness and the augmentation ablation mattered more at this
  budget) rather than left to look like an oversight.

## Presentation — 10%

- `README.md` — setup, the three headline findings, run instructions, honest
  limitations.
- `app.py` — a live Gradio demo: upload an image, drag a JPEG-quality
  slider, watch the score move, on the exact re-encode path the eval grid
  measured.
- `results/` — every table in this document is a committed, regeneratable
  file (`results/audit_sid_set.md`, `results/tpr_analysis_aug/report.md`,
  `results/error_analysis/`), not a number transcribed once and left
  unverifiable.
- `NOTES.md` — the full decision log, including what didn't work, for anyone
  who wants the "why" behind a choice this document doesn't have room for.

---

## Limitations (stated, not buried)

**810 real test images is the binding constraint on both open questions in
this submission**, not the model. The augmentation improvement and the
per-generator spread are each "not significant at this N" — that is a
statement about statistical power, not a claim that either effect is zero.
More held-out real images is the single highest-leverage next step for
resolving either one. FLUX.1-dev is the floor generator both before and
after augmentation; `results/error_analysis/` looks at what it's actually
missing. No calibration layer exists, so scores are a trained sigmoid
output, not a calibrated probability — the threshold-free metrics (AUROC,
TPR@FPR) don't have this problem, `acc@0.5` does.

---

## Built with

Python, PyTorch, open_clip (CLIP ViT-B/16, frozen), scikit-learn, Gradio,
Google Colab (T4). Data: SID_Set, JourneyDB, nano-banana, and five more
generator/real sources via HuggingFace parquet, streamed and normalized
in-memory (nothing raw ever written to disk).

*Links: [GitHub repo](#) · [Demo video](#) · [Live results](#)*
