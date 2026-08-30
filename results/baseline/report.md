# Robustness grid — baseline

Model `clip_linear(/content/drive/MyDrive/aigc/checkpoints/baseline.pt)` on `data/manifests/test.csv` (6810 images, 19 cells, seed 0).

Reproduce: `python -m src.evaluate --model clip_linear --ckpt /content/drive/MyDrive/aigc/checkpoints/baseline.pt --split test --out results/baseline/ --device cuda`

## Headline (§3.2)

| Metric | Value |
|---|---|
| Clean AUROC | 0.9810 |
| Mean transformed AUROC (family-balanced) | 0.9710 |
| **Robustness gap** ↓ | **0.0099** |
| **Worst cell AUROC** ↑ | **0.9484** (`composed_resize0.25+blur0.5+jpeg30`) |
| Mean transformed AUROC (flat, §3.2 literal) | 0.9689 |
| Robustness gap (flat) | 0.0120 |
| Mean AUROC, single transforms | 0.9714 |
| Mean AUROC, composed chains | 0.9603 |
| Cells | 19 (18 transformed, 7 families) |

`robustness_gap = AUROC(clean) − mean(family mean AUROC)`, lower is better. The headline weights each degradation family equally rather than each cell, so being good at JPEG alone (4 of 14 single cells) cannot mask fragility elsewhere; the flat cell-weighted mean §3.2 specifies is reported beside it. `worst_case = min(AUROC)` over all cells, higher is better. Clean AUROC is never to be read on its own (§13).

## Per-cell

| Cell | Chain | n | AUROC | AP | acc@0.5 | TPR@FPR=1% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `clean` | `clean` | 6810 | 0.9810 | 0.9973 | 0.9245 | 0.7398 |  |
| `jpeg_90` | `jpeg90` | 6810 | 0.9765 | 0.9966 | 0.9216 | 0.7542 |  |
| `jpeg_70` | `jpeg70` | 6810 | 0.9710 | 0.9958 | 0.9157 | 0.7250 |  |
| `jpeg_50` | `jpeg50` | 6810 | 0.9677 | 0.9953 | 0.8921 | 0.6680 |  |
| `jpeg_30` | `jpeg30` | 6810 | 0.9642 | 0.9948 | 0.8875 | 0.6352 |  |
| `blur_0.5` | `blur0.5` | 6810 | 0.9803 | 0.9971 | 0.9181 | 0.7400 |  |
| `blur_1.0` | `blur1.0` | 6810 | 0.9789 | 0.9969 | 0.8900 | 0.7415 |  |
| `blur_2.0` | `blur2.0` | 6810 | 0.9742 | 0.9963 | 0.8621 | 0.7022 |  |
| `resize_0.5` | `resize0.5` | 6810 | 0.9806 | 0.9972 | 0.8906 | 0.7645 |  |
| `resize_0.25` | `resize0.25` | 6810 | 0.9742 | 0.9963 | 0.8098 | 0.7465 |  |
| `noise_0.02` | `noise0.02` | 6810 | 0.9700 | 0.9957 | 0.9164 | 0.6705 |  |
| `noise_0.05` | `noise0.05` | 6810 | 0.9604 | 0.9943 | 0.8815 | 0.6528 |  |
| `noise_0.1` | `noise0.1` | 6810 | 0.9501 | 0.9928 | 0.8317 | 0.6222 |  |
| `jitter_0.2` | `jitter0.2` | 6810 | 0.9753 | 0.9964 | 0.9214 | 0.7157 |  |
| `center_crop_0.8` | `center_crop0.8` | 6810 | 0.9766 | 0.9967 | 0.9351 | 0.7315 |  |
| `composed_blur1.0+jpeg70` | `blur1.0+jpeg70` | 6810 | 0.9677 | 0.9953 | 0.9007 | 0.6932 |  |
| `composed_resize0.5+jpeg50` | `resize0.5+jpeg50` | 6810 | 0.9669 | 0.9952 | 0.8943 | 0.6710 |  |
| `composed_jitter0.2+jpeg30` | `jitter0.2+jpeg30` | 6810 | 0.9582 | 0.9938 | 0.8799 | 0.5978 |  |
| `composed_resize0.25+blur0.5+jpeg30` | `resize0.25+blur0.5+jpeg30` | 6810 | 0.9484 | 0.9923 | 0.7570 | 0.5590 |  |

## By family

| Family | Cells | Mean AUROC | Min AUROC |
|---|---:|---:|---:|
| `clean` | 1 | 0.9810 | 0.9810 |
| `jpeg` | 4 | 0.9699 | 0.9642 |
| `blur` | 3 | 0.9778 | 0.9742 |
| `resize` | 2 | 0.9774 | 0.9742 |
| `noise` | 3 | 0.9602 | 0.9501 |
| `jitter` | 1 | 0.9753 | 0.9753 |
| `center_crop` | 1 | 0.9766 | 0.9766 |
| `composed` | 4 | 0.9603 | 0.9484 |

## Two findings this grid doesn't show

Full methodology, bootstrap CIs, and the pre-fix comparison run:
`results/tpr_analysis/report.md`. Full per-generator table:
`results/baseline/per_generator.md`.

### (a) Aggregate AUROC hides a large operating-point spread

Per-generator clean AUROC is a tight **0.9697–0.9949** band across all seven
test generators. Per-generator clean **TPR@FPR=1%** ranges **0.511**
(FLUX.1-dev, held out) to **0.848** (MidJourney, held out) — and up to 0.893
for Aura (trained-on), the widest single point. One unseen generator is caught
barely half as often as another at a strict 1% false-positive budget, and
AUROC — the metric this grid leads with — shows none of it. Held-out status
alone does not predict this: MidJourney (held out) is the second-easiest
generator overall, ahead of four trained-on generators. See
`results/baseline/per_generator.md` for the full table.

### (b) TPR@FPR=5% is the headline robustness metric here, not AUROC

The AUROC `robustness_gap` above (0.0099) sits *under* the AUROC null SD at
this sample size (0.0108, `results/baseline/summary.json`) — statistically
indistinguishable from a model with no gap at all. TPR@FPR=5% resolves what
AUROC can't: family-balanced gap **0.0530**, 95% CI **[0.0369, 0.0632]** —
tight and clearly excluding zero, on both this run and the pre-fix run
(`results/tpr_analysis/report.md`).

TPR@FPR=1% was tried first and rejected as the headline: with 810 reals, the
1% threshold rests on ~8 images, so its gap estimate (0.0391) carries a 95% CI
of **[-0.0017, 0.1066]** — crosses zero, too noisy to report as a finding on
its own. TPR@FPR=5% rests on ~40 images at threshold and resolves cleanly;
that's the number to read as the robustness headline going forward.

Worst cell under TPR@5%: `composed_resize0.25+blur0.5+jpeg30`, **0.731** vs.
**0.911** clean — a real, resolved 18-point drop that the AUROC column for the
same cell (0.9484 vs. 0.9810, a 3-point drop) understates by 6x.

**Degradation-gap and clean-difficulty are orthogonal.** FLUX.1-dev has the
*smallest* TPR@1% degradation gap of all seven generators (0.0012, CI spans
negative) despite being the *hardest* generator clean (0.511). Aura has the
*largest* significant gap (0.0695) despite being the *easiest* clean (0.893).
Whatever makes FLUX.1-dev hard to catch is present already on clean images and
does not compound under degradation — it is a detection-difficulty problem,
not a robustness problem, and the two need separate fixes.
