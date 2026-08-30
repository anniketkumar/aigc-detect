# TPR@FPR robustness gap

Companion to `results/baseline/report.md`'s AUROC-based `robustness_gap`. See module docstring (`scripts/tpr_gap_analysis.py`) for why this metric and how the bootstrap CI is constructed.

Reproduce: `python -m scripts.tpr_gap_analysis --out results\tpr_analysis_aug` (B=800 main, B=300 per-generator, seed=0)

## baseline

`results\baseline\scores.csv` -- n=6810 (real=810, ai=6000)

| Cell | Family | TPR@1% | 95% CI | TPR@5% | 95% CI |
|---|---|---:|---|---:|---|
| `clean` | clean | 0.7398 | [0.6628, 0.8337] | 0.9107 | [0.8853, 0.9270] |
| `jpeg_90` | jpeg | 0.7542 | [0.6210, 0.7923] | 0.8910 | [0.8685, 0.9045] |
| `jpeg_70` | jpeg | 0.7250 | [0.5888, 0.7834] | 0.8615 | [0.8337, 0.8952] |
| `jpeg_50` | jpeg | 0.6680 | [0.5870, 0.7312] | 0.8575 | [0.8193, 0.8822] |
| `jpeg_30` | jpeg | 0.6352 | [0.5873, 0.7097] | 0.8372 | [0.8065, 0.8613] |
| `blur_0.5` | blur | 0.7400 | [0.6550, 0.8307] | 0.9087 | [0.8752, 0.9227] |
| `blur_1.0` | blur | 0.7415 | [0.6390, 0.8107] | 0.9005 | [0.8840, 0.9157] |
| `blur_2.0` | blur | 0.7022 | [0.5598, 0.7933] | 0.8782 | [0.8373, 0.9038] |
| `resize_0.5` | resize | 0.7645 | [0.6917, 0.8222] | 0.8983 | [0.8813, 0.9147] |
| `resize_0.25` | resize | 0.7465 | [0.6725, 0.7957] | 0.8785 | [0.8542, 0.8967] |
| `noise_0.02` | noise | 0.6705 | [0.6318, 0.7613] | 0.8563 | [0.8188, 0.8715] |
| `noise_0.05` | noise | 0.6528 | [0.6002, 0.7185] | 0.8153 | [0.7882, 0.8368] |
| `noise_0.1` | noise | 0.6222 | [0.5677, 0.6847] | 0.7670 | [0.7452, 0.8000] |
| `jitter_0.2` | jitter | 0.7157 | [0.6438, 0.7877] | 0.8645 | [0.8443, 0.8967] |
| `center_crop_0.8` | center_crop | 0.7315 | [0.6025, 0.8042] | 0.8777 | [0.8393, 0.9065] |
| `composed_blur1.0+jpeg70` | composed | 0.6932 | [0.5499, 0.7537] | 0.8358 | [0.8183, 0.8678] |
| `composed_resize0.5+jpeg50` | composed | 0.6710 | [0.4952, 0.7308] | 0.8443 | [0.8023, 0.8712] |
| `composed_jitter0.2+jpeg30` | composed | 0.5978 | [0.4854, 0.6650] | 0.8000 | [0.7590, 0.8349] |
| `composed_resize0.25+blur0.5+jpeg30` | composed | 0.5590 | [0.4601, 0.6380] | 0.7308 | [0.7068, 0.7907] |

**Family-balanced gap** (`clean - mean(family means)`, same shape as the AUROC gap):

- TPR@1% gap = **0.0391**, bootstrap median 0.0471, 95% CI [-0.0017, 0.1066]
- TPR@5% gap = **0.0530**, bootstrap median 0.0503, 95% CI [0.0369, 0.0632]

### Per generator (real pool shared)

| Generator | n | Held out | clean TPR@1% | clean TPR@5% | TPR@1% gap | 95% CI | TPR@5% gap | 95% CI |
|---|---:|---|---:|---:|---:|---|---:|---|
| Aura | 300 |  | 0.8933 | 0.9733 | 0.0695 | [0.0400, 0.1121] | 0.0396 | [0.0239, 0.0546] |
| FLUX.1-dev | 800 | yes | 0.5112 | 0.8375 | 0.0012 | [-0.0585, 0.1256] | 0.0770 | [0.0400, 0.1000] |
| Gemini-nano-banana | 2000 | yes | 0.6720 | 0.8635 | 0.0508 | [0.0083, 0.1205] | 0.0724 | [0.0483, 0.0903] |
| MidJourney | 2000 | yes | 0.8480 | 0.9740 | 0.0437 | [-0.0031, 0.1026] | 0.0338 | [0.0236, 0.0418] |
| Mobius | 300 |  | 0.8267 | 0.9500 | 0.0217 | [-0.0233, 0.0691] | 0.0298 | [0.0152, 0.0515] |
| RealVisXL-V4.0 | 300 |  | 0.7967 | 0.9167 | 0.0379 | [-0.0002, 0.1030] | 0.0469 | [0.0222, 0.0663] |
| SDXL | 300 |  | 0.7833 | 0.8900 | 0.0206 | [-0.0047, 0.0754] | 0.0299 | [0.0076, 0.0451] |

## aug

`results\aug\scores.csv` -- n=6810 (real=810, ai=6000)

| Cell | Family | TPR@1% | 95% CI | TPR@5% | 95% CI |
|---|---|---:|---|---:|---|
| `clean` | clean | 0.7590 | [0.6160, 0.8225] | 0.8995 | [0.8727, 0.9128] |
| `jpeg_90` | jpeg | 0.7303 | [0.6466, 0.8100] | 0.8837 | [0.8580, 0.9037] |
| `jpeg_70` | jpeg | 0.7142 | [0.6163, 0.7925] | 0.8768 | [0.8482, 0.8990] |
| `jpeg_50` | jpeg | 0.6978 | [0.5880, 0.7590] | 0.8518 | [0.8355, 0.8785] |
| `jpeg_30` | jpeg | 0.7063 | [0.5840, 0.7363] | 0.8492 | [0.8183, 0.8732] |
| `blur_0.5` | blur | 0.7440 | [0.6364, 0.8277] | 0.8955 | [0.8657, 0.9133] |
| `blur_1.0` | blur | 0.7522 | [0.6395, 0.8173] | 0.8920 | [0.8727, 0.9047] |
| `blur_2.0` | blur | 0.7313 | [0.5903, 0.8018] | 0.8698 | [0.8495, 0.8937] |
| `resize_0.5` | resize | 0.7705 | [0.6786, 0.8177] | 0.8908 | [0.8652, 0.9085] |
| `resize_0.25` | resize | 0.7495 | [0.6383, 0.7885] | 0.8733 | [0.8435, 0.8953] |
| `noise_0.02` | noise | 0.7308 | [0.6472, 0.7564] | 0.8533 | [0.8248, 0.8792] |
| `noise_0.05` | noise | 0.6788 | [0.6118, 0.7487] | 0.8250 | [0.7923, 0.8522] |
| `noise_0.1` | noise | 0.6647 | [0.5765, 0.7092] | 0.7957 | [0.7630, 0.8175] |
| `jitter_0.2` | jitter | 0.7148 | [0.6205, 0.7733] | 0.8623 | [0.8308, 0.8947] |
| `center_crop_0.8` | center_crop | 0.7230 | [0.6178, 0.7646] | 0.8653 | [0.8298, 0.8937] |
| `composed_blur1.0+jpeg70` | composed | 0.6923 | [0.6087, 0.7850] | 0.8623 | [0.8332, 0.8828] |
| `composed_resize0.5+jpeg50` | composed | 0.7130 | [0.5760, 0.7869] | 0.8463 | [0.8150, 0.8772] |
| `composed_jitter0.2+jpeg30` | composed | 0.6277 | [0.5234, 0.7144] | 0.8228 | [0.7885, 0.8430] |
| `composed_resize0.25+blur0.5+jpeg30` | composed | 0.6408 | [0.5618, 0.6855] | 0.7875 | [0.7635, 0.8260] |

**Family-balanced gap** (`clean - mean(family means)`, same shape as the AUROC gap):

- TPR@1% gap = **0.0429**, bootstrap median 0.0519, 95% CI [-0.0217, 0.0931]
- TPR@5% gap = **0.0402**, bootstrap median 0.0377, 95% CI [0.0253, 0.0511]

### Per generator (real pool shared)

| Generator | n | Held out | clean TPR@1% | clean TPR@5% | TPR@1% gap | 95% CI | TPR@5% gap | 95% CI |
|---|---:|---|---:|---:|---:|---|---:|---|
| Aura | 300 |  | 0.9133 | 0.9633 | 0.0720 | [0.0215, 0.1065] | 0.0250 | [0.0124, 0.0399] |
| FLUX.1-dev | 800 | yes | 0.5375 | 0.8375 | 0.0290 | [-0.0646, 0.1306] | 0.0720 | [0.0331, 0.0958] |
| Gemini-nano-banana | 2000 | yes | 0.6725 | 0.8315 | 0.0543 | [-0.0139, 0.1092] | 0.0546 | [0.0320, 0.0690] |
| MidJourney | 2000 | yes | 0.8865 | 0.9760 | 0.0388 | [-0.0388, 0.0866] | 0.0213 | [0.0129, 0.0287] |
| Mobius | 300 |  | 0.8333 | 0.9433 | 0.0281 | [-0.0227, 0.0614] | 0.0221 | [0.0085, 0.0440] |
| RealVisXL-V4.0 | 300 |  | 0.8033 | 0.9100 | 0.0285 | [-0.0055, 0.0703] | 0.0347 | [0.0098, 0.0546] |
| SDXL | 300 |  | 0.8033 | 0.8900 | 0.0325 | [-0.0123, 0.0689] | 0.0231 | [-0.0015, 0.0369] |

## Paired bootstrap: aug vs baseline

Same 6810 images scored by both runs; one resampled image set per replicate applied to both, B=2000, seed=0. `diff = aug - baseline`, negative means smaller (more robust) under `aug`.

**Family-balanced gap (primary target):**

| Metric | baseline gap | aug gap | point diff | bootstrap median diff | 95% CI | 95% CI |
|---|---:|---:|---:|---:|---|---|
| TPR@1% gap | 0.0391 | 0.0429 | +0.0038 | -0.0043 | [-0.0432, +0.0416] | includes zero |
| TPR@5% gap | 0.0530 | 0.0402 | -0.0128 | -0.0126 | [-0.0252, +0.0019] | includes zero |

**Per-generator clean-TPR spread, max-min (secondary target):**

| Metric | baseline spread | aug spread | point diff | bootstrap median diff | 95% CI | 95% CI |
|---|---:|---:|---:|---:|---|---|
| clean TPR@1% spread | 0.3821 | 0.3758 | -0.0062 | +0.0021 | [-0.0508, +0.0429] | includes zero |
| clean TPR@5% spread | 0.1365 | 0.1445 | +0.0080 | +0.0088 | [-0.0244, +0.0410] | includes zero |
