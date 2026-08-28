# Robustness grid — dummy_random

Model `dummy_random(seed=0)` on `data/fixtures/dummy/manifest.csv` (400 images, 19 cells, seed 0).

Reproduce: `python -m src.evaluate --model dummy_random --manifest data/fixtures/dummy/manifest.csv --out results/dummy --name dummy_random --quiet`

## Headline (§3.2)

| Metric | Value |
|---|---|
| Clean AUROC | 0.5428 |
| Mean transformed AUROC (family-balanced) | 0.4994 |
| **Robustness gap** ↓ | **0.0434** |
| **Worst cell AUROC** ↑ | **0.4714** (`composed_resize0.25+blur0.5+jpeg30`) |
| Mean transformed AUROC (flat, §3.2 literal) | 0.4997 |
| Robustness gap (flat) | 0.0430 |
| Mean AUROC, single transforms | 0.4984 |
| Mean AUROC, composed chains | 0.5044 |
| Cells | 19 (18 transformed, 7 families) |

`robustness_gap = AUROC(clean) − mean(family mean AUROC)`, lower is better. The headline weights each degradation family equally rather than each cell, so being good at JPEG alone (4 of 14 single cells) cannot mask fragility elsewhere; the flat cell-weighted mean §3.2 specifies is reported beside it. `worst_case = min(AUROC)` over all cells, higher is better. Clean AUROC is never to be read on its own (§13).

## Per-cell

| Cell | Chain | n | AUROC | AP | acc@0.5 | TPR@FPR=1% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `clean` | `clean` | 400 | 0.5428 | 0.5315 | 0.5475 | 0.0150 |  |
| `jpeg_90` | `jpeg90` | 400 | 0.4879 | 0.5202 | 0.4875 | 0.0350 |  |
| `jpeg_70` | `jpeg70` | 400 | 0.4987 | 0.4875 | 0.5175 | 0.0000 |  |
| `jpeg_50` | `jpeg50` | 400 | 0.5149 | 0.5233 | 0.5175 | 0.0250 |  |
| `jpeg_30` | `jpeg30` | 400 | 0.4825 | 0.4999 | 0.4875 | 0.0300 |  |
| `blur_0.5` | `blur0.5` | 400 | 0.4786 | 0.4846 | 0.4850 | 0.0050 |  |
| `blur_1.0` | `blur1.0` | 400 | 0.4720 | 0.4995 | 0.4750 | 0.0100 |  |
| `blur_2.0` | `blur2.0` | 400 | 0.4809 | 0.4894 | 0.4925 | 0.0100 |  |
| `resize_0.5` | `resize0.5` | 400 | 0.5173 | 0.5067 | 0.5100 | 0.0100 |  |
| `resize_0.25` | `resize0.25` | 400 | 0.5093 | 0.4964 | 0.5000 | 0.0050 |  |
| `noise_0.02` | `noise0.02` | 400 | 0.5356 | 0.5437 | 0.5350 | 0.0400 |  |
| `noise_0.05` | `noise0.05` | 400 | 0.5361 | 0.5221 | 0.5100 | 0.0100 |  |
| `noise_0.1` | `noise0.1` | 400 | 0.4748 | 0.4778 | 0.4825 | 0.0000 |  |
| `jitter_0.2` | `jitter0.2` | 400 | 0.5023 | 0.5012 | 0.5050 | 0.0100 |  |
| `center_crop_0.8` | `center_crop0.8` | 400 | 0.4867 | 0.4795 | 0.4825 | 0.0000 |  |
| `composed_blur1.0+jpeg70` | `blur1.0+jpeg70` | 400 | 0.4968 | 0.5018 | 0.5050 | 0.0100 |  |
| `composed_resize0.5+jpeg50` | `resize0.5+jpeg50` | 400 | 0.5203 | 0.5173 | 0.5250 | 0.0100 |  |
| `composed_jitter0.2+jpeg30` | `jitter0.2+jpeg30` | 400 | 0.5292 | 0.5459 | 0.4975 | 0.0400 |  |
| `composed_resize0.25+blur0.5+jpeg30` | `resize0.25+blur0.5+jpeg30` | 400 | 0.4714 | 0.5102 | 0.4750 | 0.0200 |  |

## By family

| Family | Cells | Mean AUROC | Min AUROC |
|---|---:|---:|---:|
| `clean` | 1 | 0.5428 | 0.5428 |
| `jpeg` | 4 | 0.4960 | 0.4825 |
| `blur` | 3 | 0.4772 | 0.4720 |
| `resize` | 2 | 0.5133 | 0.5093 |
| `noise` | 3 | 0.5155 | 0.4748 |
| `jitter` | 1 | 0.5023 | 0.5023 |
| `center_crop` | 1 | 0.4867 | 0.4867 |
| `composed` | 4 | 0.5044 | 0.4714 |
