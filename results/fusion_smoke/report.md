# Robustness grid — fusion_smoke

Model `clip_freq_fusion(runs\fusion_smoke.pt)` on `data/manifests/test.csv` (113 images, 19 cells, seed 0).

Reproduce: `python -m src.evaluate --model clip_freq_fusion --ckpt runs/fusion_smoke.pt --manifest data/manifests/test.csv --out results/fusion_smoke/ --device cpu`

## Headline (§3.2)

| Metric | Value |
|---|---|
| Clean AUROC | 0.7873 |
| Mean transformed AUROC (family-balanced) | 0.7874 |
| **Robustness gap** ↓ | **-0.0001** |
| **Worst cell AUROC** ↑ | **0.7277** (`noise_0.1`) |
| Mean transformed AUROC (flat, §3.2 literal) | 0.7889 |
| Robustness gap (flat) | -0.0016 |
| Mean AUROC, single transforms | 0.7875 |
| Mean AUROC, composed chains | 0.7937 |
| Cells | 19 (18 transformed, 7 families) |

`robustness_gap = AUROC(clean) − mean(family mean AUROC)`, lower is better. The headline weights each degradation family equally rather than each cell, so being good at JPEG alone (4 of 14 single cells) cannot mask fragility elsewhere; the flat cell-weighted mean §3.2 specifies is reported beside it. `worst_case = min(AUROC)` over all cells, higher is better. Clean AUROC is never to be read on its own (§13).

## Per-cell

| Cell | Chain | n | AUROC | AP | acc@0.5 | TPR@FPR=1% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `clean` | `clean` | 113 | 0.7873 | 0.9343 | 0.1858 | 0.1304 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jpeg_90` | `jpeg90` | 113 | 0.7800 | 0.9318 | 0.1858 | 0.1304 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jpeg_70` | `jpeg70` | 113 | 0.7666 | 0.9268 | 0.1858 | 0.1522 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jpeg_50` | `jpeg50` | 113 | 0.7821 | 0.9303 | 0.1858 | 0.1522 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jpeg_30` | `jpeg30` | 113 | 0.7635 | 0.9223 | 0.1858 | 0.1196 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `blur_0.5` | `blur0.5` | 113 | 0.7971 | 0.9374 | 0.1858 | 0.1304 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `blur_1.0` | `blur1.0` | 113 | 0.8282 | 0.9476 | 0.1858 | 0.1630 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `blur_2.0` | `blur2.0` | 113 | 0.8732 | 0.9651 | 0.1858 | 0.3261 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `resize_0.5` | `resize0.5` | 113 | 0.8095 | 0.9423 | 0.1858 | 0.1630 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `resize_0.25` | `resize0.25` | 113 | 0.8390 | 0.9525 | 0.1858 | 0.2174 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `noise_0.02` | `noise0.02` | 113 | 0.7754 | 0.9324 | 0.1858 | 0.1630 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `noise_0.05` | `noise0.05` | 113 | 0.7448 | 0.9210 | 0.1858 | 0.1739 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `noise_0.1` | `noise0.1` | 113 | 0.7277 | 0.9149 | 0.1858 | 0.1522 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jitter_0.2` | `jitter0.2` | 113 | 0.7469 | 0.9250 | 0.1858 | 0.1739 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `center_crop_0.8` | `center_crop0.8` | 113 | 0.7914 | 0.9386 | 0.1858 | 0.1413 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `composed_blur1.0+jpeg70` | `blur1.0+jpeg70` | 113 | 0.8173 | 0.9427 | 0.1858 | 0.2065 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `composed_resize0.5+jpeg50` | `resize0.5+jpeg50` | 113 | 0.8002 | 0.9360 | 0.1858 | 0.1957 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `composed_jitter0.2+jpeg30` | `jitter0.2+jpeg30` | 113 | 0.7355 | 0.9122 | 0.1858 | 0.1087 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `composed_resize0.25+blur0.5+jpeg30` | `resize0.25+blur0.5+jpeg30` | 113 | 0.8219 | 0.9397 | 0.1858 | 0.1739 | TPR@FPR=1% degenerate: 21 reals < 100 |

## By family

| Family | Cells | Mean AUROC | Min AUROC |
|---|---:|---:|---:|
| `clean` | 1 | 0.7873 | 0.7873 |
| `jpeg` | 4 | 0.7730 | 0.7635 |
| `blur` | 3 | 0.8328 | 0.7971 |
| `resize` | 2 | 0.8243 | 0.8095 |
| `noise` | 3 | 0.7493 | 0.7277 |
| `jitter` | 1 | 0.7469 | 0.7469 |
| `center_crop` | 1 | 0.7914 | 0.7914 |
| `composed` | 4 | 0.7937 | 0.7355 |
