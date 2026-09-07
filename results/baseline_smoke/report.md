# Robustness grid — baseline_smoke

Model `clip_linear(runs\baseline_smoke.pt)` on `data/manifests/test.csv` (113 images, 19 cells, seed 0).

Reproduce: `python -m src.evaluate --model clip_linear --ckpt runs/baseline_smoke.pt --manifest data/manifests/test.csv --out results/baseline_smoke/ --device cpu`

## Headline (§3.2)

| Metric | Value |
|---|---|
| Clean AUROC | 0.8587 |
| Mean transformed AUROC (family-balanced) | 0.8487 |
| **Robustness gap** ↓ | **0.0100** |
| **Worst cell AUROC** ↑ | **0.7935** (`noise_0.1`) |
| Mean transformed AUROC (flat, §3.2 literal) | 0.8453 |
| Robustness gap (flat) | 0.0134 |
| Mean AUROC, single transforms | 0.8504 |
| Mean AUROC, composed chains | 0.8275 |
| Cells | 19 (18 transformed, 7 families) |

`robustness_gap = AUROC(clean) − mean(family mean AUROC)`, lower is better. The headline weights each degradation family equally rather than each cell, so being good at JPEG alone (4 of 14 single cells) cannot mask fragility elsewhere; the flat cell-weighted mean §3.2 specifies is reported beside it. `worst_case = min(AUROC)` over all cells, higher is better. Clean AUROC is never to be read on its own (§13).

## Per-cell

| Cell | Chain | n | AUROC | AP | acc@0.5 | TPR@FPR=1% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `clean` | `clean` | 113 | 0.8587 | 0.9624 | 0.2035 | 0.4457 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jpeg_90` | `jpeg90` | 113 | 0.8685 | 0.9670 | 0.2035 | 0.5000 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jpeg_70` | `jpeg70` | 113 | 0.8504 | 0.9654 | 0.2124 | 0.6413 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jpeg_50` | `jpeg50` | 113 | 0.8468 | 0.9633 | 0.2035 | 0.5109 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jpeg_30` | `jpeg30` | 113 | 0.8514 | 0.9642 | 0.1947 | 0.5435 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `blur_0.5` | `blur0.5` | 113 | 0.8561 | 0.9615 | 0.1947 | 0.3913 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `blur_1.0` | `blur1.0` | 113 | 0.8644 | 0.9636 | 0.1947 | 0.3261 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `blur_2.0` | `blur2.0` | 113 | 0.8799 | 0.9685 | 0.1858 | 0.4022 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `resize_0.5` | `resize0.5` | 113 | 0.8608 | 0.9645 | 0.1947 | 0.4348 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `resize_0.25` | `resize0.25` | 113 | 0.8416 | 0.9564 | 0.1858 | 0.2717 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `noise_0.02` | `noise0.02` | 113 | 0.8597 | 0.9664 | 0.1947 | 0.4891 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `noise_0.05` | `noise0.05` | 113 | 0.8137 | 0.9497 | 0.1858 | 0.3261 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `noise_0.1` | `noise0.1` | 113 | 0.7935 | 0.9405 | 0.1858 | 0.2609 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `jitter_0.2` | `jitter0.2` | 113 | 0.8437 | 0.9605 | 0.2035 | 0.4348 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `center_crop_0.8` | `center_crop0.8` | 113 | 0.8753 | 0.9693 | 0.1858 | 0.4891 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `composed_blur1.0+jpeg70` | `blur1.0+jpeg70` | 113 | 0.8318 | 0.9583 | 0.1947 | 0.4457 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `composed_resize0.5+jpeg50` | `resize0.5+jpeg50` | 113 | 0.8427 | 0.9588 | 0.1947 | 0.3804 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `composed_jitter0.2+jpeg30` | `jitter0.2+jpeg30` | 113 | 0.8028 | 0.9480 | 0.1947 | 0.3370 | TPR@FPR=1% degenerate: 21 reals < 100 |
| `composed_resize0.25+blur0.5+jpeg30` | `resize0.25+blur0.5+jpeg30` | 113 | 0.8328 | 0.9479 | 0.1947 | 0.2174 | TPR@FPR=1% degenerate: 21 reals < 100 |

## By family

| Family | Cells | Mean AUROC | Min AUROC |
|---|---:|---:|---:|
| `clean` | 1 | 0.8587 | 0.8587 |
| `jpeg` | 4 | 0.8543 | 0.8468 |
| `blur` | 3 | 0.8668 | 0.8561 |
| `resize` | 2 | 0.8512 | 0.8416 |
| `noise` | 3 | 0.8223 | 0.7935 |
| `jitter` | 1 | 0.8437 | 0.8437 |
| `center_crop` | 1 | 0.8753 | 0.8753 |
| `composed` | 4 | 0.8275 | 0.8028 |
