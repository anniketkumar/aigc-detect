# Robustness grid — fusion

Model `clip_freq_fusion(/content/drive/MyDrive/aigc/checkpoints/fusion.pt)` on `data/manifests/test.csv` (6810 images, 19 cells, seed 0).

Reproduce: `python -m src.evaluate --model clip_freq_fusion --ckpt /content/drive/MyDrive/aigc/checkpoints/fusion.pt --split test --out results/fusion/ --device cuda`

## Headline (§3.2)

| Metric | Value |
|---|---|
| Clean AUROC | 0.9786 |
| Mean transformed AUROC (family-balanced) | 0.9692 |
| **Robustness gap** ↓ | **0.0095** |
| **Worst cell AUROC** ↑ | **0.9474** (`composed_resize0.25+blur0.5+jpeg30`) |
| Mean transformed AUROC (flat, §3.2 literal) | 0.9674 |
| Robustness gap (flat) | 0.0113 |
| Mean AUROC, single transforms | 0.9698 |
| Mean AUROC, composed chains | 0.9589 |
| Cells | 19 (18 transformed, 7 families) |

`robustness_gap = AUROC(clean) − mean(family mean AUROC)`, lower is better. The headline weights each degradation family equally rather than each cell, so being good at JPEG alone (4 of 14 single cells) cannot mask fragility elsewhere; the flat cell-weighted mean §3.2 specifies is reported beside it. `worst_case = min(AUROC)` over all cells, higher is better. Clean AUROC is never to be read on its own (§13).

## Per-cell

| Cell | Chain | n | AUROC | AP | acc@0.5 | TPR@FPR=1% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `clean` | `clean` | 6810 | 0.9786 | 0.9970 | 0.9122 | 0.7763 |  |
| `jpeg_90` | `jpeg90` | 6810 | 0.9746 | 0.9964 | 0.9140 | 0.7620 |  |
| `jpeg_70` | `jpeg70` | 6810 | 0.9689 | 0.9956 | 0.9010 | 0.6918 |  |
| `jpeg_50` | `jpeg50` | 6810 | 0.9664 | 0.9952 | 0.8690 | 0.6823 |  |
| `jpeg_30` | `jpeg30` | 6810 | 0.9640 | 0.9948 | 0.8545 | 0.6878 |  |
| `blur_0.5` | `blur0.5` | 6810 | 0.9781 | 0.9969 | 0.9060 | 0.7790 |  |
| `blur_1.0` | `blur1.0` | 6810 | 0.9775 | 0.9967 | 0.8830 | 0.7338 |  |
| `blur_2.0` | `blur2.0` | 6810 | 0.9742 | 0.9963 | 0.8874 | 0.7075 |  |
| `resize_0.5` | `resize0.5` | 6810 | 0.9781 | 0.9969 | 0.8705 | 0.7658 |  |
| `resize_0.25` | `resize0.25` | 6810 | 0.9733 | 0.9962 | 0.8211 | 0.7197 |  |
| `noise_0.02` | `noise0.02` | 6810 | 0.9682 | 0.9954 | 0.9135 | 0.6790 |  |
| `noise_0.05` | `noise0.05` | 6810 | 0.9591 | 0.9941 | 0.8833 | 0.6760 |  |
| `noise_0.1` | `noise0.1` | 6810 | 0.9484 | 0.9925 | 0.8272 | 0.6320 |  |
| `jitter_0.2` | `jitter0.2` | 6810 | 0.9720 | 0.9959 | 0.9059 | 0.7287 |  |
| `center_crop_0.8` | `center_crop0.8` | 6810 | 0.9741 | 0.9963 | 0.9244 | 0.7630 |  |
| `composed_blur1.0+jpeg70` | `blur1.0+jpeg70` | 6810 | 0.9652 | 0.9950 | 0.8865 | 0.6895 |  |
| `composed_resize0.5+jpeg50` | `resize0.5+jpeg50` | 6810 | 0.9648 | 0.9949 | 0.8698 | 0.6930 |  |
| `composed_jitter0.2+jpeg30` | `jitter0.2+jpeg30` | 6810 | 0.9581 | 0.9939 | 0.8471 | 0.6282 |  |
| `composed_resize0.25+blur0.5+jpeg30` | `resize0.25+blur0.5+jpeg30` | 6810 | 0.9474 | 0.9921 | 0.7670 | 0.5607 |  |

## By family

| Family | Cells | Mean AUROC | Min AUROC |
|---|---:|---:|---:|
| `clean` | 1 | 0.9786 | 0.9786 |
| `jpeg` | 4 | 0.9685 | 0.9640 |
| `blur` | 3 | 0.9766 | 0.9742 |
| `resize` | 2 | 0.9757 | 0.9733 |
| `noise` | 3 | 0.9586 | 0.9484 |
| `jitter` | 1 | 0.9720 | 0.9720 |
| `center_crop` | 1 | 0.9741 | 0.9741 |
| `composed` | 4 | 0.9589 | 0.9474 |
