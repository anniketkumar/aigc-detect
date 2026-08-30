# Robustness grid — baseline

Model `clip_linear(/content/drive/MyDrive/aigc/checkpoints/baseline.pt)` on `data/manifests/test.csv` (6810 images, 19 cells, seed 0).

Reproduce: `python -m src.evaluate --model clip_linear --ckpt /content/drive/MyDrive/aigc/checkpoints/baseline.pt --split test --out results/baseline/ --device cuda`

## Headline (§3.2)

| Metric | Value |
|---|---|
| Clean AUROC | 0.9640 |
| Mean transformed AUROC (family-balanced) | 0.9498 |
| **Robustness gap** ↓ | **0.0142** |
| **Worst cell AUROC** ↑ | **0.9089** (`noise_0.1`) |
| Mean transformed AUROC (flat, §3.2 literal) | 0.9464 |
| Robustness gap (flat) | 0.0176 |
| Mean AUROC, single transforms | 0.9496 |
| Mean AUROC, composed chains | 0.9351 |
| Cells | 19 (18 transformed, 7 families) |

`robustness_gap = AUROC(clean) − mean(family mean AUROC)`, lower is better. The headline weights each degradation family equally rather than each cell, so being good at JPEG alone (4 of 14 single cells) cannot mask fragility elsewhere; the flat cell-weighted mean §3.2 specifies is reported beside it. `worst_case = min(AUROC)` over all cells, higher is better. Clean AUROC is never to be read on its own (§13).

## Per-cell

| Cell | Chain | n | AUROC | AP | acc@0.5 | TPR@FPR=1% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `clean` | `clean` | 6810 | 0.9640 | 0.9946 | 0.8930 | 0.6770 |  |
| `jpeg_90` | `jpeg90` | 6810 | 0.9604 | 0.9941 | 0.9031 | 0.5932 |  |
| `jpeg_70` | `jpeg70` | 6810 | 0.9500 | 0.9925 | 0.9110 | 0.5768 |  |
| `jpeg_50` | `jpeg50` | 6810 | 0.9433 | 0.9915 | 0.8931 | 0.5423 |  |
| `jpeg_30` | `jpeg30` | 6810 | 0.9389 | 0.9908 | 0.8887 | 0.4657 |  |
| `blur_0.5` | `blur0.5` | 6810 | 0.9629 | 0.9945 | 0.8837 | 0.6623 |  |
| `blur_1.0` | `blur1.0` | 6810 | 0.9614 | 0.9942 | 0.8402 | 0.6320 |  |
| `blur_2.0` | `blur2.0` | 6810 | 0.9565 | 0.9936 | 0.8107 | 0.6283 |  |
| `resize_0.5` | `resize0.5` | 6810 | 0.9651 | 0.9948 | 0.8535 | 0.6650 |  |
| `resize_0.25` | `resize0.25` | 6810 | 0.9573 | 0.9937 | 0.7897 | 0.6418 |  |
| `noise_0.02` | `noise0.02` | 6810 | 0.9447 | 0.9915 | 0.8833 | 0.5460 |  |
| `noise_0.05` | `noise0.05` | 6810 | 0.9280 | 0.9889 | 0.8370 | 0.4538 |  |
| `noise_0.1` | `noise0.1` | 6810 | 0.9089 | 0.9859 | 0.7590 | 0.4143 |  |
| `jitter_0.2` | `jitter0.2` | 6810 | 0.9587 | 0.9939 | 0.8893 | 0.6225 |  |
| `center_crop_0.8` | `center_crop0.8` | 6810 | 0.9579 | 0.9937 | 0.9115 | 0.5908 |  |
| `composed_blur1.0+jpeg70` | `blur1.0+jpeg70` | 6810 | 0.9448 | 0.9917 | 0.8984 | 0.5372 |  |
| `composed_resize0.5+jpeg50` | `resize0.5+jpeg50` | 6810 | 0.9424 | 0.9913 | 0.8959 | 0.5447 |  |
| `composed_jitter0.2+jpeg30` | `jitter0.2+jpeg30` | 6810 | 0.9316 | 0.9896 | 0.8758 | 0.5562 |  |
| `composed_resize0.25+blur0.5+jpeg30` | `resize0.25+blur0.5+jpeg30` | 6810 | 0.9217 | 0.9880 | 0.8449 | 0.4678 |  |

## By family

| Family | Cells | Mean AUROC | Min AUROC |
|---|---:|---:|---:|
| `clean` | 1 | 0.9640 | 0.9640 |
| `jpeg` | 4 | 0.9481 | 0.9389 |
| `blur` | 3 | 0.9603 | 0.9565 |
| `resize` | 2 | 0.9612 | 0.9573 |
| `noise` | 3 | 0.9272 | 0.9089 |
| `jitter` | 1 | 0.9587 | 0.9587 |
| `center_crop` | 1 | 0.9579 | 0.9579 |
| `composed` | 4 | 0.9351 | 0.9217 |
