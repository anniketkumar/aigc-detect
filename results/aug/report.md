# Robustness grid — aug

Model `clip_linear(/content/drive/MyDrive/aigc/checkpoints/aug.pt)` on `data/manifests/test.csv` (6810 images, 19 cells, seed 0).

Reproduce: `python -m src.evaluate --model clip_linear --ckpt /content/drive/MyDrive/aigc/checkpoints/aug.pt --split test --out results/aug/ --device cuda`

## Headline (§3.2)

| Metric | Value |
|---|---|
| Clean AUROC | 0.9779 |
| Mean transformed AUROC (family-balanced) | 0.9702 |
| **Robustness gap** ↓ | **0.0077** |
| **Worst cell AUROC** ↑ | **0.9526** (`noise_0.1`) |
| Mean transformed AUROC (flat, §3.2 literal) | 0.9690 |
| Robustness gap (flat) | 0.0089 |
| Mean AUROC, single transforms | 0.9705 |
| Mean AUROC, composed chains | 0.9639 |
| Cells | 19 (18 transformed, 7 families) |

`robustness_gap = AUROC(clean) − mean(family mean AUROC)`, lower is better. The headline weights each degradation family equally rather than each cell, so being good at JPEG alone (4 of 14 single cells) cannot mask fragility elsewhere; the flat cell-weighted mean §3.2 specifies is reported beside it. `worst_case = min(AUROC)` over all cells, higher is better. Clean AUROC is never to be read on its own (§13).

## Per-cell

| Cell | Chain | n | AUROC | AP | acc@0.5 | TPR@FPR=1% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `clean` | `clean` | 6810 | 0.9779 | 0.9968 | 0.9203 | 0.7590 |  |
| `jpeg_90` | `jpeg90` | 6810 | 0.9752 | 0.9964 | 0.9198 | 0.7303 |  |
| `jpeg_70` | `jpeg70` | 6810 | 0.9715 | 0.9959 | 0.9195 | 0.7142 |  |
| `jpeg_50` | `jpeg50` | 6810 | 0.9677 | 0.9954 | 0.8946 | 0.6978 |  |
| `jpeg_30` | `jpeg30` | 6810 | 0.9668 | 0.9952 | 0.8969 | 0.7063 |  |
| `blur_0.5` | `blur0.5` | 6810 | 0.9774 | 0.9967 | 0.9175 | 0.7440 |  |
| `blur_1.0` | `blur1.0` | 6810 | 0.9761 | 0.9966 | 0.9034 | 0.7522 |  |
| `blur_2.0` | `blur2.0` | 6810 | 0.9724 | 0.9960 | 0.9075 | 0.7313 |  |
| `resize_0.5` | `resize0.5` | 6810 | 0.9772 | 0.9967 | 0.9032 | 0.7705 |  |
| `resize_0.25` | `resize0.25` | 6810 | 0.9721 | 0.9960 | 0.8825 | 0.7495 |  |
| `noise_0.02` | `noise0.02` | 6810 | 0.9699 | 0.9957 | 0.9162 | 0.7308 |  |
| `noise_0.05` | `noise0.05` | 6810 | 0.9618 | 0.9946 | 0.8962 | 0.6788 |  |
| `noise_0.1` | `noise0.1` | 6810 | 0.9526 | 0.9932 | 0.8627 | 0.6647 |  |
| `jitter_0.2` | `jitter0.2` | 6810 | 0.9730 | 0.9961 | 0.9185 | 0.7148 |  |
| `center_crop_0.8` | `center_crop0.8` | 6810 | 0.9728 | 0.9961 | 0.9291 | 0.7230 |  |
| `composed_blur1.0+jpeg70` | `blur1.0+jpeg70` | 6810 | 0.9694 | 0.9956 | 0.9153 | 0.6923 |  |
| `composed_resize0.5+jpeg50` | `resize0.5+jpeg50` | 6810 | 0.9681 | 0.9954 | 0.9023 | 0.7130 |  |
| `composed_jitter0.2+jpeg30` | `jitter0.2+jpeg30` | 6810 | 0.9616 | 0.9944 | 0.8921 | 0.6277 |  |
| `composed_resize0.25+blur0.5+jpeg30` | `resize0.25+blur0.5+jpeg30` | 6810 | 0.9566 | 0.9937 | 0.8467 | 0.6408 |  |

## By family

| Family | Cells | Mean AUROC | Min AUROC |
|---|---:|---:|---:|
| `clean` | 1 | 0.9779 | 0.9779 |
| `jpeg` | 4 | 0.9703 | 0.9668 |
| `blur` | 3 | 0.9753 | 0.9724 |
| `resize` | 2 | 0.9746 | 0.9721 |
| `noise` | 3 | 0.9614 | 0.9526 |
| `jitter` | 1 | 0.9730 | 0.9730 |
| `center_crop` | 1 | 0.9728 | 0.9728 |
| `composed` | 4 | 0.9639 | 0.9566 |
