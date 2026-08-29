# Phase 2 leakage audit

> **Sampling correction (see `results/audit_sampling_verification.md`).** The
> sample below was drawn from the *head of the validation split* -- shards 0-5 of
> 34, row groups 0-2 of 9 -- and touched none of the 249 `train` shards. That was
> not a valid draw. It has since been re-done: 20,000 rows at 189 independent
> positions across the whole train split. **Every headline finding replicates**
> (`is_1024sq` 0.9814 -> 0.9807, `megapixels` 0.9814 -> 0.9802), and the parquet
> is not class-ordered (class share varies by at most 1.6 pp across deciles of
> file position). The numbers here stand; the method that produced them does not.

Sample: `data\audit_sample\sid_set\sample_manifest.csv` -- 1800 images, 981 MB.

## Headline -- what a metadata-only classifier gets

AUROC of a *single* metadata feature, direction-agnostic, computed with the Phase 1 metric code. 0.5 = no leak, 1.0 = the feature alone separates the classes perfectly.

| Feature | real vs AI (binary, §4.2) | real vs tampered | real vs full_synthetic |
|---|---|---|---|
| `container_is_png` | 0.7574 ⚠️ | 0.5334 | 1.0000 🚨 |
| `n_bytes` | 0.6037 | 0.7520 ⚠️ | 0.9890 🚨 |
| `megapixels` | 0.9814 🚨 | 0.9814 🚨 | 0.9814 🚨 |
| `is_1024sq` | 0.9814 🚨 | 0.9814 🚨 | 0.9814 🚨 |
| `is_square` | 0.9806 🚨 | 0.9806 🚨 | 0.9806 🚨 |
| `bytes_per_pixel` | 0.5341 | 0.8452 ⚠️ | 0.9447 🚨 |
| `height` | 0.8966 ⚠️ | 0.8966 ⚠️ | 0.8966 ⚠️ |
| `has_icc_int` | 0.7027 ⚠️ | 0.7027 ⚠️ | 0.7027 ⚠️ |
| `width` | 0.5897 | 0.5897 | 0.5897 |
| `n_exif_tags` | 0.5000 | 0.5000 | 0.5000 |
| `n_text_chunks` | 0.5000 | 0.5000 | 0.5000 |

One-line rule `if container == PNG: predict AI` scores **100.00% accuracy** on real vs full_synthetic (1186 images). No pixels read.

## (a) Container format by class

| klass | jpeg | png | n |
|---|---|---|---|
| full_synthetic | 0 (0.0%) | 567 (100.0%) | 567 |
| real | 619 (100.0%) | 0 (0.0%) | 619 |
| tampered | 573 (93.3%) | 41 (6.7%) | 614 |

Mode (PIL) by class:

| klass | CMYK | L | RGB | n |
|---|---|---|---|---|
| full_synthetic | 0 (0.0%) | 0 (0.0%) | 567 (100.0%) | 567 |
| real | 1 (0.2%) | 14 (2.3%) | 604 (97.6%) | 619 |
| tampered | 0 (0.0%) | 0 (0.0%) | 614 (100.0%) | 614 |

## (b) Resolution by class

| klass          |   n |   w_mean |   w_std |   h_mean |   h_std |   mp_med |   pct_1024sq |   uniq_shapes |
|:---------------|----:|---------:|--------:|---------:|--------:|---------:|-------------:|--------------:|
| full_synthetic | 567 |   1024   |    0    |  1024    |    0    |     1.05 |       100    |             1 |
| real           | 619 |    972.2 |  115.67 |   778.68 |  142.48 |     0.7  |         3.72 |           142 |
| tampered       | 614 |   1024   |    0    |  1024    |    0    |     1.05 |       100    |             1 |

Top 6 exact (w×h) shapes per class:

- **full_synthetic**: 1024×1024 (100%)
- **real**: 1024×768 (24%), 1024×683 (18%), 683×1024 (4%), 768×1024 (4%), 1024×681 (4%), 1024×1024 (4%)
- **tampered**: 1024×1024 (100%)

## (c) EXIF and PNG text chunks by class

| klass          |   n |   pct_exif |   mean_exif_tags |   pct_icc |   mean_text_chunks |   pct_any_text |   pct_gen_marker |   pct_metadata_keys |
|:---------------|----:|-----------:|-----------------:|----------:|-------------------:|---------------:|-----------------:|--------------------:|
| full_synthetic | 567 |          0 |                0 |      0    |                  0 |              0 |                0 |                   0 |
| real           | 619 |          0 |                0 |     40.55 |                  0 |              0 |                0 |                   0 |
| tampered       | 614 |          0 |                0 |      0    |                  0 |              0 |                0 |                   0 |

- suspicious metadata keys seen: none
- generator tool markers seen: none
- EXIF orientation values: none

## (d) Content-type features (first 120/class)

| klass          |   uniq_colors_k |   flat_frac |   pure_bw_frac |   sat_mean |   hf_energy |   aspect |
|:---------------|----------------:|------------:|---------------:|-----------:|------------:|---------:|
| full_synthetic |           2.435 |       0.534 |          0.012 |      0.394 |       7.595 |    1     |
| real           |           2.928 |       0.298 |          0.023 |      0.312 |      11.832 |    1.333 |
| tampered       |           3.341 |       0.22  |          0.024 |      0.299 |      13.526 |    1     |

`uniq_colors_k` = thousands of distinct 8-level-quantised colours; `flat_frac` = fraction of horizontally flat pixel pairs; `pure_bw_frac` = fraction of pure black/white pixels; `hf_energy` = mean abs gradient. Illustration and screenshot material scores low on colours and high on flat/pure-bw.

## (e) Duplicates and near-duplicates

Threshold 6 is calibrated, not guessed. On 150 images from this sample: resize 0.25x, JPEG q30 and blur sigma=2 each moved the hash by at most **2** bits, while 11,175 unrelated pairs never came closer than **10** (p1 = 18, mean 30). So 6 catches every rescaled or recompressed repost with zero false positives. It does **not** catch crops: a 10%-per-side crop moves the hash ~20 bits, indistinguishable from an unrelated image.

- exact phash collisions: 0 images in 0 groups
- near-duplicate pairs (phash Hamming ≤ 6): **2 within-class**, **0 across-class**
- pairs compared: 1619100

Examples:

- `full_synthetic:full_synthetic_002612.png ~ full_synthetic:full_synthetic_001990.png (d=6)`
- `full_synthetic:full_synthetic_007719.png ~ full_synthetic:full_synthetic_005062.png (d=6)`

## Load failures

- unreadable: 0 / 1800
- truncated but recoverable: 0

