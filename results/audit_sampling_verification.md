# Did the Phase 2 audit sample across the dataset, or just read the head?

**Verdict: the sampling was wrong in method, and the findings are nonetheless
real.** The audit read the head of one split. Re-drawn uniformly at ~10×
coverage, every headline finding replicates to within 0.001 AUROC. Nothing
downstream needs to change; the audit document needs a correction note.

---

## 1. What the audit actually read

`scripts/fetch_audit_sample.py::fetch_sid` loops

```python
for shard in range(shards):        # shards = 2, raised to 6 on the real run
    for rg in range(row_groups):   # row_groups = 3
```

so it reads shards `0..5` and row groups `0..2` — the head of each file, and
only the first 6 files. Reconstructed from the sample manifest's own
`shard`/`row_group` columns:

| | value |
|---|---|
| shards read | 0, 1, 2, 3, 4, 5 (of 34) |
| row groups per shard | 0, 1, 2 (of 9) |
| images | 1,800 |

It also read **`validation-*.parquet` only**. The repo has:

| split | shards | rows |
|---|---:|---:|
| `train` | 249 | 210,207 |
| `validation` | 34 | 30,022 |

`SID_VAL_SHARDS = 34` is hardcoded and there is no `--split` flag. So the audit
covered **0.75% of one split and 0% of the split that will actually be trained
on** — and the user's suspicion about the 249 shards was well founded: not one
of them was touched.

## 2. Re-sample

`scripts/audit_resample.py`, two passes sized to what each finding costs.

**Geometry** — HF datasets-server `/rows`, which serves arbitrary row offsets as
JSON. 200 blocks of 100 rows at 200 independent random offsets over the whole
train split.

| | audit | re-sample |
|---|---|---|
| split | validation | **train** |
| rows | 1,800 | **20,000** (18,909 unique) |
| coverage | 0.75% of validation | **9.5% of train** |
| positions | 6, all at file head | **189 independent** |

**Container format** — needs the original bytes, which only exist inside the
parquet. Snappy declines to compress already-compressed image data
(`image.bytes`: 50.58 MB compressed, 50.58 MB uncompressed), so the encoded
files sit **effectively verbatim** in the parquet and their magic bytes can be
counted by raw HTTP range reads. 18 windows of 1.5 MB at random offsets in
random shards, 27 MB total, no parquet decode.

## 3. Results

### 3a. The parquet is not class-ordered — the thing that would have made this an artifact

Class share by decile of position within the split:

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| full_synthetic | .339 | .321 | .348 | .324 | .339 | .343 | .317 | .331 | .339 | .321 |
| real | .342 | .349 | .328 | .344 | .336 | .328 | .346 | .336 | .336 | .337 |
| tampered | .319 | .330 | .324 | .332 | .324 | .329 | .337 | .333 | .325 | .342 |

Maximum deviation of any class share from its own mean: **1.6 pp**. The three
classes are interleaved uniformly from the first row to the last. Head-reading
therefore did not skew the class mix — which is the specific failure mode that
would have invalidated everything.

### 3b. Geometry: replicates almost exactly

| feature | audit (1,800, val head) | re-sample (13,405, train-wide) |
|---|---|---|
| `is_1024sq` | 0.9814 | **0.9807** |
| `megapixels` | 0.9814 | **0.9802** |
| `is_square`  | 0.9806 | **0.9787** |
| `height`     | 0.8966 | **0.8874** |
| `width`      | 0.5897 | **0.5970** |

(real vs full_synthetic; n_real 6,779, n_ai 6,626)

| class | n | % 1024² | unique shapes |
|---|---:|---:|---:|
| full_synthetic | 6,626 | **100.00** | **1** |
| real | 6,779 | 3.85 | 667 |
| tampered | 6,595 | 97.73 | 2 |

The AI class is 1024×1024 and nothing else, across 189 positions spanning the
entire train split.

One near-miss worth recording: the first mid-range train shard sampled
(`train-00137`, row 0) held a **label-0 image at 1024×1024**, which looked like
the finding collapsing. It is the 3.85% tail. That tail is flat across position
— per-decile it runs 2.9%–5.7% with no trend — so it is a genuine property of
the real class, not an ordering effect.

### 3c. Container format: consistent with 100% PNG for AI

Per-class attribution needs decoded row groups, which is bandwidth this machine
does not have. The pooled test is still decisive, because the audit makes a
sharp numerical prediction: if AI (⅓ of rows) is 100% PNG and real + tampered
(⅔) are ~96% JPEG, the **pooled PNG share must be ≈ 0.35**.

18 windows, shards `train-00010` … `train-00247` and `validation-00008` …
`validation-00024`:

```
PNG signatures  15
JPEG signatures 31
PNG share       0.326      (predicted ≈ 0.35)
```

Both formats appear throughout the shard index range, so format is not ordered
by file either. This corroborates rather than proves; the direct per-class
confirmation is the `bytes` pass of `audit_resample.py`, left un-run here
because at 1.25 MB/s it is a multi-hour download for a finding already
supported from three directions.

## 4. What this changes

Nothing in the data strategy. The leaks are real and the normalization in
PLAN.md §4 (crop at native resolution, strip all metadata, constant-quality
single JPEG pass) is aimed correctly.

Two process corrections:

1. `fetch_audit_sample.py` should not be used again as-is. `audit_resample.py`
   replaces it and defaults to random shards and random row groups.
2. `results/audit_sid_set.md` now carries a header noting its sample was drawn
   from the head of the validation split, and pointing here.

## 5. Reproduce

```bash
python -m scripts.audit_resample geometry --split train --n-batches 200 --seed 0
#   -> results/audit_resample_geometry_train.csv          (~14 min, ~6 MB)
python -m scripts.audit_resample bytes --n-shards 12 --seed 0
#   -> data/audit_sample/sid_set_resampled/               (~700 MB, hours at 1.25 MB/s)
```
