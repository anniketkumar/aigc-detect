# Robust AIGC Image Detection — Build Plan

> This file is the spec. If you are a coding agent working in this repo, read this
> top to bottom before writing code, and follow the phase order. Do not skip
> Phase 1. Do not start Phase 5 until Phase 4's numbers are recorded.

---

## 0. Context

Hackathon problem statement 5: distinguish AI-generated images from authentic
ones, with accuracy that **survives real-world post-processing** (compression,
blur, resize, noise, color shifts, cropping).

### Hard constraints

| Constraint | Value |
|---|---|
| Model size | **< 2B parameters total** (all branches combined) |
| Compute | hackathon-scale; assume single GPU, possibly Colab T4/A100 |
| Forbidden training data | the WildFake reference subset (COCO val2017 4998 imgs + DALL·E Advanced 8843 imgs) |
| Output format | JSON, one entry per image: `image_path`, `pred` (float 0–1) |

### Design thesis (put this in the README and the video intro)

> Detectors don't fail because AI images are hard to spot. They fail because JPEG
> re-encoding, resizing and blur destroy the low-level artifacts they were trained
> to read. So: train on the degraded distribution, and give the model an explicit
> estimate of *how* degraded its input is.

Every architectural choice below traces back to that sentence.

---

## 1. Repo layout

```
aigc-detect/
├── README.md
├── PLAN.md                     # this file
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── baseline_clip.yaml
│   ├── aug.yaml
│   └── full.yaml
├── src/
│   ├── transforms.py           # Phase 1 — the eval transform grid
│   ├── metrics.py              # Phase 1 — AUROC / AP / acc / robustness gap
│   ├── evaluate.py             # Phase 1 — runs a model over the grid
│   ├── data/
│   │   ├── datasets.py         # torch Datasets for SID_Set / WildFake / CIFAKE
│   │   ├── manifest.py         # builds train/val/test CSV manifests
│   │   └── augment.py          # Phase 4 — training-time RandAug sampler
│   ├── models/
│   │   ├── clip_backbone.py    # frozen CLIP feature extractor + cache
│   │   ├── semantic_head.py    # MLP over CLIP features
│   │   ├── artifact_branch.py  # small CNN over DCT / high-pass residual
│   │   ├── degradation_head.py # predicts applied transform type + severity
│   │   └── fusion.py           # gated combination of the two branches
│   ├── losses.py               # BCE + consistency + auxiliary degradation loss
│   ├── train.py
│   └── calibrate.py            # Phase 6 — per-bucket temperature scaling
├── scripts/
│   ├── download_data.py
│   ├── cache_features.py       # precompute CLIP embeddings → .npy
│   └── make_report.py          # emits the robustness table as markdown
├── predict.py                  # ← THE DELIVERABLE. image dir → preds.json
├── app.py                      # Gradio demo for the video
├── results/                    # committed. tables, plots, ablation logs
└── tests/
    └── test_predict_smoke.py
```

---

## 2. Environment

```
torch, torchvision
open_clip_torch          # or transformers CLIPModel
numpy, pandas, pillow, opencv-python
scikit-learn             # AUROC / AP / logistic head sanity checks
pyyaml, tqdm
gradio                   # demo only
matplotlib               # plots for the report
```

Pin versions in `requirements.txt`. Do not add a dependency that isn't used in
the final pipeline — reviewers open this file.

---

## 3. Phase 1 — Evaluation harness (BUILD THIS FIRST)

No modelling until this runs. You cannot iterate on robustness you cannot
measure, and this phase produces deliverable #4 directly.

### 3.1 `src/transforms.py`

Implement each transform as `f(PIL.Image) -> PIL.Image`. Exact severities from
the problem statement:

```python
TRANSFORM_GRID = {
    "clean":       [None],
    "jpeg":        [90, 70, 50, 30],           # quality
    "blur":        [0.5, 1.0, 2.0],            # gaussian sigma
    "resize":      [0.5, 0.25],                # downscale then upscale back
    "noise":       [0.02, 0.05, 0.10],         # gaussian sigma, on [0,1] pixels
    "jitter":      [0.20],                     # brightness/contrast/saturation ±20%
    "center_crop": [0.80],                     # keep 80% then resize back
}

COMPOSED = [                                   # realistic redistribution chains
    ("blur", 1.0, "jpeg", 70),                 # phone photo → messaging app
    ("resize", 0.5, "jpeg", 50),               # thumbnail → repost
    ("jitter", 0.20, "jpeg", 30),              # filter app → heavy re-encode
    ("resize", 0.25, "blur", 0.5, "jpeg", 30), # worst case
]
```

Notes that matter:
- JPEG must round-trip through an actual encoder (`BytesIO` + `Image.save(fmt="JPEG", quality=q)`), not a simulation.
- `resize` = downscale to `s×` then upscale back to original size. The information loss is the point.
- Apply transforms **before** the model's own preprocessing/normalization.
- Seed everything. Noise and jitter must be reproducible run to run.

### 3.2 `src/metrics.py`

Per (transform, severity) cell, compute: `AUROC`, `AP`, `acc@0.5`, `TPR@FPR=1%`.

Plus the headline number:

```
robustness_gap = AUROC(clean) - mean(AUROC(all transformed cells))
worst_case     = min(AUROC over all cells)
```

`robustness_gap` and `worst_case` are the metrics you optimize. Clean accuracy
alone is not the objective and should never be reported alone.

### 3.3 `src/evaluate.py`

CLI: `python -m src.evaluate --ckpt <path> --split test --out results/<name>/`

Emits `results/<name>/grid.csv` and a markdown table. Cache transformed images
to disk on first run or evaluation will dominate your wall-clock time.

**Acceptance:** running the harness against a random-scoring dummy model
produces AUROC ≈ 0.5 in every cell and a complete markdown table.

---

## 4. Phase 2 — Data

> **Rewritten 2026-08-29** after the leakage audit and the source verification
> that followed it. The original §4 named SID_Set, WildFake and CIFAKE; all
> three are gone from the training path and the reasons are recorded below.
> Every repo path here was checked against the live Hub, not against a README.
> See `results/audit_sid_set.md`, `results/audit_sampling_verification.md`,
> `results/data_stats.md`.

### 4.0 Why the original plan changed

The audit found that on raw SID_Set a classifier **reading no pixels at all**
scores 0.98 AUROC: the AI class is 100% PNG and 100% 1024×1024, the real class
is 100% JPEG across 667 shapes, and ICC profiles appear in 40.6% of reals and 0%
of AI. That sample was itself drawn from the head of one split, so it was
re-done across the whole train split — the findings replicated to within 0.001.

Three consequences:

1. **Volume is not the constraint; generator diversity is.** A linear probe on
   frozen CLIP features saturates around 5–10k images per class, so a second
   generator is worth far more than a second 50k images from the first. Target:
   ~2k per generator across six, streamed.
2. **Every image goes through one canonical decode path** (§4.4), so container,
   geometry and metadata carry zero information.
3. **Tampered images are excluded from training** and retained as a third eval
   axis. A real photo with an inpainted region is not a generated image, and
   folding it into class 1 changes what the detector is being asked to do.

### 4.1 Sources — `src/data/sources.py`

18,200 images, 19.3 GB transferred, **1.7 GB resident** after normalization.

**AI (6 generators + 1 unseen), all 1024px native:**

| Generator | Repo | Family | Quota | Train? |
|---|---|---|---|---|
| SDXL | `bitmind/bm-subnet-stable-diffusion-xl-base-1.0` | sdxl | 2000 | yes |
| Mobius | `bitmind/bm-subnet-mobius` | sdxl-derivative | 2000 | yes |
| RealVisXL V4.0 | `bitmind/bm-subnet-RealVisXL_V4.0` | sdxl-derivative | 2000 | yes |
| Aura | `bitmind/bm-aura-imagegen` | aura | 2000 | yes |
| MidJourney | `bitmind/JourneyDB` | midjourney | 2000 | **held out** |
| Gemini (nano-banana) | `bitmind/nano-banana` | gemini | 2000 | **held out** |
| FLUX.1-dev | `saberzl/SID_Set` (validation, label 1) | flux | 800 | **held out** |

**Real (3 sources, deliberately):**

| Source | Repo | Character | Quota |
|---|---|---|---|
| OpenImagesV7 | `saberzl/SID_Set` (validation, label 0) | amateur web | 1800 |
| Megalith-Flickr | `bitmind/megalith-small` | amateur Flickr | 1800 |
| Unsplash | `wtcherr/unsplash_5k` | **professional** | 1800 |

Three real sources, not one, is the fix for the audit's semantic finding: the AI
class is polished and cinematic while SID_Set's real class is amateur. A model
can score well by learning "is this well lit", and that shortcut survives JPEG,
so `robustness_gap` would look excellent while the model had learned nothing
about generation. Unsplash is the control — professional photography, labelled
real — so "polished" no longer predicts "fake".

**Datasets ruled out, and why:**

| Dataset | Verdict |
|---|---|
| GenImage (`bitmind/GenImage_*`) | **Unusable.** Every mirrored generator is fixed low-res — BigGAN 128², ADM/glide/VQDM all 256², 100% of sampled rows. Real sources are 768–1152. A 512 crop yields zero images; a 224 crop makes resolution a perfect class signal again, merely inverted. The only fix would be upscaling one class — the exact per-class resampling signature §4.4 exists to prevent. |
| WildFake (ModelScope) | ~700 GB of zips, no streaming path. Its *file lists* were used to locate the forbidden reference subset (`results/audit_wildfake_paths.md`). No images pulled. |
| SID_Set as primary train | Demoted to eval-only. Single-generator (FLUX.1-dev) and the subject of the whole audit. |
| CIFAKE | 32×32. Below the crop floor, and the resize cells are meaningless there. |
| Pexels | **Dropped after measurement.** 18 KB/image at source, so its normalized files came out at a median 16 KB against 66–103 KB elsewhere — separable from every other real source by file size alone at AUROC **1.0000**, and it pushed real-vs-AI `n_bytes` to 0.7120. Replaced by Unsplash: 0.5017. Caught by the §4.5 regression test. |

### 4.2 Manifest — `src/data/manifest.py`

`data/manifests/{train,val,test}.csv`, exactly:

```
image_path, label (0=real,1=ai), generator, source_dataset, split
```

plus `detail.csv` alongside carrying hashes, source geometry and duplicate
cluster ids, so the required schema stays clean without discarding the audit
trail.

### 4.3 Splitting rules — non-negotiable

1. **Hold out entire generators.** Three, not two: MidJourney, Gemini and
   FLUX.1-dev — three distinct families, all closed commercial generators, which
   is the realistic deployment case. They go to `test` whole; not even a val
   slice, or hyperparameter choice leaks across the boundary carrying the claim.
2. **No forbidden image in `train.csv`**, enforced by **content hash**, not path
   — WildFake renames COCO files to `img000000.jpg`. Two layers: sha256 (exact)
   and pHash ≤ 6 (survives the re-encode WildFake applied). Blocklist built by
   `scripts/build_blocklist.py`: **5,000 COCO val2017 hashes**.
   **Known gap:** DALL·E Advanced (8,843 imgs) has no public standalone
   distribution and is not hashed. Recorded in the blocklist's `gaps` field,
   surfaced in `data_stats.md`, and mitigated by a registry-level DALL·E
   denylist test.
3. **No near-duplicate spans a split boundary.** pHash + LSH banding, clusters
   assigned whole. 8 bands × 8 bits makes the bucketing *exact* at threshold 6
   (pigeonhole needs `n_bands > threshold`; an earlier 4-band version was wrong
   and a brute-force test caught it).
4. **Class balance per split**, logged to `results/data_stats.md`. Train/val are
   bounded to 25–75% AI; test skews AI because the holdout generators land there.

### 4.4 Normalization — the canonical decode path

`src/data/imageio.py` → `src/data/normalize.py`. Identical for both classes:

```
decode → apply EXIF orientation → strip ALL metadata (EXIF, ICC, text, XMP)
       → RGB → [class 1 only: synthetic first JPEG generation]
       → fixed 512×512 CROP at native resolution, no resampling
       → one JPEG pass, q95 4:2:0, both classes
```

- **Crop, not resize.** Resizing applies different scale factors per class, and
  a resampling kernel leaves a signature in the exact band the Phase 5 artifact
  branch reads.
- **Crop offsets are multiples of 16** (the 4:2:0 MCU), so the inherited DCT
  grid phase cannot differ by class.
- **Random crop location, not centre.** Generators centre their subject; a
  centre crop would frame subject for one class and background for the other.
- **Constant quality, not distribution-matched.** A constant has AUROC 0.5 by
  construction — provable, not merely measured.
- **Double JPEG.** Real photos arrive as JPEG and leave as JPEG: two
  generations. AI images arrive as PNG, so they get a synthetic first generation
  at a quality *and chroma subsampling* sampled from the real class's measured
  distribution (median q93, spread 68–100). Both classes end at two.

### 4.5 The audit as a regression test

`tests/test_normalization_audit.py` re-runs the metadata probe on the normalized
manifest and asserts every channel is under 0.60 AUROC:

| channel | raw SID_Set | normalized |
|---|---|---|
| `container_is_png` | 0.7574 | **0.5000** (constant) |
| `megapixels` / `is_1024sq` | 0.9814 | **0.5000** (constant) |
| `is_square` | 0.9806 | **0.5000** (constant) |
| `height` | 0.8966 | **0.5000** (constant) |
| `has_icc` | 0.7027 | **0.5000** (constant) |
| `n_bytes` | 0.6037 | **0.5357** |
| `n_exif_tags`, `n_text_chunks` | 0.5000 | **0.5000** (constant) |

Twelve of thirteen channels are *constant* after normalization, which is a
stronger guarantee than a measured bound — there is no sample size at which they
could come out otherwise. `n_bytes` cannot be pinned: at fixed quality it
measures compressibility, which is a property of the pixels rather than the
container. It is bounded instead, and per-source fingerprinting by file size is
asserted separately.

## 5. Phase 3 — Baseline (deliberately dumb)

Frozen CLIP ViT-L/14, no augmentation, linear head.

- `scripts/cache_features.py`: run CLIP once over the train set, save embeddings
  as `.npy`. Training the head then takes seconds and you can iterate freely.
- `src/models/semantic_head.py`: start with `nn.Linear(d, 1)`. Upgrade to a
  2-layer MLP only if the linear probe underfits.
- Train with BCE. No augmentation. This is the control.

Run the Phase 1 harness. **Expected result: strong clean AUROC, sharp collapse
under `jpeg=30` and `resize=0.25`.** Save that grid as
`results/baseline/grid.csv` — the collapse is your "before" picture and it goes
in the video.

If compute allows later, compare ViT-B/16 vs ViT-L/14 and report it. Both are far
under the 2B cap (~150M and ~428M).

---

## 6. Phase 4 — Augmentation (the biggest single lever)

`src/data/augment.py` — a RandAugment-style sampler over the same six families:

- sample 1–3 transforms per image, in random order
- **severities drawn continuously and wider than the eval grid** (JPEG q ∈ [20, 95],
  blur σ ∈ [0, 3.0], noise σ ∈ [0, 0.12], scale ∈ [0.2, 1.0], jitter ∈ [0, 0.3],
  crop ∈ [0.7, 1.0])
- apply with p ≈ 0.8; leave ~20% clean
- return `(image, degradation_label_vector)` — the applied transform types and
  severities. These labels are free and Phase 5 consumes them.

⚠️ Feature caching is incompatible with online augmentation. Either
(a) precompute embeddings for K augmented copies per image, or (b) run CLIP live
in the dataloader. Start with (a), K≈4 — it's much faster and nearly as good.

Retrain the head. Re-run the harness → `results/aug/grid.csv`.

**Expected: robustness gap drops sharply, clean AUROC dips slightly.** If the gap
doesn't close, the bug is in the augmentation, not the model. Fix it before
proceeding.

---

## 7. Phase 5 — Full model

Only start once Phase 4's numbers are in `results/`.

```
                     ┌─────────────────────┐
   image ──┬────────►│ CLIP ViT-L/14 (frozen)├──► semantic feats ──► head_s ──┐
           │         └─────────────────────┘                                  │
           │                                                                  ▼
           ├────────► DCT / high-pass residual ──► small CNN ──► head_a ──► gated
           │                                                                  ▲
           └────────► degradation head ──► ŝ (transform type + severity) ─────┘
```

### 7.1 Artifact branch — `artifact_branch.py`
Small CNN (≤ 5M params) over either the 2D DCT log-magnitude spectrum or a
high-pass residual (image minus median-filtered image). Expect: excellent clean
accuracy, near-chance under `jpeg=30`. That asymmetry is the whole reason the
gate exists.

### 7.2 Degradation head — `degradation_head.py`
Tiny CNN or a head on early CLIP features. Predicts applied transform type
(multi-label) + severity (regression). Trained on the free labels from Phase 4.
Self-supervised, no annotation cost. **This is the innovation hook — say so
explicitly in the writeup.**

### 7.3 Fusion — `fusion.py`
```
w = sigmoid(MLP(degradation_embedding))     # scalar or per-branch weights
logit = w * logit_artifact + (1 - w) * logit_semantic
```
Log `w` at eval time. Plotting `w` against JPEG quality should show it sliding
from artifact-trust to semantic-trust as quality drops. **That plot is the single
best figure in your submission** — it proves the mechanism works, not just the
metric.

### 7.4 Losses — `losses.py`
```
L = BCE
  + λ_c * KL(p(view_1) || p(view_2))    # two augmented views, same image
  + λ_d * degradation_loss              # auxiliary
```
Start λ_c = 1.0, λ_d = 0.5. The consistency term is what makes robustness
structural rather than incidental.

### 7.5 Parameter budget check
Write a `scripts/param_count.py` that prints total trainable + frozen params and
**asserts < 2B**. Put its output in the README. Judges will look for this.

---

## 8. Phase 6 — Calibration

The deliverable asks for a *confidence score*. Most teams will submit raw
uncalibrated logits. Don't.

- `src/calibrate.py`: temperature scaling fit on the validation split.
- Fit a **separate temperature per degradation bucket** (clean / mild / heavy,
  bucketed by the degradation head's own prediction).
- Report ECE and a reliability diagram, before and after.

This costs an hour and buys you points under both Technical Execution and
Feasibility, and it feeds the error analysis.

---

## 9. Phase 7 — Deliverable surfaces

### 9.1 `predict.py` — treat this as the most important file in the repo

```bash
python predict.py --image_dir path/to/images --out preds.json
```

```json
[{"image_path": "path/to/images/a.jpg", "pred": 0.873}, ...]
```

Requirements:
- recurse the directory; accept jpg/jpeg/png/webp/bmp
- **handle PNGs with alpha, grayscale images, CMYK JPEGs, and truncated files** without crashing — convert to RGB, and on genuine failure emit `pred: null` with a warning rather than dying mid-run
- batch inference, `--device` flag, sensible default batch size
- deterministic: same input dir → same output
- `tests/test_predict_smoke.py` runs it against a 10-image fixture dir containing at least one alpha PNG and one grayscale image

Test it on a directory the model has never seen before submission. A crashing
inference script is the most common way a good project scores badly.

### 9.2 `app.py` — Gradio demo (this is the video)

Upload an image, then a **JPEG-quality slider from 95 → 30**. Show live:
- the baseline model's confidence sliding into wrongness
- the full model's confidence holding
- the gate weight `w` shifting

Fifteen seconds of that clip communicates more than any table.

---

## 10. Results to produce

`results/ablation.md`, generated by `scripts/make_report.py`:

| Variant | Clean AUROC | Mean transformed AUROC | Robustness gap ↓ | Worst cell ↓ | Params |
|---|---|---|---|---|---|
| CLIP linear probe (baseline) | | | | | |
| + augmentation | | | | | |
| + consistency loss | | | | | |
| + artifact branch & gate | | | | | |
| + calibration | | | | | |

Plus:
- full transform × severity grid for the final model
- cross-generator held-out results
- gate weight vs JPEG quality plot
- reliability diagram

## 11. Error analysis note (deliverable #5)

Pull the highest-confidence mistakes from the test set and group them. Expect:
- **False positives:** heavily compressed real photos, screenshots, digital art and illustration, over-filtered phone photos
- **False negatives:** thumbnails, 4th-generation reposts, small crops of large fakes, generators absent from training

Name the failure modes explicitly and state which you would fix first. Judges
reward knowing your own weaknesses far more than they reward hiding them.

## 12. Impact framing (20% of the score)

Do not pitch "detect fake images." Pitch **moderation triage**: a calibrated,
degradation-aware score lets a platform choose a different operating point for a
pristine upload than for a fourth-generation repost, and route uncertain cases to
human review instead of hard-blocking. That is the difference between a
classifier and a deployable system — and it doubles as your Feasibility answer.

---

## 13. Guardrails for the coding agent

- **Do not** reorder the phases. Effort-to-payoff ranking is: eval harness >
  augmentation > calibration > fusion gate > backbone choice.
- **Do not** fine-tune the CLIP backbone before the frozen-probe pipeline is
  fully working end to end.
- **Do not** add architectural complexity that isn't measured in the ablation
  table. If it doesn't get a row, it doesn't get merged.
- **Do not** report clean accuracy without the robustness gap beside it.
- **Do not** train on the forbidden WildFake reference subset. Assert it in code.
- **Do** commit `results/` at every phase. The intermediate numbers are the story.
- **Do** keep a `NOTES.md` log of what was tried and what failed — it becomes the
  README's limitations section for free.

## 14. Open items

- [x] Confirm submission deadline → **under 72 h from 2026-08-29.**
- [x] Confirm available compute → **Google Colab**, ~50 GB disk. Strict
      linear probe; no backbone fine-tuning fits the budget.
- [x] Decide which generators to hold out → **MidJourney, Gemini
      (nano-banana), FLUX.1-dev** — three families, all closed commercial
      (§4.1).
- [ ] **Run the download on Colab, not locally.** Measured here: 1.25 MB/s,
      so 19 GB is 4+ hours and there is only 9.5 GB free. On Colab it is
      minutes. `python -m scripts.bench_throughput` re-measures it there
      first.
- [ ] Devpost draft, README, YouTube upload (allow more time than feels necessary — presentation + impact is 30% of the score)
