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

### 4.1 Sources

| Dataset | Use | Note |
|---|---|---|
| SID_Set (HF: `saberzl/SID_Set`) | primary train | high-res, real / synthetic / tampered splits |
| WildFake (ModelScope) | train + generator diversity | **exclude the reference subset** |
| CIFAKE (Kaggle) | optional hard low-res test only | 32×32 — resize transforms are meaningless here, do not train on it |

### 4.2 Manifest

`scripts/… → data/manifests/{train,val,test}.csv` with columns:

```
image_path, label (0=real,1=ai), generator, source_dataset, split
```

### 4.3 Splitting rules — non-negotiable

1. **Hold out entire generators** from training (e.g. leave out two diffusion
   families). Cross-generator AUROC is a stronger claim than cross-transform and
   costs nothing to obtain.
2. **Assert** at manifest-build time that no image from the forbidden WildFake
   reference subset appears in `train.csv`. Hash-check it. Write the assertion as
   a test. This is a disqualification risk, not a style point.
3. Balance real/fake per split. Log the counts into `results/data_stats.md`.

---

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

- [ ] Confirm submission deadline → set phase cutoffs
- [ ] Confirm available compute → decides ViT-L/14 fine-tuning vs strict linear probe
- [ ] Decide which generators to hold out
- [ ] Devpost draft, README, YouTube upload (allow more time than feels necessary — presentation + impact is 30% of the score)
