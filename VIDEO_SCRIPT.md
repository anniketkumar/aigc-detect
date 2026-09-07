# Demo Video Script — Robust AIGC Image Detection

Target runtime: **3:45–4:15** (~600 words of VO at a demo pace, plus screen
time). Timestamps are guides, not hard marks — pace to what's on screen.
Sourced from [WRITEUP.md](WRITEUP.md) and [README.md](README.md); every
number below is already committed and regeneratable, so read it straight off
those files if anything here drifts.

Recording checklist before you hit record:
- `uvicorn app:app --port 8000` running, `npm --prefix frontend run dev` running
- Terminal font size bumped up, one clean shell, `cd` into the repo root
- Have `results/tpr_analysis_aug/report.md`, `results/audit_sid_set.md`,
  and `README.md`'s results table open in tabs, ready to flash on screen
- A test image on hand for the UI (a real photo works better than an AI one —
  the JPEG-quality slider demo reads clearer on it)

---

## 0:00–0:25 — Cold open: the finding, not the pitch

**On screen:** `results/audit_sid_set.md` or a slide with the one-liner in
large text: `if container == PNG: predict AI` → **100.00% accuracy, zero
pixels read.**

**VO:**
> Here's a detector that gets 100% accuracy on a standard AI-image benchmark
> without reading a single pixel. The real images in this dataset are 100%
> JPEG. The AI images are 100% PNG. So the file extension *is* the label.
> That's not a model problem — it's a leaderboard problem, and it's the
> first thing we found when we started measuring instead of training.

---

## 0:25–1:00 — The real problem: aggregate metrics lie by omission

**On screen:** README results table, then cut to the per-generator TPR
numbers (FLUX.1-dev 0.5375 vs MidJourney 0.8865).

**VO:**
> Once you close that leak properly — canonical decode, stripped metadata,
> every image through the same pipeline — our detector, a frozen CLIP
> backbone with a trained linear head, scores 97 to 99% AUROC across seven
> generators, three of them held completely out of training. Read as one
> number, that's a solved problem.
>
> It isn't. AUROC is threshold-free — it doesn't describe the one operating
> point a real moderation system actually uses. At a fixed 1% false-positive
> budget, this same model catches **88.65%** of MidJourney images and just
> **53.75%** of FLUX.1-dev images. Same detector, same threshold, a 35-point
> swing the aggregate number completely hides.

---

## 1:00–1:40 — What we built to find that

**On screen:** quick file-tree / architecture flash — `src/eval/`,
`src/data/normalize.py`, `runs/baseline.pt`, `runs/aug.pt`.

**VO:**
> So this project is really two things, in order. First, a measurement
> layer: a 19-cell robustness grid — clean images plus JPEG, blur, resize,
> noise, jitter, and crop, individually and composed — validated against a
> null model and a planted-weakness model before we trusted a single number
> out of it. Second, a leak-audited data pipeline that forces every image
> through one canonical decode, closing 12 of the 13 leak channels that
> audit found.
>
> Only after that measurement layer existed did we train the actual model:
> a frozen CLIP ViT-B/16 backbone plus a 513-parameter linear head. That's
> the entire trainable surface — 513 numbers, against a 2-billion-parameter
> budget.

---

## 1:40–2:50 — Live demo

**On screen:** switch to terminal.

**VO:**
> Here's the deliverable that matters most — `predict.py`. Point it at a
> folder of images, no training, no data download, checkpoint's already
> committed.

**Action:** run
```bash
python predict.py --image_dir path/to/images --out preds.json --ckpt runs/baseline.pt
```
**VO (while it runs / on the JSON output):**
> One JSON entry per image, sorted so the output's byte-identical across
> machines. Never crashes on a bad file — a decode failure gets `pred:
> null` and a warning, not a stack trace.

**Action:** switch to browser, `localhost:5173`, upload the test image.

**VO:**
> And for something you can actually feel, not just read: the same scoring
> path, live. Upload an image —

**Action:** show the score appear.

**VO:**
> — and drag this JPEG-quality slider from 95 down to 30.

**Action:** drag slider, let the score visibly move.

**VO:**
> That's not a simulated blur filter — it's a real JPEG re-encode, the exact
> same operation the 19-cell robustness grid measured at population scale.
> What you're watching happen to one image is what happened to 6,810 of them
> to produce the numbers in the last section.

---

## 2:50–3:30 — The honest result on the "obvious" fix

**On screen:** the three-checkpoint comparison table from README/WRITEUP.

**VO:**
> The obvious fix for that FLUX.1-dev gap is training-time augmentation —
> blur and JPEG noise injected during training. We tried it. It moves the
> robustness gap the right direction, about a 24% reduction. But we ran a
> paired bootstrap, not just eyeballing overlapping confidence intervals —
> and at this sample size, that improvement is not statistically
> significant. It also does not touch the per-generator spread at all.
> FLUX.1-dev stays the floor, augmented or not.
>
> On the composite score that weights clean and robust accuracy equally,
> the plain baseline actually beats the augmented checkpoint by 0.0020 — so
> the baseline is what ships, and it's already `predict.py`'s default. I'm
> not hiding the checkpoint that scores worse on my own headline table.

---

## 3:30–4:00 — Why this matters, and what's left

**On screen:** slide or README "Impact" section.

**VO:**
> The pitch isn't "we detect fake images" — everyone says that. It's that
> a platform needs to know *where* its detector is weak, not just that it's
> strong on average — route the
> FLUX.1-dev-confidence range to human review, auto-action the
> MidJourney-confidence range. A hard cutoff throws that information away;
> a triage policy can use it. That only works if someone measures the
> spread first.
>
> The frequency-fusion branch is the other half of this story. I measured
> its precondition first — a texture probe whose per-generator difficulty
> ordering is uncorrelated with CLIP's, rho = -0.143 — built it on that
> evidence, trained and evaluated it at full scale, and it scored 0.9739
> against the baseline's 0.9760. No gain. A necessary precondition wasn't a
> sufficient one, and that's in the repo as a result rather than as "built,
> evaluation pending." 810 real test images is the constraint behind every
> open question here, that one included.

---

## 4:00–4:15 — Close

**On screen:** GitHub repo URL, `README.md` scrolling.

**VO:**
> Everything here — the leak audit, the robustness grid, all three
> checkpoints, the demo — is in the repo, and every number in this video is
> a committed, regeneratable file, not a slide typed once. Thanks for
> watching.

---

## Word-count sanity check

Section VO totals roughly 560–620 words. At a demo-narration pace of
~140–150 wpm that's **~4:00–4:20** of speaking time, plus the terminal/UI
action beats — lands the whole video in the 3:45–4:30 range. If you need to
cut for time, cut from 3:30–4:00 (Impact) first — it's the section that
restates rather than shows.
