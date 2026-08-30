# Handoff — Phase 3

Deadline: ~31 Aug. Phases 1-2 done. No model exists yet.

## What exists
- **Phase 1** — evaluation harness. 19-cell transform grid (JPEG/blur/resize/
  noise/jitter/crop + 4 composed chains), metrics, `robustness_gap` as a mean of
  family means. Validated against a null model across 20 seeds and against a
  planted-weakness model that fails in the correct cell. See `results/dummy/`.
- **Phase 2** — leak-free data pipeline. See `results/audit_sid_set.md` and
  `results/audit_sampling_verification.md`.

## Why the data pipeline looks paranoid
SID_Set separates perfectly on container format alone: real 100% JPEG, AI 100%
PNG. A model reading file extensions scores 100% having never seen a pixel.
Same for geometry (all AI exactly 1024²) and ICC profiles.

Everything now passes one canonical decode path — strip all metadata, crop at
native resolution (no resampling), single JPEG pass at constant quality. 12 of
13 leak channels are now provably constant.

Do not bypass this path. Three separate leaks were introduced *by fixes for
other leaks* during Phase 2, including a "real photo" source that separated at
AUROC 1.0 on file size alone.

## Non-negotiable
- The WildFake reference subset (4998 COCO val2017 + 8843 DALL·E Advanced) must
  never enter training. **Disqualification risk.** Filenames are renamed and
  cannot be trusted, and enforcement is not uniform across the two halves:
  - COCO val2017 — content hash. `data/forbidden/blocklist.json` carries 5000
    sha256 + 5000 phash (all of val2017, a superset of the 4998), so a
    re-encoded copy is still caught.
  - DALL·E Advanced — **not hashed.** It has no standalone distribution; it
    ships only inside WildFake's ~700 GB ModelScope zips. See the `gaps` field
    in the blocklist. Enforced instead by a source-registry denylist
    (`tests/test_manifest.py::test_no_dalle_derived_source_in_the_registry`),
    which is the weaker guarantee — it blocks a declared source, not content.
- At least one generator stays fully held out of train.
- Both are blocking tests. Don't skip them to unblock yourself.

## Scope
Phase 5 (artifact branch + fusion gate) is CUT. Do not reintroduce it.
Test suite is frozen at 261 — no new tests unless something breaks.

## Workflow
Agent runs on your laptop. Training runs on Colab. Git is the bridge.
`scripts/colab_setup.ipynb` clones, downloads, caches features, trains,
evaluates, and pushes `results/` back. Runtime → T4 GPU.
Images stay ephemeral in `/content`; features and checkpoints go to Drive.

## Your job
Phase 3 — baseline only. Frozen CLIP ViT-B/16 + linear head, no augmentation.
It's the deliberate control, not a serious model.

Expected: clean AUROC 0.85-0.95, sharp collapse at jpeg=30 and resize=0.25,
held-out generator notably lower. **Clean AUROC above 0.99 means a leak, not a
good model — stop and flag it.**

**Spec correction (post Phase 3):** none of the three predictions landed as
written, and none is a leak.
- Clean AUROC came in at 0.9810 — above the 0.85-0.95 band but under the 0.99
  leak line. `results/aesthetic_probe/`'s two-feature texture probe scores
  0.6085 on the same split and its per-generator ordering has ~zero
  correlation (Spearman rho -0.14) with the CLIP head's — the number is not
  aesthetics in disguise.
- No sharp collapse at jpeg=30/resize=0.25 (0.9642, 0.9742 vs. clean 0.9810).
  That prediction was written for the artifact branch (PLAN.md §7.1), which
  was cut — frozen CLIP features are semantic, and semantic evidence survives
  compression. See PLAN.md §5's spec-correction note.
- "Held-out generator notably lower" is true, but not on AUROC (0.9697-0.9949,
  a tight band) and not uniformly across the three held-out generators — it
  shows up as a 0.511-0.893 spread in per-generator TPR@FPR=1%, and the
  hardest generator (FLUX.1-dev) and easiest (Aura) are respectively held-out
  and trained-on. See `results/baseline/per_generator.md`.

The metric that actually resolves a robustness gap here is TPR@FPR=5%, not
AUROC: the AUROC gap (0.0099) sits under its own null SD (0.0108); TPR@FPR=5%
gap is 0.0530 with a 95% CI clearly excluding zero. See
`results/tpr_analysis/report.md`.
