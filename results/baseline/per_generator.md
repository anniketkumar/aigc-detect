# Per-generator breakdown — post-fix baseline

`clean` cell only. AUROC from `results/aesthetic_probe/aesthetic_probe.md` (same
`scores.csv`); TPR@FPR from `results/tpr_analysis/tpr_per_generator_post-fix.csv`
(bootstrap CI, B=300, seed 0). Real pool (n=810) is shared across every row —
each generator is scored against all test reals, not a subset.

Sorted by TPR@FPR=1%, ascending, to surface the spread AUROC hides.

| Generator | n | Held out | Clean AUROC | Clean TPR@1% | Clean TPR@5% | TPR@1% degradation gap | 95% CI |
|---|---:|---|---:|---:|---:|---:|---|
| FLUX.1-dev | 800 | yes | 0.9697 | **0.5112** | 0.8375 | 0.0012 | [-0.059, 0.126] |
| Gemini-nano-banana | 2000 | yes | 0.9708 | 0.6720 | 0.8635 | 0.0508 | [0.008, 0.121] |
| SDXL | 300 | | 0.9778 | 0.7833 | 0.8900 | 0.0206 | [-0.005, 0.075] |
| RealVisXL-V4.0 | 300 | | 0.9847 | 0.7967 | 0.9167 | 0.0379 | [-0.000, 0.103] |
| Mobius | 300 | | 0.9893 | 0.8267 | 0.9500 | 0.0217 | [-0.023, 0.069] |
| MidJourney | 2000 | yes | 0.9922 | 0.8480 | 0.9740 | 0.0437 | [-0.003, 0.103] |
| Aura | 300 | | 0.9949 | **0.8933** | 0.9733 | 0.0695 | [0.040, 0.112] |

**AUROC band: 0.9697–0.9949 (spread 0.0252).** **TPR@FPR=1% band: 0.5112–0.8933
(spread 0.3821).** Same seven generators, same scores, same real pool — AUROC
compresses a 15x-wider spread down to something that reads as "uniformly
strong." At a 1% false-positive budget, FLUX.1-dev is caught on barely half its
images while Aura is caught on nearly 90%.

The two held-out generators at the bottom (FLUX.1-dev, Gemini-nano-banana) are
also the two hardest overall — but MidJourney, the third held-out generator, is
the *second-easiest*, ahead of four trained-on generators. Held-out status
alone does not predict clean-cell difficulty; whatever makes FLUX.1-dev hard is
a property of FLUX.1-dev, not of being unseen during training.

The degradation gap (last two columns) is a different axis and does **not**
track clean difficulty: FLUX.1-dev has the smallest gap of all seven
(essentially zero, CI spans negative) despite being the hardest generator
clean, while Aura has the largest significant gap despite being the easiest
generator clean. See `results/tpr_analysis/report.md` for the full per-cell,
per-generator table and methodology.
