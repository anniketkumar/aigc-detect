"""The source registry: which repos, which class, which generator, how many.

Separated from ``download_data.py`` so the manifest builder, the disk-footprint
estimator and the tests all read the same table instead of three copies drifting
apart.

Every path here was verified against the live Hub on 2026-08-29 (row counts and
parquet sizes come from the datasets-server ``/size`` endpoint, not from a
README), because PLAN.md §4.1 named two datasets whose paths turned out not to
be what the plan assumed.

Strategy, per the Phase 2 decisions
-----------------------------------
Small, diverse, multi-generator, streamed. A linear probe on frozen CLIP
features saturates somewhere around 5-10k images per class, so past that point
another 50k images from the *same* generator buys nothing while another
generator buys a lot. Hence ~2k per generator across six, rather than 12k from
one.

Three real sources, not one, and that is a correction rather than a nicety. The
audit found the AI class is polished and cinematic while SID_Set's real class is
amateur OpenImages material -- a semantic shortcut that survives JPEG and would
make the robustness numbers look excellent while the model had learned "is this
photo well lit". Adding Pexels (professional stock) breaks the correlation
between "polished" and "fake"; adding Megalith (Flickr) keeps the amateur end
populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """One HF parquet dataset contributing to one class."""

    name: str                 # manifest `generator` value
    repo: str
    label: int                # 0 = real, 1 = ai
    family: str               # generator family, for held-out-family splitting
    config: str = "default"
    split: str = "train"
    image_column: str = "image"
    #: Rows in the source, from the datasets-server /size endpoint.
    n_rows: int = 0
    #: Parquet bytes for the whole split, ditto. Used for the footprint estimate.
    total_gb: float = 0.0
    quota: int = 2000
    #: For sources whose split mixes classes. ``saberzl/SID_Set`` puts real (0),
    #: full_synthetic (1) and tampered (2) in the *same* rows, so a source that
    #: does not filter would stamp its own label on all three. That bug shipped
    #: once: "OpenImagesV7" came back with a median geometry of 1024x1024 when
    #: SID_Set's real class is only 3.85% square -- i.e. two thirds of the
    #: supposedly-real images were synthetic. Caught by the source-geometry
    #: table in data_stats.md.
    label_column: str = ""
    label_values: tuple[int, ...] = ()
    #: Held fully out of train, per PLAN.md §4.3.1.
    holdout: bool = False
    notes: str = ""

    @property
    def bytes_per_image(self) -> float:
        return self.total_gb * 1e9 / max(self.n_rows, 1)


# --------------------------------------------------------------------------- #
# AI sources
# --------------------------------------------------------------------------- #
# PLAN.md §4.1 and the Phase 2 decision note both pointed at GenImage. GenImage
# does not survive contact with the data: every one of its mirrored generators
# is fixed low resolution -- BigGAN 128x128, ADM / glide / VQDM all 256x256,
# 100% of sampled rows -- while every usable real source is 768-1152. Under a
# 512 crop none of them yields a single image; under a 224 crop, resolution
# becomes a *perfect* class signal again, merely inverted from the SID_Set case,
# and the only fix would be upscaling one class, which is precisely the
# per-class resampling signature the normalization exists to prevent.
#
# So GenImage is out and these are in: six generators whose native output is
# 1024px, which is the same range as the real sources. Verified against the live
# Hub on 2026-08-29 by sampling actual row dimensions, not by reading a README.
#
# Secondary benefit: these are the generators that matter in 2026. A detector
# that beats 256px ADM output proves very little.

AI_SOURCES: tuple[Source, ...] = (
    Source("SDXL", "bitmind/bm-subnet-stable-diffusion-xl-base-1.0", 1, "sdxl",
           split="2024_10_13_weekly", n_rows=8_978, total_gb=12.5),
    Source("Mobius", "bitmind/bm-subnet-mobius", 1, "sdxl-derivative",
           split="2024_10_13_weekly", n_rows=8_842, total_gb=13.5),
    Source("RealVisXL-V4.0", "bitmind/bm-subnet-RealVisXL_V4.0", 1, "sdxl-derivative",
           split="2024_10_13_weekly", n_rows=8_860, total_gb=12.6),
    Source("Aura", "bitmind/bm-aura-imagegen", 1, "aura",
           n_rows=12_860, total_gb=6.5),

    # --- held out of train entirely (PLAN.md §4.3.1) ----------------------- #
    # Three, not two. All three are closed commercial generators, which is the
    # realistic deployment case: the model will meet generators nobody released
    # weights for. Cross-generator AUROC on these is the headline claim.
    Source("MidJourney", "bitmind/JourneyDB", 1, "midjourney",
           n_rows=670_368, total_gb=503.5, holdout=True,
           notes="held out: closed commercial"),
    Source("Gemini-nano-banana", "bitmind/nano-banana", 1, "gemini",
           n_rows=9_457, total_gb=14.9, holdout=True,
           notes="held out: closed commercial, 2025-era"),
    Source("FLUX.1-dev", "saberzl/SID_Set", 1, "flux",
           split="validation", n_rows=30_022, total_gb=17.0,
           label_column="label", label_values=(1,),
           quota=800, holdout=True,
           notes="held out: SID_Set full_synthetic. Kept eval-only both because "
                 "it is a third unseen generator and because its raw form is "
                 "the leak-confounded set the Phase 2 audit was written about"),
)

# --------------------------------------------------------------------------- #
# Real sources -- three, deliberately
# --------------------------------------------------------------------------- #
# The audit's semantic finding: SID_Set's AI class is polished and cinematic,
# its real class is amateur OpenImages. A model can score well by learning "is
# this well lit", and that shortcut survives JPEG, so robustness_gap would look
# excellent while the model had learned nothing about generation. Pexels is the
# control: professional stock photography, labelled real. With it in the mix,
# "polished" no longer predicts "fake". Task G measures whether that worked.

REAL_SOURCES: tuple[Source, ...] = (
    Source("OpenImagesV7", "saberzl/SID_Set", 0, "openimages",
           split="validation", n_rows=30_022, total_gb=17.0, quota=1800,
           label_column="label", label_values=(0,),
           notes="SID_Set real class; amateur web photography, 667 shapes"),
    Source("Megalith-Flickr", "bitmind/megalith-small", 0, "flickr",
           config="chunk_0000", n_rows=10_000, total_gb=1.94, quota=1800,
           notes="Flickr, permissive licences; 1024x683-ish, the amateur end"),
    Source("Unsplash", "wtcherr/unsplash_5k", 0, "unsplash",
           n_rows=5_000, total_gb=0.65, quota=1800,
           notes="professional photography -- the task-G control that stops "
                 "'polished' from meaning 'fake'. 130 KB/image at source, the "
                 "same range as Megalith (194 KB) and OpenImages, which is why "
                 "it replaced Pexels: see the note below"),

    # Pexels (cj-mills/pexels-110k-768p-min-jpg-depth-anything-large-hf) was the
    # first choice here and was **removed after measurement**. Its sources are
    # 18 KB/image at 768p -- already heavily downscaled and recompressed -- so
    # after the canonical pass its files came out at a median 16 KB against
    # 66-103 KB for every other source. The consequences, measured on a
    # 300-image corpus:
    #
    #     n_bytes AUROC, real vs AI, with Pexels     0.7120
    #     n_bytes AUROC, real vs AI, without Pexels  0.5038
    #     Pexels vs other real sources, by file size 1.0000
    #
    # That is a perfect source fingerprint, and because Pexels was the only
    # polished real source it would have been perfectly predictive of "polished
    # AND real" -- replacing the shortcut task G exists to remove with a new
    # one. Found by the task-F regression test, which is the whole point of
    # having it.
)

SOURCES: tuple[Source, ...] = AI_SOURCES + REAL_SOURCES

BY_NAME: dict[str, Source] = {s.name: s for s in SOURCES}


def train_sources() -> tuple[Source, ...]:
    return tuple(s for s in SOURCES if not s.holdout)


def holdout_sources() -> tuple[Source, ...]:
    return tuple(s for s in SOURCES if s.holdout)


def estimate_footprint(
    sources=SOURCES, crop: int = 512, jpeg_quality: int = 95,
    keep_raw: bool = False, yield_rate: float = 0.85,
) -> dict:
    """Disk footprint, reported *before* anything is downloaded.

    Three numbers, because they are wildly different and only one of them is
    what people mean by "how big is the dataset":

    ``transferred``  bytes pulled over the network. Row groups are the unit of
                     retrieval, so this is the quota inflated by how much of a
                     row group gets discarded.
    ``normalized``   what stays on disk: a 512x512 q95 JPEG each, ~110 KB.
    ``peak``         normalized plus the largest single row group in flight.
                     Only exceeds ``normalized`` meaningfully if ``keep_raw``.

    ``yield_rate`` is the fraction of fetched rows that survive the loader and
    the 512 floor. 0.85 is a placeholder for everything except BigGAN, whose
    128px sources mostly do not survive at all -- see the per-source note.
    """
    # A 512x512 q95 4:2:0 JPEG of photographic content: measured at 55-115 KB on
    # the audit sample, call it 95 KB.
    per_norm = 95_000
    rows = []
    for s in sources:
        want = s.quota
        fetched = want / max(yield_rate, 0.01)
        transferred = fetched * s.bytes_per_image
        rows.append({
            "name": s.name,
            "label": s.label,
            "quota": want,
            "rows_fetched": round(fetched),
            "src_mb_per_img": round(s.bytes_per_image / 1e6, 3),
            "transfer_gb": round(transferred / 1e9, 2),
            "normalized_gb": round(want * per_norm / 1e9, 3),
        })
    total_tx = sum(r["transfer_gb"] for r in rows)
    total_norm = sum(r["normalized_gb"] for r in rows)
    return {
        "per_source": rows,
        "transfer_gb": round(total_tx, 2),
        "normalized_gb": round(total_norm, 2),
        "peak_gb": round(total_norm + (total_tx if keep_raw else 0.2), 2),
        "n_images": sum(r["quota"] for r in rows),
        "assumptions": {
            "crop": crop, "jpeg_quality": jpeg_quality,
            "bytes_per_normalized_image": per_norm, "yield_rate": yield_rate,
            "keep_raw": keep_raw,
        },
    }
