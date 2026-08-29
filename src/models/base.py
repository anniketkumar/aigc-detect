"""The interface the evaluation harness scores through.

Anything the harness can evaluate is a :class:`Scorer`: it takes a batch of
already-transformed RGB PIL images and returns one score in [0, 1] per image,
or ``None`` for an image it cannot score (§9.1 permits ``pred: null``).

The harness owns loading, transforming, batching, and metrics. A scorer owns
preprocessing and the forward pass, and nothing else. Keeping that line sharp is
what lets Phase 3-5 models drop in without touching src/evaluate.py.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from PIL import Image


@runtime_checkable
class Scorer(Protocol):
    """A model that produces P(AI-generated) per image."""

    name: str

    def score(
        self, images: Sequence[Image.Image], image_ids: Sequence[str]
    ) -> list[float | None]:
        """Score a batch.

        Args:
            images: RGB PIL images, transforms already applied. A scorer must
                apply its own resize/normalization here, not before.
            image_ids: stable per-image identifiers, same order and length.
                Provided so a scorer can be deterministic without hashing
                pixels; most real models will ignore them.

        Returns:
            One value per image: a float in [0, 1] where 1 means AI-generated,
            or ``None`` if the image could not be scored.
        """
        ...


#: Populated by the concrete model modules. Kept here so src/evaluate.py can
#: resolve ``--model`` without importing every backbone.
MODEL_REGISTRY: dict[str, str] = {
    "dummy_random": "src.models.dummy:RandomScorer",
    "dummy_brightness": "src.models.dummy:BrightnessScorer",
    "clip_linear": "src.models.clip_baseline:CLIPLinearScorer",
}


def load_model(name: str, **kwargs):
    """Instantiate a registered scorer by name.

    ``kwargs`` are filtered to what the target class actually accepts, so the
    CLI can pass ``seed``/``ckpt``/``device`` uniformly to every model.
    """
    import importlib
    import inspect

    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {name!r}. available: {sorted(MODEL_REGISTRY)}"
        )
    module_path, _, cls_name = MODEL_REGISTRY[name].partition(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    params = inspect.signature(cls).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return cls(**accepted)
