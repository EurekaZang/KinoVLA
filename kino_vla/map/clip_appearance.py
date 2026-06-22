"""Real CLIP appearance encoder for the semantic traversability map (spec §7).

This replaces the named-class surrogate (:mod:`kino_vla.map.appearance`) and the network-free
colour-histogram stand-in (:mod:`kino_vla.map.pixel_appearance`) with a genuine open-vocabulary
CLIP~\\cite{radford2021clip} encoder. It satisfies the same drop-in contract the surrogates
documented---``embed(rgb)`` returns an L2-normalised feature whose cosine tracks visual
homogeneity---but the feature is a real CLIP image embedding, and the encoder additionally exposes
CLIP's open-vocabulary text alignment (``classify``), which is what makes the map *semantic*: a
region is labelled by matching its pixels against a material vocabulary rather than by a hand-set
class string.

Two operations the §7 map needs, both real:

- **Open-vocab labelling** (``classify``): CLIP scores an RGB region against free-form material
  prompts (``"a photo of ice"``, ``"a photo of mud"``, ...) and returns the best label and its
  probability. This is the open-vocabulary segmentation front-end the spec names.
- **Visual-homogeneity feature** (``embed``): the L2-normalised CLIP image embedding, whose cosine
  drives the costmap's similarity propagation ("one broken thin-ice cell condemns the homogeneous
  sheet").

The backbone (``openai/clip-vit-base-patch32`` by default, $512$-d) is loaded once and cached at
module scope, on GPU when available. ``HF_HUB_OFFLINE`` is honoured so a cached model runs without
network.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_CLIP_ID = "openai/clip-vit-base-patch32"

# Material vocabulary for open-vocab labelling, aligned to the benchmark's surfaces. The prompt
# template is the CLIP-standard "a photo of {}"; the label is what the map stores.
DEFAULT_VOCAB: dict[str, str] = {
    "ice": "a photo of a slippery sheet of ice",
    "mud": "a photo of wet brown mud",
    "adhesive": "a photo of a sticky yellow adhesive board",
    "solid_ground": "a photo of solid grey concrete ground",
    "grass": "a photo of green grass",
    "metal": "a photo of a metal grate",
}


@lru_cache(maxsize=2)
def _load_clip(model_id: str) -> tuple:
    """Load (and cache) the CLIP model + processor on the best available device."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor, device


def _to_uint8_image(rgb: np.ndarray) -> object:
    """An (H, W, 3) float/uint8 array → a PIL RGB image (CLIP's expected input)."""
    from PIL import Image

    x = np.asarray(rgb)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError(f"expected (H, W, >=3) RGB, got {x.shape}")
    x = x[..., :3].astype(np.float64)
    if x.max() <= 1.5:  # 0..1 floats → 0..255
        x = x * 255.0
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


class ClipAppearanceEncoder:
    """Real CLIP image features + open-vocabulary labelling (spec §7 perception front-end)."""

    def __init__(
        self, model_id: str = DEFAULT_CLIP_ID, vocabulary: dict[str, str] | None = None
    ) -> None:
        self.model_id = str(model_id)
        self.vocabulary = dict(vocabulary) if vocabulary is not None else dict(DEFAULT_VOCAB)
        self._model = self._processor = self._device = None
        self._text_feats = None  # cached (n_labels, dim) CLIP text features for the vocabulary
        self._labels: list[str] = list(self.vocabulary)

    # ------------------------------------------------------------------ lazy init
    def _ensure(self) -> None:
        if self._model is None:
            self._model, self._processor, self._device = _load_clip(self.model_id)

    @property
    def embed_dim(self) -> int:
        """The CLIP projection dimension (512 for ViT-B/32)."""
        self._ensure()
        return int(self._model.config.projection_dim)

    # ------------------------------------------------------------------ image feature
    def _image_features(self, images: list) -> np.ndarray:
        import torch

        self._ensure()
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            # Canonical, version-robust CLIP feature: vision encoder pooled output → visual
            # projection (the same path get_image_features wraps), then L2-normalise.
            vis = self._model.vision_model(pixel_values=inputs["pixel_values"])
            feats = self._model.visual_projection(vis.pooler_output)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        """Encode one ``(H, W, 3)`` RGB crop to an L2-normalised CLIP feature (drop-in for §7)."""
        return self._image_features([_to_uint8_image(rgb)])[0].astype(np.float64)

    def embed_batch(self, rgbs: list[np.ndarray]) -> np.ndarray:
        return self._image_features([_to_uint8_image(r) for r in rgbs]).astype(np.float64)

    # ------------------------------------------------------------------ open-vocab labelling
    def _vocab_text_features(self) -> np.ndarray:
        import torch

        if self._text_feats is None:
            self._ensure()
            prompts = [self.vocabulary[k] for k in self._labels]
            inputs = self._processor(text=prompts, return_tensors="pt", padding=True).to(
                self._device
            )
            with torch.no_grad():
                txt = self._model.text_model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                tf = self._model.text_projection(txt.pooler_output)
                tf = torch.nn.functional.normalize(tf.float(), dim=-1)
            self._text_feats = tf.cpu().numpy()
        return self._text_feats

    def classify(self, rgb: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Open-vocabulary label for an RGB region: ``(best_label, prob, all_probs)``.

        Cosine of the CLIP image feature against the vocabulary's text features, softmaxed with
        CLIP's learned temperature---the standard zero-shot CLIP classifier. This is the §7
        open-vocab segmentation: the surface is named from pixels against free-form prompts.
        """
        self._ensure()
        img = self.embed(rgb)[None, :]  # (1, dim) already normalised
        txt = self._vocab_text_features()  # (L, dim) normalised
        logit_scale = float(self._model.logit_scale.exp().item())
        logits = logit_scale * (img @ txt.T)[0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        best = int(np.argmax(probs))
        all_probs = {lab: float(probs[i]) for i, lab in enumerate(self._labels)}
        return self._labels[best], float(probs[best]), all_probs
