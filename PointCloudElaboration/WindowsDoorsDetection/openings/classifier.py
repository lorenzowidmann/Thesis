"""Zero-shot door/window/other classification with CLIP.

Same structure as EmissivityCalculation/emissivity/classifier.py's
MaterialClassifier -- text prompts encoded once at construction, images
scored in one batched forward pass per frame, CUDA + fp16 auto-detected --
against an OpeningTable instead of an EmissivityTable.

Kept as a separate class rather than parameterising MaterialClassifier: that
one is coupled to EmissivityTable (it returns table.materials and is used by
the emissivity gate), and EmissivityCalculation is a sibling module this one
should not have to modify.
"""

import numpy as np
from PIL import Image

from .table import OpeningTable

# Default matches classify_session.py's measured best: on SAM masks ViT-H/14's
# confidences are calibrated enough to be worth weighting votes by, which is
# exactly what the consensus stage does. ~2.0 s/region on CPU, ~3.9 GB on
# first download.
MODEL_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def _is_cached(model_name: str) -> bool:
    """Best-effort check for a local snapshot, just to print an honest message
    -- from_pretrained() below still handles the actual cache lookup/download."""
    from huggingface_hub import scan_cache_dir

    try:
        repos = scan_cache_dir().repos
    except Exception:
        return False
    return any(r.repo_id == model_name and r.size_on_disk > 0 for r in repos)


class OpeningClassifier:
    def __init__(self, table: OpeningTable, model_name: str = MODEL_NAME,
                 device: str | None = None):
        # Heavy imports kept local so the rest of the package loads fast.
        import torch
        from transformers import CLIPModel, CLIPProcessor

        print(
            "Loading cached CLIP model..." if _is_cached(model_name)
            else f"Downloading CLIP model {model_name} (first run only)..."
        )

        self._torch = torch
        self.table = table

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.device.type == "cpu":
            # Leave one core free so a long session can't starve the rest of
            # the system.
            import os
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

        self.model = CLIPModel.from_pretrained(model_name).to(self.device, dtype=self.dtype)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print(f"OpeningClassifier: {self.device} ({self.dtype}), "
              f"classes: {', '.join(table.classes)}")

        # The prompt list never changes at runtime, so encode it once here;
        # classify_batch() then only runs the image encoder.
        text_inputs = self.processor(text=table.prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_feats = self.model.get_text_features(**text_inputs)
        # transformers >= 5 returns BaseModelOutputWithPooling with the
        # projected embeddings in pooler_output; < 5 returns the tensor.
        if not torch.is_tensor(text_feats):
            text_feats = text_feats.pooler_output
        self._text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    def _image_inputs(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.dtype == self._torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        return inputs

    def _rank(self, probs_row, top_k: int) -> list[tuple[str, float]]:
        classes = self.table.classes
        top = probs_row.argsort(descending=True)[:top_k]
        return [(classes[i], float(probs_row[i])) for i in top]

    def classify(self, image: np.ndarray | Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        """Classify one crop. Returns [(class, confidence), ...], best first."""
        return self.classify_batch([image], top_k=top_k)[0]

    def classify_batch(self, images: list[np.ndarray | Image.Image],
                       top_k: int = 1) -> list[list[tuple[str, float]]]:
        """Score every crop in one batched pass through the CLIP image encoder.

        A frame's worth of SAM masks (dozens to ~100) run one at a time is the
        dominant cost of the per-frame stage on CPU; batching the encoder call
        is the difference between a session finishing in minutes and in hours.

        Returns one [(class, confidence), ...] list per input, same order.
        """
        if not images:
            return []
        pil_images = [Image.fromarray(im) if isinstance(im, np.ndarray) else im for im in images]

        inputs = self._image_inputs(pil_images)
        with self._torch.no_grad():
            image_feats = self.model.get_image_features(**inputs)
        if not self._torch.is_tensor(image_feats):
            image_feats = image_feats.pooler_output
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        # Same computation CLIPModel.forward does for logits_per_image, just
        # against the cached text features.
        # logit_scale is a Parameter, so the product carries requires_grad even
        # though the encoder ran under no_grad -- detach before the softmax so
        # float() on a probability does not warn.
        logits = self.model.logit_scale.exp().detach() * image_feats @ self._text_feats.T
        probs = logits.softmax(dim=1)

        return [self._rank(row, top_k) for row in probs]
