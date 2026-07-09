"""Zero-shot material classification with CLIP.

Uses openai/clip-vit-base-patch32 through Hugging Face transformers.
Class prompts come from the EmissivityTable, so every class the model can
predict has a corresponding tabulated emissivity value.
"""

import numpy as np
from PIL import Image

from .table import EmissivityTable

MODEL_NAME = "openai/clip-vit-base-patch32"


class MaterialClassifier:
    def __init__(self, table: EmissivityTable, model_name: str = MODEL_NAME):
        # Heavy imports kept local so the rest of the package loads fast.
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.table = table
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def classify(
        self, image: np.ndarray | Image.Image, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Classify the material in the image.

        Args:
            image: RGB image as HxWx3 numpy array or PIL Image.
            top_k: number of (material, confidence) pairs to return.

        Returns:
            List of (material_name, confidence) sorted by confidence.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.processor(
            text=self.table.prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        materials = self.table.materials
        top = probs.argsort(descending=True)[:top_k]
        return [(materials[i], float(probs[i])) for i in top]
