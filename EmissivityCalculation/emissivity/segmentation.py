"""Superpixel segmentation: the "smallest possible zone" for material
classification when true per-pixel labeling isn't feasible (CLIP classifies
whole crops, not individual pixels).

Uses SLIC (skimage.segmentation.slic) instead of a fixed grid so zones
respect real visual edges (wall/window/radiator boundaries) instead of
cutting objects in half at arbitrary cell lines.
"""

import numpy as np
from skimage.segmentation import slic


def superpixel_segments(image: np.ndarray, n_segments: int = 100, compactness: float = 10.0) -> np.ndarray:
    """RGB HxWx3 uint8 image -> int label raster (HxW), one id per superpixel."""
    return slic(image, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)


def segment_boxes(labels: np.ndarray) -> list[dict]:
    """Per-segment id, bounding box (x0,y0,x1,y1), centroid, and area, in the
    same style as main.py's grid_boxes() -- used to crop each superpixel for
    classification without needing to mask/warp the image itself."""
    segments = []
    for seg_id in np.unique(labels):
        ys, xs = np.nonzero(labels == seg_id)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        segments.append({
            "id": int(seg_id),
            "bbox": (x0, y0, x1, y1),
            "centroid_px": [float(xs.mean()), float(ys.mean())],
            "area_px": int(len(xs)),
        })
    return segments
