"""TUM-FACADE semantic classes: id -> name and id -> RGB color.

Class ids and names follow the TUM-FACADE benchmark README (annotated ver1).
Colors are an arbitrary but distinct palette (0-1 floats) used to paint points
and voxels in the viewer. Id 0 is treated as unclassified.
"""

import numpy as np

# id -> (name, (r, g, b)) with r,g,b in 0..1
CLASSES: dict[int, tuple[str, tuple[float, float, float]]] = {
    0: ("unclassified", (0.50, 0.50, 0.50)),
    1: ("wall", (0.85, 0.75, 0.60)),
    2: ("window", (0.20, 0.55, 0.90)),
    3: ("door", (0.60, 0.30, 0.10)),
    4: ("balcony", (0.95, 0.55, 0.15)),
    5: ("molding", (0.80, 0.80, 0.35)),
    6: ("deco", (0.90, 0.40, 0.75)),
    7: ("column", (0.55, 0.40, 0.70)),
    8: ("arch", (0.35, 0.25, 0.55)),
    9: ("drainpipe", (0.30, 0.65, 0.65)),
    10: ("stairs", (0.75, 0.20, 0.20)),
    11: ("ground surface", (0.45, 0.35, 0.25)),
    12: ("terrain", (0.40, 0.60, 0.25)),
    13: ("roof", (0.70, 0.15, 0.30)),
    14: ("blinds", (0.15, 0.70, 0.45)),
    15: ("outer ceiling surface", (0.65, 0.65, 0.75)),
    16: ("interior", (0.90, 0.85, 0.55)),
    17: ("other", (0.25, 0.25, 0.25)),
    # 18 is NOT a TUM-FACADE class: it is a synthetic marker for enclosed voids
    # in a smoothed surface that could not be resolved to a single neighbouring
    # class (see smoothing.fill_enclosed_cells). Vivid magenta so it stands out.
    18: ("unknown", (0.95, 0.10, 0.90)),
}

MAX_CLASS_ID = max(CLASSES)

# Class id assigned to a filled void whose bordering cells disagree (mixed
# classes). Single source of truth, reused by smoothing.fill_enclosed_cells.
UNKNOWN_CLASS_ID = 18


def class_name(class_id: int) -> str:
    return CLASSES.get(int(class_id), ("other", None))[0]


def _color_lut() -> np.ndarray:
    """(MAX_CLASS_ID+1, 3) float array mapping class id -> RGB."""
    lut = np.full((MAX_CLASS_ID + 1, 3), 0.25, dtype=float)
    for cid, (_, rgb) in CLASSES.items():
        lut[cid] = rgb
    return lut


_LUT = _color_lut()


def colorize(labels: np.ndarray) -> np.ndarray:
    """Map an (N,) array of class ids to an (N,3) float RGB array."""
    labels = np.asarray(labels)
    clipped = np.clip(labels, 0, MAX_CLASS_ID).astype(int)
    return _LUT[clipped]
