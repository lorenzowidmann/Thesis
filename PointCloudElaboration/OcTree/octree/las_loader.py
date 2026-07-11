"""Load a TUM-FACADE .las point cloud into numpy arrays.

Returns the XYZ coordinates and a per-point semantic class label. TUM-FACADE
stores the annotation in the LAS `classification` field; if that field is
uniform (non-annotated cloud) the loader falls back to a scalar extra
dimension named like class/label/scalar_Classification, so it stays robust to
how a particular file was exported.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Extra-dimension names TUM-FACADE / CloudCompare exports have used for labels.
_LABEL_EXTRA_DIMS = ("classification", "class", "label", "scalar_Classification")


@dataclass
class PointCloud:
    points: np.ndarray  # (N, 3) float64
    labels: np.ndarray  # (N,) int32 semantic class ids

    def __len__(self) -> int:
        return len(self.points)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)


def _extract_labels(las) -> np.ndarray:
    """Best-effort per-point class id from a laspy LasData object."""
    n = len(las.x)

    # Standard LAS classification field, if it actually carries the annotation.
    classification = np.asarray(las.classification, dtype=np.int64)
    if np.unique(classification).size > 1:
        return classification.astype(np.int32)

    # Otherwise look for a labelled extra dimension.
    available = set(las.point_format.dimension_names)
    for name in _LABEL_EXTRA_DIMS:
        if name in available and name != "classification":
            vals = np.asarray(las[name], dtype=np.int64)
            if np.unique(vals).size > 1:
                return vals.astype(np.int32)

    # No usable labels: everything unclassified.
    return np.zeros(n, dtype=np.int32)


def load_las(path: str | Path) -> PointCloud:
    """Load a .las file into a PointCloud (points in the file's own CRS)."""
    import laspy

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Point cloud not found: {path}\n"
            "Run extract_sample.py first to unpack a .las from the TUM-FACADE .7z."
        )

    las = laspy.read(str(path))
    points = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    labels = _extract_labels(las)
    return PointCloud(points=points, labels=labels)
