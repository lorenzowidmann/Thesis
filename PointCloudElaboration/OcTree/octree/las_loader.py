"""Load a TUM-FACADE .las point cloud into numpy arrays.

Returns the XYZ coordinates and a per-point semantic class label. TUM-FACADE
stores the annotation in the LAS `classification` field; if that field is
uniform (non-annotated cloud) the loader falls back to a scalar extra
dimension named like class/label/scalar_Classification, so it stays robust to
how a particular file was exported.

If the file carries a per-point temperature scalar (from an upstream
LiDAR<->thermal co-registration, once that exists), it is loaded too, from an
extra dimension named like temperature/temp/scalar_Temperature/t_obj. It is
None when absent; downstream code substitutes a synthetic field for the demo.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Extra-dimension names TUM-FACADE / CloudCompare exports have used for labels.
_LABEL_EXTRA_DIMS = ("classification", "class", "label", "scalar_Classification")

# Extra-dimension names a co-registered thermal export might use for per-point
# temperature (degrees C). None of these exist on the current TUM-FACADE sample.
_TEMPERATURE_EXTRA_DIMS = ("temperature", "temp", "scalar_Temperature", "t_obj")


@dataclass
class PointCloud:
    points: np.ndarray  # (N, 3) float64
    labels: np.ndarray  # (N,) int32 semantic class ids
    temperature: np.ndarray | None = None  # (N,) float64 per-point scalar, or None

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


def _extract_temperature(las, dim: str | None = None) -> np.ndarray | None:
    """Best-effort per-point temperature from a laspy LasData object.

    `dim` forces a specific extra-dimension name; otherwise the usual thermal
    export names are tried. Returns None if none is present (the current
    sample), leaving the caller to fall back to a synthetic field.
    """
    available = set(las.point_format.dimension_names)
    candidates = (dim,) if dim else _TEMPERATURE_EXTRA_DIMS
    for name in candidates:
        if name and name in available:
            return np.asarray(las[name], dtype=np.float64)
    return None


def load_las(path: str | Path, temperature_dim: str | None = None) -> PointCloud:
    """Load a .las file into a PointCloud (points in the file's own CRS).

    temperature_dim optionally names the LAS extra dimension holding a
    per-point temperature scalar; when omitted, common thermal names are tried.
    """
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
    temperature = _extract_temperature(las, temperature_dim)
    return PointCloud(points=points, labels=labels, temperature=temperature)
