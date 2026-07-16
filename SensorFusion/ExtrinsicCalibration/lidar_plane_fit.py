#!/usr/bin/env python3
"""RANSAC plane fit of the calibration board in a Livox HAP point cloud.

The LiDAR cannot see the chessboard *pattern* (no intensity feature is used
here), only the physical planarity of the board/panel: given the raw cloud
and a ROI that isolates the board, this fits the dominant plane and returns
its (normal, distance) parameters plus the inlier points — the LiDAR-side
half of one point-plane calibration observation (Zhang & Pless / Pandey
et al. style, see lidar_camera_extrinsic.py).

The RANSAC/MSAC fit is REUSED from the OcTree module
(PointCloudElaboration/OcTree/octree/smoothing.py, fit_plane_ransac): it is
pure numpy, already validated on real scans, and keeps this module free of
open3d. Only the fit primitive is imported — the voxel-grid machinery around
it (_smooth_surface_ransac, _select_plane_for_axis) is specific to building
facades and not needed for a single cropped board.

Plane convention (shared with camera_plane_pose.py): unit normal `n` oriented
TOWARD the sensor origin, signed distance `d = n . p` for any point `p` on
the plane (so `d < 0` for a board in front of the sensor).

Usage (library; the CLI lives in main.py):
    plane = fit_board_plane(points, roi=(xmin, xmax, ymin, ymax, zmin, zmax))
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THESIS_DIR = Path(__file__).resolve().parents[2]
_OCTREE_DIR = _THESIS_DIR / "PointCloudElaboration" / "OcTree"
sys.path.insert(0, str(_OCTREE_DIR))

from octree.smoothing import fit_plane_ransac  # noqa: E402


@dataclass
class PlaneObservation:
    """One fitted calibration plane, in the observing sensor's frame.

    normal: (3,) unit normal, oriented toward the sensor origin.
    distance: signed offset, normal . p = distance for p on the plane.
    points: (M, 3) inlier points on the plane (LiDAR) or reconstructed board
        corners (camera); used for point-to-plane residuals. May be None.
    rms: RMS point-to-plane distance of `points` (m), or reprojection error
        (px) for a camera observation — a per-observation quality indicator.
    """

    normal: np.ndarray
    distance: float
    points: np.ndarray | None = None
    rms: float = 0.0


def crop_roi(points: np.ndarray, roi: tuple[float, float, float, float, float, float]) -> np.ndarray:
    """Points inside an axis-aligned bounding box (xmin, xmax, ymin, ymax, zmin, zmax).

    Manual ROI selection: pick the box around the board once per pose (e.g.
    from a viewer) so RANSAC fits the board and not the wall behind it.
    """
    pts = np.asarray(points, np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    xmin, xmax, ymin, ymax, zmin, zmax = roi
    mask = (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        & (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    )
    return pts[mask]


def orient_toward_origin(normal: np.ndarray, point_on_plane: np.ndarray) -> tuple[np.ndarray, float]:
    """Flip `normal` so it points toward the sensor origin; return (normal, distance).

    RANSAC/SVD normals have arbitrary sign; a consistent orientation across
    all observations (and across sensors) is what makes the normals usable as
    correspondences in the extrinsic solve.
    """
    n = np.asarray(normal, np.float64)
    n = n / np.linalg.norm(n)
    d = float(n @ np.asarray(point_on_plane, np.float64))
    if d > 0:  # normal points away from the origin: flip
        n, d = -n, -d
    return n, d


def fit_board_plane(
    points: np.ndarray,
    roi: tuple[float, float, float, float, float, float] | None = None,
    threshold: float = 0.02,
    iters: int = 500,
    seed: int = 0,
) -> PlaneObservation:
    """Fit the calibration board's plane in a LiDAR cloud.

    points: (N, 3) LiDAR points in the sensor frame (generic numpy input; the
        Livox loader is plugged in later).
    roi: optional axis-aligned crop applied first (see crop_roi). Without it,
        `points` must already contain (mostly) just the board.
    threshold: RANSAC inlier distance in metres. 0.02 suits the Livox HAP's
        ~cm-level range noise on a board at a few metres.
    iters / seed: hypothesis search budget and RNG seed (deterministic),
        forwarded to octree.smoothing.fit_plane_ransac.

    Returns a PlaneObservation with the inlier points attached (needed by the
    point-to-plane solve in lidar_camera_extrinsic.py).
    """
    pts = np.asarray(points, np.float64)
    if roi is not None:
        pts = crop_roi(pts, roi)
    if len(pts) < 3:
        raise ValueError(f"only {len(pts)} points after ROI crop — need >= 3")

    plane = fit_plane_ransac(pts, threshold=threshold, iters=iters, seed=seed)
    normal, distance = orient_toward_origin(plane.normal, plane.point)

    inliers = pts[plane.inliers]
    rms = float(np.sqrt(np.mean((inliers @ normal - distance) ** 2)))
    return PlaneObservation(normal=normal, distance=distance, points=inliers, rms=rms)
