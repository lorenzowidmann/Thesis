#!/usr/bin/env python3
"""LiDAR -> camera extrinsics from N co-observed board planes (point-plane method).

Zhang & Pless' camera-LRF plane-constraint calibration, extended to a 3-D
LiDAR (Pandey et al.): for each pose i of the same physical board, the LiDAR
sees a plane (from lidar_plane_fit.fit_board_plane) and the camera sees the
same plane (from camera_plane_pose.board_plane_from_image). The rigid
transform p_cam = R @ p_lidar + t must map every LiDAR inlier point onto the
camera-frame plane, so the point-to-plane error

    r = n_cam_i . (R @ p + t) - d_cam_i

over all poses i and all (subsampled) LiDAR inliers p is minimized with
scipy.optimize.least_squares (Levenberg-Marquardt). A closed-form seed comes
from the plane parameters alone: Kabsch on the normal pairs for R, linear
least squares on the offsets for t.

Sensor-agnostic on the camera side: pass ZED 2i planes or FLIR planes to get
LiDAR->ZED or LiDAR->thermal with the same code.

Degeneracy note: the poses must span >= 3 significantly different board
orientations, otherwise R (and t along the unobserved directions) is not
constrained. Tilt the board around both axes between captures.

Usage (library; the CLI lives in main.py):
    result = solve_lidar_camera(lidar_planes, camera_planes)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lidar_plane_fit import PlaneObservation

# Cap on LiDAR inlier points per pose used in the refinement: a few hundred
# points constrain a plane just as well as tens of thousands and keep the
# Jacobian small. Deterministic subsample (fixed seed).
_MAX_POINTS_PER_POSE = 500

_MIN_POSES = 3  # fewer poses cannot constrain a 6-DoF transform via planes


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, np.float64)
    T[:3, 3] = np.asarray(t, np.float64).ravel()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (transpose the rotation, re-project t)."""
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def _initial_guess(
    lidar_planes: list[PlaneObservation], camera_planes: list[PlaneObservation]
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form (R, t) seed from the plane parameters alone.

    R: Kabsch/SVD alignment of the LiDAR normals onto the camera normals
    (both are oriented toward their sensor, so no sign ambiguity). t: with
    n_cam ~ R @ n_lidar, a plane transforms as d_cam = d_lidar + n_cam . t,
    so t solves the stacked linear system n_cam_i . t = d_cam_i - d_lidar_i
    (least squares; needs >= 3 non-parallel normals for a unique solution).
    """
    N_l = np.stack([p.normal for p in lidar_planes])   # (N, 3)
    N_c = np.stack([p.normal for p in camera_planes])  # (N, 3)

    H = N_l.T @ N_c
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T

    b = np.array([c.distance - l.distance for l, c in zip(lidar_planes, camera_planes)])
    t, *_ = np.linalg.lstsq(N_c, b, rcond=None)
    return R, t


def _pose_points(lidar_planes: list[PlaneObservation], seed: int = 0) -> list[np.ndarray]:
    """Per-pose LiDAR points for the residuals, subsampled to _MAX_POINTS_PER_POSE.

    Falls back to a single plane-centroid point when a PlaneObservation
    carries no inlier points (plane-parameter-only input still works, just
    with weaker per-pose constraints).
    """
    rng = np.random.default_rng(seed)
    out = []
    for plane in lidar_planes:
        if plane.points is None or len(plane.points) == 0:
            out.append((plane.normal * plane.distance)[None, :])  # closest plane point to origin
            continue
        pts = np.asarray(plane.points, np.float64)
        if len(pts) > _MAX_POINTS_PER_POSE:
            pts = pts[rng.choice(len(pts), _MAX_POINTS_PER_POSE, replace=False)]
        out.append(pts)
    return out


def solve_lidar_camera(
    lidar_planes: list[PlaneObservation],
    camera_planes: list[PlaneObservation],
    seed: int = 0,
) -> dict:
    """Solve p_cam = R @ p_lidar + t from N co-observed board planes.

    lidar_planes[i] and camera_planes[i] must describe the SAME physical board
    pose, both with the toward-the-sensor normal convention. The rotation is
    parametrized as a rotation vector (3 DoF, singularity-free for the small
    corrections after the closed-form seed) and refined together with t by
    Levenberg-Marquardt on the point-to-plane residuals.

    Returns a JSON-ready dict: R (3x3), t (3,), transform (4x4, LiDAR->camera),
    rmse_point_to_plane (m), per_pose_rmse (m), n_poses, n_residuals.
    """
    if len(lidar_planes) != len(camera_planes):
        raise ValueError(
            f"pose count mismatch: {len(lidar_planes)} LiDAR vs {len(camera_planes)} camera planes"
        )
    if len(lidar_planes) < _MIN_POSES:
        raise ValueError(f"need >= {_MIN_POSES} board poses, got {len(lidar_planes)}")

    R0, t0 = _initial_guess(lidar_planes, camera_planes)
    x0 = np.concatenate([Rotation.from_matrix(R0).as_rotvec(), t0])

    pose_points = _pose_points(lidar_planes, seed=seed)
    normals = [p.normal for p in camera_planes]
    distances = [p.distance for p in camera_planes]

    def residuals(x: np.ndarray) -> np.ndarray:
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        t = x[3:]
        res = [pts @ R.T @ n + (n @ t - d) for pts, n, d in zip(pose_points, normals, distances)]
        return np.concatenate(res)

    sol = least_squares(residuals, x0, method="lm")
    R = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    t = sol.x[3:]

    final = residuals(sol.x)
    per_pose, k = [], 0
    for pts in pose_points:
        chunk = final[k : k + len(pts)]
        per_pose.append(float(np.sqrt(np.mean(chunk**2))))
        k += len(pts)

    return {
        "R": R.tolist(),
        "t": t.tolist(),
        "transform": make_transform(R, t).tolist(),
        "rmse_point_to_plane": float(np.sqrt(np.mean(final**2))),
        "per_pose_rmse": per_pose,
        "n_poses": len(lidar_planes),
        "n_residuals": len(final),
    }
