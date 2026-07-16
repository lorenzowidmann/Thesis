#!/usr/bin/env python3
"""ZED 2i RGB -> FLIR thermal extrinsics via cv2.stereoCalibrate.

Both cameras see the chessboard *pattern* (RGB directly, thermal via the
heated board's active contrast), so unlike the LiDAR pairs this is a plain
corner-correspondence stereo calibration: for each pose where BOTH cameras
detected the full board, the corner sets share the same known 3-D board
geometry, and stereoCalibrate solves the rigid transform between the two
cameras with the (separately calibrated) intrinsics held fixed.

Convention: the returned (R, t) maps RGB-frame points to the thermal frame,
p_thermal = R @ p_rgb + t — same direction convention as
lidar_camera_extrinsic.py (source sensor -> target sensor).

Usage (library; the CLI lives in main.py):
    result = solve_rgb_thermal(rgb_images, thermal_images, K_rgb, dist_rgb, K_th, dist_th)
"""

from __future__ import annotations

import cv2
import numpy as np

from intrinsics_rgb import (
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_SIZE,
    chessboard_object_points,
    detect_corners,
    to_gray,
)
from intrinsics_thermal import preprocess_thermal
from lidar_camera_extrinsic import make_transform

_MIN_POSES = 3


def solve_rgb_thermal(
    rgb_images: list[np.ndarray],
    thermal_images: list[np.ndarray],
    K_rgb: np.ndarray,
    dist_rgb: np.ndarray,
    K_thermal: np.ndarray,
    dist_thermal: np.ndarray,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN,
    square_size: float = DEFAULT_SQUARE_SIZE,
    invert_thermal: bool = False,
) -> dict:
    """Stereo-calibrate the RGB -> thermal rigid transform from paired images.

    rgb_images[i] and thermal_images[i] must show the SAME board pose
    (synchronized capture, or a static board per pose). Poses where either
    camera misses the full pattern are skipped. Intrinsics are fixed
    (CALIB_FIX_INTRINSIC): they come from the dedicated single-camera
    calibrations, which use more views than the paired subset here.

    Returns a JSON-ready dict: R (3x3), t (3,), transform (4x4, RGB->thermal),
    rms reprojection error, n_poses_used / n_poses_total.
    """
    if len(rgb_images) != len(thermal_images):
        raise ValueError(
            f"pose count mismatch: {len(rgb_images)} RGB vs {len(thermal_images)} thermal images"
        )

    obj = chessboard_object_points(pattern_size, square_size)
    obj_points, rgb_points, thermal_points = [], [], []
    for rgb, thermal in zip(rgb_images, thermal_images):
        c_rgb = detect_corners(rgb, pattern_size)
        c_th = detect_corners(preprocess_thermal(thermal, invert=invert_thermal), pattern_size)
        if c_rgb is not None and c_th is not None:
            obj_points.append(obj)
            rgb_points.append(c_rgb)
            thermal_points.append(c_th)

    if len(obj_points) < _MIN_POSES:
        raise RuntimeError(
            f"board found in both cameras in only {len(obj_points)}/{len(rgb_images)} poses "
            f"(need >= {_MIN_POSES}; check thermal contrast and pattern size)"
        )

    h, w = to_gray(thermal_images[0]).shape[:2]
    rms, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
        obj_points, rgb_points, thermal_points,
        K_rgb, dist_rgb, K_thermal, dist_thermal, (w, h),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    t = np.asarray(t, np.float64).ravel()

    return {
        "R": np.asarray(R).tolist(),
        "t": t.tolist(),
        "transform": make_transform(R, t).tolist(),
        "rms_reprojection_error": float(rms),
        "n_poses_used": len(obj_points),
        "n_poses_total": len(rgb_images),
    }
