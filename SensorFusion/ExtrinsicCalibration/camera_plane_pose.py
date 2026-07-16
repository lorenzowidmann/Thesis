#!/usr/bin/env python3
"""Chessboard plane pose in the camera frame, via solvePnP.

Given one image where the chessboard is detected (RGB directly, thermal after
intrinsics_thermal.preprocess_thermal) and the camera's intrinsics, solvePnP
recovers the board pose; the board plane's (normal, distance) in the camera
frame is the camera-side half of one point-plane calibration observation.

Plane convention (shared with lidar_plane_fit.py): unit normal oriented
toward the camera origin, signed distance d = n . p.

Usage (library; the CLI lives in main.py):
    plane = board_plane_from_image(image, K, dist)
"""

from __future__ import annotations

import cv2
import numpy as np

from intrinsics_rgb import (
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_SIZE,
    chessboard_object_points,
    detect_corners,
)
from lidar_plane_fit import PlaneObservation, orient_toward_origin


def board_plane_from_corners(
    corners: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN,
    square_size: float = DEFAULT_SQUARE_SIZE,
) -> PlaneObservation:
    """Board plane in the camera frame from already-detected chessboard corners.

    corners: (cols*rows, 1, 2) float32 as returned by detect_corners.
    K, dist: camera intrinsics (e.g. intrinsics_rgb.load_intrinsics).

    solvePnP gives the board->camera pose (R, t); the board lies in its own
    z = 0 plane, so the plane normal in the camera frame is R's third column
    and t is a point on the plane. The corners, mapped through (R, t), are
    attached as `points` so the extrinsic solve can also use them for
    point-to-plane residuals. `rms` is the reprojection error in pixels.
    """
    obj = chessboard_object_points(pattern_size, square_size)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist)
    if not ok:
        raise RuntimeError("solvePnP failed on the detected corners")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    normal, distance = orient_toward_origin(R[:, 2], t)

    board_points = (R @ obj.T).T + t  # corners in the camera frame, on the plane

    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    # reshape both: cornerSubPix output can be (N, 1, 2) or (N, 2) by version
    err = projected.reshape(-1, 2) - np.asarray(corners).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(err**2)))
    return PlaneObservation(normal=normal, distance=distance, points=board_points, rms=rms)


def board_plane_from_image(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN,
    square_size: float = DEFAULT_SQUARE_SIZE,
) -> PlaneObservation | None:
    """detect_corners + board_plane_from_corners on one already-loaded image.

    `image` must be detection-ready: RGB frames as-is, thermal frames after
    intrinsics_thermal.preprocess_thermal. Returns None when the full board
    is not found (the pose is then skipped, matching the intrinsics scripts).
    """
    corners = detect_corners(image, pattern_size)
    if corners is None:
        return None
    return board_plane_from_corners(corners, K, dist, pattern_size, square_size)
