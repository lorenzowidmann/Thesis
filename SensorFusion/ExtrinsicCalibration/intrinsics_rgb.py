#!/usr/bin/env python3
"""Intrinsic calibration of the ZED 2i RGB camera from chessboard images.

Standard OpenCV pipeline: findChessboardCorners (+ sub-pixel refinement) on a
folder of multi-pose chessboard images, then calibrateCamera. Outputs the
camera matrix K and the distortion coefficients as JSON.

The detection/calibration functions take already-loaded images (numpy arrays),
so the same code is reused by intrinsics_thermal.py and by the stereo step in
camera_camera_extrinsic.py; only the CLI reads files from disk.

Usage:
    py intrinsics_rgb.py --images path/to/rgb_chessboard_dir --out rgb_intrinsics.json
    py intrinsics_rgb.py --images dir --pattern 9 6 --square-size 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Inner-corner count (cols, rows) and square edge in metres. Override per
# target via --pattern / --square-size; the physical square size only scales
# the translation part of the poses, K is unaffected.
DEFAULT_PATTERN = (9, 6)
DEFAULT_SQUARE_SIZE = 0.05

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Sub-pixel refinement termination: standard OpenCV recommendation.
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def chessboard_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    """3-D chessboard corner coordinates in the board's own frame.

    (cols*rows, 3) float32, z = 0, row-major in the same order OpenCV's
    findChessboardCorners reports the 2-D corners.
    """
    cols, rows = pattern_size
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size)
    return obj


def to_gray(image: np.ndarray) -> np.ndarray:
    """Single-channel uint8 view of an image (BGR/RGB 3-channel or already gray)."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        # e.g. 16-bit thermal raw: rescale full range to 8 bit for detection.
        img = image.astype(np.float64)
        lo, hi = img.min(), img.max()
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        return ((img - lo) * scale).astype(np.uint8)
    return image


def _subpix_half_window(corners: np.ndarray, pattern_size: tuple[int, int]) -> int:
    """Refinement half-window from the detected corner spacing, in pixels.

    A fixed (11, 11) window spans more than a whole square when the board is
    small in the image (low-res thermal frames, distant boards) and drags
    corners toward neighbouring saddle points — verified to shift them by
    ~half a square. Keep the window safely inside one square instead.
    """
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    spacing = min(
        float(np.linalg.norm(np.diff(grid, axis=1), axis=2).min()),  # along rows
        float(np.linalg.norm(np.diff(grid, axis=0), axis=2).min()),  # along columns
    )
    return int(np.clip(spacing / 2 - 1, 2, 11))


def detect_corners(
    image: np.ndarray, pattern_size: tuple[int, int] = DEFAULT_PATTERN
) -> np.ndarray | None:
    """Chessboard inner corners of one image, sub-pixel refined.

    Returns (cols*rows, 1, 2) float32 in OpenCV's corner order, or None when
    the full pattern is not found (partial detections are rejected: the
    calibration needs every corner of every used view).
    """
    gray = to_gray(image)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None
    half = _subpix_half_window(corners, pattern_size)
    return cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), _SUBPIX_CRITERIA)


def calibrate_intrinsics(
    images: list[np.ndarray],
    pattern_size: tuple[int, int] = DEFAULT_PATTERN,
    square_size: float = DEFAULT_SQUARE_SIZE,
) -> dict:
    """Calibrate K and distortion from a list of chessboard views.

    Runs detect_corners on every image, keeps the views where the full board
    was found, and solves cv2.calibrateCamera. Returns a JSON-ready dict:
    camera_matrix (3x3), dist_coeffs (k1 k2 p1 p2 k3), image_size (w, h),
    rms reprojection error, n_views_used / n_views_total.
    """
    if not images:
        raise ValueError("no images given")
    h, w = to_gray(images[0]).shape[:2]

    obj = chessboard_object_points(pattern_size, square_size)
    obj_points, img_points = [], []
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            obj_points.append(obj)
            img_points.append(corners)

    if len(obj_points) < 3:
        raise RuntimeError(
            f"chessboard found in only {len(obj_points)}/{len(images)} images "
            "(need >= 3; check pattern size and image contrast)"
        )

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    return {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "image_size": [w, h],
        "rms_reprojection_error": float(rms),
        "n_views_used": len(obj_points),
        "n_views_total": len(images),
        "pattern_size": list(pattern_size),
        "square_size": float(square_size),
    }


def save_intrinsics(result: dict, path: Path | str) -> None:
    """Write a calibrate_intrinsics() result to a JSON file."""
    Path(path).write_text(json.dumps(result, indent=2))


def load_intrinsics(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """(K, dist) numpy arrays from a JSON file written by save_intrinsics()."""
    data = json.loads(Path(path).read_text())
    return np.asarray(data["camera_matrix"], np.float64), np.asarray(data["dist_coeffs"], np.float64)


def load_images_from_dir(folder: Path | str) -> list[np.ndarray]:
    """Load every readable image in a folder (sorted by name), unchanged depth."""
    folder = Path(folder)
    images = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    if not images:
        raise FileNotFoundError(f"no readable images in {folder}")
    return images


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZED 2i RGB intrinsic calibration (chessboard)")
    p.add_argument("--images", required=True, help="Folder of chessboard images (multi-pose)")
    p.add_argument("--out", default="rgb_intrinsics.json", help="Output JSON path")
    p.add_argument(
        "--pattern", type=int, nargs=2, default=list(DEFAULT_PATTERN), metavar=("COLS", "ROWS"),
        help=f"Inner-corner count (default {DEFAULT_PATTERN[0]} {DEFAULT_PATTERN[1]})",
    )
    p.add_argument(
        "--square-size", type=float, default=DEFAULT_SQUARE_SIZE, metavar="M",
        help=f"Chessboard square edge in metres (default {DEFAULT_SQUARE_SIZE})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    images = load_images_from_dir(args.images)
    result = calibrate_intrinsics(images, tuple(args.pattern), args.square_size)
    save_intrinsics(result, args.out)
    print(f"Used {result['n_views_used']}/{result['n_views_total']} views, "
          f"RMS reprojection error {result['rms_reprojection_error']:.4f} px")
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
