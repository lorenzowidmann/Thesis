#!/usr/bin/env python3
"""Intrinsic calibration of the FLIR Vue Pro R thermal camera from chessboard images.

Same OpenCV pipeline as intrinsics_rgb.py (the detection/calibration functions
are imported from there), with thermal-specific preprocessing in front.

IMPORTANT — thermal chessboard detection only works with ACTIVE CONTRAST:
a plain printed chessboard is thermally uniform and invisible in LWIR. During
acquisition the board must be heated (heating panel behind it, or a lamp
shining on it) so the dark squares absorb/emit more than the light ones and
the pattern appears as a temperature difference. Capture while the contrast
is strong; it fades as the board equalizes. Expect a noisier, lower-contrast
pattern than RGB — the preprocessing below (normalize + CLAHE, optional
inversion) exists to recover it.

Usage:
    py intrinsics_thermal.py --images path/to/thermal_chessboard_dir --out thermal_intrinsics.json
    py intrinsics_thermal.py --images dir --invert   # if hot squares appear where cold are expected
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from intrinsics_rgb import (
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_SIZE,
    calibrate_intrinsics,
    load_images_from_dir,
    save_intrinsics,
    to_gray,
)


def preprocess_thermal(image: np.ndarray, invert: bool = False, clahe_clip: float = 3.0) -> np.ndarray:
    """Enhance a thermal frame so findChessboardCorners can see the pattern.

    - Rescales to 8-bit over the frame's own min/max (handles 14/16-bit
      radiometric raw as well as already-8-bit AGC output).
    - CLAHE to stretch the (usually small) hot/cold difference of the heated
      chessboard into usable local contrast.
    - Optional inversion: depending on whether the dark squares run hotter or
      colder than the light ones, the thermal pattern can come out with the
      opposite polarity of a visual chessboard; the detector tolerates either,
      but inverting can help borderline frames.
    """
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if invert:
        gray = cv2.bitwise_not(gray)
    return gray


def calibrate_thermal_intrinsics(
    images: list[np.ndarray],
    pattern_size: tuple[int, int] = DEFAULT_PATTERN,
    square_size: float = DEFAULT_SQUARE_SIZE,
    invert: bool = False,
) -> dict:
    """calibrate_intrinsics on thermally-preprocessed frames (see preprocess_thermal)."""
    prepped = [preprocess_thermal(img, invert=invert) for img in images]
    return calibrate_intrinsics(prepped, pattern_size, square_size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FLIR Vue Pro R intrinsic calibration (heated chessboard, active contrast)"
    )
    p.add_argument("--images", required=True, help="Folder of thermal chessboard images (multi-pose)")
    p.add_argument("--out", default="thermal_intrinsics.json", help="Output JSON path")
    p.add_argument(
        "--pattern", type=int, nargs=2, default=list(DEFAULT_PATTERN), metavar=("COLS", "ROWS"),
        help=f"Inner-corner count (default {DEFAULT_PATTERN[0]} {DEFAULT_PATTERN[1]})",
    )
    p.add_argument(
        "--square-size", type=float, default=DEFAULT_SQUARE_SIZE, metavar="M",
        help=f"Chessboard square edge in metres (default {DEFAULT_SQUARE_SIZE})",
    )
    p.add_argument(
        "--invert", action="store_true",
        help="Invert the thermal image polarity before detection",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    images = load_images_from_dir(args.images)
    result = calibrate_thermal_intrinsics(
        images, tuple(args.pattern), args.square_size, invert=args.invert
    )
    save_intrinsics(result, args.out)
    print(f"Used {result['n_views_used']}/{result['n_views_total']} views, "
          f"RMS reprojection error {result['rms_reprojection_error']:.4f} px")
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
