#!/usr/bin/env python3
"""Extrinsic calibration pipeline: Livox HAP LiDAR + ZED 2i RGB + FLIR Vue Pro R.

Orchestrates the phase modules over a captured dataset and merges every
estimated transform into one calibration JSON. Each phase can be run alone
(the output file is updated incrementally) or all in sequence with `all`.

Expected dataset layout (loaders are generic — images via cv2.imread, LiDAR
clouds as .npy (N, 3) arrays; plug the real Livox/ZED/FLIR loaders in later
by converting to this layout):

    data/
      intrinsics_rgb/       multi-pose chessboard images (RGB)
      intrinsics_thermal/   multi-pose chessboard images (thermal, heated board)
      poses/
        pose_000/
          rgb.png           ZED 2i frame of the board
          thermal.png       FLIR frame of the same (static) board pose
          lidar.npy         Livox cloud of the same pose, sensor frame
          roi.json          optional [xmin, xmax, ymin, ymax, zmin, zmax]
                            crop isolating the board (overrides --roi)
        pose_001/ ...

Usage:
    py main.py all --data data --out calibration.json
    py main.py intrinsics-rgb --data data
    py main.py lidar-rgb --data data --roi -1 1 -1 1 1 4
    py main.py check --out calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from camera_camera_extrinsic import solve_rgb_thermal
from camera_plane_pose import board_plane_from_image
from consistency_check import check_consistency, print_report
from intrinsics_rgb import (
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_SIZE,
    calibrate_intrinsics,
    load_images_from_dir,
)
from intrinsics_thermal import calibrate_thermal_intrinsics, preprocess_thermal
from lidar_camera_extrinsic import solve_lidar_camera
from lidar_plane_fit import fit_board_plane

_ALL_PHASES = ("intrinsics-rgb", "intrinsics-thermal", "lidar-rgb", "lidar-thermal",
               "rgb-thermal", "check")


# --------------------------------------------------------------------------- #
# Calibration file: one JSON accumulating every phase's result.               #
# --------------------------------------------------------------------------- #
def _load_calibration(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_calibration(calib: dict, path: Path) -> None:
    path.write_text(json.dumps(calib, indent=2))
    print(f"Updated -> {path}")


def _require(calib: dict, key: str, phase: str) -> dict:
    if key not in calib:
        raise SystemExit(f"'{phase}' needs '{key}' in the calibration file — run that phase first")
    return calib[key]


def _intrinsics(calib: dict, key: str, phase: str) -> tuple[np.ndarray, np.ndarray]:
    data = _require(calib, key, phase)
    return np.asarray(data["camera_matrix"], np.float64), np.asarray(data["dist_coeffs"], np.float64)


# --------------------------------------------------------------------------- #
# Dataset access                                                              #
# --------------------------------------------------------------------------- #
def _pose_dirs(data_dir: Path) -> list[Path]:
    poses = sorted(p for p in (data_dir / "poses").iterdir() if p.is_dir())
    if not poses:
        raise FileNotFoundError(f"no pose folders in {data_dir / 'poses'}")
    return poses


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return img


def _pose_roi(pose_dir: Path, cli_roi: list[float] | None) -> tuple | None:
    roi_file = pose_dir / "roi.json"
    if roi_file.exists():
        return tuple(json.loads(roi_file.read_text()))
    return tuple(cli_roi) if cli_roi else None


def _lidar_camera_planes(pose_dirs: list[Path], image_name: str, K, dist, args, thermal: bool):
    """Matched (lidar_planes, camera_planes) over the poses where both succeed."""
    lidar_planes, camera_planes = [], []
    for pose_dir in pose_dirs:
        image = _read_image(pose_dir / image_name)
        if thermal:
            image = preprocess_thermal(image, invert=args.invert_thermal)
        cam = board_plane_from_image(image, K, dist, tuple(args.pattern), args.square_size)
        if cam is None:
            print(f"  {pose_dir.name}: chessboard not found in {image_name}, skipped")
            continue
        points = np.load(pose_dir / "lidar.npy")
        lidar = fit_board_plane(points, roi=_pose_roi(pose_dir, args.roi),
                                threshold=args.ransac_threshold, seed=args.seed)
        lidar_planes.append(lidar)
        camera_planes.append(cam)
        print(f"  {pose_dir.name}: {len(lidar.points)} LiDAR inliers "
              f"(plane rms {lidar.rms*1000:.1f} mm), reproj {cam.rms:.2f} px")
    return lidar_planes, camera_planes


# --------------------------------------------------------------------------- #
# Phases                                                                      #
# --------------------------------------------------------------------------- #
def run_intrinsics_rgb(calib: dict, args) -> None:
    images = load_images_from_dir(Path(args.data) / "intrinsics_rgb")
    result = calibrate_intrinsics(images, tuple(args.pattern), args.square_size)
    calib["rgb_intrinsics"] = result
    print(f"RGB intrinsics: {result['n_views_used']}/{result['n_views_total']} views, "
          f"RMS {result['rms_reprojection_error']:.4f} px")


def run_intrinsics_thermal(calib: dict, args) -> None:
    images = load_images_from_dir(Path(args.data) / "intrinsics_thermal")
    result = calibrate_thermal_intrinsics(images, tuple(args.pattern), args.square_size,
                                          invert=args.invert_thermal)
    calib["thermal_intrinsics"] = result
    print(f"Thermal intrinsics: {result['n_views_used']}/{result['n_views_total']} views, "
          f"RMS {result['rms_reprojection_error']:.4f} px")


def run_lidar_rgb(calib: dict, args) -> None:
    K, dist = _intrinsics(calib, "rgb_intrinsics", "lidar-rgb")
    lidar, cam = _lidar_camera_planes(_pose_dirs(Path(args.data)), "rgb.png", K, dist,
                                      args, thermal=False)
    result = solve_lidar_camera(lidar, cam, seed=args.seed)
    calib["lidar_to_rgb"] = result
    print(f"LiDAR->RGB: {result['n_poses']} poses, "
          f"point-to-plane RMSE {result['rmse_point_to_plane']*1000:.1f} mm")


def run_lidar_thermal(calib: dict, args) -> None:
    K, dist = _intrinsics(calib, "thermal_intrinsics", "lidar-thermal")
    lidar, cam = _lidar_camera_planes(_pose_dirs(Path(args.data)), "thermal.png", K, dist,
                                      args, thermal=True)
    result = solve_lidar_camera(lidar, cam, seed=args.seed)
    calib["lidar_to_thermal"] = result
    print(f"LiDAR->thermal: {result['n_poses']} poses, "
          f"point-to-plane RMSE {result['rmse_point_to_plane']*1000:.1f} mm")


def run_rgb_thermal(calib: dict, args) -> None:
    K_rgb, dist_rgb = _intrinsics(calib, "rgb_intrinsics", "rgb-thermal")
    K_th, dist_th = _intrinsics(calib, "thermal_intrinsics", "rgb-thermal")
    pose_dirs = _pose_dirs(Path(args.data))
    rgb_images = [_read_image(p / "rgb.png") for p in pose_dirs]
    thermal_images = [_read_image(p / "thermal.png") for p in pose_dirs]
    result = solve_rgb_thermal(rgb_images, thermal_images, K_rgb, dist_rgb, K_th, dist_th,
                               tuple(args.pattern), args.square_size,
                               invert_thermal=args.invert_thermal)
    calib["rgb_to_thermal"] = result
    print(f"RGB->thermal: {result['n_poses_used']}/{result['n_poses_total']} poses, "
          f"RMS {result['rms_reprojection_error']:.4f} px")


def run_check(calib: dict, args) -> None:
    report = check_consistency(
        np.asarray(_require(calib, "lidar_to_rgb", "check")["transform"]),
        np.asarray(_require(calib, "rgb_to_thermal", "check")["transform"]),
        np.asarray(_require(calib, "lidar_to_thermal", "check")["transform"]),
    )
    calib["consistency"] = report
    print_report(report)


_RUNNERS = {
    "intrinsics-rgb": run_intrinsics_rgb,
    "intrinsics-thermal": run_intrinsics_thermal,
    "lidar-rgb": run_lidar_rgb,
    "lidar-thermal": run_lidar_thermal,
    "rgb-thermal": run_rgb_thermal,
    "check": run_check,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LiDAR / RGB / thermal extrinsic calibration pipeline (plane-based bridge "
        "solution — see README.md)",
    )
    p.add_argument("phase", choices=_ALL_PHASES + ("all",),
                   help="Which phase to run ('all' = every phase in order)")
    p.add_argument("--data", default="data", help="Dataset root (see module docstring for layout)")
    p.add_argument("--out", default="calibration.json",
                   help="Calibration JSON, updated incrementally across phases")
    p.add_argument("--pattern", type=int, nargs=2, default=list(DEFAULT_PATTERN),
                   metavar=("COLS", "ROWS"), help="Chessboard inner-corner count")
    p.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE, metavar="M",
                   help="Chessboard square edge in metres")
    p.add_argument("--roi", type=float, nargs=6, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="LiDAR crop isolating the board (per-pose roi.json overrides this)")
    p.add_argument("--ransac-threshold", type=float, default=0.02, metavar="M",
                   help="LiDAR plane-fit inlier distance in metres (default 0.02)")
    p.add_argument("--invert-thermal", action="store_true",
                   help="Invert thermal image polarity before chessboard detection")
    p.add_argument("--seed", type=int, default=0, help="RANSAC/subsampling RNG seed (deterministic)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    calib = _load_calibration(out)

    phases = _ALL_PHASES if args.phase == "all" else (args.phase,)
    for phase in phases:
        print(f"\n=== {phase} ===")
        _RUNNERS[phase](calib, args)
        _save_calibration(calib, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
