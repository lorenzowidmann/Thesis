#!/usr/bin/env python3
"""Extrinsic calibration seed: LiDAR -> stereo camera rigid transform.

No calibration target yet, so this is a placeholder: it grabs one synced
camera frame + one LiDAR window as a sanity check that both sensors are
live and roughly co-pointed, then prints the current extrinsics (identity
until a target exists). Swap `solve_extrinsics()` for a real
correspondence-based solve once the target is available -- e.g. paired 3D
points (LiDAR + stereo depth) via Kabsch/SVD, or 2D image points + known 3D
target geometry via cv2.solvePnP.

Usage:
    py extrinsic_calibration.py
    py extrinsic_calibration.py --image photo.jpg
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THESIS_DIR = Path(__file__).resolve().parent.parent
_EMISSIVITY_DIR = _THESIS_DIR / "EmissivityCalculation"
_LIDAR_DIR = _THESIS_DIR / "LidarDistance"

sys.path.insert(0, str(_EMISSIVITY_DIR))
sys.path.insert(0, str(_LIDAR_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emissivity_main = _load_module("_sensorfusion_emissivity_main", _EMISSIVITY_DIR / "main.py")
lidar_main = _load_module("_sensorfusion_lidar_main", _LIDAR_DIR / "main.py")

from emissivity.sources import ImageSource, WebcamSource, ZedSource, ZedUvcSource  # noqa: E402
from livox import DEFAULT_DATA_PORT, LivoxReceiver, compute_stats  # noqa: E402


@dataclass
class Extrinsics:
    """Rigid transform mapping a LiDAR-frame point to the camera frame:
    p_cam = R @ p_lidar + t. Identity/zero until solve_extrinsics() below
    is replaced with a real target-based solve."""

    R: np.ndarray
    t: np.ndarray

    @classmethod
    def identity(cls) -> "Extrinsics":
        return cls(R=np.eye(3), t=np.zeros(3))

    def __str__(self) -> str:
        rows = "\n".join(f"    [{r[0]:+.4f} {r[1]:+.4f} {r[2]:+.4f}]" for r in self.R)
        return f"R =\n{rows}\nt = [{self.t[0]:+.4f} {self.t[1]:+.4f} {self.t[2]:+.4f}]"


def solve_extrinsics(frame: np.ndarray, xyz: np.ndarray) -> Extrinsics:
    """TODO once a calibration target is available: collect matching point
    correspondences between `frame` (image) and `xyz` (LiDAR points, sensor
    frame) and solve for the rigid transform -- e.g. Kabsch/SVD on paired 3D
    points, or cv2.solvePnP on 2D image points + known 3D target geometry.
    Returns identity for now; there's nothing to solve without a target."""
    return Extrinsics.identity()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LiDAR <-> stereo camera extrinsic calibration (no GUI, target TBD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("camera source (default: --zed-uvc)")
    g = src.add_mutually_exclusive_group()
    g.add_argument("--image", help="Still image instead of a live camera")
    g.add_argument("--webcam", action="store_true")
    g.add_argument("--zed", action="store_true", help="Needs pyzed + NVIDIA GPU/CUDA")
    g.add_argument("--zed-uvc", action="store_true", help="Plain UVC, no SDK/GPU -- default")
    src.add_argument("--camera-index", default="0", metavar="N")

    lidar = p.add_argument_group("LiDAR (Livox)")
    lidar.add_argument("--host-ip", default="0.0.0.0", help="Local NIC IP to bind (default: all)")
    lidar.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    lidar.add_argument("--timeout", type=float, default=3.0, help="Seconds to wait for data")
    lidar.add_argument("--duration", type=float, default=0.5,
                        help="Seconds of LiDAR points to accumulate (default 0.5)")
    return p.parse_args(argv)


def make_camera_source(args: argparse.Namespace):
    if args.image:
        return ImageSource(args.image)
    index = emissivity_main.parse_camera_index(args.camera_index)
    if args.webcam:
        return WebcamSource(index)
    if args.zed:
        return ZedSource()
    return ZedUvcSource(index)  # default, incl. explicit --zed-uvc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        with make_camera_source(args) as source, LivoxReceiver(args.host_ip, args.data_port, args.timeout) as rx:
            frame = source.grab()
            xyz = lidar_main.collect_window(rx, args.duration)
    except KeyboardInterrupt:
        return 0

    print(f"Camera frame: {frame.shape[1]}x{frame.shape[0]} px")
    if xyz.shape[0] == 0:
        print("LiDAR: no data received (device streaming to this host?)", file=sys.stderr)
    else:
        # Rough "both sensors see roughly the same patch" sanity check, reusing
        # the same central-square defaults as sensor_fusion.py/LidarDistance --
        # not part of the calibration itself.
        half_rad = lidar_main.half_angle_rad(argparse.Namespace(square_deg=None, fov_deg=40.0, fraction=0.5))
        stats = compute_stats(xyz, half_rad, 0.1, 100.0)
        print(f"LiDAR: {xyz.shape[0]} points this window, {stats.format()}")

    extrinsics = solve_extrinsics(frame, xyz)
    print("\nExtrinsics (LiDAR -> camera), placeholder until a target is used:")
    print(extrinsics)
    print("\nNo calibration target yet -- see solve_extrinsics() TODO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
