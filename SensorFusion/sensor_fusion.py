#!/usr/bin/env python3
"""Headless sensor fusion seed: emissivity (camera+CLIP) + distance (LiDAR).

No GUI -- built for the onboard rover PC, whose GPU can't carry the
EmissivityCalculation --show/--live overlay path. Reuses the existing
EmissivityCalculation and LidarDistance modules as-is (loaded from their own
main.py files) instead of reimplementing classification or range-gating.
Every cycle prints emissivity + distance for the same central square to the
terminal. No files, no JSON, no logging -- that comes once this is validated.

Usage:
    py sensor_fusion.py                       # zed-uvc + livox, continuous
    py sensor_fusion.py --once                # single measurement
    py sensor_fusion.py --image photo.jpg --once
    py sensor_fusion.py --precision accurate  # heavier CLIP model
    py sensor_fusion.py --shared              # read the left eye from a running
                                               # CameraServer/camera_server.py instead
                                               # of opening the camera directly -- use
                                               # this to run alongside DriveView (Windows
                                               # only lets one process own a UVC device)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

_THESIS_DIR = Path(__file__).resolve().parent.parent
_EMISSIVITY_DIR = _THESIS_DIR / "EmissivityCalculation"
_LIDAR_DIR = _THESIS_DIR / "LidarDistance"
_CAMERASERVER_DIR = _THESIS_DIR / "CameraServer"

# Sibling modules are plain directories (not installed packages), and
# EmissivityCalculation/LidarDistance both happen to have their own main.py,
# so they're loaded under distinct module names via importlib instead of a
# bare `import main` that would collide. Their packages ("emissivity",
# "livox") don't collide and are added to sys.path normally.
sys.path.insert(0, str(_EMISSIVITY_DIR))
sys.path.insert(0, str(_LIDAR_DIR))
sys.path.insert(0, str(_CAMERASERVER_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emissivity_main = _load_module("_sensorfusion_emissivity_main", _EMISSIVITY_DIR / "main.py")
lidar_main = _load_module("_sensorfusion_lidar_main", _LIDAR_DIR / "main.py")

from emissivity import EmissivityTable, MaterialClassifier  # noqa: E402
from emissivity.sources import ImageSource, WebcamSource, ZedSource, ZedUvcSource  # noqa: E402
from livox import DEFAULT_DATA_PORT, LivoxReceiver  # noqa: E402
from shared_frame import SharedZedSource  # noqa: E402

PRECISION_MODELS = {
    "fast": "openai/clip-vit-base-patch32",       # lighter, default -- weak-GPU rover PC
    "accurate": "openai/clip-vit-large-patch14",  # heavier, matches original GUI tool's default
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless emissivity + LiDAR distance fusion (no GUI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("camera source (default: --zed-uvc)")
    src_group = src.add_mutually_exclusive_group()
    src_group.add_argument("--image", help="Path to a still image instead of a live camera")
    src_group.add_argument("--webcam", action="store_true", help="Use the default webcam")
    src_group.add_argument(
        "--zed", action="store_true",
        help="ZED 2i via the ZED SDK (needs pyzed + NVIDIA GPU/CUDA -- avoid on the rover PC)",
    )
    src_group.add_argument(
        "--zed-uvc", action="store_true",
        help="ZED 2i as a plain UVC webcam (OpenCV only, no SDK/GPU -- default)",
    )
    src_group.add_argument(
        "--shared", action="store_true",
        help="Read the left eye from a running CameraServer/camera_server.py instead of "
             "opening the camera directly -- use this to run alongside DriveView, since "
             "Windows only lets one process own a UVC device at a time",
    )
    src.add_argument("--camera-index", default="0", metavar="N",
                      help="Device index/path for --webcam / --zed-uvc (default 0)")
    src.add_argument("--roi", help="Explicit crop cx,cy,w,h; overrides the default central square")
    src.add_argument("--fraction", type=float, default=0.5,
                      help="Central square as a fraction of the frame's short side, and of the "
                           "LiDAR's --fov-deg (default 0.5, shared between both sensors so they "
                           "look at the same patch)")

    clip = p.add_argument_group("emissivity model")
    clip.add_argument("--precision", choices=sorted(PRECISION_MODELS), default="fast",
                       help="fast = openai/clip-vit-base-patch32 (default, light enough for a "
                            "weak GPU); accurate = openai/clip-vit-large-patch14 (heavier)")
    clip.add_argument("--clip-model", default=None,
                       help="Override with an explicit HF CLIP model name (takes precedence "
                            "over --precision)")
    clip.add_argument("--table", default=None, help="Path to a custom emissivity CSV")
    clip.add_argument("--top-k", type=int, default=3, help="Matches considered (default 3)")

    lidar = p.add_argument_group("LiDAR (Livox)")
    lidar.add_argument("--host-ip", default="0.0.0.0", help="Local NIC IP to bind (default: all)")
    lidar.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    lidar.add_argument("--timeout", type=float, default=3.0, help="Seconds to wait for data")
    lidar.add_argument("--fov-deg", type=float, default=40.0,
                        help="Nominal full angular FOV the square is a fraction of (default 40)")
    lidar.add_argument("--square-deg", type=float, default=None,
                        help="Set the square's angular width directly (overrides --fov-deg/--fraction)")
    lidar.add_argument("--min-range", type=float, default=0.1)
    lidar.add_argument("--max-range", type=float, default=100.0)
    lidar.add_argument("--duration", type=float, default=0.5,
                        help="Seconds of LiDAR points to accumulate per cycle (default 0.5)")

    p.add_argument("--once", action="store_true", help="Single measurement instead of continuous")
    return p.parse_args(argv)


def make_camera_source(args: argparse.Namespace):
    if args.image:
        return ImageSource(args.image)
    if args.shared:
        return SharedZedSource(eye="left")
    index = emissivity_main.parse_camera_index(args.camera_index)
    if args.webcam:
        return WebcamSource(index)
    if args.zed:
        return ZedSource()
    return ZedUvcSource(index)  # default, incl. explicit --zed-uvc


def run_cycle(source, classifier, table, rx, half_rad, args) -> None:
    frame = source.grab()
    roi = args.roi or emissivity_main.default_center_roi(frame, args.fraction)
    crop = emissivity_main.crop_roi(frame, roi)
    best, _confidence = emissivity_main.classify_frame(crop, classifier, table, args.top_k)

    xyz = lidar_main.collect_window(rx, args.duration)
    if xyz.shape[0] == 0:
        print("LiDAR: no data received (device streaming to this host?)", file=sys.stderr)
        print(f"[fusion] emissivity={best.emissivity:.2f} ({best.material})  distance=N/A")
        return

    stats = lidar_main.compute_stats(xyz, half_rad, args.min_range, args.max_range)
    print(f"LiDAR {stats.format()}")
    distance = f"{stats.median:.3f} m" if stats.n else "N/A"
    print(f"[fusion] emissivity={best.emissivity:.2f} ({best.material})  distance={distance}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    half_rad = lidar_main.half_angle_rad(args)

    table = EmissivityTable(args.table) if args.table else EmissivityTable()
    model_name = args.clip_model or PRECISION_MODELS[args.precision]
    classifier = MaterialClassifier(table, model_name=model_name)

    try:
        with make_camera_source(args) as source, LivoxReceiver(args.host_ip, args.data_port, args.timeout) as rx:
            while True:
                run_cycle(source, classifier, table, rx, half_rad, args)
                if args.once:
                    return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
