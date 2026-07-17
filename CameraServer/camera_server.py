#!/usr/bin/env python3
"""Own the camera exclusively, publish frames over shared memory.

Windows locks a UVC device to whichever process opens it first (verified: a
second cv2.VideoCapture on the same index gets zero successful frame reads
while the first is active). Run this once, then point sensor_fusion.py /
drive_view.py / extrinsic_calibration.py at it with --shared instead of
having them open the camera directly -- lets both the headless emissivity
pipeline and the live drive view read the ZED at the same time.

Also self-closes when idle: if neither eye has been read (attached or
read() called) for --idle-timeout seconds, the server shuts itself down and
releases the camera instead of holding it forever after both
sensor_fusion.py and drive_view.py have exited.

Usage:
    py camera_server.py                  # ZED / webcam at index 0
    py camera_server.py --camera-index 1
    py camera_server.py --idle-timeout 60  # wait longer before self-closing
    py camera_server.py --idle-timeout 0   # never self-close (old behavior)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_THESIS_DIR = Path(__file__).resolve().parent.parent
_CAMERASERVER_DIR = Path(__file__).resolve().parent
_EMISSIVITY_DIR = _THESIS_DIR / "EmissivityCalculation"

sys.path.insert(0, str(_CAMERASERVER_DIR))
sys.path.insert(0, str(_EMISSIVITY_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emissivity_main = _load_module("_cameraserver_emissivity_main", _EMISSIVITY_DIR / "main.py")

from emissivity.sources import WebcamSource, find_v4l2_capture_index  # noqa: E402
from shared_frame import FrameWriter  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-index", default=None, metavar="N",
                   help="Device index/path. Omit to auto-detect the ZED capture "
                        "node on Linux (falls back to 0).")
    p.add_argument("--idle-timeout", type=float, default=30.0, metavar="SEC",
                    help="Self-close after this many seconds with no reader activity on "
                         "either eye (default 30; 0 disables self-close, runs until Ctrl-C)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.camera_index is None:
        index = find_v4l2_capture_index(prefer="ZED")
        if index is None:
            index = 0
            print("No V4L2 capture device auto-detected; using index 0.")
        else:
            print(f"Auto-selected camera index {index} (V4L2 capture node).")
    else:
        index = emissivity_main.parse_camera_index(args.camera_index)

    with WebcamSource(index) as source:
        frame = source.grab()
        writer = FrameWriter(frame.shape)
        print(f"Publishing {frame.shape[1]}x{frame.shape[0]} frames "
              f"(shared memory '{writer.header_shm.name}' / '{writer.data_shm.name}'). Ctrl-C to stop.")
        if args.idle_timeout > 0:
            print(f"Self-closing after {args.idle_timeout:.0f}s with no reader activity "
                  f"on either eye (--idle-timeout 0 to disable).")
        try:
            while True:
                writer.publish(source.grab())
                if args.idle_timeout > 0 and writer.idle_seconds() > args.idle_timeout:
                    print(f"No reader activity for {args.idle_timeout:.0f}s -- "
                          f"neither eye is in use, closing.")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
