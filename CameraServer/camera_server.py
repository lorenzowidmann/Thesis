#!/usr/bin/env python3
"""Own the camera exclusively, publish frames over shared memory.

Windows locks a UVC device to whichever process opens it first (verified: a
second cv2.VideoCapture on the same index gets zero successful frame reads
while the first is active). Run this once, then point sensor_fusion.py /
drive_view.py / extrinsic_calibration.py at it with --shared instead of
having them open the camera directly -- lets both the headless emissivity
pipeline and the live drive view read the ZED at the same time.

Usage:
    py camera_server.py                  # ZED / webcam at index 0
    py camera_server.py --camera-index 1
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

from emissivity.sources import WebcamSource  # noqa: E402
from shared_frame import FrameWriter  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-index", default="0", metavar="N", help="Device index/path (default 0)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index = emissivity_main.parse_camera_index(args.camera_index)

    with WebcamSource(index) as source:
        frame = source.grab()
        writer = FrameWriter(frame.shape)
        print(f"Publishing {frame.shape[1]}x{frame.shape[0]} frames "
              f"(shared memory '{writer.header_shm.name}' / '{writer.data_shm.name}'). Ctrl-C to stop.")
        try:
            while True:
                writer.publish(source.grab())
        except KeyboardInterrupt:
            pass
        finally:
            writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
