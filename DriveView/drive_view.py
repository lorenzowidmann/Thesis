#!/usr/bin/env python3
"""Live driving view from the ZED 2i's second lens.

The ZED 2i's left eye is used for material classification in
EmissivityCalculation (now run headless via SensorFusion, no window). This
just shows the *right* eye live in a plain cv2 window so you can see where
the rover is going -- a separate, cheap raw video feed with no CLIP/GPU load,
so it can run alongside the headless SensorFusion process.

Usage:
    py drive_view.py                  # ZED UVC, right eye (default)
    py drive_view.py --eye left       # left eye instead
    py drive_view.py --camera-index 1 # if the ZED isn't device 0
    py drive_view.py --webcam         # plain webcam, for dev without a ZED
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_THESIS_DIR = Path(__file__).resolve().parent.parent
_EMISSIVITY_DIR = _THESIS_DIR / "EmissivityCalculation"
sys.path.insert(0, str(_EMISSIVITY_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emissivity_main = _load_module("_driveview_emissivity_main", _EMISSIVITY_DIR / "main.py")

from emissivity.sources import WebcamSource, ZedUvcSource  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--webcam", action="store_true", help="Use a plain webcam instead of the ZED UVC feed")
    p.add_argument("--camera-index", default="0", metavar="N",
                    help="Device index/path (default 0)")
    p.add_argument("--eye", choices=["left", "right"], default="right",
                    help="Which ZED lens to show (default: right, the second lens)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index = emissivity_main.parse_camera_index(args.camera_index)

    source = WebcamSource(index) if args.webcam else ZedUvcSource(index, eye=args.eye)

    import cv2

    window = f"Drive view ({'webcam' if args.webcam else args.eye + ' eye'}) -- press 'q' to quit"
    print(f"{window}. Press Ctrl+C to stop.")
    try:
        with source:
            while True:
                frame = source.grab()
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(window, bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
