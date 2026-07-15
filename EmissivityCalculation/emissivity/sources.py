"""Frame sources: still image, webcam, and ZED 2i stereo camera.

All sources return RGB numpy arrays (HxWx3, uint8) from grab().
The ZED source imports pyzed lazily so the package works without the SDK.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def grab(self) -> np.ndarray:
        """Return the next frame as an RGB HxWx3 uint8 array."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ImageSource(FrameSource):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Image not found: {self.path}")

    def grab(self) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.path).convert("RGB"))


class WebcamSource(FrameSource):
    def __init__(self, index: int = 0):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {index}")

    def grab(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to grab frame from webcam")
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


class ZedUvcSource(FrameSource):
    """ZED 2i left-eye RGB frames via plain UVC (OpenCV), no ZED SDK/GPU needed.

    Over USB the ZED 2i exposes itself as one wide webcam whose frame is the
    left+right stereo pair concatenated side by side (unrectified). This just
    opens it like any other webcam and crops the left half -- no depth, no
    rectification, but that's not needed for a color crop fed to CLIP.
    """

    def __init__(self, index: int = 0):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open ZED camera (UVC) at index {index}")

    def grab(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to grab frame from ZED camera (UVC)")
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        left, _right = np.split(rgb, 2, axis=1)
        return left

    def close(self) -> None:
        self.cap.release()


class ZedSource(FrameSource):
    """ZED 2i left-eye RGB frames via the ZED SDK Python API (pyzed)."""

    def __init__(self):
        try:
            import pyzed.sl as sl
        except ImportError:
            raise RuntimeError(
                "ZED SDK not installed. Install the ZED SDK from "
                "https://www.stereolabs.com/developers/release/ and then the "
                "pyzed Python API (run the SDK's get_python_api.py). "
                "Requires an NVIDIA GPU with CUDA."
            ) from None

        self._sl = sl
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.depth_mode = sl.DEPTH_MODE.NONE  # depth not needed for emissivity lookup
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Could not open ZED camera: {status}")
        self._mat = sl.Mat()

    def grab(self) -> np.ndarray:
        sl = self._sl
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab frame from ZED camera")
        self.zed.retrieve_image(self._mat, sl.VIEW.LEFT)
        bgra = self._mat.get_data()
        return np.ascontiguousarray(bgra[:, :, [2, 1, 0]])  # BGRA -> RGB

    def close(self) -> None:
        self.zed.close()
