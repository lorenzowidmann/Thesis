"""Seqlock-protected shared-memory frame buffer: one writer, many readers.

Windows locks a UVC device (e.g. the ZED 2i) to whichever process opens it
first -- verified empirically: a second cv2.VideoCapture on the same index
gets zero successful frame reads while the first is active. camera_server.py
is the one process that owns the real camera; everything else
(drive_view.py, ...) attaches as a FrameReader/SharedZedSource instead of
opening the device itself.

The seqlock (odd seq = write in progress, even = stable) avoids readers ever
seeing a torn frame without needing a real cross-process mutex.

Readers also stamp a shared heartbeat on attach and on every read(), so
camera_server.py can tell when neither eye is actually being consumed and
shut itself down -- see FrameWriter.idle_seconds() / --idle-timeout.
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

HEADER_NAME = "sensorfusion_zed_header"
DATA_NAME = "sensorfusion_zed_data"
HEADER_DTYPE = np.int64  # [seq, height, width, last_activity_ms]


def _cleanup_stale(name: str) -> None:
    """Remove a leftover segment from a previous, uncleanly-stopped server."""
    try:
        stale = shared_memory.SharedMemory(name=name)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


class FrameWriter:
    """Server side: owns the shared memory, publishes one frame at a time."""

    def __init__(self, shape: tuple[int, int, int]):
        h, w, c = shape
        if c != 3:
            raise ValueError(f"expected an HxWx3 frame, got shape {shape}")

        _cleanup_stale(HEADER_NAME)
        _cleanup_stale(DATA_NAME)
        self.header_shm = shared_memory.SharedMemory(name=HEADER_NAME, create=True, size=4 * 8)
        self.data_shm = shared_memory.SharedMemory(name=DATA_NAME, create=True, size=h * w * c)
        self.header = np.ndarray((4,), dtype=HEADER_DTYPE, buffer=self.header_shm.buf)
        # last_activity starts at "now" so a fresh server gets one full
        # idle-timeout grace period even if no client ever attaches.
        self.header[:] = (0, h, w, _now_ms())
        self.shape = shape
        self._data = np.ndarray(shape, dtype=np.uint8, buffer=self.data_shm.buf)

    def publish(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape:
            raise ValueError(f"frame shape {frame.shape} != {self.shape} (camera resolution changed?)")
        seq = int(self.header[0])
        self.header[0] = seq + 1  # odd: write in progress
        self._data[:] = frame
        self.header[0] = seq + 2  # even: stable, new frame ready

    def idle_seconds(self) -> float:
        """Seconds since any reader last attached or called read() -- i.e.
        since either eye was last actually consumed."""
        return (_now_ms() - int(self.header[3])) / 1000.0

    def close(self) -> None:
        self.header_shm.close()
        self.header_shm.unlink()
        self.data_shm.close()
        self.data_shm.unlink()


class FrameReader:
    """Client side: attaches to an already-running camera_server.py."""

    def __init__(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.header_shm = shared_memory.SharedMemory(name=HEADER_NAME)
                self.data_shm = shared_memory.SharedMemory(name=DATA_NAME)
                break
            except FileNotFoundError:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"No camera_server.py publishing '{HEADER_NAME}' -- start it first "
                        "(see CameraServer/README.md)"
                    ) from None
                time.sleep(0.1)

        self.header = np.ndarray((4,), dtype=HEADER_DTYPE, buffer=self.header_shm.buf)
        _, h, w, _ = (int(x) for x in self.header)
        self.shape = (h, w, 3)
        self._data = np.ndarray(self.shape, dtype=np.uint8, buffer=self.data_shm.buf)
        self.header[3] = _now_ms()  # attaching counts as activity too

    def read(self) -> np.ndarray:
        """Return the latest published frame, retrying on a torn read."""
        while True:
            s1 = int(self.header[0])
            if s1 % 2 == 1:
                time.sleep(0.001)
                continue
            frame = self._data.copy()
            s2 = int(self.header[0])
            if s1 == s2:
                self.header[3] = _now_ms()
                return frame

    def close(self) -> None:
        self.header_shm.close()
        self.data_shm.close()


class SharedZedSource:
    """Drop-in replacement for ZedUvcSource when another process (camera_server
    .py) already owns the camera: reads the full stereo frame from shared
    memory and returns one eye, same eye="left"|"right" convention."""

    def __init__(self, eye: str = "left", timeout: float = 5.0):
        if eye not in ("left", "right"):
            raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")
        self.eye = eye
        self._reader = FrameReader(timeout=timeout)

    def grab(self) -> np.ndarray:
        frame = self._reader.read()
        left, right = np.split(frame, 2, axis=1)
        return left if self.eye == "left" else right

    def close(self) -> None:
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
