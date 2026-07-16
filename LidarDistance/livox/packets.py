"""Parse Livox SDK2 (Mid-360 / HAP) point-cloud UDP packets.

Only the point-data packet is decoded -- no control/IMU frames. Two point
formats are supported, matching the Livox SDK2 Ethernet spec:

    data_type 0x01  Cartesian high  int32 x,y,z (mm) + reflectivity + tag  (14 B/pt)
    data_type 0x02  Cartesian low   int16 x,y,z (cm) + reflectivity + tag  ( 8 B/pt)

Everything is little-endian. Coordinates are returned in metres in the Livox
sensor frame (x forward, y left, z up).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# LivoxLidarEthernetPacket header, 36 bytes:
#   version B, length H, time_interval H, dot_num H, udp_cnt H,
#   frame_cnt B, data_type B, time_type B, rsvd 12x, crc32 I, timestamp 8s
_HEADER = struct.Struct("<B4H3B12xI8s")
HEADER_SIZE = _HEADER.size  # 36

DATA_TYPE_CARTESIAN_HIGH = 0x01
DATA_TYPE_CARTESIAN_LOW = 0x02

# Per-point byte size and metres-per-unit scale for each supported data_type.
_POINT_FMT = {
    DATA_TYPE_CARTESIAN_HIGH: (np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                                         ("refl", "u1"), ("tag", "u1")]), 1e-3),
    DATA_TYPE_CARTESIAN_LOW: (np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<i2"),
                                        ("refl", "u1"), ("tag", "u1")]), 1e-2),
}


@dataclass
class PointBlock:
    """Points from one UDP packet, in metres (sensor frame)."""

    xyz: np.ndarray      # (N, 3) float32, metres
    reflectivity: np.ndarray  # (N,) uint8
    tag: np.ndarray      # (N,) uint8

    def __len__(self) -> int:
        return self.xyz.shape[0]


def parse_packet(payload: bytes) -> PointBlock | None:
    """Decode one Livox point-data UDP payload. Returns None for packets that
    are too short or use an unsupported data_type (e.g. spherical/IMU)."""
    if len(payload) < HEADER_SIZE:
        return None

    (_ver, _length, _t_interval, dot_num, _udp_cnt,
     _frame_cnt, data_type, _time_type, _crc32, _ts) = _HEADER.unpack_from(payload)

    fmt = _POINT_FMT.get(data_type)
    if fmt is None or dot_num == 0:
        return None
    dtype, scale = fmt

    need = HEADER_SIZE + dot_num * dtype.itemsize
    if len(payload) < need:
        return None

    raw = np.frombuffer(payload, dtype=dtype, count=dot_num, offset=HEADER_SIZE)
    xyz = np.empty((dot_num, 3), dtype=np.float32)
    xyz[:, 0] = raw["x"] * scale
    xyz[:, 1] = raw["y"] * scale
    xyz[:, 2] = raw["z"] * scale
    return PointBlock(xyz=xyz, reflectivity=raw["refl"].copy(), tag=raw["tag"].copy())
