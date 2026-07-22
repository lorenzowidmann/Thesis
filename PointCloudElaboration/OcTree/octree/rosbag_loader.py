"""Load a ROS2 rosbag2 (.db3) LiDAR recording into a PointCloud.

Targets sensor_msgs/msg/PointCloud2 messages (CDR-serialized) written by
rosbag2's sqlite3 storage backend -- e.g. the /cloud_registered topic from a
FAST-LIO/Point-LIO-style SLAM stack -- and also raw livox_ros_driver2/msg/CustomMsg
scans (e.g. /livox/lidar straight from the Livox driver, un-registered). All scan
messages on the chosen topic are merged into a single cloud.

Only stdlib (sqlite3, struct) + numpy are used -- no ROS install or extra
CDR/rosbag package required.
"""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import struct

import numpy as np

from .las_loader import PointCloud

_POINT_CLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
_CUSTOM_MSG_TYPE = "livox_ros_driver2/msg/CustomMsg"
_SUPPORTED_TYPES = (_POINT_CLOUD2_TYPE, _CUSTOM_MSG_TYPE)

# livox_ros_driver2 CustomPoint: uint32 offset_time, float32 x/y/z, uint8
# reflectivity/tag/line. Max member alignment is 4, so each element is padded
# to a 20-byte stride in the CDR sequence.
_CUSTOM_POINT_STRIDE = 20

# sensor_msgs/msg/PointField datatype constants we know how to read.
_FLOAT32 = 7
_FLOAT64 = 8
_FIELD_DTYPES = {_FLOAT32: "f4", _FLOAT64: "f8"}


@dataclass
class _PointField:
    name: str
    offset: int
    datatype: int
    count: int


class _CdrReader:
    """Cursor over a CDR-encoded buffer, positioned after the 4-byte header.

    Every PointCloud2 field up to (and including) the point `data` blob is at
    most 4-byte aligned, so alignment can be tracked as a plain absolute
    offset into the buffer (the 4-byte encapsulation header is itself a
    multiple of 4, so no origin correction is needed).
    """

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 4  # skip encapsulation header

    def _align(self, n: int) -> None:
        rem = self.pos % n
        if rem:
            self.pos += n - rem

    def _align_origin(self, n: int) -> None:
        # CDR alignment is relative to the start of the data, i.e. after the
        # 4-byte encapsulation header. Matters only for n > 4 (e.g. uint64).
        rem = (self.pos - 4) % n
        if rem:
            self.pos += n - rem

    def u64(self) -> int:
        self._align_origin(8)
        val = struct.unpack_from("<Q", self.buf, self.pos)[0]
        self.pos += 8
        return val

    def skip(self, n: int) -> None:
        self.pos += n

    def u8(self) -> int:
        val = self.buf[self.pos]
        self.pos += 1
        return val

    def bool_(self) -> bool:
        return self.u8() != 0

    def i32(self) -> int:
        self._align(4)
        val = struct.unpack_from("<i", self.buf, self.pos)[0]
        self.pos += 4
        return val

    def u32(self) -> int:
        self._align(4)
        val = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return val

    def string(self) -> str:
        length = self.u32()  # includes trailing NUL
        raw = self.buf[self.pos:self.pos + length]
        self.pos += length
        return raw[:-1].decode("utf-8")

    def bytes_(self) -> bytes:
        length = self.u32()
        raw = self.buf[self.pos:self.pos + length]
        self.pos += length
        return raw


def _read_point_field(r: _CdrReader) -> _PointField:
    name = r.string()
    offset = r.u32()
    datatype = r.u8()
    count = r.u32()
    return _PointField(name=name, offset=offset, datatype=datatype, count=count)


def _parse_point_cloud2(msg_bytes: bytes):
    """Parse a sensor_msgs/msg/PointCloud2 CDR message.

    Returns (n_points, point_step, fields, is_bigendian, data_bytes).
    """
    r = _CdrReader(msg_bytes)

    # std_msgs/Header header
    r.i32()  # stamp.sec
    r.u32()  # stamp.nanosec
    r.string()  # frame_id

    height = r.u32()
    width = r.u32()

    n_fields = r.u32()
    fields = [_read_point_field(r) for _ in range(n_fields)]

    is_bigendian = r.bool_()
    point_step = r.u32()
    r.u32()  # row_step
    data = r.bytes_()
    r.bool_()  # is_dense

    return width * height, point_step, fields, is_bigendian, data


def _xyz_from_message(n_points, point_step, fields, is_bigendian, data, point_stride: int = 1) -> np.ndarray:
    by_name = {f.name: f for f in fields}
    missing = [name for name in ("x", "y", "z") if name not in by_name]
    if missing:
        raise ValueError(f"PointCloud2 message missing field(s): {missing}")

    endian = ">" if is_bigendian else "<"
    offsets = []
    formats = []
    for name in ("x", "y", "z"):
        f = by_name[name]
        if f.datatype not in _FIELD_DTYPES:
            raise ValueError(
                f"Unsupported datatype {f.datatype} for field '{name}' "
                f"(only FLOAT32/FLOAT64 are handled)"
            )
        offsets.append(f.offset)
        formats.append(endian + _FIELD_DTYPES[f.datatype])

    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": formats,
        "offsets": offsets,
        "itemsize": point_step,
    })
    arr = np.frombuffer(data, dtype=dtype, count=n_points)[::point_stride]
    pts = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
    return pts[np.isfinite(pts).all(axis=1)]


def _parse_custom_msg(msg_bytes: bytes):
    """Parse a livox_ros_driver2/msg/CustomMsg CDR message.

    Returns (n_points, data_bytes) where data_bytes is the raw CustomPoint
    sequence (20-byte stride per point).
    """
    r = _CdrReader(msg_bytes)

    # std_msgs/Header header
    r.i32()  # stamp.sec
    r.u32()  # stamp.nanosec
    r.string()  # frame_id

    r.u64()  # timebase
    point_num = r.u32()
    r.u8()  # lidar_id
    r.skip(3)  # rsvd[3] uint8

    seq_len = r.u32()  # CustomPoint[] length prefix
    n_points = max(point_num, seq_len)
    data = r.buf[r.pos:]
    return n_points, data


def _xyz_from_custom(n_points, data, point_stride: int = 1) -> np.ndarray:
    # CDR applies no trailing pad after the final element, so the buffer can be
    # up to (stride - member_size) bytes short; pad it out so frombuffer reads
    # the last point cleanly.
    need = n_points * _CUSTOM_POINT_STRIDE
    if len(data) < need:
        data = data + b"\x00" * (need - len(data))

    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": ["<f4", "<f4", "<f4"],
        "offsets": [4, 8, 12],
        "itemsize": _CUSTOM_POINT_STRIDE,
    })
    arr = np.frombuffer(data, dtype=dtype, count=n_points)[::point_stride]
    pts = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
    return pts[np.isfinite(pts).all(axis=1)]


def load_db3(
    path: str | Path, topic: str | None = None, stride: int = 1, point_stride: int = 1,
) -> PointCloud:
    """Load and merge all LiDAR scans on one topic of a rosbag2 .db3 file.

    Supports sensor_msgs/msg/PointCloud2 and livox_ros_driver2/msg/CustomMsg.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rosbag not found: {path}")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, type FROM topics WHERE type IN (?, ?)",
            _SUPPORTED_TYPES,
        )
        candidates = cur.fetchall()
        if topic is not None:
            candidates = [c for c in candidates if c[1] == topic]
            if not candidates:
                raise ValueError(f"No supported LiDAR topic named '{topic}' in {path}")
        elif not candidates:
            supported = " / ".join(_SUPPORTED_TYPES)
            raise ValueError(f"No supported LiDAR topics ({supported}) found in {path}")
        elif len(candidates) > 1:
            names = ", ".join(f"{c[1]} [{c[2]}]" for c in candidates)
            raise ValueError(
                f"Multiple LiDAR topics in {path}: {names}. Pass --db3-topic to pick one."
            )

        topic_id, _topic_name, msg_type = candidates[0]
        # Order by rowid (primary key), not timestamp: some rosbag2 recordings
        # left unfinalized ship a corrupt timestamp_idx that faults ORDER BY
        # timestamp with "database disk image is malformed". All messages are
        # merged into one cloud, so scan ordering does not matter here anyway.
        cur.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY id",
            (topic_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"Topic has no messages in {path}")

    chunks = []
    for (blob,) in rows[::stride]:
        if msg_type == _CUSTOM_MSG_TYPE:
            n_points, data = _parse_custom_msg(bytes(blob))
            if n_points == 0:
                continue
            chunks.append(_xyz_from_custom(n_points, data, point_stride))
        else:
            n_points, point_step, fields, is_bigendian, data = _parse_point_cloud2(bytes(blob))
            if n_points == 0:
                continue
            chunks.append(_xyz_from_message(n_points, point_step, fields, is_bigendian, data, point_stride))

    if not chunks:
        raise ValueError(f"No points decoded from topic in {path}")

    points = np.concatenate(chunks, axis=0)
    labels = np.zeros(len(points), dtype=np.int32)
    return PointCloud(points=points, labels=labels)
