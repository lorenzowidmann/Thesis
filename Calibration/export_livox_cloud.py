"""Export raw Livox `CustomMsg` scans from a ROS 2 bag, for MATLAB or CloudCompare.

The extrinsic-calibration bags record `/livox/lidar` as
`livox_ros_driver2/msg/CustomMsg`, not `sensor_msgs/PointCloud2`. MATLAB's
`ros2bagreader` cannot deserialise that without first building custom message
support (`ros2genmsg`, which needs a C++ toolchain), and the bags carry an empty
message definition, so nothing can recover the layout from the file alone. This
script decodes it here and hands MATLAB a plain `.mat`.

`PlayCloudBuild.m` opens the `.mat` directly:

    PlayCloudBuild('frames.mat')

The points are in the **sensor** frame, not a map frame: no SLAM pose is applied
and none is needed for a calibration recording, where the LiDAR is bolted down
and only the board moves. Accumulating frames there thickens one static scene
and shows the board jumping between poses. Do not expect the corridor-style
build-up of a `/cloud_registered` bag, which is already registered by FAST-LIO.

Decoding is a direct numpy read of the CDR buffer rather than a generic
deserialiser: the point array is a fixed 20-byte stride, so one `frombuffer` replaces
90k Python objects per scan. Verified bit-identical to `rosbags`'
`deserialize_cdr` on this bag, and ~120x faster.

Usage:
    py export_livox_cloud.py --bag BAG --output frames.mat
    py export_livox_cloud.py --bag BAG --output pose1.mat --start 1 --end 109 --step 5
    py export_livox_cloud.py --bag BAG --output frames.mat --voxel 0.05 --ply cloud.ply

Pair it with check_bag_rate.py, which reports the pose windows to pass to
--start/--end.
"""

from __future__ import annotations

import argparse
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import savemat

try:
    from rosbags.rosbag2 import Reader
except ImportError:  # pragma: no cover - environment problem, not a bug
    raise SystemExit(
        "error: the 'rosbags' package is required (pip install rosbags)"
    ) from None

# livox_ros_driver2/msg/CustomPoint, as laid out by CDR: 19 bytes of members
# rounded up to the 4-byte alignment of the widest one. The trailing pad byte is
# real for every element but the last, which CDR leaves off.
POINT_DTYPE = np.dtype([
    ("offset_time", "<u4"),
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("reflectivity", "u1"), ("tag", "u1"), ("line", "u1"),
    ("_pad", "u1"),
])

# CDR alignment is counted from the start of the payload, i.e. after the
# 4-byte encapsulation header -- not from the start of the buffer.
_ENCAPSULATION = 4


def _align(offset: int, boundary: int) -> int:
    rel = offset - _ENCAPSULATION
    return _ENCAPSULATION + ((rel + boundary - 1) & ~(boundary - 1))


def parse_custom_msg(raw: bytes) -> tuple[np.ndarray, float]:
    """Decode one CustomMsg to (xyz Nx3 float32, header stamp in seconds).

    Layout: Header, uint64 timebase, uint32 point_num, uint8 lidar_id,
    uint8[3] rsvd, CustomPoint[] points.
    """
    off = _ENCAPSULATION
    sec, nsec, frame_len = struct.unpack_from("<iII", raw, off)
    off += 12 + frame_len          # stamp (8) + frame_id length (4) + its chars

    off = _align(off, 8)
    off += 8                       # timebase, not used: header stamp is the clock
    point_num, = struct.unpack_from("<I", raw, off)
    off += 4
    off += 4                       # lidar_id (1) + rsvd (3)

    off = _align(off, 4)
    n, = struct.unpack_from("<I", raw, off)
    off += 4
    if n != point_num:
        raise ValueError(
            f"CustomMsg layout mismatch: array length {n} != point_num {point_num}")

    need = n * POINT_DTYPE.itemsize
    avail = len(raw) - off
    if avail < need:
        # Only ever short by the last element's trailing pad byte.
        if need - avail > POINT_DTYPE.itemsize:
            raise ValueError(f"CustomMsg truncated: {avail} bytes for {need} needed")
        pts = np.frombuffer(bytes(raw[off:]) + b"\x00" * (need - avail),
                            dtype=POINT_DTYPE, count=n)
    else:
        pts = np.frombuffer(raw, dtype=POINT_DTYPE, count=n, offset=off)

    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 0] = pts["x"]
    xyz[:, 1] = pts["y"]
    xyz[:, 2] = pts["z"]
    return xyz, sec + nsec * 1e-9


def voxel_first(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """One point per occupied voxel: the first seen, not the cell average.

    Cheaper than averaging and the difference is under half a voxel, which the
    MATLAB side re-grids anyway. Order is preserved so the scan pattern stays
    recognisable.
    """
    if voxel <= 0 or xyz.size == 0:
        return xyz
    key = np.floor(xyz / voxel).astype(np.int64)
    # Unique on a structured view: one pass over rows instead of a lexsort per axis.
    view = np.ascontiguousarray(key).view(
        np.dtype([("a", "<i8"), ("b", "<i8"), ("c", "<i8")])).ravel()
    _, idx = np.unique(view, return_index=True)
    return xyz[np.sort(idx)]


def write_ply(path: Path, xyz: np.ndarray) -> None:
    """Binary little-endian PLY, the accumulated cloud in one file."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment generated by {Path(__file__).name}\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(np.ascontiguousarray(xyz, dtype="<f4").tobytes())


def read_frames(bag: Path, topic: str, step: int, max_frames: int | None,
                start: float | None, end: float | None, voxel: float,
                verbose: bool) -> tuple[list[np.ndarray], np.ndarray]:
    """Decoded scans and their header stamps, after windowing and decimation."""
    if not bag.exists():
        raise FileNotFoundError(f"bag not found: {bag}")

    frames: list[np.ndarray] = []
    stamps: list[float] = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = sorted({c.topic for c in reader.connections})
            raise RuntimeError(
                f"topic {topic!r} not in bag. Available: " + ", ".join(available))
        msgtype = conns[0].msgtype
        if "CustomMsg" not in msgtype:
            raise RuntimeError(
                f"topic {topic!r} is {msgtype}, not a Livox CustomMsg. "
                "PointCloud2 topics can be read by MATLAB directly.")

        t0 = None
        kept = 0
        for i, (_, _, raw) in enumerate(reader.messages(connections=conns)):
            # Window on the header stamp, which costs 12 bytes to read, before
            # deciding whether the scan is worth decoding at all.
            sec, nsec = struct.unpack_from("<iI", raw, _ENCAPSULATION)
            stamp = sec + nsec * 1e-9
            if t0 is None:
                t0 = stamp
            rel = stamp - t0
            if start is not None and rel < start:
                continue
            if end is not None and rel > end:
                break
            if i % step:
                continue

            xyz, stamp = parse_custom_msg(raw)
            xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
            xyz = voxel_first(xyz, voxel)
            frames.append(xyz)
            stamps.append(stamp)
            kept += 1
            if verbose:
                print(f"  frame {kept}: t={rel:8.2f} s  {len(xyz)} points")
            elif kept % 25 == 0:
                print(f"  {kept} frames, {sum(len(f) for f in frames)} points")
            if max_frames is not None and kept >= max_frames:
                break

    if not frames:
        raise RuntimeError("no scans matched the requested window")
    return frames, np.asarray(stamps, dtype=float)


def parse_args():
    p = argparse.ArgumentParser(
        description="Export Livox CustomMsg scans from a ROS 2 bag to .mat / .ply",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bag", required=True, metavar="DIR",
                   help="rosbag2 directory (the folder holding metadata.yaml)")
    p.add_argument("--topic", default="/livox/lidar", help="CustomMsg topic to export")
    p.add_argument("--output", default="frames.mat", metavar="PATH",
                   help="Destination .mat, readable by PlayCloudBuild.m")
    p.add_argument("--step", type=int, default=10,
                   help="Keep one scan every N. A static scene needs very few")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many kept scans")
    p.add_argument("--start", type=float, default=None, metavar="SEC",
                   help="Window start, seconds from the first scan")
    p.add_argument("--end", type=float, default=None, metavar="SEC",
                   help="Window end, seconds from the first scan")
    p.add_argument("--voxel", type=float, default=0.05, metavar="M",
                   help="Per-scan voxel decimation in metres (0 disables)")
    p.add_argument("--ply", default=None, metavar="PATH",
                   help="Also write the accumulated cloud as a binary PLY")
    p.add_argument("--verbose", action="store_true",
                   help="Report every kept scan, not one line per 25")
    args = p.parse_args()
    if args.step < 1:
        p.error("--step must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        p.error("--max-frames must be >= 1")
    if args.start is not None and args.end is not None and args.end <= args.start:
        p.error("--end must follow --start")
    if args.voxel < 0:
        p.error("--voxel must be >= 0")
    return args


def main():
    args = parse_args()
    bag = Path(args.bag)

    print(f"Reading {args.topic} from {bag}")
    try:
        frames, stamps = read_frames(
            bag, args.topic, args.step, args.max_frames,
            args.start, args.end, args.voxel, args.verbose)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    counts = np.array([len(f) for f in frames], dtype=np.int64)
    xyz = np.vstack(frames).astype(np.float32)
    rel = stamps - stamps[0]

    print(f"\n{len(frames)} scans, {len(xyz)} points "
          f"({counts.mean():.0f} per scan), {rel[-1]:.1f} s span")
    print(f"  X: {xyz[:, 0].min():7.2f} {xyz[:, 0].max():7.2f}")
    print(f"  Y: {xyz[:, 1].min():7.2f} {xyz[:, 1].max():7.2f}")
    print(f"  Z: {xyz[:, 2].min():7.2f} {xyz[:, 2].max():7.2f}")

    out = Path(args.output)
    savemat(out, {
        "xyz": xyz,
        "counts": counts.reshape(-1, 1),
        "stamps": rel.reshape(-1, 1),
        "topic": args.topic,
        "bag": str(bag.resolve()),
        "voxel_size": float(args.voxel),
        "frame": "sensor",
        "tool": Path(__file__).name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }, do_compression=True)
    print(f"Saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  MATLAB:  PlayCloudBuild('{out.resolve()}')")

    if args.ply:
        ply = Path(args.ply)
        write_ply(ply, xyz)
        print(f"Saved -> {ply}  ({ply.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
