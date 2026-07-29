"""
Automatically detect the board poses in a capture session by counting the
"stable" segments between one board move and the next.

Idea: during a pose the board is still -> the difference between consecutive
frames is low. When the board is moved -> a peak of difference appears. The
peaks separate the poses; the stable segments between two peaks are the real
poses.

Three input types are supported:

  * FLIR thermal RJPG:  YYYYMMDD_HHMMSS_R.jpg
      The timestamp lives in the filename (which is also temporal order).

  * ZED right-eye PNG:  right_NNNNNN.png (or any <prefix>_NNNNNN.png)
      No timestamp in the name; frames are ordered by their numeric index.
      Per-frame times are read from a sibling metadata.json
      (schema "zed_record/v1": recording.frame_interval_s, frames[].t_offset_s,
      session.started_utc). If metadata.json is missing, times fall back to
      index / --fps.

  * LiDAR rosbag2 (--db3):  a .db3 file or a rosbag2 folder containing one.
      Reads livox_ros_driver2/msg/CustomMsg (or sensor_msgs/msg/PointCloud2)
      scans. "Difference" between consecutive scans = voxel-occupancy Jaccard
      distance: low while the scene is static (a held board), a peak when the
      board is moved. Pure stdlib (sqlite3 + struct), no ROS install needed.

For images, accepts multiple folders: a long session may be split into separate
folders, which are treated as a single stream.

Usage:
    py detect_board_poses.py <folder1> [<folder2> ...]
    py detect_board_poses.py <folder1> --threshold 8.0 --min-pose-frames 20
    py detect_board_poses.py --db3 <rosbag2_dir_or_file>
    py detect_board_poses.py --db3 <bag> --voxel 0.10 --db3-msg-stride 10

Output: list of the detected poses with index, start/end time, duration in
seconds and number of frames (LiDAR: sampled scans). Also useful to build the
-s/-u windows for `rosbag play`.

NOTE on thermal cameras: the RJPG preview uses an auto-scaled palette (AGC) and
performs a periodic flat-field correction (NUC). Both cause a full-frame
brightness change even when the board is perfectly still, producing spurious
"movement" spikes. To avoid over-counting the poses, only a RUN of at least
--min-move-frames consecutive above-threshold frames is treated as a real board
move; isolated single-frame spikes are ignored.
"""

import argparse
import json
import re
import sqlite3
import struct
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python required:  py -m pip install opencv-python")


IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


def parse_flir_timestamp(name: str):
    """Extract the datetime from a FLIR filename: YYYYMMDD_HHMMSS_R.jpg"""
    stem = name.split("_R")[0]
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def extract_index(name: str):
    """Extract the trailing frame index from names like right_000123.png -> 123."""
    m = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(m.group(1)) if m else None


def collect_images(dirs):
    files = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            sys.exit(f"Folder not found: {p}")
        found = [f for f in p.iterdir() if f.suffix in IMG_EXTS]
        if not found:
            print(f"WARNING: no image found in {p}")
        files.extend(found)

    # Sort: prefer FLIR timestamp, then numeric frame index, then plain name.
    # The category tag keeps the tuple comparisons type-safe.
    def sort_key(f):
        ts = parse_flir_timestamp(f.name)
        if ts:
            return (0, ts.timestamp(), f.name)
        idx = extract_index(f.name)
        if idx is not None:
            return (1, idx, f.name)
        return (2, 0, f.name)

    files.sort(key=sort_key)
    return files


def load_metadata_times(dirs):
    """Look for a zed_record metadata.json in each dir or its parent.

    Returns (offset_by_name, abs_start, path) or None.
      offset_by_name: {filename -> t_offset_s (float)}
      abs_start: datetime of session start (tz-aware) or None
    """
    seen = set()
    for d in dirs:
        for cand in (Path(d) / "metadata.json", Path(d).parent / "metadata.json"):
            if cand in seen or not cand.is_file():
                continue
            seen.add(cand)
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            frames = data.get("frames")
            if not frames:
                continue
            offset_by_name = {
                fr["file"]: float(fr["t_offset_s"])
                for fr in frames
                if "file" in fr and "t_offset_s" in fr
            }
            if not offset_by_name:
                continue
            abs_start = None
            started = (data.get("session") or {}).get("started_utc")
            if started:
                try:
                    abs_start = datetime.fromisoformat(started.replace("Z", "+00:00"))
                except ValueError:
                    abs_start = None
            return offset_by_name, abs_start, cand
    return None


def build_time_index(files, dirs, fps):
    """Build {filename -> offset_seconds} and the absolute session start.

    Priority: FLIR filename timestamp -> metadata.json -> index / fps.
    Returns (offset_by_name, abs_start, source_str).
    """
    # FLIR: timestamps embedded in the filename take priority. metadata.json is
    # a ZED-only sidecar and must never be applied to a FLIR session.
    if parse_flir_timestamp(files[0].name) is not None:
        t0 = parse_flir_timestamp(files[0].name)
        offsets = {}
        for f in files:
            ts = parse_flir_timestamp(f.name)
            offsets[f.name] = (ts - t0).total_seconds() if ts else 0.0
        return offsets, t0, "FLIR filename timestamps"

    meta = load_metadata_times(dirs)
    if meta is not None:
        offset_by_name, abs_start, path = meta
        # Only keep entries for files we actually loaded; warn on gaps.
        missing = [f.name for f in files if f.name not in offset_by_name]
        if missing:
            print(f"WARNING: {len(missing)} frame(s) not listed in metadata.json "
                  f"(e.g. {missing[0]}); those fall back to index/fps.")
        base = min(offset_by_name.values()) if offset_by_name else 0.0
        offsets = {}
        for i, f in enumerate(files):
            if f.name in offset_by_name:
                offsets[f.name] = offset_by_name[f.name] - base
            else:
                offsets[f.name] = i / fps
        return offsets, abs_start, f"metadata.json ({path})"

    # Fallback: assume constant frame rate
    offsets = {f.name: i / fps for i, f in enumerate(files)}
    return offsets, None, f"constant {fps} fps (no metadata)"


def frame_difference(prev_gray, cur_gray):
    """Mean absolute difference between two grayscale frames."""
    diff = cv2.absdiff(prev_gray, cur_gray)
    return float(np.mean(diff))


def fmt_clock(abs_start, offset):
    """HH:MM:SS at abs_start+offset, or the raw offset if no absolute start."""
    if abs_start is not None:
        return (abs_start + timedelta(seconds=offset)).strftime("%H:%M:%S")
    return f"{offset:.1f}s"


# ----------------------------------------------------------------------------
# LiDAR rosbag2 (.db3) support
# ----------------------------------------------------------------------------
_CUSTOM_MSG_TYPE = "livox_ros_driver2/msg/CustomMsg"
_POINT_CLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
_CUSTOM_POINT_STRIDE = 20  # CustomPoint: u32 offset_time, f32 x/y/z, 3x u8 -> 20


def _resolve_db3(path):
    """Accept a .db3 file or a rosbag2 folder; return the .db3 Path."""
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("*.db3"))
        if not cands:
            sys.exit(f"No .db3 file found in {p}")
        return cands[0]
    if not p.is_file():
        sys.exit(f"Rosbag not found: {p}")
    return p


def _custom_msg_xyz(buf, point_stride):
    """Parse livox CustomMsg CDR bytes -> (N,3) float32 xyz."""
    pos = 4                                   # skip encapsulation header
    pos += 4 + 4                              # header.stamp sec + nanosec
    slen = struct.unpack_from("<I", buf, pos)[0]; pos += 4 + slen  # frame_id
    rem = (pos - 4) % 8                       # align u64 (timebase)
    if rem:
        pos += 8 - rem
    pos += 8                                  # timebase
    point_num = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    pos += 1 + 3                              # lidar_id + rsvd[3]
    seq_len = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    n = max(point_num, seq_len)
    data = buf[pos:]
    need = n * _CUSTOM_POINT_STRIDE
    if len(data) < need:                      # CDR pads no trailing element
        data = data + b"\x00" * (need - len(data))
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4", "<f4", "<f4"],
                   "offsets": [4, 8, 12], "itemsize": _CUSTOM_POINT_STRIDE})
    arr = np.frombuffer(data, dtype=dt, count=n)[::point_stride]
    p = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return p[np.isfinite(p).all(axis=1)]


def _pointcloud2_xyz(buf, point_stride):
    """Parse sensor_msgs/PointCloud2 CDR bytes -> (N,3) float32 xyz (f32 x/y/z)."""
    pos = 4
    pos += 4 + 4
    slen = struct.unpack_from("<I", buf, pos)[0]; pos += 4 + slen  # frame_id
    height = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    width = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    n_fields = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    fields = {}
    for _ in range(n_fields):
        fl = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        name = buf[pos:pos + fl][:-1].decode(); pos += fl
        offset = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        datatype = buf[pos]; pos += 1
        pos += 4                              # count
        fields[name] = (offset, datatype)
    pos += 1                                  # is_bigendian
    point_step = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    pos += 4                                  # row_step
    dlen = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    data = buf[pos:pos + dlen]
    n = width * height
    ox, oy, oz = fields["x"][0], fields["y"][0], fields["z"][0]
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4", "<f4", "<f4"],
                   "offsets": [ox, oy, oz], "itemsize": point_step})
    arr = np.frombuffer(data, dtype=dt, count=n)[::point_stride]
    p = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return p[np.isfinite(p).all(axis=1)]


def _voxel_keys(points, voxel):
    """Set of occupied-voxel keys (int64) for a cloud."""
    q = np.floor(points / voxel).astype(np.int64) + (1 << 19)  # keep positive
    keys = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
    return set(np.unique(keys).tolist())


def lidar_profile(db3, topic, msg_stride, point_stride, voxel):
    """Build the scan-to-scan difference profile of a rosbag2 LiDAR recording.

    Returns (ts, values, abs_start, source, n_total, eff_rate).
      ts     : offset seconds of each sampled scan (len = n_samples)
      values : voxel-occupancy Jaccard distance vs the previous sample
               (len = n_samples - 1)
    """
    conn = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT id, name, type FROM topics WHERE type IN (?, ?)",
            (_CUSTOM_MSG_TYPE, _POINT_CLOUD2_TYPE),
        ).fetchall()
        if topic is not None:
            rows = [r for r in rows if r[1] == topic]
        if not rows:
            sys.exit(f"No supported LiDAR topic ({topic or 'CustomMsg/PointCloud2'}) "
                     f"in {db3}")
        if len(rows) > 1:
            names = ", ".join(f"{r[1]} [{r[2]}]" for r in rows)
            sys.exit(f"Multiple LiDAR topics: {names}. Pass --db3-topic to pick one.")
        topic_id, topic_name, msg_type = rows[0]

        idx = cur.execute(
            "SELECT id, timestamp FROM messages WHERE topic_id=? ORDER BY id",
            (topic_id,),
        ).fetchall()
        if len(idx) < 2:
            sys.exit("Topic has fewer than 2 scans.")
        sel = idx[::msg_stride]
        t0_ns = idx[0][1]
        abs_start = datetime.fromtimestamp(t0_ns / 1e9, tz=timezone.utc)
        parse = _custom_msg_xyz if msg_type == _CUSTOM_MSG_TYPE else _pointcloud2_xyz

        print(f"LiDAR topic: {topic_name} [{msg_type}]")
        print(f"Scans: {len(idx)}  sampled: {len(sel)} (every {msg_stride})")

        ts, values = [], []
        prev = None
        for k, (mid, ns) in enumerate(sel):
            blob = cur.execute("SELECT data FROM messages WHERE id=?", (mid,)).fetchone()[0]
            keys = _voxel_keys(parse(bytes(blob), point_stride), voxel)
            ts.append((ns - t0_ns) / 1e9)
            if prev is not None:
                uni = len(prev | keys)
                values.append(1.0 - len(prev & keys) / uni if uni else 0.0)
            prev = keys
            if (k + 1) % 100 == 0:
                print(f"  ...{k + 1}/{len(sel)}")
    finally:
        conn.close()

    ts = np.array(ts)
    dur = ts[-1] - ts[0]
    eff_rate = (len(ts) - 1) / dur if dur > 0 else 0.0
    src = f"rosbag2 {Path(db3).name} ({topic_name})"
    return ts, np.array(values), abs_start, src, len(idx), eff_rate


def segment(values, threshold, min_move_frames, min_pose_frames):
    """Debounced segmentation shared by the image and LiDAR paths.

    Returns (poses, transitions): index ranges into `values`.
    """
    moving = values > threshold
    n = len(moving)
    transitions = []
    i = 0
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            if j - i >= min_move_frames:
                transitions.append((i, j))
            i = j
        else:
            i += 1
    poses = []
    prev_end = 0
    for a, b in transitions:
        if a - prev_end >= min_pose_frames:
            poses.append((prev_end, a))
        prev_end = b
    if n - prev_end >= min_pose_frames:
        poses.append((prev_end, n))
    return poses, transitions


def _pose_intervals(value_times, values, args, min_move, min_pose, mad_k):
    """Segment + duration-filter -> list of (start_s, end_s) pose intervals."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    thr = (args.threshold if args.threshold is not None
           else median + mad_k * (mad if mad > 1e-6 else 1.0))
    poses, _ = segment(values, thr, min_move, min_pose)

    def span(a, b):
        return value_times[a], value_times[min(b, len(value_times)) - 1]

    intervals = [span(a, b) for a, b in poses]
    if args.min_duration is not None:
        intervals = [(s, e) for (s, e) in intervals if e - s >= args.min_duration]
    return intervals


def sensor_pose_intervals(kind, args):
    """Detect poses for one sensor. Returns (name, intervals) where intervals is
    a list of (start_s, end_s) relative to that sensor's own recording start."""
    if kind == "lidar":
        db3 = _resolve_db3(args.db3)
        print(f"[LiDAR] {db3.name}")
        vt_all, values, _abs, _src, _n, eff = lidar_profile(
            db3, args.db3_topic, args.db3_msg_stride, args.db3_point_stride, args.voxel)
        vt = vt_all[1:]
        min_move = args.min_move_frames if args.min_move_frames is not None else 1
        min_pose = (args.min_pose_frames if args.min_pose_frames is not None
                    else max(4, round(10.0 * eff)))
        mad_k = args.mad_k if args.mad_k is not None else 4.0
        name = "LiDAR"
    else:
        dirs = args.flir if kind == "flir" else args.zed
        name = "FLIR" if kind == "flir" else "ZED"
        print(f"[{name}] {', '.join(dirs)}")
        files = collect_images(dirs)
        if len(files) < 2:
            sys.exit(f"{name}: at least 2 images required.")
        vt, values, _abs, _src, _labels = image_profile(files, dirs, args)
        min_move = args.min_move_frames if args.min_move_frames is not None else 3
        min_pose = args.min_pose_frames if args.min_pose_frames is not None else 15
        mad_k = args.mad_k if args.mad_k is not None else 6.0

    intervals = _pose_intervals(vt, values, args, min_move, min_pose, mad_k)
    return name, intervals


def compare_sensors(args):
    """Detect poses on each provided sensor and compare inter-pose gaps."""
    sensors = []
    if args.flir:
        sensors.append(sensor_pose_intervals("flir", args))
    if args.zed:
        sensors.append(sensor_pose_intervals("zed", args))
    if args.db3:
        sensors.append(sensor_pose_intervals("lidar", args))
    if len(sensors) < 2:
        sys.exit("--compare needs at least two of --flir / --zed / --db3.")

    print()
    print("=" * 78)
    print("POSES DETECTED PER SENSOR")
    for name, iv in sensors:
        print(f"  {name:<6} {len(iv)} poses")
    counts = {len(iv) for _, iv in sensors}
    n = min(len(iv) for _, iv in sensors)
    if len(counts) > 1:
        print(f"  WARNING: pose counts differ -> aligning the first {n} by order; "
              "a missed pose in one sensor will misalign the rest.")
    print("=" * 78)

    if n < 2:
        sys.exit("Need at least 2 poses per sensor to have an inter-pose gap.")

    names = [name for name, _ in sensors]
    print("Inter-pose gap = seconds from one pose's END to the next pose's START")
    print("(relative within each sensor, so different clocks do not matter).")
    print()
    header = f"{'gap':>7}  " + "  ".join(f"{nm:>8}" for nm in names) + "   | maxdiff"
    print(header)
    print("-" * len(header))
    for i in range(n - 1):
        gaps = []
        for _, iv in sensors:
            gaps.append(iv[i + 1][0] - iv[i][1])
        maxdiff = max(gaps) - min(gaps)
        cells = "  ".join(f"{g:7.1f}s" for g in gaps)
        print(f"{i + 1:>3}->{i + 2:<2}  {cells}   | {maxdiff:6.1f}s")
    print()
    print("NOTE: large maxdiff on a row = the sensors disagree on that interval, "
          "usually a missed/extra pose in one stream. Tune per-sensor detection "
          "with --threshold / --mad-k / --min-duration.")


def image_profile(files, dirs, args):
    """Frame-difference profile of an image session.

    Returns (value_times, values, abs_start, source, labels) where value_times
    and values are aligned per consecutive-frame pair.
    """
    offsets, abs_start, source = build_time_index(files, dirs, args.fps)
    print(f"Images found: {len(files)}   Time source: {source}")
    print(f"First: {files[0].name}   Last: {files[-1].name}")

    print("Computing differences between consecutive frames...")
    vt, values, labels = [], [], []
    prev = None
    skipped = 0
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1               # 0-byte / corrupt frame
            continue
        if args.downscale > 1:
            img = cv2.resize(img, (img.shape[1] // args.downscale,
                                   img.shape[0] // args.downscale))
        if prev is not None:
            values.append(frame_difference(prev, img))
            vt.append(offsets[f.name])
            labels.append(f.name)
        prev = img
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(files)}")
    if skipped:
        print(f"  Skipped {skipped}/{len(files)} unreadable (empty/corrupt) frames.")
    if not values:
        sys.exit("No difference could be computed.")
    return np.array(vt), np.array(values), abs_start, source, labels


def main():
    ap = argparse.ArgumentParser(
        description="Count the board poses in a capture session (images or LiDAR)."
    )
    ap.add_argument("dirs", nargs="*", help="One or more image folders (in order)")
    ap.add_argument("--db3", default=None,
                    help="LiDAR mode: a rosbag2 .db3 file or folder. Detects poses "
                         "from scan-to-scan voxel-occupancy change instead of images.")
    ap.add_argument("--compare", action="store_true",
                    help="Cross-check 2-3 sensors: detect poses on each of --flir / "
                         "--zed / --db3 and report the gap between consecutive poses "
                         "per sensor (clock-independent, so FLIR's different clock is "
                         "fine).")
    ap.add_argument("--flir", nargs="+", default=None,
                    help="FLIR image folder(s) for --compare.")
    ap.add_argument("--zed", nargs="+", default=None,
                    help="ZED image folder(s) for --compare.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Movement threshold. If omitted, estimated as "
                         "median + K*MAD (see --mad-k).")
    ap.add_argument("--mad-k", type=float, default=None,
                    help="Auto-threshold robustness factor (default 6 for images, "
                         "4 for LiDAR).")
    ap.add_argument("--min-pose-frames", type=int, default=None,
                    help="Min stable frames/scans for a pose (default 15 images; "
                         "LiDAR ~10 s worth).")
    ap.add_argument("--min-move-frames", type=int, default=None,
                    help="Min consecutive above-threshold frames/scans for a real "
                         "move (default 3 images, 1 LiDAR).")
    ap.add_argument("--min-duration", type=float, default=None,
                    help="Keep only poses lasting at least this many SECONDS "
                         "(applies to images and LiDAR alike).")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Fallback frame rate for images without timestamps.")
    ap.add_argument("--downscale", type=int, default=2,
                    help="Image downscale factor (default 2).")
    # LiDAR-only
    ap.add_argument("--db3-topic", default=None,
                    help="LiDAR topic name if the bag has more than one.")
    ap.add_argument("--db3-msg-stride", type=int, default=10,
                    help="Use every Nth scan (default 10). Lower = finer, slower.")
    ap.add_argument("--db3-point-stride", type=int, default=4,
                    help="Subsample every Nth point per scan (default 4).")
    ap.add_argument("--voxel", type=float, default=0.10,
                    help="Voxel size (m) for the occupancy difference (default 0.10).")
    ap.add_argument("--show-profile", action="store_true",
                    help="Print the full sample-by-sample difference profile.")
    args = ap.parse_args()

    if args.compare:
        compare_sensors(args)
        return

    is_lidar = args.db3 is not None
    if is_lidar and args.dirs:
        sys.exit("Pass either image folders or --db3, not both.")
    if not is_lidar and not args.dirs:
        sys.exit("Nothing to do: pass image folder(s), --db3 <rosbag>, or --compare.")

    if is_lidar:
        db3 = _resolve_db3(args.db3)
        vt_all, values, abs_start, source, n_total, eff_rate = lidar_profile(
            db3, args.db3_topic, args.db3_msg_stride, args.db3_point_stride, args.voxel)
        value_times = vt_all[1:]                       # value i compares scan i-1,i
        count_unit = "scans"
        # a board move spans ~1 sampled scan at coarse msg-stride, so no debounce
        min_move = args.min_move_frames if args.min_move_frames is not None else 1
        min_pose = (args.min_pose_frames if args.min_pose_frames is not None
                    else max(4, round(10.0 * eff_rate)))
        mad_k = args.mad_k if args.mad_k is not None else 4.0
        print(f"Effective sample rate: {eff_rate:.2f} Hz "
              f"(min-pose-frames={min_pose} ~ {min_pose/max(eff_rate,1e-6):.0f} s)")
    else:
        files = collect_images(args.dirs)
        if len(files) < 2:
            sys.exit("At least 2 images are required.")
        value_times, values, abs_start, source, labels = image_profile(
            files, args.dirs, args)
        count_unit = "frames"
        min_move = args.min_move_frames if args.min_move_frames is not None else 3
        min_pose = args.min_pose_frames if args.min_pose_frames is not None else 15
        mad_k = args.mad_k if args.mad_k is not None else 6.0

    dur_total = value_times[-1] - value_times[0]
    print(f"Total session duration: {dur_total:.0f} s ({dur_total/60:.1f} min)")
    if abs_start is not None:
        print(f"Session start: {abs_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if args.threshold is None:
        threshold = median + mad_k * (mad if mad > 1e-6 else 1.0)
    else:
        threshold = args.threshold

    print(f"Difference statistics: median={median:.3f}  MAD={mad:.3f}")
    print(f"Movement threshold used: {threshold:.3f}"
          f"{f'  (auto: median+{mad_k:g}*MAD)' if args.threshold is None else '  (provided)'}")
    print()

    if args.show_profile:
        print("Difference profile:")
        for i, v in enumerate(values):
            marker = "  <-- MOVEMENT" if v > threshold else ""
            print(f"  {fmt_clock(abs_start, value_times[i])}  {v:8.3f}{marker}")
        print()

    poses, transitions = segment(values, threshold, min_move, min_pose)

    def pose_duration(a, b):
        return value_times[min(b, len(value_times)) - 1] - value_times[a]

    dropped = 0
    if args.min_duration is not None:
        kept = [(a, b) for (a, b) in poses if pose_duration(a, b) >= args.min_duration]
        dropped = len(poses) - len(kept)
        poses = kept

    print("=" * 78)
    print(f"DETECTED POSES: {len(poses)}   (real transitions: {len(transitions)}"
          f"{f'; {dropped} shorter than {args.min_duration:g}s dropped' if dropped else ''})")
    print("=" * 78)
    print(f"{'#':>3}  {'start':<10} {'end':<10} {'duration':>8}  {count_unit:>6}")
    print("-" * 78)

    for k, (a, b) in enumerate(poses, start=1):
        off_a = value_times[a]
        off_b = value_times[min(b, len(value_times)) - 1]
        dur = off_b - off_a
        print(f"{k:>3}  {fmt_clock(abs_start, off_a):<10} "
              f"{fmt_clock(abs_start, off_b):<10} "
              f"{dur:>7.0f}s  {b - a:>6}     "
              f"[session offset: {off_a:.0f}s - {off_b:.0f}s]")

    print()
    print("NOTES:")
    print(f"  - Stable segments shorter than {min_pose} {count_unit} were discarded")
    print("    (likely transitions, not real poses). Tune with --min-pose-frames.")
    print(f"  - Above-threshold bursts shorter than {min_move} {count_unit} were")
    print("    ignored as noise. Tune with --min-move-frames.")
    if args.min_duration is not None:
        print(f"  - Only poses lasting >= {args.min_duration:g}s are shown "
              "(--min-duration).")
    print("  - If the pose count is off, try --threshold or --mad-k "
          "(higher = fewer moves).")
    if is_lidar:
        print("  - LiDAR 'difference' = voxel-occupancy Jaccard distance per scan; "
              "tune --voxel / --db3-msg-stride.")
    print("  - Use --show-profile to inspect the sample-by-sample detail.")
    print("  - Offsets are relative to the start of THIS session. The camera and")
    print("    LiDAR recordings start at different moments and must be aligned by")
    print("    comparing their absolute start times.")


if __name__ == "__main__":
    main()
