"""
Automatically detect the board poses in a capture session by counting the
"stable" segments between one board move and the next.

Idea: during a pose the board is still -> the difference between consecutive
frames is low. When the board is moved -> a peak of difference appears. The
peaks separate the poses; the stable segments between two peaks are the real
poses.

Two capture formats are supported (auto-detected):

  * FLIR thermal RJPG:  YYYYMMDD_HHMMSS_R.jpg
      The timestamp lives in the filename (which is also temporal order).

  * ZED right-eye PNG:  right_NNNNNN.png (or any <prefix>_NNNNNN.png)
      No timestamp in the name; frames are ordered by their numeric index.
      Per-frame times are read from a sibling metadata.json
      (schema "zed_record/v1": recording.frame_interval_s, frames[].t_offset_s,
      session.started_utc). If metadata.json is missing, times fall back to
      index / --fps.

Accepts multiple folders: a long session may be split into separate folders,
which are treated as a single stream.

Usage:
    py detect_board_poses.py <folder1> [<folder2> ...]
    py detect_board_poses.py <folder1> --threshold 8.0 --min-pose-frames 20

Output: list of the detected poses with index, start/end time, duration in
seconds and number of frames. Also useful to build the -s/-u windows for
`rosbag play`.

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
import sys
from pathlib import Path
from datetime import datetime, timedelta

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


def main():
    ap = argparse.ArgumentParser(
        description="Count the board poses in a capture session via frame differencing."
    )
    ap.add_argument("dirs", nargs="+", help="One or more image folders (in order)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Difference threshold above which a frame is considered "
                         "'movement'. If omitted, it is estimated automatically.")
    ap.add_argument("--min-pose-frames", type=int, default=15,
                    help="Minimum number of stable frames for a segment to count "
                         "as a pose (default 15). Scale with the frame rate: at "
                         "5 Hz, 15 frames = 3 s.")
    ap.add_argument("--min-move-frames", type=int, default=3,
                    help="Minimum number of CONSECUTIVE above-threshold frames for "
                         "a real board move. Shorter bursts are treated as noise "
                         "(thermal AGC/NUC spikes) and ignored. Default 3.")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Frame rate used only as a fallback when no metadata.json "
                         "and no FLIR timestamps are available (default 5).")
    ap.add_argument("--downscale", type=int, default=2,
                    help="Downscale factor to speed things up (default 2)")
    ap.add_argument("--show-profile", action="store_true",
                    help="Print the full frame-by-frame difference profile")
    args = ap.parse_args()

    files = collect_images(args.dirs)
    if len(files) < 2:
        sys.exit("At least 2 images are required.")

    offsets, abs_start, source = build_time_index(files, args.dirs, args.fps)

    print(f"Images found: {len(files)}")
    print(f"First: {files[0].name}")
    print(f"Last: {files[-1].name}")
    print(f"Time source: {source}")

    dur_total = offsets[files[-1].name] - offsets[files[0].name]
    print(f"Total session duration: {dur_total:.0f} s ({dur_total/60:.1f} min)")
    if abs_start is not None:
        print(f"Session start: {abs_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    # --- compute the difference profile ---
    print("Computing differences between consecutive frames...")
    diffs = []
    prev = None
    skipped = 0
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # empty (0-byte) or corrupt file: the capture created the name but
            # never wrote pixels. Count it, don't spam one line per file.
            skipped += 1
            continue
        if args.downscale > 1:
            img = cv2.resize(img, (img.shape[1] // args.downscale,
                                   img.shape[0] // args.downscale))
        if prev is not None:
            diffs.append((i, f, frame_difference(prev, img)))
        prev = img
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(files)}")

    if skipped:
        print(f"  Skipped {skipped}/{len(files)} unreadable (empty/corrupt) frames.")

    if not diffs:
        sys.exit("No difference could be computed.")

    values = np.array([d[2] for d in diffs])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if args.threshold is None:
        # robust threshold: median + 6*MAD (MAD resists outliers)
        threshold = median + 6.0 * (mad if mad > 1e-6 else 1.0)
    else:
        threshold = args.threshold

    print()
    print(f"Difference statistics: median={median:.3f}  MAD={mad:.3f}")
    print(f"Movement threshold used: {threshold:.3f}"
          f"{'  (auto-estimated)' if args.threshold is None else '  (provided)'}")
    print()

    if args.show_profile:
        print("Difference profile:")
        for idx, f, v in diffs:
            marker = "  <-- MOVEMENT" if v > threshold else ""
            print(f"  {f.name}  {v:8.3f}{marker}")
        print()

    # --- segmentation into poses ---
    # A frame is above threshold => candidate movement. But a real board move
    # lasts several frames, while thermal AGC/NUC produces isolated 1-frame
    # spikes. So only RUNS of >= min-move-frames consecutive above-threshold
    # frames are accepted as real transitions; everything else stays "stable".
    moving = values > threshold
    n = len(moving)

    transitions = []  # (start, end) index ranges of real board moves
    i = 0
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            if j - i >= args.min_move_frames:
                transitions.append((i, j))
            i = j
        else:
            i += 1

    # the stable segments between the accepted transitions are the poses
    poses = []
    prev_end = 0
    for a, b in transitions:
        if a - prev_end >= args.min_pose_frames:
            poses.append((prev_end, a))
        prev_end = b
    if n - prev_end >= args.min_pose_frames:
        poses.append((prev_end, n))

    print("=" * 78)
    print(f"DETECTED POSES: {len(poses)}   (real transitions: {len(transitions)})")
    print("=" * 78)
    print(f"{'#':>3}  {'start':<10} {'end':<10} {'duration':>8}  {'frames':>6}")
    print("-" * 78)

    for k, (a, b) in enumerate(poses, start=1):
        f_start = diffs[a][1]
        f_end = diffs[min(b, len(diffs) - 1)][1]
        off_a = offsets[f_start.name]
        off_b = offsets[f_end.name]
        dur = off_b - off_a
        print(f"{k:>3}  {fmt_clock(abs_start, off_a):<10} "
              f"{fmt_clock(abs_start, off_b):<10} "
              f"{dur:>7.0f}s  {b - a:>6}     "
              f"[session offset: {off_a:.0f}s - {off_b:.0f}s]")

    print()
    print("NOTES:")
    print(f"  - Stable segments shorter than {args.min_pose_frames} frames were discarded")
    print("    (likely transitions, not real poses). Tune with --min-pose-frames.")
    print(f"  - Above-threshold bursts shorter than {args.min_move_frames} frames were")
    print("    ignored as noise (thermal AGC/NUC spikes). Tune with --min-move-frames.")
    print("  - If the pose count does not match reality, try changing --threshold")
    print("    (higher = fewer movements detected).")
    print("  - Use --show-profile to inspect the frame-by-frame detail.")
    print("  - The offsets in seconds are relative to the start of THIS session,")
    print("    NOT to the LiDAR bag: the two start at different moments and must be")
    print("    aligned by comparing the absolute recording start times.")


if __name__ == "__main__":
    main()
