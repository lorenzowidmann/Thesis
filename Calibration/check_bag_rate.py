"""Check the recording rate of a ROS 2 bag, overall and over per-pose windows.

Answers two questions about a bag before it is used for calibration:
    1. did every topic actually record at its nominal rate?
    2. how many scans fell inside each static-board pose?

A pose held for ~100 s should contribute ~100 x rate scans. If a window comes
out short, that pose was recorded through a drop and the frames averaged for it
are fewer than assumed -- which is exactly the kind of thing that quietly
degrades an extrinsic calibration.

Timestamps are the bag's *receive* times (the ns key of each message), not the
message header stamps: reading them costs nothing, while header stamps would
mean deserialising every PointCloud2 in a multi-GB bag. For checking whether
the recording kept up this is the right clock anyway -- it is the one that shows
scans lost between the sensor and the disk.

Rate conventions, deliberately different in the two tables:
- overall, rate = (count - 1) / span, the mean interval over the messages seen;
- per window, rate = count / (end - start), scans per second of *wall time in
  that window*, so a window emptied by a drop reads low instead of hiding it.
Both tables also report the median-interval rate, which ignores gaps, so
rate << median_hz is the signature of dropped messages rather than a slow sensor.

Usage:
    py check_bag_rate.py --bag path/to/rosbag2_2026_07_30-15_02_17
    py check_bag_rate.py --bag BAG --topic /livox/lidar
    py check_bag_rate.py --bag BAG --topic /livox/lidar --expected-hz 10
    py check_bag_rate.py --bag BAG --topic /livox/lidar \
        --windows 1:109 111:221 247:350 352:461 478:571 573:663 \
                  683:768 807:914 942:1035 1050:1154 1158:1241 1270:1375
    py check_bag_rate.py --bag BAG --topic /livox/lidar --windows 1:109 --output rate.json

Exit status is 1 if --expected-hz is given and any reported rate falls outside
--tolerance, so the check can gate a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from rosbags.rosbag2 import Reader
except ImportError:  # pragma: no cover - environment problem, not a bug
    raise SystemExit(
        "error: the 'rosbags' package is required (pip install rosbags)"
    ) from None

# An interval longer than this many median intervals is a gap, not jitter.
# 3x is loose enough to survive normal scheduling noise on a Windows recorder
# and tight enough to catch a single missing scan.
_GAP_FACTOR = 3.0


# --------------------------------------------------------------------------- #
# Results                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class RateStats:
    """Rate of one topic, over the whole bag or over a single window."""

    label: str
    count: int
    span_s: float
    rate_hz: float
    median_hz: float
    jitter_ms: float
    max_gap_s: float
    n_gaps: int
    missing_est: int


def rate_stats(stamps: np.ndarray, label: str, span_s: float | None = None) -> RateStats:
    """Rate statistics for one run of timestamps (seconds, sorted).

    `span_s` is the window length when the messages were counted inside a fixed
    interval; left out, the span is measured from the messages themselves.
    """
    n = int(stamps.size)
    if n == 0:
        return RateStats(label, 0, float(span_s or 0.0), 0.0, 0.0, 0.0, 0.0, 0, 0)
    if n == 1:
        return RateStats(label, 1, float(span_s or 0.0), 0.0, 0.0, 0.0, 0.0, 0, 0)

    measured = float(stamps[-1] - stamps[0])
    span = float(span_s) if span_s is not None else measured
    # Counting inside a fixed window: every message belongs to that window, so
    # divide by n. Measuring a free run: n messages bound n-1 intervals.
    rate = (n / span) if span_s is not None else ((n - 1) / measured if measured > 0 else 0.0)

    dt = np.diff(stamps)
    med = float(np.median(dt))
    gaps = dt[dt > _GAP_FACTOR * med] if med > 0 else np.empty(0)
    # Each gap stands in for the messages that would have filled it.
    missing = int(np.sum(np.round(gaps / med) - 1)) if med > 0 and gaps.size else 0

    return RateStats(
        label=label,
        count=n,
        span_s=span,
        rate_hz=float(rate),
        median_hz=(1.0 / med) if med > 0 else 0.0,
        jitter_ms=float(np.std(dt) * 1e3),
        max_gap_s=float(dt.max()),
        n_gaps=int(gaps.size),
        missing_est=missing,
    )


# --------------------------------------------------------------------------- #
# Bag reading                                                                  #
# --------------------------------------------------------------------------- #
def read_stamps(bag: Path, topic: str | None) -> tuple[dict[str, np.ndarray], float]:
    """Receive timestamps per topic, in seconds, plus the bag start time.

    With `topic` set only that topic's rows are pulled out of the bag, which on
    a multi-GB LiDAR bag is the difference between seconds and minutes.
    """
    if not bag.exists():
        raise FileNotFoundError(f"bag not found: {bag}")

    with Reader(bag) as reader:
        conns = [c for c in reader.connections if topic is None or c.topic == topic]
        if not conns:
            available = sorted({c.topic for c in reader.connections})
            raise RuntimeError(
                f"topic {topic!r} not in bag. Available: " + ", ".join(available)
            )
        bag_start = reader.start_time / 1e9

        stamps: dict[str, list[float]] = {}
        for conn in conns:
            stamps.setdefault(conn.topic, [])
        for conn, ts, _ in reader.messages(connections=conns):
            stamps[conn.topic].append(ts / 1e9)

    # Sorted defensively: a bag written by several recorders is not guaranteed
    # to be monotonic across connections, and every statistic here assumes it is.
    return {t: np.sort(np.asarray(v, dtype=float)) for t, v in stamps.items()}, bag_start


def topic_counts(bag: Path) -> dict[str, str]:
    """Message type per topic, for the listing header."""
    with Reader(bag) as reader:
        return {c.topic: c.msgtype for c in reader.connections}


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
_HEADER = (f"{'topic / window':<28}{'msgs':>7}{'span_s':>9}{'rate_hz':>9}"
           f"{'median_hz':>11}{'jitter_ms':>11}{'max_gap_s':>11}{'gaps':>6}{'missing':>9}")


def format_row(s: RateStats) -> str:
    return (f"{s.label:<28}{s.count:>7}{s.span_s:>9.1f}{s.rate_hz:>9.2f}"
            f"{s.median_hz:>11.2f}{s.jitter_ms:>11.1f}{s.max_gap_s:>11.2f}"
            f"{s.n_gaps:>6}{s.missing_est:>9}")


def verdict(s: RateStats, expected: float | None, tol: float) -> str:
    """OK / LOW / HIGH against an expected rate, or '' when none was given."""
    if expected is None or s.count == 0:
        return ""
    dev = (s.rate_hz - expected) / expected
    if abs(dev) <= tol:
        return "  OK"
    return f"  {'LOW' if dev < 0 else 'HIGH'} ({dev * 100:+.0f}%)"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_window(text: str) -> tuple[float, float]:
    """A 'START:END' window in seconds relative to the time origin."""
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"window must be START:END in seconds (got {text!r})")
    try:
        a, b = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"window bounds must be numbers (got {text!r})") from None
    if b <= a:
        raise argparse.ArgumentTypeError(f"window end must follow its start (got {text!r})")
    return a, b


def parse_args():
    p = argparse.ArgumentParser(
        description="Check the recording rate of a ROS 2 bag, overall and per pose window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bag", required=True, metavar="DIR",
                   help="rosbag2 directory (the folder holding metadata.yaml)")
    p.add_argument("--topic", default=None,
                   help="Topic to analyse. Omitted, every topic in the bag is listed")
    p.add_argument("--windows", nargs="+", type=parse_window, default=None, metavar="START:END",
                   help="Time windows in seconds from the origin, e.g. 1:109 111:221")
    p.add_argument("--time-origin", choices=("topic", "bag"), default="topic",
                   help="Zero of the window clock: the topic's first message, or the bag start")
    p.add_argument("--window-label", default="Pose",
                   help="Name used for the windows in the report")
    p.add_argument("--expected-hz", type=float, default=None,
                   help="Nominal rate to check against; exit 1 if anything is outside --tolerance")
    p.add_argument("--tolerance", type=float, default=0.1,
                   help="Allowed relative deviation from --expected-hz (0.1 = 10%%)")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Also write the report as JSON here")
    args = p.parse_args()
    if args.windows and args.topic is None:
        p.error("--windows needs --topic (windows are counted on one topic)")
    if args.tolerance < 0:
        p.error("--tolerance must be >= 0")
    return args


def main():
    args = parse_args()
    bag = Path(args.bag)

    try:
        msgtypes = topic_counts(bag)
        stamps, bag_start = read_stamps(bag, args.topic)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from None

    print(f"Bag: {bag.resolve()}")
    if args.topic is None:
        print(f"{len(msgtypes)} topics\n")
    else:
        print(f"Topic: {args.topic}  ({msgtypes.get(args.topic, 'unknown type')})\n")

    overall = [rate_stats(v, t) for t, v in sorted(stamps.items())]
    print(_HEADER)
    failed = False
    for s in overall:
        # Only judge the analysed topic: an expected rate for the LiDAR says
        # nothing about /tf or /imu.
        mark = verdict(s, args.expected_hz, args.tolerance) if args.topic else ""
        failed = failed or mark.startswith("  LOW") or mark.startswith("  HIGH")
        print(format_row(s) + mark)

    windows: list[RateStats] = []
    if args.windows:
        t = stamps[args.topic]
        if t.size == 0:
            raise SystemExit(f"error: no messages on {args.topic}, nothing to window")
        t0 = float(t[0]) if args.time_origin == "topic" else bag_start
        rel = t - t0

        print(f"\nWindows on {args.topic} "
              f"(t=0 at the {'first message' if args.time_origin == 'topic' else 'bag start'})")
        print(_HEADER)
        for i, (a, b) in enumerate(args.windows, 1):
            sel = rel[(rel >= a) & (rel <= b)]
            s = rate_stats(sel, f"{args.window_label} {i} [{a:g}:{b:g}]", span_s=b - a)
            windows.append(s)
            mark = verdict(s, args.expected_hz, args.tolerance)
            failed = failed or mark.startswith("  LOW") or mark.startswith("  HIGH")
            print(format_row(s) + mark)

        counted = sum(s.count for s in windows)
        print(f"\n{counted}/{t.size} messages fall inside the {len(windows)} "
              f"window{'s' if len(windows) != 1 else ''} "
              f"({100 * counted / t.size:.1f}%)")

    if args.output:
        report = {
            "bag": str(bag.resolve()),
            "topic": args.topic,
            "time_origin": args.time_origin,
            "expected_hz": args.expected_hz,
            "tolerance": args.tolerance,
            "overall": [asdict(s) for s in overall],
            "windows": [asdict(s) for s in windows],
            "provenance": {
                "tool": Path(__file__).name,
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python_version": sys.version.split()[0],
                "command": " ".join(sys.argv),
            },
        }
        out = Path(args.output)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report -> {out}")

    if failed:
        print(f"\nSome rates are outside {args.tolerance * 100:.0f}% "
              f"of the expected {args.expected_hz:g} Hz")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
