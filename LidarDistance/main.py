#!/usr/bin/env python3
"""Livox distance-in-central-square measurement.

Listens to a Livox SDK2 LiDAR (Mid-360 / HAP) point-cloud UDP stream, keeps only
the returns inside a centred angular square of the field of view, and reports the
distance to that patch. No point-cloud / mapping -- just the distance, to be used
later for radiometric calibration of the thermocamera.

The central square mirrors EmissivityCalculation's `default_center_roi`: the
square's angular width is `--fraction` of `--fov-deg` (or set it directly with
`--square-deg`). The reported distance is the median range of the square's points.

Prereq: the device must already stream to this host. Bring it up once with
LivoxViewer2 (host IP 192.168.1.50), then close LivoxViewer2 and run this. See
run_commands.txt.

Examples:
    python main.py                          # one 0.5 s measurement, defaults
    python main.py --square-deg 20 --json   # 20 deg square, machine-readable
    python main.py --continuous             # keep measuring until Ctrl-C
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from livox import DEFAULT_DATA_PORT, LivoxReceiver, compute_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    net = p.add_argument_group("network")
    net.add_argument("--host-ip", default="0.0.0.0",
                     help="Local NIC IP to bind (default: 0.0.0.0 = all). Set to "
                          "192.168.1.50 to bind only the Livox interface.")
    net.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT,
                     help=f"Point-cloud UDP port (default: {DEFAULT_DATA_PORT})")
    net.add_argument("--timeout", type=float, default=3.0,
                     help="Seconds to wait for data before giving up (default: 3)")

    sq = p.add_argument_group("central square (angular FOV)")
    sq.add_argument("--fov-deg", type=float, default=40.0,
                    help="Nominal full angular width the square is a fraction of "
                         "(default: 40)")
    sq.add_argument("--fraction", type=float, default=0.5,
                    help="Square width as a fraction of --fov-deg (default: 0.5, "
                         "matching the emissivity central box)")
    sq.add_argument("--square-deg", type=float, default=None,
                    help="Set the square's full angular width directly, in degrees "
                         "(overrides --fov-deg/--fraction)")

    rng = p.add_argument_group("range gating")
    rng.add_argument("--min-range", type=float, default=0.1,
                     help="Drop returns closer than this (m), incl. zero/no-return "
                          "(default: 0.1)")
    rng.add_argument("--max-range", type=float, default=100.0,
                     help="Drop returns farther than this (m) (default: 100)")

    win = p.add_argument_group("measurement window")
    win.add_argument("--duration", type=float, default=0.5,
                     help="Seconds of points to accumulate per measurement "
                          "(default: 0.5)")
    win.add_argument("--continuous", action="store_true",
                     help="Repeat measurements until Ctrl-C")
    win.add_argument("--json", action="store_true",
                     help="Print one JSON object per measurement instead of text")
    return p.parse_args(argv)


def half_angle_rad(args: argparse.Namespace) -> float:
    full_deg = args.square_deg if args.square_deg is not None else args.fov_deg * args.fraction
    return math.radians(full_deg) / 2.0


def collect_window(rx: LivoxReceiver, duration: float) -> np.ndarray:
    """Accumulate points from the stream for `duration` seconds. Returns (N,3)."""
    chunks: list[np.ndarray] = []
    deadline = time.monotonic() + duration
    for block in rx.blocks():
        chunks.append(block.xyz)
        if time.monotonic() >= deadline:
            break
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def emit(stats, half_rad: float, as_json: bool) -> None:
    if as_json:
        print(json.dumps({
            "n": stats.n, "n_total": stats.n_total,
            "square_deg": round(math.degrees(half_rad * 2.0), 4),
            "distance_m": None if stats.n == 0 else {
                "median": stats.median, "mean": stats.mean,
                "min": stats.min, "max": stats.max, "std": stats.std,
            },
        }), flush=True)
    else:
        print(stats.format(), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    half_rad = half_angle_rad(args)

    if not args.json:
        print(f"Central square: {math.degrees(half_rad * 2.0):.1f} deg "
              f"(+/-{math.degrees(half_rad):.1f} deg off +x axis)")
        print(f"Listening on {args.host_ip}:{args.data_port} ...", flush=True)

    try:
        with LivoxReceiver(args.host_ip, args.data_port, args.timeout) as rx:
            while True:
                xyz = collect_window(rx, args.duration)
                if xyz.shape[0] == 0:
                    print("no data received (device streaming to this host?)",
                          file=sys.stderr, flush=True)
                    if not args.continuous:
                        return 1
                    continue
                stats = compute_stats(xyz, half_rad, args.min_range, args.max_range)
                emit(stats, half_rad, args.json)
                if not args.continuous:
                    return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
