# LidarDistance

Measure the **distance to a centred square patch** using a Livox SDK2 LiDAR
(Mid-360 / HAP). No point cloud, no mapping — just the distance, meant to feed
later **radiometric calibration** of the thermocamera.

The "central square" is the LiDAR analogue of EmissivityCalculation's
`default_center_roi`: instead of a box of pixels, it's a centred square of the
angular field of view around the forward (+x) axis. Only returns inside that
square are kept, and their **median range** is the reported distance.

## How it works

1. `livox/receiver.py` binds the point-cloud UDP port and receives raw packets.
2. `livox/packets.py` decodes Livox SDK2 point packets (Cartesian high `0x01` and
   low `0x02`) into metres, sensor frame (x forward, y left, z up).
3. `livox/geometry.py` masks points to the central angular square and computes
   distance stats (median / mean / min / max / std).
4. `main.py` accumulates a short window of points and prints the result.

## Requirements

- Python 3.10+ and `numpy`.
- A Livox SDK2 device **already streaming to this host**. This program is a pure
  listener — it sends no control commands. Bring the device up once with
  LivoxViewer2 (host IP `192.168.1.50`), then **close LivoxViewer2** (it holds the
  UDP port) and run this.

## Usage

```bash
# On Ubuntu, set the NIC on the Livox subnet first (interface name may differ):
sudo ifconfig enp89s0 192.168.1.50

cd Thesis/LidarDistance
python3 main.py                        # one 0.5 s measurement, 20 deg square
python3 main.py --continuous           # repeat until Ctrl-C
python3 main.py --square-deg 20 --json # explicit 20 deg square, JSON out
```

## The central square

Two ways to size it:

- `--fov-deg` (default 40) × `--fraction` (default 0.5) → 20° square. This mirrors
  the emissivity central box (`fraction = 0.5` of the frame).
- `--square-deg N` sets the full angular width directly and overrides the above.

A point is inside when both `|azimuth| ≤ half` and `|elevation| ≤ half`, where
`half = square / 2`, azimuth = `atan2(y, x)`, elevation = `atan2(z, hypot(x, y))`.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--host-ip` | `0.0.0.0` | Local NIC IP to bind (`192.168.1.50` to bind only the Livox NIC) |
| `--data-port` | `57000` | Point-cloud UDP port (Livox SDK2 default) |
| `--timeout` | `3.0` | Seconds to wait for data before giving up |
| `--fov-deg` / `--fraction` | `40` / `0.5` | Square width = fraction × fov-deg |
| `--square-deg` | — | Set square width directly (overrides fov/fraction) |
| `--min-range` / `--max-range` | `0.1` / `100` | Range gate (m); drops zero/no-return |
| `--duration` | `0.5` | Seconds of points per measurement |
| `--continuous` | off | Repeat measurements until Ctrl-C |
| `--json` | off | One JSON object per measurement |

## Output

Text:

```
distance[m]  median=2.431  mean=2.438  min=2.402  max=2.489  std=0.021  (n=1834/52210)
```

JSON (`--json`): `{"n", "n_total", "square_deg", "distance_m": {"median", ...}}`.

`median` is the robust distance estimate to use downstream. `n` is the point count
inside the square; `n_total` is everything received in the window.
