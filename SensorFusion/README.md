# SensorFusion

Headless (no GUI) companion to `EmissivityCalculation` + `LidarDistance`, for
the onboard rover PC whose GPU can't carry the `--show`/`--live` overlay path.
Two entry points:

1. **`sensor_fusion.py`** — per-cycle emissivity (camera + CLIP) and LiDAR
   distance for the same central square, printed to the terminal.
2. **`extrinsic_calibration.py`** — seed of the LiDAR ↔ stereo-camera
   extrinsic calibration; prints an identity placeholder transform until a
   calibration target is available.

Neither script duplicates logic. Both load `EmissivityCalculation/main.py`
and `LidarDistance/main.py` via `importlib` (unique module names, since both
directories ship their own `main.py` and can't both be `import main`'d by
plain name) and call straight into their existing functions —
`classify_frame`, `crop_roi`, `default_center_roi`, `collect_window`,
`half_angle_rad`, `compute_stats` — plus the `emissivity`/`livox` packages
directly for `EmissivityTable`, `MaterialClassifier`, the camera
`*Source` classes, and `LivoxReceiver`.

No file writing, no JSON, no logging framework yet — plain terminal output,
same as the two source modules' default (non-`--show`) behavior.

## sensor_fusion.py

Continuous by default: grabs a camera frame, classifies the central square
with CLIP, grabs a matching LiDAR window, and prints both plus one combined
line, looping until Ctrl-C.

```bash
cd Thesis/SensorFusion
C:\venvs\emissivity\Scripts\python.exe sensor_fusion.py                       # zed-uvc + livox, continuous
C:\venvs\emissivity\Scripts\python.exe sensor_fusion.py --once                # single measurement
C:\venvs\emissivity\Scripts\python.exe sensor_fusion.py --image photo.jpg --once
C:\venvs\emissivity\Scripts\python.exe sensor_fusion.py --precision accurate  # heavier CLIP model
```

**Running alongside `DriveView`?** Windows locks the ZED's UVC device to
whichever process opens it first, so `sensor_fusion.py --zed-uvc` and
`drive_view.py` can't both open it directly at the same time. Start
`CameraServer/camera_server.py` once, then pass `--shared` to read the right
eye from it instead:

```bash
C:\venvs\emissivity\Scripts\python.exe sensor_fusion.py --shared
```

See `../CameraServer/README.md`.

Output per cycle:

```
Material              Confidence  Emissivity  Range
------------------------------------------------------------
brick                 98.9%       0.93        0.90-0.96
...
Best estimate: brick -> emissivity = 0.93 (Common red brick)
LiDAR distance[m]  median=2.431  mean=2.438  min=2.402  max=2.489  std=0.021  (n=1834/52210)
[fusion] emissivity=0.93 (brick)  distance=2.431 m
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--image` / `--webcam` / `--zed` / `--zed-uvc` / `--shared` | `--zed-uvc` | Camera source (default needs no ZED SDK/GPU; `--shared` reads the right eye from a running `CameraServer/camera_server.py`) |
| `--camera-index` | `0` | Device index/path for `--webcam` / `--zed-uvc` |
| `--roi` | — | Explicit crop `cx,cy,w,h`, overrides the default central square |
| `--fraction` | `0.5` | Central square size, shared between the camera ROI and the LiDAR angular square (same patch on both sensors) |
| `--precision` | `fast` | `fast` = `openai/clip-vit-base-patch32` (light, weak-GPU default); `accurate` = `openai/clip-vit-large-patch14` |
| `--clip-model` | — | Explicit HF model name, overrides `--precision` |
| `--table` | — | Custom emissivity CSV |
| `--top-k` | `3` | Matches considered |
| `--host-ip` / `--data-port` / `--timeout` | `0.0.0.0` / `57000` / `3.0` | LiDAR UDP bind + wait timeout |
| `--fov-deg` / `--square-deg` | `40` / — | LiDAR angular square sizing (see `LidarDistance/README.md`) |
| `--min-range` / `--max-range` | `0.1` / `100` | LiDAR range gate (m) |
| `--duration` | `0.5` | Seconds of LiDAR points accumulated per cycle |
| `--once` | off | Single measurement instead of continuous |

## extrinsic_calibration.py

No calibration target yet, so this is a placeholder: grabs one synced camera
frame + one LiDAR window as a sanity check that both sensors are live and
roughly co-pointed, then prints the current extrinsics — identity until a
target exists.

```bash
cd Thesis/SensorFusion
C:\venvs\emissivity\Scripts\python.exe extrinsic_calibration.py
C:\venvs\emissivity\Scripts\python.exe extrinsic_calibration.py --image photo.jpg
```

Output:

```
Camera frame: 960x640 px
LiDAR: 1834 points this window, distance[m]  median=2.431  ...

Extrinsics (LiDAR -> camera), placeholder until a target is used:
R =
    [+1.0000 +0.0000 +0.0000]
    [+0.0000 +1.0000 +0.0000]
    [+0.0000 +0.0000 +1.0000]
t = [+0.0000 +0.0000 +0.0000]

No calibration target yet -- see solve_extrinsics() TODO.
```

`Extrinsics` (`R`, `t`, rigid transform `p_cam = R @ p_lidar + t`) and
`solve_extrinsics(frame, xyz)` are the extension points: once a target is
available, replace `solve_extrinsics()` with a real correspondence-based
solve — paired 3D points (LiDAR + stereo depth) via Kabsch/SVD, or 2D image
points + known 3D target geometry via `cv2.solvePnP`.

Shares `sensor_fusion.py`'s camera/LiDAR flags (`--image`/`--webcam`/`--zed`/
`--zed-uvc`/`--shared`, `--camera-index`, `--host-ip`/`--data-port`/`--timeout`,
`--duration`).

## Setup

No new dependencies — reuses `EmissivityCalculation`'s venv (`torch`,
`transformers`, `opencv-python`, `numpy`, `pandas`, `pillow`) and
`LidarDistance`'s `numpy`-only requirement. See those modules' READMEs for
venv setup.

## Structure

```
SensorFusion/
├── sensor_fusion.py          # headless emissivity + LiDAR distance, per central square
├── extrinsic_calibration.py  # LiDAR <-> camera extrinsics seed (identity until a target exists)
└── README.md
```
