# Sensor Fusion

This folder is the home for everything that synchronizes the rover's thermal
(FLIR), ZED, and LiDAR data — keep such scripts here.

- **`zed_record.py`** — record-only capture of the ZED 2i (SVO2 + mp4 + PNG
  frames).
- **`sync_manifest.py`** — post-processing: matches FLIR thermal, ZED PNG, and
  LiDAR pose streams into a triplet manifest.

## `zed_record.py` — ZED 2i recorder

Opens the ZED 2i via the ZED SDK (`pyzed`) and captures, in parallel, three
outputs into one timestamped session folder:

1. **`session.svo2`** — native SVO2 recording, the master. Records the **full
   stereo stream (both eyes)**, encoded H264 by default (`--svo-compression`).
2. **`session_<eye>.mp4`** — a plain, viewable mp4 built from one RGB eye via
   OpenCV `VideoWriter`, for quick review.
3. **`frames/<eye>_NNNNNN.png`** — a still every `--frame-interval` seconds.
   This is what `EmissivityCalculation/` consumes as input (it reads individual
   RGB image files via `emissivity.sources.ImageSource`).

This is a **pure recording tool** — no emissivity estimation, CLIP inference, or
radiometric math. That stays in `EmissivityCalculation/` and consumes a
finished session as a separate post-processing step.

### Which eye

`--eye` defaults to **right** because that is the eye `EmissivityCalculation`'s
CLIP classification runs on (both `--zed-uvc` and `--shared` there pass
`eye="right"`), so the exported frames stay consistent with the classifier's
input. The SVO always keeps both eyes regardless, so left is recoverable.

### Setup

Needs the official ZED SDK (unlike `EmissivityCalculation`'s `--zed-uvc` path,
there is no OpenCV-only fallback — SVO recording is an SDK feature):

```powershell
py -m venv C:\venvs\zedrecord
C:\venvs\zedrecord\Scripts\Activate.ps1
cd SensorFusion
# Install pyzed into the venv by running the ZED SDK's get_python_api.py
# (the SDK itself: https://www.stereolabs.com/developers/release/ — needs an
# NVIDIA GPU with CUDA). Note: pyzed may not support Python 3.13 yet; if it
# fails, recreate the venv with `py -3.12 -m venv C:\venvs\zedrecord`.
pip install -r requirements.txt
```

On the rover (Ubuntu/Jetson) the code is unchanged — Stereolabs ships
first-class Linux/Jetson support, and the tool is headless (no display needed):

```bash
python3 -m venv .venv
source .venv/bin/activate
cd SensorFusion
pip install -r requirements.txt
```

### Usage

```powershell
# Record until Ctrl+C
python zed_record.py

# Record a fixed length then auto-stop
python zed_record.py --duration 60

# Lower resolution, higher fps
python zed_record.py --resolution HD720 --fps 60

# Lossless SVO (larger files) instead of the H264 default
python zed_record.py --svo-compression lossless

# A PNG every 2 s instead of the 1 s default
python zed_record.py --frame-interval 2.0

# Export the left eye instead of the default right
python zed_record.py --eye left

# SVO only — skip the mp4 and PNG exports
python zed_record.py --no-mp4 --no-frames
```

`--duration` is the **total** recording length (omit → until Ctrl+C).
`--frame-interval` is the spacing **between** PNG stills — the two are
independent. Example: `--duration 60 --frame-interval 5` records 60 s of
SVO2 + mp4 and drops ~12 PNGs.

### Output

Each run creates one session folder under `--output-dir` (default `recordings/`
next to the script), named by UTC timestamp:

```
recordings/20260726_140311/
├── session.svo2          # full stereo master (both eyes)
├── session_right.mp4     # one-eye viewable video (--eye)
├── frames/               # one PNG per --frame-interval (emissivity input)
│   ├── right_000000.png
│   └── ...
└── metadata.json         # provenance sidecar (see below)
```

`metadata.json` is a flat provenance record (same `json.dumps(indent=2)` style
as the OcTree exports): schema/generator tag, camera info (resolution, fps,
serial number, model), recording settings (compression, export eye, frame
format/interval, frame count), session start/stop UTC timestamps + duration +
stop reason, and a **frame manifest** (each PNG's filename + capture offset in
seconds from session start, so a post-processing step can align frames to the
SVO timeline). It is written up-front (partial) so a crash still leaves
provenance, then finalized on stop.

### Errors

Camera-not-found / not-initialized / already-in-use are reported as a single
clear line, not a raw stack trace. "Already in use" usually means another
process holds the camera (a viewer, or `CameraServer/camera_server.py`).

## `sync_manifest.py` — FLIR/ZED/LiDAR triplet manifest

Post-processing (no live capture). Produces a JSON manifest matching each FLIR
thermal frame to the nearest ZED PNG and LiDAR pose, on a common clock. Two
stages:

**Stage 1 — manual FLIR↔ZED event sync.** FLIR and ZED are not hardware-synced.
A steppable viewer (matplotlib) shows each stream so you pick the frame in each
where a shared heat-source event (lighter / heat gun / hot object) appears; the
offset is the difference of the two frames' own timestamps, saved to
`<session>/flir_zed_offset.json`. Because it comes from one physical event, the
offset absorbs any constant clock/timezone difference — FLIR timestamps only
need to be internally consistent. If an offset file already exists you're asked
whether to reuse it; `--recompute-offset` forces re-selection. Fully headless:
pass `--flir-event-frame N --zed-event-frame M` to skip the viewer.

**Stage 2 — triplet matching.** Using the Stage-1 offset and the LiDAR↔ZED
relationship (`--lidar-zed-offset`), each FLIR frame (the reference stream) is
matched to the nearest ZED PNG and nearest LiDAR `/Odometry` pose in the ZED
clock. Each triplet records the three paths/timestamps, the LiDAR pose
(position + orientation quaternion), the pairwise deltas, and a
`match_status` of `low-confidence` (with which pairs) when any delta exceeds
`--max-delta`, rather than silently pairing distant frames.

> **LiDAR↔ZED clock:** not yet stored anywhere in the pipeline. `--lidar-zed-offset`
> defaults to `0.0` (shared host clock, per `RadiometricCalibration/README.md`'s
> Synchronization note) and is flagged `UNVERIFIED` in the manifest. Measure it
> on the rig and pass the real value.

```powershell
# Interactive Stage 1 viewer, then manifest
python sync_manifest.py --session-dir recordings/20260726_140311 \
    --flir-dir "C:\...\FlyrCamera\20250823_211855" --bag path\to\rosbag2

# Headless (indices instead of the viewer), tighter tolerance
python sync_manifest.py --session-dir <dir> --flir-dir <dir> --bag <dir> \
    --flir-event-frame 12 --zed-event-frame 3 --max-delta 0.05
```

Inputs: a `zed_record.py` session folder (for `metadata.json` + `frames/`), a
FLIR `*_R.jpg` folder (as read by `RadiometricCalibration/ThermalData.py`), and
a rosbag2 with an odometry topic (`--odom-topic`, default `/Odometry`; read via
`rosbags`, same as `PointCloudView/view_pointcloud.py`). Needs `flyr`,
`rosbags`, `Pillow`, `matplotlib`. Output `sync_manifest.json` lands in the
session folder.

## Feeding EmissivityCalculation

A finished session's `frames/` is directly consumable by
`EmissivityCalculation/main.py --image <path>`. Wiring it to iterate a session
folder (or extract frames straight from `session.svo2`) automatically is a
`TODO` noted in `zed_record.py`, left as a follow-up rather than built here.

## Structure

```
SensorFusion/
├── zed_record.py       # ZED 2i record-only CLI (SVO2 + mp4 + PNG frames)
├── sync_manifest.py    # FLIR/ZED/LiDAR triplet-manifest builder (2-stage sync)
├── requirements.txt
└── recordings/         # session folders (git-ignored); each also gets
                        #   flir_zed_offset.json + sync_manifest.json after sync
```
