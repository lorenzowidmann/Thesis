# Sensor Fusion

Recording utilities for the rover's onboard sensors. Right now this is a single
record-only tool for the ZED 2i (`zed_record.py`); no fusion/estimation logic
lives here yet — the name is for the eventual multi-sensor alignment step.

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

## Feeding EmissivityCalculation

A finished session's `frames/` is directly consumable by
`EmissivityCalculation/main.py --image <path>`. Wiring it to iterate a session
folder (or extract frames straight from `session.svo2`) automatically is a
`TODO` noted in `zed_record.py`, left as a follow-up rather than built here.

## Structure

```
SensorFusion/
├── zed_record.py       # ZED 2i record-only CLI (SVO2 + mp4 + PNG frames)
├── requirements.txt
└── recordings/         # session folders (git-ignored)
```
