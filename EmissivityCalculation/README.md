# Emissivity Calculation

Estimates the emissivity of the material(s) seen by a camera. A frame is
grabbed from a ZED 2i stereo camera (or an image file / webcam during
development), tiled into a grid of cells, each cell classified independently
with CLIP zero-shot image classification, and its emissivity looked up in a
table of tabulated values (`emissivity_table.csv`).

Two ways to run it:
- **`main.py`** — single live/still frame, coarse NxN grid. Good for
  development and spot checks (see Usage below).
- **`classify_session.py` + `project_to_flir.py`** — a recorded session
  (FLIR+ZED+LiDAR, synced by `../SensorFusion/sync_manifest.py`), fine-grained
  SLIC superpixels instead of a fixed grid, and the material/emissivity map
  projected onto *real FLIR pixels* via LiDAR (no direct FLIR<->ZED
  calibration needed — see "Session pipeline" below). This is what actually
  answers "what material and emissivity is in each zone of the FLIR image."

No viewing-angle-dependent emissivity correction is applied by either path —
both use tabulated *normal* emissivity only. The session pipeline does now
place that value geometrically on FLIR pixels (via LiDAR), which viewing-angle
correction would build on top of, but computing the angle itself (surface
normal vs. camera ray, from LiDAR) is still not implemented.

## Setup

```powershell
# The venv lives at a short path (C:\venvs\emissivity) because torch's
# installation exceeds the Windows 260-character path limit when the venv
# is inside this (deeply nested) project folder.
py -m venv C:\venvs\emissivity
C:\venvs\emissivity\Scripts\Activate.ps1
cd EmissivityCalculation
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Nothing in the code is Windows-specific (no OS checks, no hardcoded paths) —
on Ubuntu/Linux, skip the short-path venv workaround:

```bash
python3 -m venv .venv
source .venv/bin/activate
cd EmissivityCalculation
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA index if there's an NVIDIA GPU
pip install -r requirements.txt
```

`--camera-index` maps to `/dev/videoN` there instead of a Windows device
index, and `--zed` (official SDK) is actually easier on Linux — Stereolabs
ships first-class Linux/Jetson support. `--show` needs a display (X11/Wayland).

The first run downloads the CLIP model (~600 MB) from Hugging Face, cached at
`~/.cache/huggingface/hub` (shared across venvs/projects on this account — it
never re-downloads afterward, and doesn't expire). Every run after that prints
`Loading cached CLIP model...` and loads straight from disk in a few seconds.

## Usage

With the venv activated:

```powershell
# From an image file (development without camera)
python main.py --image test_images/brick.jpg

# From the default webcam, with a display window
python main.py --webcam --show

# From the ZED 2i via the official SDK (requires pyzed + NVIDIA GPU/CUDA, see below)
python main.py --zed --show

# From the ZED 2i as a plain UVC webcam (OpenCV only, no SDK/GPU needed)
python main.py --zed-uvc --show
python main.py --zed-uvc --camera-index 1 --show   # if it's not device 0

# On Linux, --camera-index also takes a device path -- use this if OpenCV's
# numeric index doesn't match reality (common with multi-node UVC cameras
# like the ZED 2i). Check the real node with `v4l2-ctl --list-devices` first.
python main.py --zed-uvc --camera-index /dev/video1 --show

# Coarser or finer grid (default is 3x3)
python main.py --image photo.jpg --grid-size 4

# Restrict the grid to a region: center-x, center-y, width, height (px)
python main.py --image photo.jpg --roi 320,240,200,200

# Live mode: keep grabbing + classifying frames (model loaded once) until you
# press 'q' in the window or Ctrl+C. Needs --show and a camera source.
python main.py --zed-uvc --show --live
```

**Grid classification.** Every source — `--image` included — has its frame
tiled into an NxN grid (`--grid-size`, default 3) and each cell classified
independently, so you get emissivity coverage across the whole view instead
of just one spot. `--show` draws every cell's box and its best-match label
(`material e=X.XX`) as an overlay. Pass `--roi cx,cy,w,h` to tile only an
explicit sub-region instead of the whole frame.

Output: one row per grid cell with its best material match, confidence, and
tabulated emissivity, plus the single highest-confidence cell overall, e.g.

```
Row  Col  Material            Confidence  Emissivity
------------------------------------------------------------
0    0    brick               91.2%       0.93
0    1    brick               77.4%       0.93
...
Best estimate: brick (row 0, col 0) -> emissivity = 0.93 (91%)
```

## Session pipeline (real FLIR pixels, via LiDAR fusion)

Two steps, two different venvs (see below for why):

```powershell
# 1. Material + emissivity per SLIC superpixel, on the ZED frame of every
#    (or --every-n Nth / --limit first-N) synced triplet. Uses the
#    `emissivity` venv (torch/transformers/scikit-image).
C:\venvs\emissivity\Scripts\python.exe classify_session.py `
    --session-dir path\to\ZED\<session>\fullrate `
    --n-segments 100 --overlay

# 2. Project that material map onto FLIR's own pixel grid via LiDAR (reads
#    /cloud_registered from the rosbag2, needs `rosbags` -- uses the
#    `sensorfusion` venv instead, no torch/CLIP dependency at this step,
#    it only reads step 1's saved output).
C:\venvs\sensorfusion\Scripts\python.exe project_to_flir.py `
    --session-dir path\to\ZED\<session>\fullrate `
    --bag path\to\Lidar\<bag-folder> --overlay
```

`--session-dir` must point at a ZED session folder that already has
`sync_manifest.json` (from `../SensorFusion/sync_manifest.py`) — that's what
drives which frames get processed and supplies each frame's LiDAR pose.

**Why LiDAR, not a direct FLIR<->ZED calibration:** no FLIR<->ZED extrinsic
exists (or is needed). Instead, `../Calibration/rig_calibration.yaml` holds
LiDAR<->FLIR and LiDAR<->ZED extrinsics (from the LVT2Calib board-pose
session, RMSE ~6cm) plus both cameras' intrinsics. For every LiDAR point
visible in *both* camera frustums, `project_to_flir.py` reads its material
off the ZED-side classification and writes it at that point's own FLIR
pixel — LiDAR is the bridge between the two image planes.

**Read the coverage number.** LiDAR's point density inside FLIR's narrow
32x26 deg FOV is far below FLIR's 86k pixels — expect single-digit percent
direct coverage (`stats.json`'s `coverage_pct`, ~6-7% in the one session
tested so far). Everything else is a plain nearest-neighbor fill;
`sampled_mask.npy` keeps the direct/interpolated distinction per pixel so
that split is never hidden downstream.

**Known rough edges:**
- Low-emissivity materials (e.g. `steel_polished`, e=0.07) make
  `RadiometricCalibration/main.py`'s correction numerically touchy (dividing
  by a small ε amplifies noise) — expect implausible outlier temperatures on
  metal/radiator zones specifically, not a sign the rest of the map is wrong.
- Extrinsic RMSE (~6cm both pairs) is fine for zone-level analysis, not
  precise pixel-perfect edges, especially at the corridor's far end (angular
  error amplifies with distance).
- FLIR frames must be rotated 180 deg before use (camera is mounted upside
  down; the LiDAR<->FLIR extrinsic was derived on rotated images) — see
  `rig_calibration.yaml`'s `flir.rotated_180_before_calibration` flag and
  `../SensorFusion`'s already-rotated `*_rot180/*.npy` Celsius frames for the
  convention to follow.
- `rig_calibration.yaml` is the single place to update when a calibration is
  redone — nothing downstream hardcodes numbers.

## Adding materials

Add a row to `emissivity_table.csv`:

| column | meaning |
|---|---|
| `material` | unique identifier, e.g. `steel_oxidized` |
| `emissivity` | tabulated normal emissivity (typical value) |
| `emissivity_range` | literature min–max |
| `prompt` | CLIP text prompt describing how the material *looks*, e.g. "a photo of a dark oxidized steel metal surface" |
| `notes` | free text |

The classifier's classes are generated from this table, so new rows are
immediately classifiable — no retraining.

## ZED 2i camera

Two ways to read the camera, depending on hardware:

**`--zed-uvc`** (works on this PC — no NVIDIA GPU here): the ZED 2i also
shows up as a plain USB webcam. Over UVC its frame is the left+right stereo
pair concatenated side by side (unrectified); `ZedUvcSource` just opens it
with OpenCV like any webcam and crops the right half (the eye used for CLIP
classification). No depth, no rectification — not
needed here since only a color crop is fed to CLIP. Use `--camera-index` if
it isn't device 0 (e.g. a laptop's built-in webcam is usually 0, so the ZED
may enumerate as 1 or 2).

**`--zed`** (needs the official SDK): only worth it if you later need depth
or rectified stereo. Requires:

1. Installing the ZED SDK: https://www.stereolabs.com/developers/release/
   (requires an NVIDIA GPU with CUDA — not available on this PC)
2. Running the SDK's `get_python_api.py` to install `pyzed` into the venv.
   Note: pyzed may not support Python 3.13 yet — if installation fails,
   recreate the venv with `py -3.12 -m venv C:\venvs\emissivity`.

## Structure

```
EmissivityCalculation/
├── main.py                  # CLI entry point (single frame, NxN grid)
├── classify_session.py      # CLI: session pipeline step 1 (SLIC + CLIP per triplet)
├── project_to_flir.py       # CLI: session pipeline step 2 (LiDAR-mediated fusion onto FLIR pixels)
├── emissivity_table.csv     # tabulated emissivity values + CLIP prompts
├── emissivity/
│   ├── table.py             # EmissivityTable: CSV loading + lookup
│   ├── classifier.py        # MaterialClassifier: CLIP zero-shot (classify + classify_batch)
│   ├── segmentation.py      # SLIC superpixel segmentation (skimage)
│   └── sources.py           # ImageSource / WebcamSource / ZedSource / ZedUvcSource
├── test_images/             # sample images for development
└── requirements.txt

../Calibration/              # used by project_to_flir.py, not part of this module
├── rig_calibration.yaml     # canonical FLIR/ZED intrinsics + LiDAR extrinsics (edit here)
├── rig_calibration.py       # loader
└── projection.py            # generic LiDAR -> camera pixel projection
```
