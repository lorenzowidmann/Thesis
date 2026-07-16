# Emissivity Calculation

Estimates the emissivity of the material(s) seen by a camera. A frame is
grabbed from a ZED 2i stereo camera (or an image file / webcam during
development), tiled into a grid of cells, each cell classified independently
with CLIP zero-shot image classification, and its emissivity looked up in a
table of tabulated values (`emissivity_table.csv`).

No viewing-angle correction is applied here — tabulated normal emissivity only
(surface geometry will be handled separately with lidar data).

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
with OpenCV like any webcam and crops the right half (matching the eye
SensorFusion uses for CLIP classification). No depth, no rectification — not
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
├── main.py                  # CLI entry point
├── emissivity_table.csv     # tabulated emissivity values + CLIP prompts
├── emissivity/
│   ├── table.py             # EmissivityTable: CSV loading + lookup
│   ├── classifier.py        # MaterialClassifier: CLIP zero-shot
│   └── sources.py           # ImageSource / WebcamSource / ZedSource / ZedUvcSource
├── test_images/             # sample images for development
└── requirements.txt
```
