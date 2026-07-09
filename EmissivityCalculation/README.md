# Emissivity Calculation

Estimates the emissivity of the material seen by a camera. A frame is grabbed
from a ZED 2i stereo camera (or an image file / webcam during development),
the material is classified automatically with CLIP zero-shot image
classification, and the emissivity is looked up in a table of tabulated values
(`emissivity_table.csv`).

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

The first run downloads the CLIP model (~600 MB) from Hugging Face.

## Usage

With the venv activated:

```powershell
# From an image file (development without camera)
python main.py --image test_images/brick.jpg

# From the default webcam, with a display window
python main.py --webcam --show

# From the ZED 2i (requires ZED SDK, see below)
python main.py --zed --show

# Restrict classification to a region: center-x, center-y, width, height (px)
python main.py --image photo.jpg --roi 320,240,200,200
```

Output: top-3 material matches with confidence and their tabulated emissivity
(value + range), e.g.

```
Material              Confidence  Emissivity  Range
------------------------------------------------------------
brick                 93.1%       0.93        0.90-0.96
...
Best estimate: brick -> emissivity = 0.93 (Common red brick)
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

## ZED 2i camera (later)

The `--zed` source needs the ZED SDK, which is not yet installed on this PC:

1. Install the ZED SDK: https://www.stereolabs.com/developers/release/
   (requires an NVIDIA GPU with CUDA)
2. Run the SDK's `get_python_api.py` to install `pyzed` into the venv.
   Note: pyzed may not support Python 3.13 yet — if installation fails,
   recreate the venv with `py -3.12 -m venv C:\venvs\emissivity`.

Until then, develop with `--image` or `--webcam`.

## Structure

```
EmissivityCalculation/
├── main.py                  # CLI entry point
├── emissivity_table.csv     # tabulated emissivity values + CLIP prompts
├── emissivity/
│   ├── table.py             # EmissivityTable: CSV loading + lookup
│   ├── classifier.py        # MaterialClassifier: CLIP zero-shot
│   └── sources.py           # ImageSource / WebcamSource / ZedSource
├── test_images/             # sample images for development
└── requirements.txt
```
