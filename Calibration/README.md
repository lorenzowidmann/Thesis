# Calibration

Intrinsic calibration of the **ZED 2i's right eye** — the lens the rest of the
pipeline consumes (`EmissivityCalculation` crops it for CLIP classification).
Only that eye is solved: no stereo baseline, no rectification, just a plain
monocular `cv2.calibrateCamera` problem.

The output feeds two things: our own provenance-tracked JSON record, and an
intrinsics file for [LVT2Calib](https://github.com/Clothooo/lvt2calib), which
does the LiDAR↔camera **extrinsic** calibration downstream.

Two scripts, deliberately separate:

1. **`capture_zed_right.py`** — live preview, one keypress per board pose,
   writes right-eye frames to disk. Needs a camera and a display.
2. **`zed_intrinsic_calib.py`** — folder of images in, intrinsics out. No
   camera, no display, no imports from any other module, so it runs over SSH
   on the rover.

OpenCV only (`opencv-python`, `numpy`). No ZED SDK, no CUDA, no `pyzed`.

## Setup

```bash
pip3 install opencv-python numpy
```

Check whether you even need to:

```bash
python3 -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

## 1. Capture the board

```bash
cd Thesis/Calibration
python3 capture_zed_right.py --checkerboard-size 9 6
```

| Key | Action |
|-----|--------|
| `SPACE` / `ENTER` | save the current frame |
| `u` | delete the last saved frame |
| `q` / `ESC` | quit |

Frames land in `ZedCaptures/` as `right_001.png`, `right_002.png`, … The
directory is created on first run and is resolved relative to this script, so
the same command works on the Windows box and the rover's Ubuntu checkout.
An interrupted session resumes the numbering instead of overwriting.

Only the **right half** of the ZED's side-by-side UVC frame is written, so the
output drops straight into step 2 with no cropping.

`--checkerboard-size` here is optional and only drives the green/red
`board: YES/no` overlay — frames are saved either way. The real sub-pixel
detection happens in step 2.

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--out-dir` | `ZedCaptures/` | Where to write the frames |
| `--camera-index` | `0` | UVC device index |
| `--resolution` | `4416x1242` | ZED side-by-side stereo mode; `driver` leaves the camera's own mode alone |
| `--checkerboard-size` | — | Inner corners, for the live indicator only |
| `--detect-every` | `5` | Run the preview detection every N frames |

ZED 2i modes — per-eye size in brackets:

| `--resolution` | per eye | fps |
|---|---|---|
| `1344x376` | 672×376 | 100 |
| `2560x720` | 1280×720 | 60 |
| `3840x1080` | 1920×1080 | 30 |
| `4416x1242` | 2208×1242 | 15 |

The default is the largest: left alone the ZED usually opens at the smallest,
where a square is ~15 px and corner localisation is at its noisiest. Frame rate
doesn't matter — the board is static. If the USB link can't sustain the mode
the driver silently keeps its own, and the script says so; drop to `2560x720`.

> **Calibrate at the mode you will actually run at.** ZED modes are different
> sensor crops/binnings, not scalings of one another, so `K` does *not*
> transfer exactly between them. See *Known limitations*.

### Shooting a good set

- 12–20 poses. Three is the mathematical floor, not a target.
- Large tilts about **both** board axes — a set of near-parallel views leaves
  the focal length poorly constrained.
- Vary the distance, and put the board in different parts of the frame
  (corners included: that is where distortion is largest).
- Board taped flat to rigid backing. A curled sheet is not a plane, and every
  bend is a corner the solver misplaces.
- In focus, evenly lit, no glare on the squares.

## 2. Solve the intrinsics

```bash
python3 zed_intrinsic_calib.py \
    --image-dir ZedCaptures \
    --checkerboard-size 9 6 \
    --square-size 0.025 \
    --output zed_right_intrinsics.json \
    --lvt2calib-export intrinsic.yaml
```

Pipeline per image: `findChessboardCorners` → `cornerSubPix` → then one
`calibrateCamera` over every view that survived, plus a per-image reprojection
error so a bad frame can be identified and reshot.

Images where the board isn't found are **skipped and reported**, never fatal:

```
[skip] right_004.png: checkerboard 9x6 inner corners not found
[skip] right_011.png: image size 320x200 differs from 2208x1242
[skip] right_017.png: unreadable (cv2.imread returned None)
```

Output:

```
Right eye 2208x1242, 18/20 images used
  fx=1743.201  fy=1742.884  cx=1103.552  cy=620.417
  dist (k1 k2 p1 p2 k3) = [-0.171, 0.0263, 0.00012, -8e-05, 0.0114]
  reprojection error: mean 0.1874 px, rms 0.1921 px, worst 0.3742 px (right_009.png)
```

Rule of thumb: mean **< 0.5 px** is good. **> 1 px** means bad detections are in
the set — check the named worst image and reshoot it.

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--image-dir` | *required* | Folder of right-eye checkerboard images |
| `--checkerboard-size` | *required* | **Inner** corner count, `COLS ROWS` |
| `--square-size` | *required* | Square edge in **metres** |
| `--output` | `zed_right_intrinsics.json` | Canonical JSON record |
| `--lvt2calib-export` | — | Also write an LVT2Calib intrinsics YAML here |
| `--verbose` | off | List accepted images too, not just skipped ones |

`--checkerboard-size` and `--square-size` have no defaults on purpose: a wrong
board silently produces a plausible-looking wrong `K`.

### Getting the board numbers right

**`--checkerboard-size` counts inner corners, not squares.** An inner corner is
where four squares meet; the outer border doesn't count.

```
┌───┬───┬───┬───┐
│ ■ │   │ ■ │   │     5 squares across
├───●───●───●───┤     3 squares down
│   │ ■ │   │ ■ │
├───●───●───●───┤     ● = inner corners
│ ■ │   │ ■ │   │     4 across, 2 down
└───┴───┴───┴───┘
                      --checkerboard-size 4 2
```

Squares − 1, each direction. A 10×7-square board is `9 6`.

- Order matters: if `9 6` finds nothing, try `6 9` — the board was rotated.
- Avoid equal counts (`7 7`): a square board leaves rotation ambiguous.
- Every image skipped as "not found" means the count is wrong, not the photos.

**`--square-size` is the edge of one square, in metres.** 25 mm → `0.025`.

Measure the *printed* board, not the PDF's nominal size — printers rescale.
Measure across many squares and divide: 10 squares spanning 247 mm →
`--square-size 0.0247`.

This one is forgiving: square size fixes the board **poses** but leaves `K` and
the distortion coefficients unchanged (they're scale-invariant), and the
LVT2Calib export carries no scale at all. A few percent off cannot corrupt the
extrinsic calibration. Wrong *units* is the real trap — `--square-size 25`
means 25-metre squares.

## 3. Hand off to LVT2Calib

```bash
cp intrinsic.yaml ~/catkin_ws/src/lvt2calib/data/camera_info/
```

Then pass the filename as that tool's `cam_info_filename` launch argument.

The export is an OpenCV `FileStorage` document with exactly the three fields
LVT2Calib reads:

```yaml
%YAML:1.0
---
CameraMat: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ ... ]
DistCoeff: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ ... ]
ImageSize: [ 2208, 1242 ]
```

This was confirmed from that repo (branch `ros_noetic`), not guessed — its
README only shows the layout as an image. `src/camera/cam_pattern.cpp:132-138`:

```cpp
cv::FileStorage fs_reader(camera_info_dir_, cv::FileStorage::READ);
fs_reader["CameraMat"] >> cam_intrinsic;
fs_reader["DistCoeff"] >> cam_distcoeff;
fs_reader["ImageSize"] >> img_size;
```

Two traps that cost real time if you hit them:

- **Do not use their `data/camera_info/rgb_camParam.txt` flat `K:`/`D:`
  layout.** Its parser (`ReadMatFromTxt`) sits commented out directly above the
  lines quoted here — the file would be silently ignored.
- **The `.txt` in their README is not plain text.** `cv::FileStorage` picks its
  parser by extension and treats anything that isn't `.xml`/`.json` as YAML, so
  their own `optris_demo.txt` is a YAML file with a `.txt` suffix. Either
  extension works.

`ImageSize` is not decorative: it is passed straight into
`initUndistortRectifyMap`, and must match the resolution of the frames you feed
LVT2Calib at runtime.

### JSON vs the export

|  | `--output` JSON | `--lvt2calib-export` YAML |
|---|---|---|
| Role | canonical record | compatibility handoff |
| Contains | K, distortion, mean/rms/per-image error, every filename used and skipped with reasons, checkerboard geometry, timestamp, OpenCV version, exact command | K, distortion, image size |
| Keep it | yes — this is the provenance | regenerable anytime |

The YAML is lossy. Don't treat it as the record.

## Structure

```
Calibration/
├── capture_zed_right.py    # live capture: keypress -> right-eye PNG
├── zed_intrinsic_calib.py  # headless: images -> K, distortion, JSON + LVT2Calib YAML
├── ZedCaptures/            # captured frames (gitignored)
└── README.md
```

## Known limitations

- **Neither script has been run against a real ZED.** The calibration maths is
  validated against synthetic boards rendered through a known `K`
  (fx = fy = 520, cx = 336, cy = 188, zero distortion), recovered to ~0.2 px
  with 0.043 px RMS; the capture script's frame splitting, resume numbering and
  save/undo are unit-tested with a fake camera. Real-camera behaviour — board
  detection under glare and blur, whether the USB link sustains HD2K — is
  unverified.
- **The LVT2Calib export has never been consumed by LVT2Calib itself.** It
  parses through `cv2.FileStorage` with node types identical to that repo's
  shipped `intrinsic.yaml`, but their ROS node was not run.
- **The live pipeline does not pin a resolution.**
  `EmissivityCalculation/emissivity/sources.py` opens the ZED without setting
  `CAP_PROP_FRAME_WIDTH/HEIGHT`, so it runs at whatever the driver defaults to
  — likely the smallest mode. Since `K` does not transfer exactly between ZED
  modes, calibrating at `4416x1242` while running at `1344x376` gives
  mismatched intrinsics. Either pin the live source to the same mode, or
  capture with `--resolution driver`.
- **Sub-pixel window is sized to the detected corner spacing**, not the usual
  fixed `(11, 11)`. On a small eye an 11 px half-window reaches past the
  neighbouring saddle point: measured on the synthetic set, forcing `(11, 11)`
  gave 2.87 px RMS and fx 373 against a true 520, versus 0.04 px and 519.8 with
  the adaptive window. Worth knowing before anyone "simplifies" it back.
- Only the right eye is calibrated. The left eye and the stereo baseline are
  out of scope.
