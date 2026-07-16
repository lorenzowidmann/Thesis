# DriveView

Live view from the ZED 2i's **second lens**, for driving the rover — a plain
`cv2.imshow` window, no CLIP, no LiDAR, no GPU load. Meant to run alongside
the headless `SensorFusion/sensor_fusion.py`, which uses the *left* eye for
classification and doesn't open a window.

`EmissivityCalculation`'s `ZedUvcSource` cropped only the left half of the
ZED's side-by-side UVC frame; it now takes an `eye="left"|"right"` argument,
so this reuses the same source class instead of reimplementing camera capture.

## Usage

```bash
cd Thesis/DriveView
C:\venvs\emissivity\Scripts\python.exe drive_view.py                  # ZED UVC, right eye (default)
C:\venvs\emissivity\Scripts\python.exe drive_view.py --eye left       # left eye instead
C:\venvs\emissivity\Scripts\python.exe drive_view.py --camera-index 1 # if the ZED isn't device 0
C:\venvs\emissivity\Scripts\python.exe drive_view.py --webcam         # plain webcam, for dev without a ZED
```

Press `q` in the window or Ctrl+C to stop.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--webcam` | off | Use a plain webcam instead of the ZED UVC feed |
| `--camera-index` | `0` | Device index/path |
| `--eye` | `right` | Which ZED lens to show (`left` matches SensorFusion's classified crop) |

## Structure

```
DriveView/
├── drive_view.py   # live cv2 window, right (or left) eye, no CLIP/LiDAR
└── README.md
```
