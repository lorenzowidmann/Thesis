# DriveView

Live view from the ZED 2i's **second lens**, for driving the rover — a plain
`cv2.imshow` window, no CLIP, no LiDAR, no GPU load. Meant to run alongside
the headless `SensorFusion/sensor_fusion.py`, which uses the *left* eye for
classification and doesn't open a window.

`EmissivityCalculation`'s `ZedUvcSource` cropped only the left half of the
ZED's side-by-side UVC frame; it now takes an `eye="left"|"right"` argument,
so this reuses the same source class instead of reimplementing camera capture.

**Running alongside `sensor_fusion.py`?** Windows locks the ZED's UVC device
to whichever process opens it first — verified empirically, a second
`cv2.VideoCapture` on the same index gets zero frames while the first is
active. `drive_view.py --zed-uvc`-style direct access and `sensor_fusion.py
--zed-uvc` **cannot run at the same time**. Use `--shared` instead, with
`CameraServer/camera_server.py` running, so both read the one physical
stream. See `../CameraServer/README.md`.

## Usage

```bash
cd Thesis/DriveView
C:\venvs\emissivity\Scripts\python.exe drive_view.py                  # ZED UVC, right eye (default)
C:\venvs\emissivity\Scripts\python.exe drive_view.py --eye left       # left eye instead
C:\venvs\emissivity\Scripts\python.exe drive_view.py --camera-index 1 # if the ZED isn't device 0
C:\venvs\emissivity\Scripts\python.exe drive_view.py --webcam         # plain webcam, for dev without a ZED
C:\venvs\emissivity\Scripts\python.exe drive_view.py --shared         # read from a running
                                                                       # camera_server.py, so this can
                                                                       # run alongside sensor_fusion.py
```

Press `q` in the window or Ctrl+C to stop.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--webcam` / `--shared` | off | `--webcam`: plain webcam instead of the ZED; `--shared`: read from a running `CameraServer/camera_server.py` instead of opening the camera directly |
| `--camera-index` | `0` | Device index/path (ignored with `--shared`) |
| `--eye` | `right` | Which ZED lens to show (`left` matches SensorFusion's classified crop) |

## Structure

```
DriveView/
├── drive_view.py   # live cv2 window, right (or left) eye, no CLIP/LiDAR
└── README.md
```
