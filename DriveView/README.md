# DriveView

Live view from the ZED 2i's **first lens**, for driving the rover — a plain
`cv2.imshow` window, no CLIP, no LiDAR, no GPU load. Meant to run alongside
other headless camera consumers, which use the *right* eye for
classification and don't open a window.

`EmissivityCalculation`'s `ZedUvcSource` cropped only the left half of the
ZED's side-by-side UVC frame; it now takes an `eye="left"|"right"` argument,
so this reuses the same source class instead of reimplementing camera capture.

**Running alongside another camera script?** Windows locks the ZED's UVC
device to whichever process opens it first — verified empirically, a second
`cv2.VideoCapture` on the same index gets zero frames while the first is
active. Two `--zed-uvc`-style direct accesses **cannot run at the same
time**. Use `--shared` instead, with `CameraServer/camera_server.py`
running, so both read the one physical stream. See
`../CameraServer/README.md`.

## Usage

```bash
cd Thesis/DriveView
C:\venvs\emissivity\Scripts\python.exe drive_view.py                  # ZED UVC, left eye (default)
C:\venvs\emissivity\Scripts\python.exe drive_view.py --eye right      # right eye instead
C:\venvs\emissivity\Scripts\python.exe drive_view.py --camera-index 1 # if the ZED isn't device 0
C:\venvs\emissivity\Scripts\python.exe drive_view.py --webcam         # plain webcam, for dev without a ZED
C:\venvs\emissivity\Scripts\python.exe drive_view.py --shared         # read from a running
                                                                       # camera_server.py, so this can
                                                                       # run alongside other consumers
```

Press `q` in the window or Ctrl+C to stop.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--webcam` / `--shared` | off | `--webcam`: plain webcam instead of the ZED; `--shared`: read from a running `CameraServer/camera_server.py` instead of opening the camera directly |
| `--camera-index` | `0` | Device index/path (ignored with `--shared`) |
| `--eye` | `left` | Which ZED lens to show (`right` matches the classified crop) |

## Structure

```
DriveView/
├── drive_view.py   # live cv2 window, left (or right) eye, no CLIP/LiDAR
└── README.md
```
