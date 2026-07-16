# CameraServer

Fixes a hardware constraint discovered while wiring up `SensorFusion` and
`DriveView`: **Windows locks a UVC device to whichever process opens it
first.** Tested empirically on the rover's ZED 2i (device index 0,
`1344x376` side-by-side UVC frame) — a second `cv2.VideoCapture` on the same
index gets zero successful frame reads (MSMF) or fails to even open
(DirectShow) while the first is active.

The ZED's right eye (classification, `SensorFusion`) and left eye (driving
view, `DriveView`) come from the **same physical stream**, so the fix isn't
"open the camera twice" — it's one process owns the camera and republishes
each frame for others to read.

## How it works

- `camera_server.py` opens the camera once (reuses `WebcamSource` from
  `EmissivityCalculation`, unsplit — the full left+right frame) and writes
  each frame into a `multiprocessing.shared_memory` block.
- `shared_frame.py` implements the shared memory: a seqlock (odd sequence
  number = write in progress, even = stable) so readers never see a torn
  frame without needing a real cross-process mutex, plus `FrameReader` (raw
  client) and `SharedZedSource` (drop-in `eye="left"|"right"` adapter, same
  interface as `ZedUvcSource`).
- `SensorFusion/sensor_fusion.py`, `SensorFusion/extrinsic_calibration.py`,
  and `DriveView/drive_view.py` all take a `--shared` flag that swaps in
  `SharedZedSource` instead of opening the camera directly.

## Usage

Start the server once, then run any number of `--shared` clients against it,
concurrently:

```bash
cd Thesis/CameraServer
C:\venvs\emissivity\Scripts\python.exe camera_server.py &

cd ../SensorFusion
C:\venvs\emissivity\Scripts\python.exe sensor_fusion.py --shared

cd ../DriveView
C:\venvs\emissivity\Scripts\python.exe drive_view.py --shared --eye left
```

Verified: with the server running, two independent client processes reading
the left and right eye concurrently for 4s got ~3500 frames each with
distinct, correct brightness values (the two lenses see slightly different
scenes) — versus 0 frames for the second process when both tried to open
the camera directly.

Stop the server with Ctrl-C; it unlinks the shared memory on exit. If a run
is killed uncleanly, the next `camera_server.py` start cleans up the stale
segment itself (no manual cleanup needed).

See `run_commands.txt` for the full 3-terminal copy-paste sequence (server,
then emissivity, then drive view).

## Auto-close when idle

The server also self-closes if **neither eye** has been read for
`--idle-timeout` seconds (default 30) — you don't need to remember to Ctrl-C
it once `sensor_fusion.py --shared` and `drive_view.py --shared` have both
exited. Attaching or reading counts as activity from either script, so the
timer only starts once both are gone. `--idle-timeout 0` disables this and
runs until Ctrl-C (the old behavior).

Verified: with `--idle-timeout 5`, a client reading continuously for 6s kept
the server alive past the 5s mark, then the server closed ~5s after that
client disconnected; with no client at all it closed after exactly 5s.

## When you don't need this

If only one script uses the camera at a time (e.g. just `drive_view.py`, or
just `sensor_fusion.py`), skip the server and use `--zed-uvc` /
`--webcam` / `--image` directly, same as before -- this is purely for
running two camera consumers at once.

## Structure

```
CameraServer/
├── shared_frame.py    # FrameWriter (server), FrameReader / SharedZedSource (clients)
├── camera_server.py   # CLI entry point: owns the camera, publishes frames
├── run_commands.txt   # copy-paste 3-terminal sequence
└── README.md
```
