# ExtrinsicCalibration

Extrinsic calibration between the three rover sensors — Livox HAP LiDAR,
ZED 2i RGB camera, FLIR Vue Pro R thermal camera — using a **standard planar
chessboard** instead of the circular-holes target required by L²V²T²Calib.

**This is a bridge solution.** Once the holed target is available, replace
this pipeline with L²V²T²Calib (point-to-point hole-center correspondences);
see *Known limitations* below for why.

## Method

- **Camera side (RGB and thermal):** the chessboard pattern is detected with
  OpenCV; `solvePnP` gives the board plane (normal + distance) in each camera
  frame. The thermal camera only sees the pattern with **active contrast**:
  the board is heated (heating panel / lamp) during acquisition so the squares
  differ in emitted radiation.
- **LiDAR side:** the LiDAR does **not** see the pattern — only the physical
  planarity of the panel. A RANSAC plane fit on a ROI-cropped cloud gives the
  same board plane in the LiDAR frame. The RANSAC primitive is reused from
  `PointCloudElaboration/OcTree/octree/smoothing.py` (`fit_plane_ransac`,
  pure numpy) instead of adding open3d.
- **LiDAR ↔ camera:** point-plane constraint calibration (Zhang & Pless 2004,
  extended to 3-D LiDAR by Pandey et al.): over N board poses, the rigid
  transform must place every LiDAR inlier point on the camera-frame plane.
  Solved with `scipy.optimize.least_squares` (Levenberg-Marquardt) from a
  closed-form seed (Kabsch on normals + linear solve for t). The same solver
  produces LiDAR→ZED and LiDAR→thermal, just from different image sets.
- **RGB ↔ thermal:** both cameras see the pattern, so this pair uses plain
  `cv2.stereoCalibrate` on the shared corner detections (intrinsics fixed).
- **Consistency:** `T_rgb→thermal · T_lidar→rgb ≈ T_lidar→thermal`; the
  residual rotation (deg) and translation (m) of the loop closure are
  reported. A small residual is necessary but not sufficient for accuracy.

## Structure

- `intrinsics_rgb.py` — ZED 2i intrinsics (chessboard, `calibrateCamera`)
- `intrinsics_thermal.py` — FLIR intrinsics (heated chessboard + CLAHE preprocessing)
- `lidar_plane_fit.py` — board plane from a LiDAR cloud (RANSAC, reused from OcTree)
- `camera_plane_pose.py` — board plane from a camera image (`solvePnP`)
- `lidar_camera_extrinsic.py` — point-plane LiDAR→camera solve (LM)
- `camera_camera_extrinsic.py` — RGB→thermal via `stereoCalibrate`
- `consistency_check.py` — loop-closure residual of the three transforms
- `main.py` — CLI orchestrating single phases or the full pipeline

## Setup

```
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The requirements are local to this folder (OcTree's stay untouched); the
OcTree RANSAC import needs no extra dependency (numpy only).

## Usage

Dataset layout (generic inputs for now — images readable by `cv2.imread`,
LiDAR clouds as `.npy` (N, 3) arrays in the sensor frame; the real
Livox/ZED/FLIR loaders get plugged in later):

```
data/
  intrinsics_rgb/       chessboard images, many poses (RGB)
  intrinsics_thermal/   chessboard images, many poses (thermal, heated board)
  poses/
    pose_000/
      rgb.png           board seen by the ZED 2i
      thermal.png       same static board pose, FLIR
      lidar.npy         same pose, Livox cloud
      roi.json          optional [xmin, xmax, ymin, ymax, zmin, zmax] LiDAR
                        crop isolating the board (overrides --roi)
    pose_001/ ...
```

```
py main.py all --data data --out calibration.json
py main.py intrinsics-rgb --data data          # single phase; JSON updated incrementally
py main.py lidar-rgb --data data --roi -1 1 -1 1 1 4
py main.py check
```

All transforms land in one `calibration.json`: `rgb_intrinsics`,
`thermal_intrinsics`, `lidar_to_rgb`, `lidar_to_thermal`, `rgb_to_thermal`
(each with `R`, `t` and a 4×4 `transform`, p_target = T @ p_source), plus the
`consistency` report.

Acquisition guidance: ≥ 3 board poses are the mathematical minimum for the
plane-based solves; use 8–15 poses spanning **large tilts around both board
axes** and different distances — near-parallel planes leave the transform
unconstrained.

## Known limitations (why this is a bridge solution)

- **Lower accuracy than the holed-target method.** A plane constrains only
  3 of 6 DoF per pose (the in-plane translations and the in-plane rotation
  are invisible to a featureless plane); accuracy comes from accumulating
  many well-spread poses, and residual errors are typically an order of
  magnitude above point-to-point hole correspondences.
- **LiDAR sees no pattern feature.** `lidar_plane_fit.py` uses only the
  panel's physical planarity — board-edge or hole features would anchor the
  in-plane DoF directly, which is exactly what L²V²T²Calib's circular holes
  provide.
- **The LiDAR–thermal pair is the weakest.** It combines the noisiest plane
  observations on both sides: LiDAR range noise on one, low-resolution /
  low-contrast thermal corner detection (which also decays as the heated
  board equalizes) on the other. Prefer composing LiDAR→RGB with RGB→thermal
  if the consistency check shows a large direct residual.
- **Thermal detection depends on active heating.** Corner accuracy varies
  with the thermal contrast at capture time; recapture rather than accept
  marginal detections.
- The `consistency_check` residual measures **internal coherence only** — a
  bias shared by all three estimates cancels out and stays invisible.
