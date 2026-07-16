# Thesis — true surface-temperature estimation from a rover

Software for a mobile rover instrumented with a **thermal camera**, a **ZED 2i
stereo camera**, and a **LiDAR**. The goal is to turn the raw reading of the
thermal camera into an accurate **true surface temperature** of the terrain
and objects the rover observes.

A thermal camera does not measure temperature directly — it measures incoming
infrared radiation and converts it assuming the target is a perfect black body
(emissivity ε = 1) seen through perfectly transparent air (transmission
τ = 1). Both assumptions are false, so the raw "apparent temperature" is
biased. Correcting it needs two things the thermal camera cannot supply on its
own: the **emissivity** of the material being viewed, and the **distance** to
each point (which sets how much radiation the atmosphere absorbs). The first
two modules below provide exactly those, and the second combines them into the
corrected temperature. A third module begins the **point-cloud elaboration**
that the geometry-side corrections (co-registration, viewing angle) will build
on.

## Modules

### 1. `EmissivityCalculation/` — what material am I looking at?

Estimates the **emissivity** of the surface seen by the camera. A frame from
the ZED 2i (or an image file / webcam during development) is classified by
material using CLIP zero-shot image classification, and the emissivity is
looked up in a table of tabulated literature values
(`emissivity_table.csv`). New materials are added by editing the CSV — no
retraining. Currently returns tabulated *normal* emissivity only;
viewing-angle correction is deferred.

Output: the emissivity value ε that the radiometric correction needs.

### 2. `RadiometricCalibration/` — from apparent to true temperature

Converts the thermal camera's **apparent-temperature map** into a **true
temperature map**, correcting **per pixel**. Each pixel is corrected with its
own LiDAR distance (which fixes the atmospheric transmission τ) and its own
emissivity (from module 1), while relative humidity and air temperature are
global for the frame. It implements the standard thermography measurement
chain: water-vapour content from humidity, atmospheric transmission from
distance, then inversion of the radiation balance to recover the object's true
temperature. Includes a `--show` visualization and a synthetic demo scene for
testing without hardware.

Output: the corrected true-temperature map.

### 3. `PointCloudElaboration/OcTree/` — point cloud → planar building surfaces

Groundwork for the geometry side. It loads a semantically annotated point
cloud (the TUM-FACADE benchmark, used as stand-in data) and runs a small
processing pipeline, each stage behind a toggle, in an interactive PyVista GUI:

1. **Voxelize** — sample the cloud into voxels via an **octree** subdivision;
   a metric voxel-size slider (0.05–1.0 m), semantic-class coloring, a
   raw-points overlay, and a per-change check that every voxel holds ≥1 point.
2. **Filter** — a minimum-points-per-voxel threshold (1–10) that hides sparse,
   often-disconnected scan noise.
3. **Smooth** — flatten the stepped voxels onto a plane to get **planar,
   well-formed surfaces for OpenStudio import**. The plane is found by a
   **RANSAC/MSAC dominant-plane fit** (pure numpy) that locks onto the actual
   wall at *any* orientation — tilt included — not just world x/y. A
   tolerance-band snap pulls recessed windows flush as co-planar fenestration,
   and per-class zoning is preserved (a wall subdivided into homogeneous
   sub-surfaces). The `u`/`v`/`z` selector picks *which* detected plane to
   flatten — `u` the dominant facade, `v` the perpendicular facade, `z` the
   roof/floor — with the legacy PCA-yaw voxel-layer method still available via
   `--offset-method`. An opt-in `--project-to-axis-aligned` step then derives a
   **second** surface from the fitted plane on a **world-axis-aligned** grid
   (gravity-vertical columns on a facade, X/Y on a roof) instead of the diagonal
   PCA basis, dropping colour patches smaller than `--min-side` metres as noise.
   Exports OpenStudio-friendly polygon JSON (with an optional `.osm` SDK
   adapter).

Also reads real SLAM data directly: `--db3 bag.db3` loads a rosbag2 sqlite3
bag's `PointCloud2` scans (stdlib only, no ROS install), with `--db3-stride`
and `--db3-point-stride` to subsample scans/points on the often much larger
real-world clouds.

Runs on Python 3.13 with `laspy` + `PyVista` (pure numpy for the RANSAC path).

Output: a voxel/planar-surface representation of the scene — OpenStudio-ready
building surfaces, and the foundation for the deferred LiDAR/stereo
co-registration and surface-geometry work.

### 4. `SensorFusion/` — headless emissivity + distance, and extrinsic calibration seed

A GUI-free companion to modules 1 and `LidarDistance/`, for the onboard rover
PC whose GPU can't carry the `--show`/`--live` overlay path. `sensor_fusion.py`
prints emissivity (from module 1's CLIP classifier) and LiDAR distance for the
same central square, per cycle, to the terminal — no display, no file output
yet. `extrinsic_calibration.py` is the seed of the LiDAR ↔ stereo-camera
extrinsic calibration (the co-registration step mentioned below): it grabs a
synced frame + LiDAR window as a live-sensor sanity check and prints an
identity placeholder transform until a calibration target is available. Both
scripts reuse the existing modules' functions directly rather than
duplicating them.

Output: terminal-printed emissivity + distance per cycle; later, the LiDAR →
camera extrinsic transform.

### 5. `DriveView/` — live view from the ZED 2i's second lens

Module 1 / `SensorFusion` classify the ZED's **left** eye headlessly (no
window). `DriveView/drive_view.py` shows the **right** eye live in a plain
`cv2` window — no CLIP, no LiDAR, negligible GPU load — so the rover can be
driven visually while `SensorFusion` runs headless alongside it.
`EmissivityCalculation`'s `ZedUvcSource` now takes an `eye="left"|"right"`
argument so this reuses the same capture code instead of duplicating it.

Output: a live video window for teleoperation.

## How the modules connect

Modules 1–2 form the temperature-correction chain; module 3 is a separate,
foundational strand for the scene geometry (not yet wired into the chain):

```
        ZED 2i ─▶ EmissivityCalculation ─▶ emissivity ε ─┐
                                                          │
  thermal camera ─▶ apparent temperature ────────────────┼─▶ RadiometricCalibration ─▶ true temperature
                                                          │
          LiDAR ─▶ distance map ───────────────────────▶─┤
                                                          │
     hygrometer ─▶ humidity + air temp ────────────────▶─┘
```

The modules are loosely coupled: `RadiometricCalibration` reads the emissivity
value (or looks a material up directly in `EmissivityCalculation`'s CSV), but
neither imports the other's heavy dependencies.

## Current status

All three modules are **drafts** and run today on files/values (or stand-in
datasets) instead of live sensors. Field integration is deliberately deferred
and documented in each module's README:

- **Hardware drivers** — the ZED SDK, thermal-camera SDK, LiDAR, and
  hygrometer are stubbed until the devices are available on this PC; the
  point-cloud module uses the TUM-FACADE benchmark as stand-in LiDAR data.
- **Co-registration** (spatial) — the thermal, distance, and emissivity maps
  are assumed pixel-aligned; projecting the LiDAR/ZED data onto the thermal
  image is a separate step that the point-cloud module lays groundwork for.
  `SensorFusion/extrinsic_calibration.py` is the first concrete seed of this —
  currently an identity placeholder pending a calibration target.
- **Point-cloud plane scope** — the RANSAC smoothing fits the single
  **dominant** plane at any orientation (the `u`/`v`/`z` selector picks the
  dominant facade, its perpendicular, or the roof/floor); iterating the fit to
  segment *all* walls at once (full multi-plane segmentation) is still future
  work (documented in the module's README).
- **Synchronization** (temporal) — matching each thermal frame to the
  LiDAR/ZED frame captured at the same instant on the moving rover (see the
  design note in `RadiometricCalibration/README.md`).
- **Angle correction** — emissivity currently uses tabulated normal values;
  surface-geometry correction from the stereo/LiDAR data is future work.

See each module's own `README.md` for setup, usage, and the physics.
