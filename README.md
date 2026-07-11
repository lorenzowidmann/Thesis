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
   well-formed surfaces for OpenStudio import**: a mode/tolerance-band fit
   snaps recessed windows flush as co-planar fenestration and preserves
   per-class zoning (a wall subdivided into homogeneous sub-surfaces). The
   flattening axis auto-aligns to the building's real orientation via PCA
   (`u` = dominant wall direction, `v` = perpendicular), since real buildings
   are rarely aligned with world x/y. Exports OpenStudio-friendly polygon JSON
   (with an optional `.osm` SDK adapter).

Runs on Python 3.13 with `laspy` + `PyVista`.

Output: a voxel/planar-surface representation of the scene — OpenStudio-ready
building surfaces, and the foundation for the deferred LiDAR/stereo
co-registration and surface-geometry work.

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
- **Point-cloud smoothing scope** — smoothing handles one dominant wall
  direction pair (`u`/`v`) plus vertical; a building with more than two wall
  orientations still needs full multi-plane segmentation (documented in the
  module's README).
- **Synchronization** (temporal) — matching each thermal frame to the
  LiDAR/ZED frame captured at the same instant on the moving rover (see the
  design note in `RadiometricCalibration/README.md`).
- **Angle correction** — emissivity currently uses tabulated normal values;
  surface-geometry correction from the stereo/LiDAR data is future work.

See each module's own `README.md` for setup, usage, and the physics.
