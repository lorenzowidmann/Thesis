# AlignedOctree — full-building leveling + octree voxelization

Self-contained (no cross-import from OpenStudioModel or OcTree -- the files
it reuses are copied in, see "Provenance" below). Two-step pipeline:

1. **`fit_closed_planes.py`** -- RANSAC-fits a single watertight box (floor,
   ceiling, 4 walls) from the raw LiDAR point cloud. Writes `planes.json`.
2. **`aligned_octree.py`** -- derives a 3-axis "building frame" rotation from
   that closed box, applies it (+ a translation putting the floor at z=0) as
   a rigid transform to the *entire* raw cloud, then octree-voxelizes the
   now axis-aligned result. Writes `voxels.npz` + `transform.json`.
3. **`view_voxels.py`** -- pyvista viewer for `voxels.npz`: cubes colored by
   point count (no semantic classes here), with the closed box from
   `planes.json` overlaid in the same aligned frame as a sanity check, and an
   optional raw-points overlay. Interactive window by default, or
   `--screenshot out.png` for a headless render.

## Why

Generalizes the leveling step in De Pazzi, Chiodini, Pertile (Sensors 2022),
["3D Radiometric Mapping by Means of LiDAR SLAM and Thermal Camera Data
Fusion"](https://doi.org/10.3390/s22134794) -- that paper levels against one
ground plane vs. gravity (Sec. 4.2, Eq. 14), which only corrects a tilt. This
corridor sits at a real *yaw* in the SLAM (`camera_init`) frame, not just a
tilt, so one plane isn't enough: the full closed set of walls/floor/ceiling
is used instead to define all 3 axes. The octree step itself is the paper's
actual method, unchanged: one root bin encompassing all points, recursively
subdivided into 8 occupied children per level.

No thermal/temperature averaging per voxel yet -- geometry/alignment only.

## Usage

```
C:\venvs\planefit\Scripts\python.exe fit_closed_planes.py --bag <rosbag2_folder> --out planes.json
C:\venvs\planefit\Scripts\python.exe aligned_octree.py --planes planes.json --depth 8
C:\venvs\planefit\Scripts\python.exe view_voxels.py
```

`--bag` defaults to the reference bag:
`C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\rosbag2_2026_07_30-18_12_20_filtered`
(single merged `/cloud_registered` message, ~1,066,093 points, fields
x,y,z float32, frame_id=camera_init).

`aligned_octree.py` re-reads the same bag itself (via the `bag`/`topic`/
`store` fields recorded in `planes.json`, overridable) -- the planes are used
*only* to compute the alignment transform, every point in the bag is still
loaded and transformed, none are clipped/dropped by plane membership.

See each script's module docstring (`--help`) for the full flag list.

## Output

- `planes.json` -- same schema as `fit_planes.py`: `normal`, `d`,
  `orientation` (`wall` / `floor_ceiling`), `tilt_deg`, `centroid_3d`,
  `corners_3d`, etc. Closed box only (`--close-geometry` + `cap_open_faces`
  always on), no free-standing fragments.
- `voxels.npz` -- `centers` (M,3), `counts` (M,), `voxel_size` (scalar),
  `origin` (3,), `depth` (scalar): occupied voxel centers + point counts of
  the aligned cloud, same fields `octree/voxelizer.py`'s `VoxelGrid` produces.
- `transform.json` -- `rotation` (3x3), `translation` (3,) and their inverse
  (`rotation_inv`, `translation_inv`), so voxel coordinates can be mapped
  back to the original SLAM/map (`camera_init`) frame later. Row-vector
  convention: `aligned = points @ rotation.T + translation`.

## Provenance (self-contained copies, adapted where noted)

- `fit_planes.py` -- copied verbatim from `Thesis/OpenStudioModel/fit_planes.py`.
  `fit_closed_planes.py` imports `load_merged_cloud`, `segment_planes`,
  `dedupe_planes`, `close_geometry` from it directly.
- `octree/octree.py` -- copied verbatim from
  `PointCloudElaboration/OcTree/octree/octree.py` (numpy-only, no adaptation
  needed).
- `octree/voxelizer.py` -- copied from the same module, with the
  `classes.py`-dependent `MAX_CLASS_ID` import inlined as a local constant
  (`= 0`): `classes.py` (TUM-FACADE semantic ids) was intentionally not
  copied here since this pipeline doesn't classify points. The `VoxelGrid`
  fields used (`centers`, `counts`, `voxel_size`, `origin`) are unaffected;
  the unused `labels` field is always 0.
- `octree/smoothing.py`, `viewer.py`, `classes.py`, `las_loader.py`,
  `rosbag_loader.py`, `openstudio_adapter.py` are **not** copied -- not
  needed for geometry/alignment-only voxelization.
