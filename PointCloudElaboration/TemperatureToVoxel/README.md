# TemperatureToVoxel — corrected FLIR temperature per voxel

Self-contained (no cross-import -- the files it reuses are copied in, see
"Provenance" below). Three-step pipeline:

1. **`temperature_to_voxel.py`** -- for every FLIR/LiDAR pose in
   `sync_manifest.json`, reprojects `AlignedOctree`'s raw point cloud into
   that pose's FLIR image, z-buffers out occluded points, samples
   `corrected_temperature_consensus.npy` at the surviving pixels, and
   accumulates a per-point multi-pose mean temperature. Applies
   `AlignedOctree/transform.json`'s rigid transform to the whole cloud, bins
   it into `AlignedOctree/voxels.npz`'s *existing* voxel grid, and averages
   per-voxel. Writes `voxels_temperature.npz`.
2. **`broadcast_coarse_temperature.py`** -- optional. If `voxels.npz` gets
   regenerated at a finer voxel size than temperature was assigned at (e.g.
   0.05 m vs. the 0.15 m `temperature_to_voxel.py` was run at), do NOT rerun
   step 1 directly against the finer grid: the LiDAR<->FLIR extrinsic
   calibration RMSE (~5.8cm, `rig_calibration.yaml`) and z-buffer tolerance
   (8cm) are both coarser than a 5cm voxel, so binning FLIR observations
   straight into 5cm cells would assign spatial precision the calibration
   can't back up, and would fragment the (already sparse) per-point
   observations across ~27x more bins. Instead this script broadcasts each
   *coarse* voxel's already-computed `mean_temperature` onto every *fine*
   voxel that falls inside it (containment lookup only, no new FLIR
   sampling) -- same temperature repeated across a coarse cell's children,
   so two calibration-indistinguishable regions never get assigned different
   values, while the finer grid still renders at its own resolution. Writes
   `voxels_temperature_broadcast.npz` (same schema as step 1's output, drop
   straight into `view_voxels.py --voxels`).
3. **`view_voxels.py`** -- pyvista viewer for `voxels_temperature.npz`:
   cubes colored by `mean_temperature` (voxels with no valid observation are
   dropped from the render by default, `--show-no-temp` brings them back as
   grey), with the same live min-count filter, floating-voxel declutter, and
   `planes_aligned.json` box overlay as `AlignedOctree/view_voxels.py`.

   (A `--filler` flag -- patching small floor/ceiling coverage gaps with
   synthetic voxels -- was tried and removed after several iterations that
   never fully converged on correct behavior; see `FILLER.md` for the full
   design, the debugging journey, and the removed code, kept for a future
   attempt rather than lost.)

## Why

Ports the working logic in `MATLAB_PointCloudVisualization/Piano1_CorridoioLungo.m`
section 9 ("Colorazione per temperatura FLIR") to Python, feeding
`AlignedOctree`'s voxel grid instead of a fresh point cloud + an
independently recomputed pooling voxel size. Every triplet reprojects the
*same* one already-accumulated map cloud (loaded once, matching the MATLAB
script's `xyzFilt = pc.Location`) -- unlike
`EmissivityCalculation/project_to_flir.py`, which re-fetches a fresh
single-scan LiDAR snapshot from the bag per timestamp.

`corrected_temperature_consensus.npy` is already the radiometrically
calibrated, multi-view-material-consensus result -- used as-is here, no
solar-artifact correction (that is a separate, not-yet-built step on top of
this).

## Usage

```
C:\venvs\planefit\Scripts\python.exe temperature_to_voxel.py
C:\venvs\planefit\Scripts\python.exe view_voxels.py
```

Defaults (all overridable):
- `--bag` -- the same filtered bag `AlignedOctree` used:
  `C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\rosbag2_2026_07_30-18_12_20_filtered`
- `--sync-dir` -- `C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate`
  (must contain `sync_manifest.json` and `emissivity_map/<flir_stem>/corrected_temperature_consensus.npy`)
- `--voxels` / `--transform` -- `../AlignedOctree/voxels.npz` /
  `../AlignedOctree/transform.json` (read only, never regenerated here)
- `--calibration` -- `rig_calibration.yaml` (copied into this folder)

Tunable to match `Piano1_CorridoioLungo.m` section 9 exactly (defaults are
already its values): `--range-max-m 20`, `--zbuffer-tol-m 0.08`,
`--min-depth-m 0.05`.

See each script's module docstring (`--help`) for the full flag list,
including `view_voxels.py`'s live keyboard controls (`]`/`[`/`0`/`d`, same
as `AlignedOctree/view_voxels.py`) and `--show-no-temp`.

## Output

- `voxels_temperature.npz` -- `centers` (M,3) and `counts` (M,), copied
  straight from `voxels.npz`, plus:
  - `mean_temperature` (M,) float -- mean of the voxel's points' per-point
    (already pose-averaged) temperatures; NaN where no point in that voxel
    ever got a valid observation.
  - `n_temp_samples` (M,) int -- how many individual points contributed to
    that mean. Distinct from `counts` (voxels.npz's original raw point
    count): a point can be geometrically present in a voxel (`counts`)
    without ever landing in a usable, unoccluded, non-NaN FLIR pixel in any
    pose (excluded from `n_temp_samples`).
  - `voxel_size`, `origin`, `depth` -- copied from `voxels.npz`, so this
    file is self-contained for the viewer (no need to also load the
    original `voxels.npz`).

Console output reports two coverage numbers as a sanity check: how many of
the raw points got >=1 valid observation (step 5 of the pipeline), and how
many of `voxels.npz`'s occupied voxels ended up with a valid
`mean_temperature` (step 7).

## Provenance (self-contained copies, adapted where noted)

- `rig_calibration.py`, `rig_calibration.yaml`, `projection.py` -- copied
  verbatim from `Thesis/Calibration/`. `project_lidar_to_camera` (in
  `projection.py`) already implements the same pinhole + radial/tangential
  distortion projection as `Piano1_CorridoioLungo.m`'s hand-rolled
  `projectPinholeTemp` (FLIR's `k3 = 0`, so `cv2.projectPoints`' 5-param
  model and MATLAB's 4-param formula are numerically identical) -- reused
  directly rather than reimplemented, with only the stricter
  `MIN_DEPTH_M = 0.05` cutoff (`projectPinholeTemp`'s `z > 0.05`, vs.
  `project_lidar_to_camera`'s plain `depth > 0`) layered on top in
  `temperature_to_voxel.py`.
- `z_buffer_mask` in `temperature_to_voxel.py` -- ported fresh from
  `zBufferMaskTemp` (no equivalent existed anywhere in this repo): nearest
  point per pixel wins, but any point within `ZBUFFER_TOL_M` of that pixel's
  true minimum depth also survives (not a strict single winner). This is
  what keeps an occluded point from stealing a visible surface's
  pixel/temperature -- distinct from `project_to_flir.py`'s approach, which
  has no z-buffer at all (last point in iteration order simply overwrites
  the pixel).
- `load_merged_cloud`/`read_pointcloud2` in `temperature_to_voxel.py` --
  copied from `AlignedOctree/fit_closed_planes.py` (same convention: raw,
  unaligned bag points in the SLAM/map frame).
- `voxel_index`/`voxel_centers_to_index` in `temperature_to_voxel.py` --
  reimplement `octree/voxelizer.py`'s `origin`/`voxel_size`/`depth` ->
  integer voxel index convention directly (not a copy of the whole file):
  this script only ever needs to map new points onto voxels.npz's *existing*
  grid, never build a new one, so `voxelize()`/`voxelize_octree()`/
  `VoxelGrid`/`filter_by_count` would be dead weight.
- `view_voxels.py` -- copied from `AlignedOctree/view_voxels.py`, adapted to
  color by `mean_temperature` (dropped from the render by default,
  `--show-no-temp` to bring them back grey) with a percentile-clipped color
  range instead of point count; min-count filter, declutter, and the
  `planes_aligned.json` box overlay are unchanged, except declutter now runs
  *after* the no-temp filter (not before) -- otherwise a voxel counted as
  "connected" only through a no-temp neighbour that then gets dropped from
  the render would still end up visually floating, just past the point
  declutter could still catch it. (A `--filler` flag lived here too --
  removed, see `FILLER.md`.)
