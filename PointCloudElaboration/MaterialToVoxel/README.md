# MaterialToVoxel — majority-vote material per voxel

Self-contained (no cross-import -- the files it reuses are copied in, see
"Provenance" below). Material counterpart of `../TemperatureToVoxel`, same
structure:

1. **`material_to_voxel.py`** -- same reprojection/z-buffer pipeline as
   `temperature_to_voxel.py` (range prefilter, `project_lidar_to_camera`,
   `z_buffer_mask`), but samples `EmissivityCalculation/project_to_flir.py`'s
   `segment_id.npy` per triplet instead of a temperature map, and resolves
   each pixel's segment id to a material label via that frame's
   `segments.json`. Material is categorical, so there is no mean: instead it
   votes, twice --  per-point majority across every pose that saw that
   point, then per-voxel majority among a voxel's points' (already
   point-level) labels. Writes `voxels_material.npz`.
2. **`broadcast_coarse_material.py`** -- optional, same containment-based
   broadcast as `broadcast_coarse_temperature.py`: if `voxels.npz` gets
   regenerated at a finer size than material was assigned at, don't rerun
   step 1 against the finer grid (same calibration/z-buffer-tolerance
   argument as the temperature side -- see that script's docstring -- and
   worse here, since a majority vote over only 1-2 points per 5cm voxel
   isn't a real majority). Instead broadcasts each *coarse* voxel's already-
   decided material onto every *fine* voxel that falls inside it. Writes
   `voxels_material_broadcast.npz` (same schema as step 1's output).
3. **`view_voxels.py`** -- pyvista viewer for `voxels_material.npz`: cubes
   colored by `material_id`, one fixed swatch per vocabulary material
   (matplotlib's `tab20`, up to 20 distinct colors) instead of
   `TemperatureToVoxel/view_voxels.py`'s percentile-clipped continuous
   range -- `material_id` is categorical, a continuous colormap would imply
   an ordering between materials that doesn't exist. The color bar is
   annotated with each material's name at its integer tick, doubling as the
   legend. Voxels with no valid material (`material_id == -1`) are dropped
   from the render by default, `--show-no-material` brings them back as
   grey. Same live min-count filter, floating-voxel declutter, and
   `planes_aligned.json` box overlay as the other viewers.

## Why

Puts a material label on the same voxel grid `temperature_to_voxel.py`
already puts a temperature on, so the two can be cross-referenced per voxel
(e.g. per-material temperature statistics) without a second independent
alignment/binning step.

## Usage

```
C:\venvs\planefit\Scripts\python.exe material_to_voxel.py
C:\venvs\planefit\Scripts\python.exe view_voxels.py
```

Defaults (all overridable):
- `--bag` -- the same filtered bag `AlignedOctree`/`TemperatureToVoxel` used:
  `C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\rosbag2_2026_07_30-18_12_20_filtered`
- `--sync-dir` -- `C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate`
  (must contain `sync_manifest.json` and `emissivity_map/<flir_stem>/segment_id.npy`)
- `--material-map-dir` -- `<sync-dir>/material_map_consensus` (must contain
  `<flir_stem>/segments.json`). **Confirmed with the user, not inferable
  from `temperature_to_voxel.py`** (it never reads `segments.json` at all):
  chosen over `material_map/` (classify_session.py's default SLIC run --
  69 segments on the sample frame checked, a *different* segment-id space
  than `segment_id.npy`'s observed 0-27 range, so it wouldn't even resolve)
  and `material_map_sam/` (same 28-segment SAM id space as
  `material_map_consensus`, but each frame's raw single-view CLIP call,
  pre-vote) because `temperature_to_voxel.py`'s own
  `corrected_temperature_consensus.npy` was itself produced from the
  `material_map_consensus` materials
  (`RadiometricCalibration/correct_session.py --material-map-dir
  material_map_consensus`) -- using the same source here keeps material and
  temperature voxels mutually consistent.
- `--voxels` / `--transform` -- `../AlignedOctree/voxels.npz` /
  `../AlignedOctree/transform.json` (read only, never regenerated here)
- `--calibration` -- `rig_calibration.yaml` (copied into this folder)

Tunable, same values/meaning as `temperature_to_voxel.py`: `--range-max-m
20`, `--zbuffer-tol-m 0.08`, `--min-depth-m 0.05`.

See each script's module docstring (`--help`) for the full flag list.

## Output

- `voxels_material.npz` -- `centers` (M,3) and `counts` (M,), copied
  straight from `voxels.npz`, plus:
  - `material_id` (M,) int32 -- index into `materials` (below); -1 where the
    voxel never got a single material-labeled point.
  - `material_confidence` (M,) float -- the winning material's share of that
    voxel's material-labeled points (e.g. 0.8 = 80% of the voxel's
    material-labeled points agreed on this label); NaN where `material_id`
    is -1. Same idea as `material_map_consensus/segments.json`'s own
    per-segment `consensus.agreement` field, one level up: computed at the
    voxel instead of the FLIR-segment.
  - `n_material_votes` (M,) int32 -- how many individual points contributed
    a (point-level-majority) label to that voxel's vote. Distinct from
    `counts` (voxels.npz's raw point count), same distinction
    `n_temp_samples` draws on the temperature side.
  - `materials` (K,) string array -- the vocabulary `material_id` indexes
    into, in first-seen order across the synced triplets (not alphabetical,
    not the emissivity table's own order).
  - `voxel_size`, `origin`, `depth` -- copied from `voxels.npz`, so this file
    is self-contained (no need to also load the original `voxels.npz`).

Console output reports coverage at every stage: material vocabulary size,
per-point vote coverage, and per-voxel majority coverage.

## Provenance (self-contained copies, adapted where noted)

- `rig_calibration.py`, `rig_calibration.yaml`, `projection.py` -- copied
  verbatim from `TemperatureToVoxel` (itself copied from `Thesis/Calibration/`).
- `read_pointcloud2`/`load_merged_cloud`, `z_buffer_mask`,
  `voxel_index`/`voxel_centers_to_index`, and the pack/searchsorted matching
  trick in `bin_material_into_voxels` -- copied verbatim from
  `temperature_to_voxel.py` (same self-contained convention: this folder
  doesn't cross-import from `TemperatureToVoxel` either, everything reused
  is physically copied in).
- `segment_id.npy` sampling -- new (temperature_to_voxel.py has no
  equivalent, it samples a float temperature map directly). Reads
  `EmissivityCalculation/project_to_flir.py`'s output (FLIR pixel grid,
  already nearest-fill complete) instead of re-deriving segment ids from
  scratch, same "reuse the existing artifact" approach as the temperature
  side reusing `corrected_temperature_consensus.npy`.
- Per-point and per-voxel majority voting -- new; no MATLAB or Python
  precedent in this repo (`Piano1_CorridoioLungo.m` section 9 only ever
  averages a continuous temperature, never a category).
- `view_voxels.py` -- copied from `TemperatureToVoxel/view_voxels.py`,
  adapted to color by `material_id` with a fixed qualitative colormap +
  annotated color bar instead of a percentile-clipped continuous range;
  min-count filter, declutter (still runs after the no-material filter, same
  reasoning), and the `planes_aligned.json` box overlay are unchanged.
