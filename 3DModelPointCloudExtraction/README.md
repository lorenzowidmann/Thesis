# 3DModelPointCloudExtraction

Pipeline from a FAST-LIO point cloud (loaded from a `.pcd`/bag) to an OpenStudio
building model: fit axis-aligned room boxes, QA/edit them, merge multi-session
scans, then export to `.osm`.

## MATLAB

- `BagFilter.m` — loop-closure correction on an already-processed FAST-LIO bag.
  Un-transforms `/cloud_registered` back to body frame using `/Odometry`,
  finds loops with Scan Context, verifies with ICP, builds and optimizes a
  pose graph (sequential + loop constraints), rebuilds the map with corrected
  poses.
- `BagFilter_NoLoopClosure.m` — same correction pipeline without loop search
  (no Scan Context/ICP/loop constraints). Uses gravity + temporal constraints
  and non-loop drift corrections (wall yaw, floor Z) only, plus statistical
  outlier removal (per-keyframe and on the final map).
- `ViewPCD.m` — quick viewer for a saved `.pcd` (e.g. from `SavedBag/`), with
  optional voxel and/or random downsampling before `pcshow`. Requires
  Computer Vision Toolbox / Lidar Toolbox.

## Python

Run in order:

1. `fit_boxes.py` — tiles the floor footprint of every hall in the cloud into
   axis-aligned rectangular boxes (extruded to each hall's height); writes
   `boxes.json`.
2. `interactive_boxes.py` — top-down interactive editor for `boxes.json`
   (fix/add/delete a box); saves an edited copy, never overwrites the input.
3. `merge_boxes.py` — merges multiple `boxes.json` sessions (one per scanned
   bag/run) into a single file, with click-drag translate-only alignment per
   session.
4. `show_boxes.py` — QA viewer: colors the cloud by fitted box and overlays
   each box as a wireframed cuboid (PyVista). Off-screen PNG by default,
   `--interactive` for a live window, `--live` to poll `boxes.json` and
   redraw as `fit_boxes.py` is re-tuned.
5. `to_openstudio.py` — builds an OpenStudio `.osm` model from `boxes.json`.
   Boxes sharing a `hall` id become one Space/ThermalZone; touching or
   overlapping boxes are left open to each other (no wall inserted between
   them) regardless of room membership.
6. `to_pcd.py` — exports a single merged `.pcd` from `boxes_merged.json`,
   applying to each session's cloud the same translation its boxes were
   dragged by in `merge_boxes.py` (which aligns the boxes but leaves the
   clouds behind them unaligned). Keeps only the points the boxes contain by
   default, `--all-points` to keep the whole aligned scan, `--voxel` to
   collapse duplicates where two sessions share a source cloud. Also works on
   a single-session `boxes.json`, where it simply crops to the boxes.

## Data (not tracked, see `.gitignore`)

- `SavedBag/` — saved `.pcd` point clouds, including `to_pcd.py`'s
  `merged_cloud.pcd`.
- `SavedBoxes/` — `boxes.json` / `boxes_edited.json` / `boxes_merged.json`.
- `OpenStudioModel/` — exported `.osm` models.
- `__pycache__/` — Python bytecode cache.
