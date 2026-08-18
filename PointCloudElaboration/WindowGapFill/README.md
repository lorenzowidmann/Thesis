# WindowGapFill — window openings as holes in wall coverage

Self-contained (no cross-import for the detection script; the viewer
borrows one function from `MaterialToVoxel` only for its optional `--points`
overlay, see "Provenance"). Two scripts:

1. **`find_window_gaps.py`** -- finds window openings as literal HOLES in
   `AlignedOctree`'s wall voxel coverage (not by material classification --
   see "Why") and fills them with synthetic voxel centers. Writes
   `window_gaps.npz` + `windows.json`.
2. **`view_window_gaps.py`** -- pyvista viewer: the whole building
   (translucent context), the closed box overlay, and the fill voxels
   colored one distinct color per window.

## Why

Superseded a first attempt that used `MaterialToVoxel`'s `material_id ==
"glass"` (image-classification-derived) voxels to locate windows: real
window glass barely returns LiDAR at all (specular/transparent), so that
approach found only 2 small, sparse clusters (13 and 22 voxels) on a real
run -- clearly missing real windows. A window is much more reliably a
literal ABSENCE of occupied voxels on an otherwise-solid wall, so this finds
holes directly in the wall's own voxel raster instead.

On the same data, this approach found 4 windows (63-82 cells each, ~1-2 m^2)
at a **consistent height band (z 0.96-1.71 m) across two independently
processed walls** -- a physical sanity check the material-vote clusters
never had.

Structurally similar to `TemperatureToVoxel/FILLER.md`'s abandoned floor/
ceiling gap-filler (same "find holes in a voxel raster" shape), but the
physical situation differs enough that a method FILLER.md rejected works
here: FILLER.md's strict enclosed-hole test failed for floor/ceiling because
real scan coverage is genuinely ragged right up to its true physical
boundary (most "gaps" touched the raster's own edge, so hardly anything
counted as enclosed). A wall's true extent is bounded by floor/ceiling/end-
walls -- opaque, reflective, well-scanned surfaces -- so a window really is
enclosed within the wall's own footprint. The one thing that still needed
solving here (which FILLER.md never fully resolved for its own case) was
real coverage POROSITY at 15cm resolution (radiators, pillars, uneven
returns -- see `EmissivityCalculation/classify_session.py`'s own docstring
on this corridor's wall clutter): without a light morphological closing
first, that porosity let the background bleed to the raster's edge almost
everywhere and 0 holes were found at all. `--closing-iterations` (default 2,
empirically chosen -- see the module docstring) bridges that small-scale
porosity without erasing a real, metres-scale window.

## Usage

```
C:\venvs\planefit\Scripts\python.exe find_window_gaps.py
C:\venvs\planefit\Scripts\python.exe view_window_gaps.py
```

Defaults (all overridable, see `--help`):
- `--voxels` / `--planes-aligned` -- `../AlignedOctree/voxels.npz` /
  `../AlignedOctree/planes_aligned.json` (read only, never regenerated
  here). **Thresholds (`--closing-iterations`,
  `--wall-tol-m`, `--min-hole-cells`/`--max-hole-cells`) are tuned for a
  15cm voxel grid** -- retune if `voxels.npz` is regenerated at a
  materially different size (e.g. the 5cm grid used elsewhere in this
  pipeline was NOT checked here).
- North/south walls only (`find_target_walls` -- the wall pair whose
  RANSAC-fit normal is closest to world Y in the aligned frame). The
  X-normal end walls are not handled.

## Output

- `window_gaps.npz` -- `centers` (K,3), one synthetic voxel per hole cell,
  placed AT the parent wall's own plane (not at whatever raw-point depth,
  since there mostly isn't one); `window_id` (K,) -- which detected window
  each cell belongs to; `wall_id` (K,) -- parent wall's `planes_aligned.json`
  id; `voxel_size`, `origin` -- copied from `--voxels`.
- `windows.json` -- one record per window: `window_id`, `wall_id`,
  `n_cells`, `area_m2`, `x_range`, `z_range` (world/aligned frame), plus the
  params used to find it, for provenance.

Console output reports, per wall: near-wall voxel count, footprint
occupancy, and every candidate hole found (kept or excluded, with the
reason).

## Provenance (self-contained copies, adapted where noted)

- `axis_of` in `find_window_gaps.py` -- copied from
  `AlignedOctree/aligned_octree.py`'s `_axis_of` (same convention: which
  world axis a plane's normal is nearest to).
- Hole detection (`wall_occupancy_grid`, `find_wall_holes`) -- new; grew out
  of a scratchpad prototype checked against real data before being written
  here (see "Why" for the two dead ends it avoided, both informed by
  `TemperatureToVoxel/FILLER.md`'s earlier floor/ceiling attempt).
- `view_window_gaps.py`'s box overlay (`load_box_wireframe`) -- copied
  verbatim from the other viewers in this pipeline (Aligned Octree /
  TemperatureToVoxel / MaterialToVoxel).
- `view_window_gaps.py`'s categorical per-window coloring
  (`window_color_map`, `tab20`) -- same idea as
  `MaterialToVoxel/view_voxels.py`'s per-material coloring, adapted to
  index by `window_id` instead of `material_id`.
- `view_window_gaps.py`'s `--points` overlay -- imports
  `MaterialToVoxel/material_to_voxel.py`'s `load_merged_cloud` directly
  (the one deliberate cross-import in this folder, for one optional debug
  flag -- not worth a third copy of the same ~15-line function).
