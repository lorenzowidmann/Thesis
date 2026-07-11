# OcTree — point-cloud octree voxel sampling + GUI

First draft of the point-cloud elaboration stage. It loads a semantically
annotated point cloud (TUM-FACADE benchmark), samples it into **voxels** using
an **octree** subdivision, and shows in an interactive GUI how the raw points
collapse into voxels — with a slider to change the **voxel size (0.05–1.0 m)**
live, a **minimum-points-per-voxel filter (1–10)** to hide sparse/noisy
voxels, **surface smoothing** to flatten voxels into planar OpenStudio
surfaces, and a toggle to overlay the raw points for comparison.

## How it works

**Voxel sampling.** Each point is mapped to an integer cell index
`idx = floor((xyz - origin) / voxel_size)`; points sharing a cell collapse into
one voxel. Per voxel we keep the cell center, the **majority semantic class** of
its points, and the point count. This is a vectorized numpy operation
(`octree/voxelizer.py`).

**Octree.** The root is the cubic bounding box of the cloud. Each node splits
into 8 octants; only *occupied* octants are created, recursively, down to a
`max_depth` (`octree/octree.py`). The occupied leaves of a complete octree at
depth *d* are **exactly** the voxels of edge `root_extent / 2^d` — so the GUI's
"octree depth" slider just maps depth → voxel size, and the fast voxelizer
(`voxelize_octree`) draws the identical voxel set. This equivalence is asserted
by `main.py --selftest` (leaf count == voxel count at every depth).

Coloring is by semantic class (wall, window, door, roof, …); the id→name→RGB
map is in `octree/classes.py`, taken from the dataset's class list.

**Surface smoothing → OpenStudio** (`octree/smoothing.py`). OpenStudio /
EnergyPlus need planar, well-formed surfaces, but the voxel wall is stepped.
`smooth_surface(grid, axis, ...)` flattens the voxels of one surface onto a
single plane whose normal is the chosen axis:

- **Plane offset** = the *mode* voxel layer along the axis (the most-populated
  layer) — robust to noise and to a minority of recessed/protruding voxels, and
  grid-aligned. `median` and `outer` (95th-percentile exterior face) are options.
- **Tolerance band** (±`tolerance_voxels`, default 3): voxels within the band
  **snap onto the plane**, keeping their class. A recessed window is only 1–3
  voxels deep, so it snaps flush and becomes a **co-planar sub-surface** — which
  is exactly what OpenStudio requires (fenestration must be co-planar with its
  base wall; recess depth is modeled via frame/shading objects, not geometry).
  Voxels **beyond** the band are returned as a `deviations` set (kept, not
  dropped) to treat as noise or as their own surface later.
- **Zoning** is preserved: snapped voxels project to the in-plane lattice, each
  cell takes its majority class, and same-class cells are merged into
  axis-aligned rectangles (numpy greedy cover). Each rectangle is a planar quad
  polygon; separate windows land on non-adjacent cells → separate sub-surfaces
  automatically. Output is a planar wall subdivided into homogeneous
  sub-surfaces by class, not one uniform plane.

`to_openstudio_json` writes the surfaces as JSON (planar 3-D vertex loops +
class + envelope/fenestration `role`); `octree/openstudio_adapter.py` maps that
to an `.osm` via the OpenStudio SDK (envelope → `Surface`, fenestration →
`SubSurface`). The SDK is optional — the JSON needs no extra dependency; install
`openstudio` (a cp313 wheel exists) to write `.osm` directly.

**Auto-aligned axes (`'u'` / `'v'`).** Real buildings are rarely aligned with
world x/y — this sample sits at a ~66° yaw, so flattening onto a literal world
axis cuts diagonally across the facade instead of following it (verified:
world `x`/`y` capture only 4–5% of the wall voxels within the tolerance band,
vs. ~14–19% for the auto-aligned axes below). `principal_yaw(grid, select)`
finds the building's true horizontal orientation via PCA on the voxel
footprint; axis `'u'` (the default) flattens along that dominant wall
direction, and `'v'` along the perpendicular one — the *other* main facade.
Internally this rotates the grid into the building's frame, reuses the exact
same mode/tolerance/zoning logic as the literal axes, then rotates the output
polygons back to world coordinates, so it's the same tested pipeline underneath.
A manual `rotation_deg` overrides the auto-detected yaw (e.g. for a near-square
footprint where PCA could pick either near-equal direction). Literal `'x'` /
`'y'` / `'z'` remain available (`'z'` stays literal in both cases — only yaw,
i.e. rotation about the vertical axis, is corrected, not pitch/roll).

**Limitation.** This only handles a single wall direction pair (u/v) plus
vertical (z) — a building with more than two wall directions, or a genuinely
pitched/rolled scan, needs full multi-plane segmentation (RANSAC/region-growing),
which is not built yet (see "Next steps").

## Data

Uses the TUM-FACADE benchmark (https://github.com/OloOcki/tum-facade), each
building shipped as a `.7z` containing a `.las` (with per-point labels). The
archives live **outside** this repo (e.g. `C:\Users\loren\Desktop\tum-facade`);
`extract_sample.py` unpacks a single `.las` into `data/` (git-ignored). The
default sample is `DEBY_LOD2_4959459` (~1.05M points, ~41 m cube).

## Setup

Open3D has no Python 3.13 wheels, so this module uses **laspy + PyVista**, which
do — it runs on the system Python 3.13. A dedicated venv on a short path keeps
VTK's long file paths under the Windows limit:

```powershell
py -3.13 -m venv C:\venvs\octree
C:\venvs\octree\Scripts\Activate.ps1
cd PointCloudElaboration\OcTree
pip install -r requirements.txt   # numpy, laspy[lazrs], pyvista (pulls VTK, ~100 MB)
```

Extraction uses 7-Zip at `C:\Program Files\7-Zip\7z.exe` (edit `SEVENZIP` in
`extract_sample.py` if yours is elsewhere).

## Usage

```powershell
# 1. Unpack the sample .las from its .7z
python extract_sample.py                      # or --id DEBY_LOD2_4959322 --category annotatedLocalCRS

# 2. Inspect the cloud (point count, classes, octree node counts per depth)
python main.py --info

# 3. Interactive viewer — drag the "voxel size (m)" / "min points/voxel"
#    sliders, toggle points/filter with the checkboxes
python main.py                                # --voxel-size 0.20 --max-points 2000000

# Start with the sparse-voxel filter already on
python main.py --filter --min-count 3

# Headless render to an image (no display needed)
python main.py --screenshot preview.png --voxel-size 0.20 --filter --min-count 5

# Surface smoothing: flatten to planar surfaces (GUI or screenshot).
# 'u' (default) is the PCA-auto-aligned dominant wall direction; 'v' is the
# other main facade; 'z' is the roof/floor.
python main.py --smooth                                       # interactive, axis u
python main.py --smooth --smooth-axis v                       # the other wall
python main.py --screenshot flat.png --smooth --smooth-axis u

# Manual yaw override instead of auto-detected PCA (u/v only)
python main.py --smooth --smooth-axis u --rotation-deg 66.2

# Export planar sub-surfaces as OpenStudio-friendly JSON, then exit
python main.py --export-openstudio surfaces.json --smooth-axis u --voxel-size 0.20

# Consistency checks (octree leaves == voxelizer voxels, monotonicity)
python main.py --selftest
```

**Viewer controls.** The lower slider sets the voxel edge in metres (0.05 m
finest, 1.0 m coarsest). The upper slider sets the **minimum points per
voxel** (1–10); combined with the "filter on/off" checkbox (top-left, second
row), turning the filter on hides every voxel with fewer points than the
threshold — the sparse voxels that otherwise show up as scattered,
disconnected boxes. The "points on/off" checkbox (top-left, first row)
overlays the raw points: when on, the points are drawn black and the voxels
become wireframe cages (keeping their semantic-class colors) so the points
stay visible; when off, the voxels are solid and colored by semantic class
(matching the legend). The "smooth on/off" checkbox (top-left, third row)
flattens the (filtered) voxels into planar, class-colored sub-surfaces (the
pipeline is voxelize → filter → smooth); offset method and tolerance come
from the CLI. The **axis row** below it (u / v / z checkboxes) switches which
plane is flattened onto live, without relaunching — u/v are the PCA-aligned
wall directions, z is literal vertical (a startup `--smooth-axis x`/`y` is
respected but isn't on this row, since those are rarely useful once u/v exist).

**On filtering vs. connectivity.** The minimum-points filter is a per-voxel
density threshold — it removes sparse voxels but does not check whether the
remaining voxels are spatially connected, so an isolated voxel that happens to
clear the threshold can still appear alone. A stronger follow-up (not
implemented here) would be a connected-components pass over the occupied
voxel grid, keeping only voxels that belong to a large cluster of
face/edge-adjacent neighbours.

**Non-empty check.** By construction a voxel only exists where points fall, so
every voxel holds >=1 point. After each voxel-size change the viewer re-verifies
this (`verify_nonempty`): the on-screen readout shows `(>=1 pt/voxel: OK)` and
the console logs `min pt/voxel` and that all points were binned. A voxel can
still *look* empty if points are subsampled for display, so the whole default
cloud (~1M points) is drawn without subsampling (`--max-points`, default 2,000,000);
larger clouds subsample only for display speed while the check still runs on the
full data.

Sampling granularity (default sample), from `--screenshot` at several voxel sizes:

| voxel size | voxels |
|-----------:|-------:|
| 1.00 m | 3,034 |
| 0.50 m | 11,370 |
| 0.20 m | 66,536 |
| 0.05 m | 627,551 |

## Structure

```
OcTree/
├── main.py              # CLI: --info / interactive GUI / --screenshot / --selftest
├── extract_sample.py    # unpack one .las from a TUM-FACADE .7z into data/
├── octree/
│   ├── las_loader.py         # read .las -> points + semantic labels (robust to label field)
│   ├── voxelizer.py          # voxelize(), voxelize_octree(), filter_by_count() (numpy)
│   ├── octree.py             # OctreeNode, build_octree(), level_counts(), leaf_voxels()
│   ├── smoothing.py          # smooth_surface() -> PlanarSurface, to_openstudio_json()
│   ├── openstudio_adapter.py # PlanarSurface/JSON -> .osm via the OpenStudio SDK (optional)
│   ├── classes.py            # TUM-FACADE class id -> name / RGB, colorize()
│   └── viewer.py             # PyVista GUI: sliders + points/filter/smooth toggles, legend
├── data/                     # extracted .las + preview PNGs (git-ignored)
└── requirements.txt
```

## Next steps (not in this draft)

- Multi-plane segmentation: PCA (`principal_yaw`/axes `u`/`v`) now handles a
  single dominant wall direction and its perpendicular, but a building with
  more than two wall directions still needs per-wall segmentation
  (RANSAC/region-growing) instead of one global yaw (see smoothing's
  "Limitation" above).
- Connected-components filtering: keep only voxels belonging to a large
  cluster of adjacent occupied voxels, to remove isolated survivors of the
  minimum-points filter (see "On filtering vs. connectivity" above).
- Store the octree explicitly (sparse voxel keys) for fast neighbour queries
  rather than rebuilding per depth.
- Level-of-detail: keep multiple depths and swap by camera distance.
- Feed voxel occupancy / per-voxel class into the downstream rover pipeline
  (this is the point-cloud side of the eventual LiDAR↔thermal co-registration).
