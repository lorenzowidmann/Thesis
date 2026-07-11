# OcTree — point-cloud octree voxel sampling + GUI

First draft of the point-cloud elaboration stage. It loads a semantically
annotated point cloud (TUM-FACADE benchmark), samples it into **voxels** using
an **octree** subdivision, and shows in an interactive GUI how the raw points
collapse into voxels — with a slider to change the **voxel size (0.05–1.0 m)**
live, and a toggle to overlay the raw points for comparison.

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

# 3. Interactive viewer — drag the "voxel size (m)" slider (0.05-1.0 m),
#    toggle the points on/off
python main.py                                # --voxel-size 0.20 --max-points 400000

# Headless render to an image (no display needed)
python main.py --screenshot preview.png --voxel-size 0.20

# Consistency checks (octree leaves == voxelizer voxels, monotonicity)
python main.py --selftest
```

**Viewer controls.** The bottom slider sets the voxel edge in metres (0.05 m
finest, 1.0 m coarsest). The checkbox (top-left) overlays the raw points: when
on, the points are drawn black and the voxels become wireframe cages (keeping
their semantic-class colors) so the points stay visible; when off, the voxels
are solid and colored by semantic class (matching the legend).

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
│   ├── las_loader.py    # read .las -> points + semantic labels (robust to label field)
│   ├── voxelizer.py     # voxelize() and voxelize_octree() (numpy)
│   ├── octree.py        # OctreeNode, build_octree(), level_counts(), leaf_voxels()
│   ├── classes.py       # TUM-FACADE class id -> name / RGB, colorize()
│   └── viewer.py        # PyVista GUI: points + voxel cubes, depth slider, legend
├── data/                # extracted .las + preview PNGs (git-ignored)
└── requirements.txt
```

## Next steps (not in this draft)

- Store the octree explicitly (sparse voxel keys) for fast neighbour queries
  rather than rebuilding per depth.
- Level-of-detail: keep multiple depths and swap by camera distance.
- Feed voxel occupancy / per-voxel class into the downstream rover pipeline
  (this is the point-cloud side of the eventual LiDAR↔thermal co-registration).
```
