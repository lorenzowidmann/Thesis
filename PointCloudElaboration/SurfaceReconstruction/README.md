# SurfaceReconstruction

Two scripts, run in sequence, that take `PlaneExtraction`'s raw RANSAC output
to an actual watertight 3D model:

```
<bag>_planes.bvg  --[dedupe_planes.py]-->  <bag>_planes_dedup.bvg  --[reconstruct_mesh.py]-->  model.obj
```

> **Note:** the reference numbers below (56 -> 30 -> 424 faces) predate
> `extract_planes.py` gaining `--max-tilt-deg` (default 15), which now drops
> diagonal artifact planes at the source -- a fresh run starts from 27 kept
> planes instead of 56. The two-step dedupe-then-reconstruct flow and its
> gotchas are unchanged; just expect different counts if you re-run end to
> end. See `PlaneExtraction/README.md`'s `--max-tilt-deg` section.

## Why two steps

PlaneExtraction's RANSAC splits a single real surface into several disjoint
fragments whenever there's occlusion, clutter, or (on a long SLAM corridor)
accumulated drift bowing one end of a wall out of the other end's plane --
e.g. the reference corridor bag came out of `extract_planes.py` as 56
"planes" for a structure that really has a handful of walls plus
floor/ceiling. PolyFit's own docs say it "assumes the model is closed and
all necessary planes are provided" -- feeding it that many near-duplicate
fragments doesn't just look messy, it blows up the hypothesis+selection MIP:
reconstructing directly against the raw 56-plane bag was left running past
two minutes and 790+ CPU-seconds with no result (verified by hand, not a
guess). `dedupe_planes.py` merges same-surface fragments first so PolyFit
actually has a tractable, closed-ish set of planes to work with.

## Requirements -- read before installing

```
pip install -r requirements.txt
```

Both `easy3d` and `polyfit` have the **same PyPI name-squatting trap**:
`pip install easy3d` / `pip install polyfit` install unrelated stub packages
with nothing to do with either library. Get the real wheels from each
project's GitHub Releases page instead:

```
# Python 3.12 / Windows
https://github.com/LiangliangNan/Easy3D/releases/download/v2.6.1/easy3d-2.6.1-cp312-cp312-win_amd64.whl
https://github.com/LiangliangNan/PolyFit/releases/download/v1.6.0/polyfit-1.6.0-cp312-cp312-win_amd64.whl
pip install easy3d-2.6.1-cp312-cp312-win_amd64.whl
pip install polyfit-1.6.0-cp312-cp312-win_amd64.whl
```

`C:\venvs\surfacereconstruction` already has both installed. PolyFit ships
with the SCIP solver bundled -- no separate solver install needed (Gurobi is
optional/faster, needs a license, not wired up here).

## Step 1: dedupe_planes.py

```
python dedupe_planes.py <bag>_planes.bvg [--output-dir <dir>]
    [--dedupe-normal-deg 10] [--dedupe-offset-m 0.15]
    [--min-plane-points 0]
```

Ported from `OpenStudioModel/fit_planes.py`'s `dedupe_planes` (same
greedy largest-absorbs-smaller logic), adapted to work on an already
RANSAC-segmented cloud: PCA-refits each plane id's own (normal, offset) from
its member points, merges ids within `--dedupe-normal-deg` / `--dedupe-offset-m`
of each other, optionally drops merged groups under `--min-plane-points`,
and writes the relabeled `v:primitive_index` / `v:primitive_type` back onto
the cloud. Outputs `<stem>_dedup.ply` + `<stem>_dedup.bvg` next to the input
(or under `--output-dir`).

### Reference run

```
python dedupe_planes.py rosbag2_2026_07_30-18_12_20_filtered_planes.bvg --min-plane-points 5000
```
56 planes -> merged 4 duplicate pairs -> dropped 22 fragments under 5000
points -> **30 planes** kept (biggest: 187k, 95k, 85k, 83k, 75k points --
floor/ceiling/main walls; rest are the corridor's individual pier faces,
correctly kept distinct since they're genuinely different planes, not
duplicates).

## Step 2: reconstruct_mesh.py

```
python reconstruct_mesh.py <bag>_planes_dedup.bvg [--output model.obj]
    [--weight-fitting 0.43] [--weight-coverage 0.27] [--weight-complexity 0.3]
```

Runs PolyFit's hypothesis-and-selection reconstruction (Nan & Wonka, ICCV
2017) with the SCIP solver and PolyFit's own example weight defaults. Prints
face count and elapsed time; saves an `.obj`.

**Budget minutes, not seconds** -- even on the deduped 30-plane reference
output this took **437.5s (~7.3 min)** and produced **424 faces**, saved
successfully. If it's still too slow, dedupe more aggressively first (looser
`--dedupe-normal-deg`/`--dedupe-offset-m`, or a `--min-plane-points` floor)
rather than waiting it out.

### Reference run result -- read before trusting the output blindly

The reconstructed mesh from the 30-plane reference input is a clean-looking
closed box **but it only covers roughly a 10m slice (X: 24.2-34.5) of the
corridor's ~35m length**, and it does not carve out the door/window openings
visible in the point cloud -- it reconstructed a simple solid box over the
region PolyFit found enough closing planes to work with, not the whole
scanned structure. This is not a crash or a script bug -- it's PolyFit's
weight-balanced hypothesis selection choosing a smaller, fully-explainable
closed volume over a larger, harder-to-close one (this method targets
piecewise-planar objects/building massing more than long corridors riddled
with doorways). If you need the whole corridor's extent modeled, expect to
either retune `--weight-fitting`/`--weight-coverage`/`--weight-complexity`,
or accept a partial reconstruction per corridor segment.

## Gotchas

- Feed `reconstruct_mesh.py` the **deduped** `.bvg`, not `extract_planes.py`'s
  raw output directly -- see "Why two steps" above.
- `dedupe_planes.py` writes properties back per-point via Easy3D's
  `VertexProperty.__setitem__(Vertex, value)` -- the only way that persists;
  `.vector()` returns a **copy**, mutating it silently does nothing (learned
  the hard way, see the script's docstring).
- Distinct real planes at different depths (e.g. this corridor's separate
  pier faces) are correctly kept apart by dedupe -- only near-parallel,
  near-coplanar fragments merge. If two planes you know are the same surface
  aren't merging, loosen `--dedupe-normal-deg`/`--dedupe-offset-m`; if
  genuinely distinct planes are wrongly merging, tighten them.
