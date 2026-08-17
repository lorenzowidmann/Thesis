# PlaneExtraction

RANSAC plane extraction on a rosbag2 point cloud, via [Easy3D](https://github.com/LiangliangNan/Easy3D)
(`easy3d.PrimitivesRansac`, PLANE only). Pass the bag folder -- everything
else (the `.ply` round-trip Easy3D needs, normal estimation, saving) happens
automatically.

```
read bag -> merge frames -> write intermediate .ply (Open3D)
  -> load into Easy3D -> estimate normals -> RANSAC plane detection
  -> save .bvg (PolyFit-ready) + .ply (v:primitive_type / v:primitive_index)
```

No polygon fitting here -- this stops at plane segmentation. PolyFit
(`.bvg` -> watertight polygonal mesh) is a separate, not-yet-wired-up step.

## Requirements -- read before installing

```
pip install -r requirements.txt
```

`rosbags` and `open3d` are normal PyPI packages. **`easy3d` is not** --
`pip install easy3d` installs an unrelated 3.6 KB stub (PyPI name-squatted,
"A simple 3D utility package" by a different author, camera-pose viewer
only -- no `PointCloudIO`, no `PrimitivesRansac`). The real library is
[github.com/LiangliangNan/Easy3D](https://github.com/LiangliangNan/Easy3D)
(cloned into `ClaudeCode/Easy3D`), distributed as wheels on its
[GitHub Releases](https://github.com/LiangliangNan/Easy3D/releases) page,
not PyPI:

```
# pick the wheel matching your Python + OS, e.g. Python 3.12 / Windows:
https://github.com/LiangliangNan/Easy3D/releases/download/v2.6.1/easy3d-2.6.1-cp312-cp312-win_amd64.whl
pip install easy3d-2.6.1-cp312-cp312-win_amd64.whl
```

`C:\venvs\planeextraction` already has all three installed (Python 3.12) --
tested end-to-end against the reference bag below.

## Usage

```
python extract_planes.py <bag_folder> [--output-dir <dir>]
    [--topic /cloud_registered] [--store ROS2_HUMBLE]
    [--min-support 1000] [--dist-threshold 0.005]
    [--bitmap-resolution 0.02] [--normal-threshold 0.8]
    [--overlook-probability 0.001] [--normal-k 16]
```

`<bag_folder>` is a rosbag2 folder (`metadata.yaml` + `.db3`) with a
`PointCloud2` topic -- e.g. the output of `PointCloudFilterGUI`'s
"Save filtered bag...". `--output-dir` defaults to `<bag_folder>_planes`
next to the bag.

### Example (reference bag)

```
python extract_planes.py "C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\rosbag2_2026_07_30-18_12_20_filtered"
```

~1.07M points, ran in ~13s end-to-end (5.6s normal estimation + 1.5s RANSAC
detection on this machine), found 56 planes (dominant floor/ceiling/wall
faces at hundreds of thousands of points each, down to small fragments --
raw RANSAC output, no dedupe/merge step like `OpenStudioModel/fit_planes.py`
does).

## Outputs

```
<output_dir>/
  intermediate/<bag_name>.ply     -- raw xyz, Open3D-written, Easy3D-loaded from here
  <bag_name>_planes.ply           -- segmented cloud, v:primitive_type / v:primitive_index
                                      properties for CloudCompare/Open3D inspection
  <bag_name>_planes.bvg           -- PolyFit-ready vertex-group format
```

The intermediate `.ply` is kept on disk (debugging/reuse), but it's plumbing
-- you only ever pass the bag folder in.

## The normals gotcha (read this before changing defaults)

Easy3D's RANSAC detector silently does nothing without per-point normals: it
logs `RANSAC Detector requires point cloud normals` and returns 0 -- no
exception. Worse, **saving `.bvg` on a cloud that skipped normal estimation
segfaults the whole process** (verified by hand: `PointCloudIO_vg::save_bvg`
dereferences the primitive-type property unconditionally, and `detect()`
never created it because it bailed out first). This script always calls
`easy3d.PointCloudNormals.estimate(cloud, k)` right after loading, before
RANSAC -- not optional, not skippable via CLI, only its `k` (`--normal-k`,
default 16, matching Easy3D's own C++ default) is tunable. Once normals are
estimated, saving `.bvg`/`.ply` is safe even with 0 planes found.

## RANSAC parameters

`--min-support`, `--dist-threshold`, `--bitmap-resolution`,
`--normal-threshold`, `--overlook-probability` map directly to
`easy3d::PrimitivesRansac::detect()`'s five parameters; defaults are Easy3D's
own tutorial defaults (`Tutorial_703_Cloud_PlaneExtraction`).

**`--dist-threshold` and `--bitmap-resolution` are fractions, not metres** --
per the C++ doc comments, `dist_threshold` is relative to the bounding box's
*max dimension*, `bitmap_resolution` relative to its *width* (both
effectively `GenericBox::max_range()`, the single largest axis extent -- not
the diagonal, despite how it reads at a glance). The script always prints,
before running RANSAC:

```
Bounding box: X=10.568m  Y=2.300m  Z=2.590m
  max dimension : 10.568 m  (what --dist-threshold / --bitmap-resolution scale against)
  diagonal      : 11.121 m
  --dist-threshold 0.005 -> 0.0528 m
  --bitmap-resolution 0.02 -> 0.2114 m
```

so you can sanity-check what your fractions mean in real units before waiting
on a run, and re-run with different values if e.g. 5cm is too coarse/fine for
your scene.

## Gotchas

- `<bag_folder>` must contain `metadata.yaml` directly (pass the rosbag2
  folder, not a file inside it) -- same convention as
  `PointCloudFilterGUI`/`PointCloudView`.
- Multiple messages on `--topic` are merged (with a printed note) rather than
  erroring, but this script is built around the single-merged-message case
  (e.g. `PointCloudFilterGUI` output).
- Only `x`, `y`, `z` are read from the source `PointCloud2` (first 12 bytes
  per point) -- any intensity/normal/curvature fields on the source topic are
  ignored, same as the other PointCloudElaboration tools.
