# PointCloudView

View a LiDAR/SLAM point cloud from a rosbag2 with PyVista, with optional
density downsampling and Statistical Outlier Removal (SOR) to drop "flying"
points.

## Requirements

```
pip install pyvista rosbags numpy scipy
```

- `scipy` is optional but recommended — it enables true kNN-based SOR.
  Without it, `--sor` falls back to a cruder numpy voxel-density filter.

## Usage

```
py view_pointcloud.py <bag_folder> [options]
```

`<bag_folder>` is the rosbag2 directory containing `metadata.yaml` and the
`.db3`/`.mcap` files. **All frames on the topic are merged into a single cloud.**

### Examples

```bash
# Full map, no filtering
py view_pointcloud.py C:\Users\loren\Desktop\SLAM\rosbag2_2026_07_23-16_49_16

# Full map + remove flying points
py view_pointcloud.py <bag> --sor

# Full map, downsample to 5 cm, then SOR (recommended for dense clouds)
py view_pointcloud.py <bag> --voxel 0.05 --sor

# More aggressive outlier removal
py view_pointcloud.py <bag> --sor --sor-std 1.0

# Remove disconnected islands, keep only the main cloud
py view_pointcloud.py <bag> --sor --declutter

# Keep the main cloud plus anything within 1 m of it
py view_pointcloud.py <bag> --sor --declutter --cluster-dist 1.0

# Keep every cluster with at least 200 points
py view_pointcloud.py <bag> --sor --declutter --min-cluster 200

# Draw occupied space as transparent bluish voxel cubes (5 cm), after removing flyers
py view_pointcloud.py <bag> --sor --voxel 0.05 --cubes

# Same, but solid/opaque cubes
py view_pointcloud.py <bag> --voxel 0.05 --cubes --solid
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `bag` (positional) | — | rosbag2 folder (`metadata.yaml` + `.db3`/`.mcap`). All frames on the topic are merged. |
| `--topic` | `/cloud_registered` | PointCloud2 topic to read. Wrong topic → the script prints the available ones. |
| `--store` | `ROS2_HUMBLE` | ROS typestore, used for bags with no embedded type definitions. Set to your recording's distro (e.g. `ROS2_FOXY`). |
| `--sor` | off | Statistical Outlier Removal: drops isolated / flying points. |
| `--sor-k` | `16` | Number of nearest neighbours examined per point. Higher = smoother statistics, slower. |
| `--sor-std` | `1.5` | Std-dev multiplier. **Lower = more aggressive** (1.0 strong, 2.0–3.0 gentle). |
| `--declutter` | off | Remove disconnected islands (the floating blobs far from the main body). By default keeps only the **largest** cluster. |
| `--cluster-gap` | `0.30` | Max gap (m) for points to count as the same cluster. Larger = merges nearby pieces; smaller = splits more. |
| `--min-cluster` | `0` | Keep **every** cluster with ≥ N points (instead of largest only). |
| `--cluster-dist` | `0.0` | Keep the main cluster plus any cluster whose nearest point is within this distance (m) of it. |
| `--voxel` | `0.0` (off) | Voxel size in metres; keeps one point per voxel to reduce density. e.g. `0.05` = 5 cm. Also sets cube size for `--cubes`. |
| `--cubes` | off | Overlay **transparent bluish cubes** (with edges) on the points, one per occupied voxel, like the OcTree viewer. The raw points stay visible. Cube edge = `--voxel` (or 0.05 m if unset). |
| `--solid` | off | Make the cubes solid/opaque instead of the default transparent look. |

## Processing pipeline

```
read messages
   → merge all frames
   → statistical outlier removal (--sor, --sor-k, --sor-std)
   → declutter: drop disconnected islands (--declutter, --cluster-gap, ...)
   → --cubes ? draw points + voxel cubes overlay (--voxel size)
             : voxel downsample (--voxel) → draw points
   → render (PyVista)
```

## How SOR works

For each point, the mean distance to its `k` nearest neighbours is computed.
Points whose mean distance exceeds

```
global_mean + sor_std * global_std
```

are considered outliers and removed. Flying/sparse points have large neighbour
distances, so they get dropped while dense surfaces are kept.

## How declutter works

The cloud is split into clusters: two points are in the same cluster when their
voxels (edge = `--cluster-gap`) touch (26-connectivity). A blob only becomes a
**separate** cluster once the empty gap to the main cloud exceeds roughly
`--cluster-gap` (default **0.30 m**) — closer than that and it merges into the
main body and is always kept.

Whether a separate cluster survives:

| Mode | External cluster kept if… |
|------|---------------------------|
| default (`--declutter`) | never — only the largest cluster survives (distance irrelevant) |
| `--cluster-dist X` | its nearest point is ≤ X m from the main cloud |
| `--min-cluster N` | it has ≥ N points (distance irrelevant) |

There is no fixed minimum distance — you set the threshold. To keep nearby
clouds and drop far ones, use `--cluster-dist`, e.g. keep anything within 1 m:

```bash
py view_pointcloud.py <bag> --sor --declutter --cluster-dist 1.0
```

## Tuning tips

- **Still too dense** → add/raise `--voxel` (e.g. `0.05`, `0.10`).
- **Not enough outliers removed** → lower `--sor-std` (`1.5`, `1.0`).
- **Real structure being deleted** → raise `--sor-std` (`3.0`) or raise `--sor-k`.
- **Slow on huge clouds** → downsample first with `--voxel`, then `--sor`.

## Appearance

Points render as dark grey (`#555555`) on a white background. Edit the
`cloud.plot(...)` call in `view_pointcloud.py` to change `color`, `point_size`,
or `background`.
