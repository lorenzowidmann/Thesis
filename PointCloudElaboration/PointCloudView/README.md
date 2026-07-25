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
`.db3`/`.mcap` files.

### Examples

```bash
# First frame only, no filtering
py view_pointcloud.py C:\Users\loren\Desktop\SLAM\rosbag2_2026_07_23-16_49_16

# Full map: merge every frame
py view_pointcloud.py <bag> --all

# Full map + remove flying points
py view_pointcloud.py <bag> --all --sor

# Full map, downsample to 5 cm, then SOR (recommended for dense clouds)
py view_pointcloud.py <bag> --all --voxel 0.05 --sor

# More aggressive outlier removal
py view_pointcloud.py <bag> --all --sor --sor-std 1.5
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `bag` (positional) | — | rosbag2 folder (`metadata.yaml` + `.db3`/`.mcap`). |
| `--topic` | `/cloud_registered` | PointCloud2 topic to read. Wrong topic → the script prints the available ones. |
| `--all` | off (first frame) | Merge **all** frames into a single cloud. Without it you see only the first message. |
| `--store` | `ROS2_HUMBLE` | ROS typestore, used for bags with no embedded type definitions. Set to your recording's distro (e.g. `ROS2_FOXY`). |
| `--sor` | off | Statistical Outlier Removal: drops isolated / flying points. |
| `--sor-k` | `16` | Number of nearest neighbours examined per point. Higher = smoother statistics, slower. |
| `--sor-std` | `2.0` | Std-dev multiplier. **Lower = more aggressive** (1.0–1.5 strong, 3.0 gentle). |
| `--voxel` | `0.0` (off) | Voxel size in metres; keeps one point per voxel to reduce density. e.g. `0.05` = 5 cm. |

## Processing pipeline

```
read messages
   → merge frames        (--all)
   → voxel downsample     (--voxel)
   → statistical outlier removal (--sor, --sor-k, --sor-std)
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

## Tuning tips

- **Still too dense** → add/raise `--voxel` (e.g. `0.05`, `0.10`).
- **Not enough outliers removed** → lower `--sor-std` (`1.5`, `1.0`).
- **Real structure being deleted** → raise `--sor-std` (`3.0`) or raise `--sor-k`.
- **Slow on huge clouds** → downsample first with `--voxel`, then `--sor`.

## Appearance

Points render as dark grey (`#555555`) on a white background. Edit the
`cloud.plot(...)` call in `view_pointcloud.py` to change `color`, `point_size`,
or `background`.
