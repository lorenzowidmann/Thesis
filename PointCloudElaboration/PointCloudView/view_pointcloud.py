"""Open and view a point cloud from a rosbag with PyVista.

Usage:
    python view_pointcloud.py <bag_folder> [--topic /cloud_registered] [--all]
                              [--sor] [--sor-k 16] [--sor-std 2.0] [--voxel 0.05]

    <bag_folder>   path to the rosbag2 folder (containing metadata.yaml + .db3/.mcap)
    --topic        PointCloud2 topic (default: /cloud_registered)
    --all          merge all frames instead of only the first
    --store        ROS typestore for bags without embedded type defs (default: ROS2_HUMBLE)
    --sor          apply Statistical Outlier Removal (drop "flying" points)
    --sor-k        neighbours used per point for SOR (default: 16)
    --sor-std      std-dev multiplier; lower = more aggressive (default: 2.0)
    --voxel        voxel size (m) to downsample density before SOR (default: off)
"""
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def voxel_downsample(xyz, size):
    """Keep one point per voxel of edge `size` (metres)."""
    if size <= 0:
        return xyz
    keys = np.floor(xyz / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx]


def statistical_outlier_removal(xyz, k=16, std_ratio=2.0):
    """SOR: drop points whose mean distance to k neighbours is an outlier.

    Uses scipy cKDTree when available; otherwise falls back to a voxel-grid
    density filter that removes isolated (flying) points.
    """
    if len(xyz) <= k:
        return xyz
    if _HAVE_SCIPY:
        tree = cKDTree(xyz)
        dist, _ = tree.query(xyz, k=k + 1, workers=-1)  # +1 = the point itself
        mean_d = dist[:, 1:].mean(axis=1)
        thresh = mean_d.mean() + std_ratio * mean_d.std()
        keep = mean_d < thresh
        print(f"SOR (scipy): removed {int((~keep).sum())} / {len(xyz)} points")
        return xyz[keep]
    # numpy-only fallback: radius/voxel density filter
    return _voxel_outlier_removal(xyz, min_neighbors=k, std_ratio=std_ratio)


def _voxel_outlier_removal(xyz, min_neighbors=16, std_ratio=2.0):
    """Approximate SOR without scipy: bin to a voxel grid, drop points in
    sparsely populated voxels (fewer than an adaptive count of co-voxel points)."""
    span = xyz.max(axis=0) - xyz.min(axis=0)
    size = float(np.median(span)) / 200.0 or 0.05  # ~200 voxels across the scene
    keys = np.floor(xyz / size).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    per_point = counts[inv]
    thresh = max(2, per_point.mean() - std_ratio * per_point.std())
    keep = per_point >= thresh
    print(f"SOR (numpy fallback, voxel={size:.3f}m): "
          f"removed {int((~keep).sum())} / {len(xyz)} points "
          f"(install scipy for true SOR)")
    return xyz[keep]


def read_pointcloud2(msg):
    # x,y,z as float32 at offsets 0,4,8; slice by point_step to skip extra fields
    step = msg.point_step
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    # drop NaN/inf points
    return xyz[np.isfinite(xyz).all(axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path, help="rosbag2 folder")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--all", action="store_true", help="merge all frames")
    ap.add_argument("--store", default="ROS2_HUMBLE",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--sor", action="store_true", help="apply Statistical Outlier Removal")
    ap.add_argument("--sor-k", type=int, default=16, help="neighbours per point for SOR")
    ap.add_argument("--sor-std", type=float, default=2.0,
                    help="SOR std multiplier (lower = more aggressive)")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="voxel size (m) to downsample density (0 = off)")
    args = ap.parse_args()

    typestore = get_typestore(Stores[args.store])

    frames = []
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Topic {args.topic!r} not found. Available: {topics}")
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            frames.append(read_pointcloud2(msg))
            if not args.all:
                break

    xyz = np.vstack(frames)
    print(f"{len(frames)} frame(s), {len(xyz)} points")

    if args.voxel > 0:
        xyz = voxel_downsample(xyz, args.voxel)
        print(f"after voxel {args.voxel}m: {len(xyz)} points")
    if args.sor:
        xyz = statistical_outlier_removal(xyz, k=args.sor_k, std_ratio=args.sor_std)
        print(f"after SOR: {len(xyz)} points")

    cloud = pv.PolyData(xyz)
    cloud.plot(point_size=2, render_points_as_spheres=False,
               color="#555555", background="white")


if __name__ == "__main__":
    main()
