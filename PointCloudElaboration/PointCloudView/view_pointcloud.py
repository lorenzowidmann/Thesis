"""Open and view a point cloud from a rosbag with PyVista.

Usage:
    python view_pointcloud.py <bag_folder> [--topic /cloud_registered]
                              [--sor] [--sor-k 16] [--sor-std 1.5] [--voxel 0.05]

    All frames on the topic are merged into a single cloud.

    <bag_folder>   path to the rosbag2 folder (containing metadata.yaml + .db3/.mcap)
    --topic        PointCloud2 topic (default: /cloud_registered)
    --store        ROS typestore for bags without embedded type defs (default: ROS2_HUMBLE)
    --sor          apply Statistical Outlier Removal (drop "flying" points)
    --sor-k        neighbours used per point for SOR (default: 16)
    --sor-std      std-dev multiplier; lower = more aggressive (default: 1.5)
    --voxel        voxel size (m) to downsample density (default: off)
    --cubes        draw occupied voxels as transparent bluish cubes (uses --voxel size, default 0.05)
    --solid        render cubes solid/opaque instead of transparent
    --declutter    remove disconnected islands, keep the main cloud
    --cluster-gap  max gap (m) for points to count as one cluster (default: 0.30)
    --min-cluster  keep every cluster with >= N points (instead of largest only)
    --cluster-dist keep clusters within this distance (m) of the main cloud
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


def voxel_centers(xyz, size):
    """Return the centre of every occupied voxel of edge `size` (metres)."""
    keys = np.floor(xyz / size).astype(np.int64)
    uniq = np.unique(keys, axis=0)
    return (uniq + 0.5) * size


def voxel_cubes(centers, size):
    """Build one cube mesh of edge `size` at each voxel centre."""
    cube = pv.Cube(x_length=size, y_length=size, z_length=size)
    return pv.PolyData(centers).glyph(geom=cube, scale=False, orient=False)


def statistical_outlier_removal(xyz, k=16, std_ratio=1.5):
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


def cluster_labels(xyz, gap):
    """Label points by connectivity: two points in voxels (edge `gap`) that touch
    (26-connectivity) belong to the same cluster. Returns a per-point label array."""
    keys = np.floor(xyz / gap).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    index = {tuple(v): i for i, v in enumerate(uniq)}
    parent = np.arange(len(uniq))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    neigh = [(dx, dy, dz)
             for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
             if (dx, dy, dz) != (0, 0, 0)]
    for i, v in enumerate(uniq):
        for dx, dy, dz in neigh:
            j = index.get((v[0] + dx, v[1] + dy, v[2] + dz))
            if j is not None and j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(len(uniq))])
    _, comp = np.unique(roots, return_inverse=True)
    return comp[inv]


def declutter(xyz, gap, min_size=0, keep_dist=0.0):
    """Remove disconnected islands far from the main body.

    - keep_dist > 0: keep the largest cluster plus any cluster whose nearest
      point is within `keep_dist` metres of it.
    - min_size > 0 (and keep_dist == 0): keep every cluster with >= min_size points.
    - otherwise: keep only the largest cluster.
    """
    labels = cluster_labels(xyz, gap)
    counts = np.bincount(labels)
    main = int(counts.argmax())
    n_clusters = len(counts)

    if keep_dist > 0 and _HAVE_SCIPY:
        tree = cKDTree(xyz[labels == main])
        keep = labels == main
        for c in range(n_clusters):
            if c == main:
                continue
            if min_size and counts[c] < min_size:
                continue
            pts = xyz[labels == c]
            if tree.query(pts, k=1)[0].min() <= keep_dist:
                keep |= labels == c
    elif min_size:
        big = np.where(counts >= min_size)[0]
        keep = np.isin(labels, big)
    else:
        keep = labels == main

    print(f"declutter (gap={gap}m): {n_clusters} clusters, "
          f"kept {int(keep.sum())} / {len(xyz)} points")
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
    ap.add_argument("--store", default="ROS2_HUMBLE",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--sor", action="store_true", help="apply Statistical Outlier Removal")
    ap.add_argument("--sor-k", type=int, default=16, help="neighbours per point for SOR")
    ap.add_argument("--sor-std", type=float, default=1.5,
                    help="SOR std multiplier (lower = more aggressive)")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="voxel size (m) to downsample density (0 = off)")
    ap.add_argument("--cubes", action="store_true",
                    help="draw occupied voxels as transparent cubes (uses --voxel size)")
    ap.add_argument("--solid", action="store_true",
                    help="render cubes solid/opaque instead of transparent")
    ap.add_argument("--declutter", action="store_true",
                    help="remove disconnected islands (keep the main cloud)")
    ap.add_argument("--cluster-gap", type=float, default=0.30,
                    help="max gap (m) for points to count as the same cluster")
    ap.add_argument("--min-cluster", type=int, default=0,
                    help="keep every cluster with at least N points (instead of largest only)")
    ap.add_argument("--cluster-dist", type=float, default=0.0,
                    help="keep clusters within this distance (m) of the main cloud")
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

    xyz = np.vstack(frames)
    print(f"{len(frames)} frame(s), {len(xyz)} points")

    # remove flying points first, then disconnected islands, then reduce density
    if args.sor:
        xyz = statistical_outlier_removal(xyz, k=args.sor_k, std_ratio=args.sor_std)
        print(f"after SOR: {len(xyz)} points")

    if args.declutter:
        xyz = declutter(xyz, gap=args.cluster_gap,
                        min_size=args.min_cluster, keep_dist=args.cluster_dist)

    if args.cubes:
        size = args.voxel if args.voxel > 0 else 0.05
        centers = voxel_centers(xyz, size)
        print(f"drawing {len(centers)} voxel cubes of {size}m")
        cubes = voxel_cubes(centers, size)
        p = pv.Plotter()
        # bluish cubes with visible edges; transparent by default, --solid for opaque
        opacity = 1.0 if args.solid else 0.25
        p.add_mesh(cubes, color="#4C7DB0", opacity=opacity,
                   show_edges=True, edge_color="#274966", line_width=1)
        # keep the raw points visible inside/over the cubes
        p.add_mesh(pv.PolyData(xyz), color="#303030", point_size=2,
                   render_points_as_spheres=False)
        p.set_background("white")
        p.show()
        return

    if args.voxel > 0:
        xyz = voxel_downsample(xyz, args.voxel)
        print(f"after voxel {args.voxel}m: {len(xyz)} points")

    cloud = pv.PolyData(xyz)
    cloud.plot(point_size=2, render_points_as_spheres=False,
               color="#555555", background="white")


if __name__ == "__main__":
    main()
