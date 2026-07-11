"""OcTree — sample a TUM-FACADE point cloud into voxels and visualize it.

Loads a .las (extract one first with extract_sample.py), then either prints
octree/voxel statistics, launches the interactive GUI, or renders a headless
screenshot.

Usage:
    py main.py                          # launch the interactive viewer (default sample)
    py main.py --info                   # print point/octree/voxel stats, no GUI
    py main.py --screenshot out.png     # headless render to an image
    py main.py --las data/DEBY_LOD2_4959459.las --depth 7
    py main.py --selftest               # octree<->voxel consistency checks
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from octree import (
    build_octree,
    class_name,
    leaf_voxels,
    level_counts,
    load_las,
    root_extent,
    voxelize_octree,
)

DEFAULT_LAS = Path(__file__).resolve().parent / "data" / "DEBY_LOD2_4959459.las"


def parse_args():
    p = argparse.ArgumentParser(description="Octree voxel sampling + viewer")
    p.add_argument("--las", default=str(DEFAULT_LAS), help="Path to a .las file")
    p.add_argument(
        "--voxel-size", type=float, default=0.20,
        help="Initial voxel edge in metres (GUI slider range 0.05-1.0)",
    )
    p.add_argument("--info", action="store_true", help="Print statistics and exit (no GUI)")
    p.add_argument("--screenshot", default=None, help="Render to this image file (headless)")
    p.add_argument("--selftest", action="store_true", help="Run octree<->voxel checks and exit")
    p.add_argument("--max-points", type=int, default=2_000_000, help="Max points drawn in the GUI")
    return p.parse_args()


def print_info(pc):
    lo, hi = pc.bounds
    extent = root_extent(pc.points)
    print(f"\nPoints: {len(pc):,}")
    print(f"Bounds min: {np.round(lo, 2)}")
    print(f"Bounds max: {np.round(hi, 2)}")
    print(f"Cubic root extent: {extent:.2f} m")

    ids, counts = np.unique(pc.labels, return_counts=True)
    print("\nSemantic classes present:")
    for cid, cnt in sorted(zip(ids.tolist(), counts.tolist()), key=lambda x: -x[1]):
        print(f"  {cid:>2} {class_name(cid):<22} {cnt:>10,}  ({cnt / len(pc):5.1%})")

    print("\nOctree occupied nodes per depth (root=depth 0):")
    root = build_octree(pc.points, max_depth=8)
    for d, n in enumerate(level_counts(root)):
        print(f"  depth {d}:  {n:>8,} nodes   (<= 8^{d} = {8**d:,})")


def selftest(pc):
    """Occupied octree leaves at depth d == voxelizer voxels at the matching size."""
    print("\nOctree <-> voxelizer consistency:")
    ok = True
    for d in range(3, 8):
        n_vox = len(voxelize_octree(pc.points, pc.labels, d))
        root = build_octree(pc.points, max_depth=d)
        centers, _ = leaf_voxels(root)
        n_leaf = len(centers)
        match = n_vox == n_leaf
        ok &= match
        print(f"  depth {d}: voxelizer={n_vox:>7,}  octree_leaves={n_leaf:>7,}  {'OK' if match else 'MISMATCH'}")

    # Monotonicity: finer voxels -> more voxels, but never more than points.
    ns = [len(voxelize_octree(pc.points, pc.labels, d)) for d in range(3, 9)]
    mono = all(a <= b for a, b in zip(ns, ns[1:])) and ns[-1] <= len(pc)
    ok &= mono
    print(f"  voxel counts by depth 3..8: {ns}  monotonic&bounded: {'OK' if mono else 'FAIL'}")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    args = parse_args()

    print(f"Loading {args.las} ...")
    pc = load_las(args.las)

    if args.info:
        print_info(pc)
        return 0

    if args.selftest:
        return 0 if selftest(pc) else 1

    if args.screenshot:
        from octree.viewer import render_screenshot

        grid = render_screenshot(
            pc.points, pc.labels, args.screenshot,
            voxel_size=args.voxel_size, max_points=args.max_points,
        )
        print(f"Wrote {args.screenshot}  (voxel {args.voxel_size:.2f} m, {len(grid)} voxels)")
        return 0

    from octree.viewer import launch

    print("Opening viewer — drag the 'voxel size (m)' slider to change voxel resolution.")
    launch(pc.points, pc.labels, voxel_size=args.voxel_size, max_points=args.max_points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
