"""OcTree — sample a TUM-FACADE point cloud into voxels and visualize it.

Loads a .las (extract one first with extract_sample.py), then either prints
octree/voxel statistics, launches the interactive GUI, or renders a headless
screenshot.

Usage:
    py main.py                          # launch the interactive viewer (default sample)
    py main.py --info                   # print point/octree/voxel stats, no GUI
    py main.py --screenshot out.png     # headless render to an image
    py main.py --smooth                 # flatten to planar surfaces (auto-aligned axis 'u')
    py main.py --smooth --smooth-axis v # the other main wall direction
    py main.py --export-openstudio surfaces.json --smooth-axis u
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
    p.add_argument(
        "--min-count", type=int, default=1,
        help="Initial minimum points/voxel filter threshold (1-10)",
    )
    p.add_argument(
        "--filter", action="store_true",
        help="Start with the minimum-points filter enabled",
    )
    p.add_argument(
        "--smooth", action="store_true",
        help="Flatten voxels to a plane (OpenStudio surfaces) instead of showing cubes",
    )
    p.add_argument(
        "--smooth-axis", choices=("x", "y", "z", "u", "v"), default="u",
        help="Surface-normal axis to flatten along: literal x/y/z, or the "
        "PCA-auto-aligned dominant wall direction 'u' (default) / "
        "perpendicular direction 'v'",
    )
    p.add_argument(
        "--rotation-deg", type=float, default=None,
        help="Manual yaw override in degrees, only valid with --smooth-axis u/v "
        "(default: auto-detected via PCA on the horizontal footprint)",
    )
    p.add_argument(
        "--offset-method", choices=("mode", "median", "outer"), default="mode",
        help="How the plane offset is chosen along the axis (default mode)",
    )
    p.add_argument(
        "--tolerance", type=int, default=3,
        help="Half-width in voxels of the band snapped onto the plane (default 3)",
    )
    p.add_argument(
        "--export-openstudio", default=None, metavar="PATH.json",
        help="Smooth and write OpenStudio-friendly polygon JSON, then exit",
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

    effective_min = args.min_count if args.filter else 1

    if args.export_openstudio:
        from octree import smooth_surface, to_openstudio_json, voxelize
        from octree.voxelizer import filter_by_count

        grid = voxelize(pc.points, pc.labels, args.voxel_size)
        grid = filter_by_count(grid, effective_min)
        surface = smooth_surface(
            grid, args.smooth_axis, args.offset_method, args.tolerance,
            rotation_deg=args.rotation_deg,
        )
        path = to_openstudio_json(surface, args.export_openstudio)
        yaw_note = f" (yaw {surface.rotation_deg:.1f} deg)" if args.smooth_axis in ("u", "v") else ""
        print(
            f"Smoothed along {args.smooth_axis}{yaw_note} (plane {surface.plane_coord:+.2f} m): "
            f"{surface.n_inliers:,} inliers, {surface.n_deviations:,} deviations, "
            f"{len(surface.subsurfaces)} sub-surfaces, {surface.n_polygons} polygons.\n"
            f"Wrote {path}"
        )
        return 0

    if args.screenshot:
        from octree.viewer import render_screenshot

        result = render_screenshot(
            pc.points, pc.labels, args.screenshot,
            voxel_size=args.voxel_size, max_points=args.max_points,
            min_count=effective_min, smooth=args.smooth, smooth_axis=args.smooth_axis,
            offset_method=args.offset_method, tolerance=args.tolerance,
            rotation_deg=args.rotation_deg,
        )
        print(f"Wrote {args.screenshot}  (voxel {args.voxel_size:.2f} m)")
        return 0

    from octree.viewer import launch

    print("Opening viewer — sliders: 'voxel size (m)', 'min points/voxel'; "
          "checkboxes: points / filter / smooth; axis: u / v / z.")
    launch(
        pc.points, pc.labels, voxel_size=args.voxel_size, max_points=args.max_points,
        min_count=args.min_count, filter_on=args.filter,
        smooth_on=args.smooth, smooth_axis=args.smooth_axis,
        offset_method=args.offset_method, tolerance=args.tolerance,
        rotation_deg=args.rotation_deg,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
