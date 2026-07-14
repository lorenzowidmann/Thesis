"""OcTree — sample a TUM-FACADE point cloud into voxels and visualize it.

Loads a .las (extract one first with extract_sample.py), then either prints
octree/voxel statistics, launches the interactive GUI, or renders a headless
screenshot.

Usage:
    py main.py                          # launch the interactive viewer (default sample)
    py main.py --info                   # print point/octree/voxel stats, no GUI
    py main.py --screenshot out.png     # headless render to an image
    py main.py --smooth                 # flatten to a RANSAC-fitted plane (dominant facade 'u')
    py main.py --smooth --smooth-axis v # the perpendicular facade
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
        help="Which detected plane to flatten onto: 'u' dominant facade (default), "
        "'v' perpendicular facade, 'z' roof/floor, 'x'/'y' nearest that world axis",
    )
    p.add_argument(
        "--offset-method", choices=("ransac", "mode", "median", "outer"), default="ransac",
        help="How the plane is found: 'ransac' (default) fits the dominant plane "
        "at any orientation; 'mode'/'median'/'outer' are the legacy voxel-layer "
        "picks along a fixed axis",
    )
    p.add_argument(
        "--ransac-threshold", type=float, default=None, metavar="M",
        help="RANSAC plane-fit inlier distance in metres (default ~0.5*voxel-size)",
    )
    p.add_argument(
        "--ransac-iters", type=int, default=500,
        help="RANSAC hypothesis iterations per plane (default 500)",
    )
    p.add_argument("--seed", type=int, default=0, help="RANSAC RNG seed (deterministic)")
    p.add_argument(
        "--rotation-deg", type=float, default=None,
        help="Manual yaw override in degrees, legacy --offset-method + --smooth-axis "
        "u/v only (default: auto-detected via PCA on the horizontal footprint)",
    )
    p.add_argument(
        "--tolerance", type=int, default=3,
        help="Half-width in voxels of the band snapped onto the plane (default 3)",
    )
    p.add_argument(
        "--project-to-axis-aligned", action="store_true",
        help="After the RANSAC smooth, derive a second surface on a world-axis-"
        "aligned grid (vertical columns on a facade, X/Y on a roof) instead of "
        "the diagonal PCA basis; sparse colour blobs are dropped (see --min-side). "
        "RANSAC offset-method only",
    )
    p.add_argument(
        "--min-side", type=float, default=1.0, metavar="M",
        help="Axis-aligned projection: keep a colour blob only if its bounding "
        "box reaches this many metres on at least one side (default 1.0)",
    )
    p.add_argument(
        "--merge-adjacent", action="store_true",
        help="Merge touching same-class rectangles into single rectangles after "
        "smoothing (base merging: exact zero-gap edge contact; corner-only "
        "contact never counts). Runs before OpenStudio export/render",
    )
    p.add_argument(
        "--merge-gap-tolerance", type=float, default=0.0, metavar="M",
        help="Extended merging (needs --merge-adjacent): also bridge rectangles "
        "separated by a gap of at most this many metres along their shared edge "
        "(rounded up to whole raster cells -- any value > 0 bridges >= 1 cell). "
        "Ignored with a warning if --merge-adjacent is not set. Try ~0.5x "
        "voxel-size as a starting point (default 0.0: off)",
    )
    p.add_argument(
        "--merge-fit-strategy", choices=("max_inscribed", "bounding_box"), default="max_inscribed",
        help="How to reshape a non-rectangular merged group: 'max_inscribed' "
        "(default) shrinks to the largest rectangle within the true detected "
        "footprint (never overclaims area, can drop real cells); 'bounding_box' "
        "grows to contain the whole group (never loses detected area, can "
        "absorb gap cells -- guarded against encroaching on another class)",
    )
    p.add_argument(
        "--merge-min-coverage", type=float, default=0.8, metavar="FRACTION",
        help="Reject a reshaped merge (leaving it as its original, unmerged "
        "rectangles) unless it retains at least this fraction of the group's "
        "true area (0,1]. On real data a dominant class like wall usually "
        "forms one sprawling touching component spanning the whole surface; "
        "without this gate reshaping it to one rectangle would discard or "
        "overclaim most of it (default 0.8)",
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

    if args.merge_gap_tolerance and not args.merge_adjacent:
        print("--merge-gap-tolerance has no effect without --merge-adjacent; ignoring.", file=sys.stderr)

    if args.export_openstudio:
        from octree import smooth_surface, to_openstudio_json, voxelize
        from octree.voxelizer import filter_by_count

        grid = voxelize(pc.points, pc.labels, args.voxel_size)
        grid = filter_by_count(grid, effective_min)
        surface = smooth_surface(
            grid, args.smooth_axis, args.offset_method, args.tolerance,
            rotation_deg=args.rotation_deg, ransac_threshold=args.ransac_threshold,
            ransac_iters=args.ransac_iters, seed=args.seed,
        )
        if args.project_to_axis_aligned:
            from octree import project_axis_aligned

            if surface.normal is None:
                print("--project-to-axis-aligned needs --offset-method ransac", file=sys.stderr)
                return 2
            surface = project_axis_aligned(
                grid, surface, min_side_m=args.min_side, tolerance_voxels=args.tolerance,
            )
            print(f"Re-projected onto world-axis-aligned grid (min-side {args.min_side:.2f} m).")
        if args.merge_adjacent:
            from octree import merge_planar_surface

            if surface.normal is None and args.smooth_axis in ("u", "v"):
                print(
                    "--merge-adjacent needs --offset-method ransac when --smooth-axis "
                    "is 'u'/'v' (legacy PCA-yaw u/v can't be reversed to its raster grid)",
                    file=sys.stderr,
                )
                return 2
            surface, merge_summary = merge_planar_surface(
                surface, gap_tolerance=args.merge_gap_tolerance, fit_strategy=args.merge_fit_strategy,
                min_coverage=args.merge_min_coverage,
            )
            merge_summary.print_report()
        path = to_openstudio_json(surface, args.export_openstudio)
        if surface.normal is not None:
            n = surface.normal
            plane_note = f" (normal {n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})"
        else:
            plane_note = f" (yaw {surface.rotation_deg:.1f} deg)" if args.smooth_axis in ("u", "v") else ""
        print(
            f"Fitted plane '{args.smooth_axis}'{plane_note} (offset {surface.plane_coord:+.2f} m): "
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
            rotation_deg=args.rotation_deg, ransac_threshold=args.ransac_threshold,
            ransac_iters=args.ransac_iters, seed=args.seed,
            axis_aligned=args.project_to_axis_aligned, min_side=args.min_side,
            merge_adjacent=args.merge_adjacent, merge_gap_tolerance=args.merge_gap_tolerance,
            merge_fit_strategy=args.merge_fit_strategy, merge_min_coverage=args.merge_min_coverage,
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
        rotation_deg=args.rotation_deg, ransac_threshold=args.ransac_threshold,
        ransac_iters=args.ransac_iters, seed=args.seed,
        axis_aligned_on=args.project_to_axis_aligned, min_side=args.min_side,
        merge_adjacent_on=args.merge_adjacent, merge_gap_tolerance=args.merge_gap_tolerance,
        merge_fit_strategy=args.merge_fit_strategy, merge_min_coverage=args.merge_min_coverage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
