"""RANSAC-fit a closed wall/floor/ceiling box from the raw LiDAR point cloud.

Thin wrapper around the RANSAC plane-fitting logic copied into fit_planes.py
(same folder): load_merged_cloud, segment_planes, dedupe_planes,
close_geometry, tilt_from_structure_deg are reused directly, not rewritten.

This is the geometry step behind the leveling idea in De Pazzi, Chiodini,
Pertile (Sensors 2022), "3D Radiometric Mapping by Means of LiDAR SLAM and
Thermal Camera Data Fusion", generalized from their single ground-plane
gravity-leveling (Sec. 4.2, Eq. 14) to a full closed set of walls/floor/
ceiling -- needed because this corridor sits at a real yaw in the SLAM frame,
not just a tilt, so one plane isn't enough to define an orientation.

Unlike fit_planes.py's own CLI, there is intentionally no --snap-axis here:
snapping every normal to the nearest *world* axis would erase the corridor's
real yaw, which is exactly what aligned_octree.py (script 2) needs from the
wall planes to build its rotation. --close-geometry's own per-axis grouping
(largest plane per side per axis of the *original*, unsnapped normals) still
works to pair opposing walls as long as the yaw is well under 45 deg, which
holds for this corridor.

close_geometry always runs with cap_open_faces=True here: the whole point of
this script is a single watertight box (floor + ceiling + all 4 walls) for
script 2 to level against, not a partial plane set.

Usage:
    python fit_closed_planes.py [--bag <rosbag2_folder>] [--topic /cloud_registered]
        [--distance-threshold 0.02] [--ransac-n 3] [--num-iterations 1000]
        [--min-inliers 500] [--max-planes 20] [--max-attempts 0]
        [--max-tilt-deg 15] [--rect-cluster-gap 0.3]
        [--no-dedupe] [--dedupe-normal-deg 10] [--dedupe-offset-m 0.15]
        [--close-retry-margin 0.5] [--close-retry-min-inliers 150]
        [--close-retry-min-density-ratio 0.4]
        [--out planes.json]

Venv: C:\\venvs\\planefit (Python 3.12 -- open3d has no cp313 wheel yet, see
requirements.txt), same as fit_planes.py.
"""
import argparse
import json
from pathlib import Path

from fit_planes import close_geometry, dedupe_planes, load_merged_cloud, segment_planes

DEFAULT_BAG = Path(
    r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20"
    r"\rosbag2_2026_07_30-18_12_20_filtered"
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=DEFAULT_BAG, help="rosbag2 folder")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--distance-threshold", type=float, default=0.02,
                    help="RANSAC inlier distance (m)")
    ap.add_argument("--ransac-n", type=int, default=3)
    ap.add_argument("--num-iterations", type=int, default=1000)
    ap.add_argument("--min-inliers", type=int, default=500,
                    help="stop once the next best plane has fewer inliers than this")
    ap.add_argument("--max-planes", type=int, default=20, help="safety cap")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="total RANSAC tries (accepted + rejected); 0 = --max-planes * 5")
    ap.add_argument("--max-tilt-deg", type=float, default=15.0,
                    help="reject a candidate plane if its normal is more than this many "
                         "degrees off vertical (floor/ceiling) or horizontal (wall) -- "
                         "discarded as noise, not returned as a plane (0 = accept any orientation)")
    ap.add_argument("--rect-cluster-gap", type=float, default=0.3,
                    help="per-plane rectangle trim: max gap (m) for a plane's inlier "
                         "points to count as the same 2D patch (0 = off)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep every RANSAC fragment as its own plane instead of merging "
                         "near-duplicate fragments of the same physical surface")
    ap.add_argument("--dedupe-normal-deg", type=float, default=10.0,
                    help="max angle (deg) between normals to count as the same surface")
    ap.add_argument("--dedupe-offset-m", type=float, default=0.15,
                    help="max offset difference (m) to count as the same surface")
    ap.add_argument("--close-retry-margin", type=float, default=0.5,
                    help="thickness (m) of the slab searched at an open boundary "
                         "when retrying a fit to close it")
    ap.add_argument("--close-retry-min-inliers", type=int, default=150,
                    help="relaxed --min-inliers used only for the retry fit")
    ap.add_argument("--close-retry-min-density-ratio", type=float, default=0.4,
                    help="reject a retry fit whose point density (inliers/area) is below "
                         "this fraction of the confirmed real walls' density")
    ap.add_argument("--out", type=Path, default=Path("planes.json"))
    args = ap.parse_args()

    xyz = load_merged_cloud(args.bag, args.topic, args.store)

    print(f"RANSAC input: {len(xyz)} points")
    planes = segment_planes(
        xyz, args.distance_threshold, args.ransac_n, args.num_iterations,
        args.min_inliers, args.max_planes, rect_cluster_gap=args.rect_cluster_gap,
        max_tilt_deg=args.max_tilt_deg, max_attempts=args.max_attempts,
        align_to_structure=True, snap_axis=False)

    if not args.no_dedupe:
        planes = dedupe_planes(planes, normal_deg=args.dedupe_normal_deg,
                                offset_m=args.dedupe_offset_m)

    planes = close_geometry(
        planes, xyz=xyz, distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n, num_iterations=args.num_iterations,
        retry_min_inliers=args.close_retry_min_inliers,
        retry_margin=args.close_retry_margin,
        retry_min_density_ratio=args.close_retry_min_density_ratio,
        rect_cluster_gap=args.rect_cluster_gap, align_to_structure=True,
        snap_axis=False, roi=None, cap_open_faces=True)

    args.out.write_text(json.dumps(
        {"bag": str(args.bag), "topic": args.topic, "store": args.store, "planes": planes},
        indent=2))
    print(f"wrote {len(planes)} plane(s) to {args.out}")


if __name__ == "__main__":
    main()
