"""Level the raw point cloud against the closed wall/floor/ceiling box from
fit_closed_planes.py, then octree-voxelize the now axis-aligned cloud.

Generalizes the gravity-leveling rotation in De Pazzi, Chiodini, Pertile
(Sensors 2022), "3D Radiometric Mapping by Means of LiDAR SLAM and Thermal
Camera Data Fusion" (Sec. 4.2, Eq. 14) from one ground-plane tilt correction
to a full 3-axis "building frame": the floor plane's normal gives the up
axis, the dominant wall pair's normal gives a second axis, and their cross
product gives the third. Applying that rigid transform to the whole raw
cloud turns the corridor's real SLAM-frame yaw into a Manhattan-aligned box,
so the octree's voxel grid comes out straight -- purely because the input
points were pre-aligned here, not because octree.py/voxelizer.py know
anything about planes.

The planes are used ONLY to compute the alignment transform. Every point in
the bag is loaded and transformed -- nothing is clipped or dropped based on
plane membership. The octree step itself follows the paper's actual method:
one root bin encompassing all points, recursively subdivided into 8 occupied
children per level (build_octree, octree/octree.py).

No temperature/thermal averaging per voxel here -- geometry/alignment only.

Usage:
    python aligned_octree.py [--planes planes.json] [--tilt-only]
        [--bag <rosbag2_folder>] [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--depth 8] | [--voxel-size 0.15]
        [--voxels-out voxels.npz] [--transform-out transform.json]
        [--planes-aligned-out planes_aligned.json]
        [--refit-box-from-voxels [--refit-distance-threshold M] [--refit-min-inliers N]
         [--refit-max-tilt-deg 15] [--planes-from-voxels-out planes_from_voxels.json]]

Also writes planes_aligned.json: the closed box re-derived *in* the aligned
frame (align_and_reclose_planes), not just planes.json's corners rotated by
the transform -- see that function's docstring for why the naive rotation
leaves a small residual tilt. view_voxels.py overlays this file.

--refit-box-from-voxels (off by default) additionally RANSAC-fits a second,
independent closed box directly to the occupied voxel centers -- not the
raw points -- at the very end of the pipeline, after voxels.npz is written.
See refit_box_from_voxels' docstring for why this can trace the voxelized
data's actual footprint more closely than planes_aligned.json.

Venv: C:\\venvs\\planefit (same as fit_closed_planes.py -- rosbags is enough
for this script, but it shares the venv for convenience).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_planes import canonical_normal_offset, close_geometry, dedupe_planes, load_merged_cloud, segment_planes
from octree import build_octree, leaf_voxels, level_counts, verify_nonempty, voxelize, voxelize_octree


def _axis_of(plane):
    """Index (0/1/2 = X/Y/Z) of the world axis a plane's normal is nearest
    to -- same convention as fit_planes.close_geometry's axis grouping."""
    return int(np.argmax(np.abs(np.asarray(plane["normal"]))))


def compute_building_frame(planes):
    """Derive an orthonormal building-frame rotation + translation from a
    closed set of wall/floor/ceiling planes.

    up       : floor plane's normal, sign-corrected to point from floor
               towards the ceiling (or towards +Z of the original frame if
               there's no ceiling plane -- LiDAR SLAM map frames are usually
               already roughly Z-up, so this only fixes the sign, same
               assumption the paper's single-plane tilt correction relies on).
    forward  : the dominant wall pair's normal (the wall axis-group -- X-normal
               or Y-normal walls -- with the larger total inlier_count, i.e.
               the more densely/reliably fit pair; not necessarily the
               corridor's longer walls by area, e.g. a densely-scanned end
               wall can outweigh a sparser long side wall), Gram-Schmidt-
               orthogonalized against up.
    right    : cross(up, forward), so (forward, right, up) is right-handed
               (forward x right == up).

    Returns (R, t, info): R is the (3,3) rotation with rows [forward, right,
    up] (i.e. aligned_col = R @ world_col + t), t is the (3,) translation
    that puts the floor plane at z=0. `info` records which planes/axis were
    used, for the transform.json audit trail.
    """
    floor_ceiling = [p for p in planes if p["orientation"] == "floor_ceiling"]
    walls = [p for p in planes if p["orientation"] == "wall"]
    if not floor_ceiling:
        raise SystemExit("no floor/ceiling plane in planes.json -- cannot build a frame")
    if not walls:
        raise SystemExit("no wall plane in planes.json -- cannot build a frame")

    floor_ceiling = sorted(floor_ceiling, key=lambda p: p["centroid_3d"][2])
    floor_plane = floor_ceiling[0]
    ceiling_plane = floor_ceiling[-1] if len(floor_ceiling) > 1 else None

    up = np.asarray(floor_plane["normal"], dtype=float)
    up = up / np.linalg.norm(up)
    if ceiling_plane is not None:
        direction = np.asarray(ceiling_plane["centroid_3d"]) - np.asarray(floor_plane["centroid_3d"])
        if np.dot(up, direction) < 0:
            up = -up
    elif up[2] < 0:
        up = -up

    # dominant wall pair: the wall axis-group (of the *original*, unsnapped
    # normals) with the most total inliers -- the more reliably fit pair.
    wall_axis_totals = {}
    for p in walls:
        ax = _axis_of(p)
        wall_axis_totals[ax] = wall_axis_totals.get(ax, 0) + p["inlier_count"]
    dominant_axis = max(wall_axis_totals, key=wall_axis_totals.get)
    wall_group = [p for p in walls if _axis_of(p) == dominant_axis]

    if len(wall_group) >= 2:
        canon_normals = [canonical_normal_offset(p["normal"], p["d"])[0] for p in wall_group]
        forward_raw = np.mean(canon_normals, axis=0)
    else:
        forward_raw = np.asarray(wall_group[0]["normal"], dtype=float)

    forward = forward_raw - np.dot(forward_raw, up) * up
    norm_forward = np.linalg.norm(forward)
    if norm_forward < 1e-6:
        raise SystemExit("dominant wall normal is parallel to the floor normal -- degenerate frame")
    forward /= norm_forward

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    forward = np.cross(right, up)  # re-orthogonalize against fp error
    forward /= np.linalg.norm(forward)

    R = np.stack([forward, right, up])  # rows; aligned_col = R @ world_col + t

    floor_height = float(np.dot(up, floor_plane["centroid_3d"]))
    t = np.array([0.0, 0.0, -floor_height])

    info = {
        "floor_plane_id": floor_plane["id"],
        "ceiling_plane_id": ceiling_plane["id"] if ceiling_plane is not None else None,
        "dominant_wall_axis": "XYZ"[dominant_axis],
        "dominant_wall_plane_ids": [p["id"] for p in wall_group],
        "floor_height_before_translation": floor_height,
    }
    return R, t, info


def compute_tilt_only_frame(planes):
    """Same up/translation as compute_building_frame (floor plane's normal,
    sign-corrected towards the ceiling; floor put at z=0), but NO wall-based
    yaw correction: forward/right are the minimal-rotation choice (world X,
    Gram-Schmidt'd against up; world Y if X is ~parallel to up) instead of
    the dominant wall pair's normal.

    Why: compute_building_frame snaps the whole cloud's yaw to ONE dominant
    wall pair's normal, forcing a single rectangular box axis-aligned end to
    end. A corridor that actually bends/snakes along its length (real here,
    not a fitting artifact) then has its far segments rotated away from that
    one wall's angle, landing visibly outside the (straight, axis-aligned)
    voxel/box overlay -- see the picture in the thread that prompted this:
    red box straight, point cloud curving out past both ends of it. Only
    correcting the floor/ceiling tilt (this function) leaves the cloud's
    real yaw/curvature alone, so nothing gets forced into a box shape it
    doesn't actually have.

    Still requires a floor plane (raises the same SystemExit if missing);
    does NOT require any wall plane at all (unlike compute_building_frame).
    """
    floor_ceiling = [p for p in planes if p["orientation"] == "floor_ceiling"]
    if not floor_ceiling:
        raise SystemExit("no floor/ceiling plane in planes.json -- cannot build a frame")

    floor_ceiling = sorted(floor_ceiling, key=lambda p: p["centroid_3d"][2])
    floor_plane = floor_ceiling[0]
    ceiling_plane = floor_ceiling[-1] if len(floor_ceiling) > 1 else None

    up = np.asarray(floor_plane["normal"], dtype=float)
    up = up / np.linalg.norm(up)
    if ceiling_plane is not None:
        direction = np.asarray(ceiling_plane["centroid_3d"]) - np.asarray(floor_plane["centroid_3d"])
        if np.dot(up, direction) < 0:
            up = -up
    elif up[2] < 0:
        up = -up

    # minimal-rotation forward: project world X onto the plane perpendicular
    # to up: keeps the cloud's original yaw (and its real curvature) intact,
    # only removing tilt. Falls back to world Y if up is nearly parallel to
    # world X (forward would otherwise be near-degenerate).
    world_x = np.array([1.0, 0.0, 0.0])
    forward = world_x - np.dot(world_x, up) * up
    if np.linalg.norm(forward) < 1e-3:
        world_y = np.array([0.0, 1.0, 0.0])
        forward = world_y - np.dot(world_y, up) * up
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    forward = np.cross(right, up)  # re-orthogonalize against fp error
    forward /= np.linalg.norm(forward)

    R = np.stack([forward, right, up])  # rows; aligned_col = R @ world_col + t

    floor_height = float(np.dot(up, floor_plane["centroid_3d"]))
    t = np.array([0.0, 0.0, -floor_height])

    info = {
        "mode": "tilt_only",
        "floor_plane_id": floor_plane["id"],
        "ceiling_plane_id": ceiling_plane["id"] if ceiling_plane is not None else None,
        "dominant_wall_axis": None,
        "dominant_wall_plane_ids": [],
        "floor_height_before_translation": floor_height,
    }
    return R, t, info


def apply_rigid(points, R, t):
    """aligned = points @ R.T + t (row-vector points, (N,3))."""
    return points @ R.T + t


def align_and_reclose_planes(planes, R, t):
    """Rotate every plane's geometry into the aligned frame, then re-close
    the box with fit_planes.close_geometry -- reused, not reimplemented.

    close_geometry's box in planes.json was already axis-snapped once, but
    only against the *original* (pre-alignment) frame's X/Y/Z: it groups
    planes by axis_of() = argmax(abs(normal)) and stretches each rectangle to
    a shared bounding box on that assumption. compute_building_frame's R is a
    more precise rotation (built from the true measured wall/floor normals,
    not an axis-snapped approximation), so simply rotating planes.json's
    corners by R leaves a small residual tilt baked in -- e.g. here R has a
    real ~2.6 deg yaw correction, so an edge that was "already" treated as
    X-aligned pre-rotation comes out with a small but visible Y/Z component
    after rotation. The fix is to redo the axis grouping and bounding-box
    stretch *after* rotating, when the normals are actually close to exact
    world axes (by construction of R) -- close_geometry does exactly that,
    called a second time on the rotated planes. No missing sides are
    expected (fit_closed_planes.py already closed every side with
    cap_open_faces=True), so no retry-fit xyz is needed here.
    """
    rotated = []
    for p in planes:
        normal = np.asarray(p["normal"], dtype=float) @ R.T
        centroid = np.asarray(p["centroid_3d"], dtype=float) @ R.T + t
        corners = np.asarray(p["corners_3d"], dtype=float) @ R.T + t
        rotated.append({
            **p,
            "normal": normal.tolist(),
            "d": float(-np.dot(normal, centroid)),
            "centroid_3d": centroid.tolist(),
            "corners_3d": corners.tolist(),
        })
    return close_geometry(rotated, xyz=None, cap_open_faces=True,
                           align_to_structure=True, snap_axis=False)


def refit_box_from_voxels(voxel_centers, voxel_size, distance_threshold=None,
                           min_inliers=None, max_tilt_deg=15.0):
    """Independent second box, RANSAC-fit directly to the occupied voxel
    *centers* instead of the original 1M-point raw cloud.

    Reuses fit_planes.segment_planes/dedupe_planes/close_geometry exactly as
    fit_closed_planes.py does -- they operate on any (N,3) point array, so
    voxel centers work as-is, just at a much smaller N (thousands, not
    millions) and coarser spacing (voxel_size, not raw LiDAR noise).

    Why this can differ from planes_aligned.json (the point-fit box, just
    rotated+reclosed into the aligned frame, see align_and_reclose_planes):
    that box is fit from RANSAC's dominant-plane search over the *raw*
    cloud, which only captures the largest/straightest wall fragment per
    side -- real corridor irregularity (alcoves, uneven floor/ceiling,
    weakly-scanned end caps) sits outside it despite being real occupied
    voxels (see the "why the voxel dont follow the closed planes" thread --
    ~80% of voxels fell outside that box on one real run). A box fit to the
    voxel grid *itself*, after alignment, is answering a different question:
    "what's the dominant flat surface *of the voxelized result*", which can
    trace the data's actual footprint more closely, at the cost of being
    RANSAC-noisy at voxel resolution instead of raw-point resolution.

    distance_threshold/min_inliers default to something reasonable relative
    to voxel_size/voxel count when not given explicitly (RANSAC's raw-point
    defaults, 0.02 m / 500 inliers, are much too tight for a coarse,
    small-N voxel-center cloud).
    """
    if distance_threshold is None:
        distance_threshold = max(voxel_size * 0.6, 0.02)
    if min_inliers is None:
        min_inliers = max(20, int(0.01 * len(voxel_centers)))

    print(f"refit-box-from-voxels: {len(voxel_centers)} voxel centers, "
          f"distance_threshold={distance_threshold:.4f} m, min_inliers={min_inliers}")

    planes = segment_planes(
        voxel_centers, distance_threshold, ransac_n=3, num_iterations=1000,
        min_inliers=min_inliers, max_planes=20, rect_cluster_gap=0.0,
        max_tilt_deg=max_tilt_deg, max_attempts=0,
        align_to_structure=True, snap_axis=False)
    planes = dedupe_planes(planes)
    planes = close_geometry(
        planes, xyz=voxel_centers, distance_threshold=distance_threshold,
        ransac_n=3, num_iterations=1000, retry_min_inliers=min_inliers,
        retry_margin=max(voxel_size * 3, 0.5), rect_cluster_gap=0.0,
        align_to_structure=True, snap_axis=False, roi=None, cap_open_faces=True)
    return planes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=Path, default=Path("planes.json"),
                    help="output of fit_closed_planes.py")
    ap.add_argument("--tilt-only", action="store_true",
                    help="level against floor/ceiling only (fix roll/pitch), skip the "
                         "wall-based yaw correction -- use when the corridor genuinely "
                         "bends/snakes along its length, so forcing the whole cloud's "
                         "yaw to one dominant wall pair pushes far segments outside the "
                         "axis-aligned voxel/box overlay. See compute_tilt_only_frame's "
                         "docstring. No wall plane required in --planes when this is set.")
    ap.add_argument("--bag", type=Path, default=None,
                    help="rosbag2 folder (default: the 'bag' field recorded in --planes)")
    ap.add_argument("--topic", default=None,
                    help="default: the 'topic' field recorded in --planes")
    ap.add_argument("--store", default=None,
                    help="default: the 'store' field recorded in --planes, or ROS2_HUMBLE")
    ap.add_argument("--depth", type=int, default=8,
                    help="octree max depth / voxelize_octree depth (ignored if "
                         "--voxel-size is given)")
    ap.add_argument("--voxel-size", type=float, default=None,
                    help="metric voxel edge in metres (e.g. 0.15). Uses the plain "
                         "uniform-grid voxelizer (octree.voxelize) instead of the "
                         "power-of-two octree lattice, since an arbitrary size like "
                         "0.15 generally isn't root_extent/2**depth for any integer "
                         "depth -- build_octree/voxelize_octree are skipped in this case")
    ap.add_argument("--voxels-out", type=Path, default=Path("voxels.npz"))
    ap.add_argument("--transform-out", type=Path, default=Path("transform.json"))
    ap.add_argument("--planes-aligned-out", type=Path, default=Path("planes_aligned.json"),
                    help="closed box re-derived in the aligned frame (see "
                         "align_and_reclose_planes) -- this is what view_voxels.py "
                         "overlays; it is NOT just planes.json rotated by transform.json, "
                         "see the function docstring for why that would carry a small tilt")
    ap.add_argument("--refit-box-from-voxels", action="store_true",
                    help="also RANSAC-fit a second, independent closed box directly to "
                         "the occupied voxel centers (not the raw points) -- can trace "
                         "the voxelized data's actual footprint more closely than "
                         "planes_aligned.json, which is just the raw-point-fit box "
                         "rotated+reclosed, see refit_box_from_voxels' docstring")
    ap.add_argument("--refit-distance-threshold", type=float, default=None,
                    help="RANSAC inlier distance (m) for --refit-box-from-voxels "
                         "(default: max(0.6 * voxel edge, 0.02))")
    ap.add_argument("--refit-min-inliers", type=int, default=None,
                    help="min inliers to accept a plane for --refit-box-from-voxels "
                         "(default: max(20, 1%% of occupied voxel count))")
    ap.add_argument("--refit-max-tilt-deg", type=float, default=15.0,
                    help="--max-tilt-deg equivalent for --refit-box-from-voxels")
    ap.add_argument("--planes-from-voxels-out", type=Path, default=Path("planes_from_voxels.json"),
                    help="output of --refit-box-from-voxels")
    args = ap.parse_args()

    record = json.loads(args.planes.read_text())
    planes = record["planes"]
    bag = args.bag if args.bag is not None else Path(record["bag"])
    topic = args.topic if args.topic is not None else record["topic"]
    store = args.store if args.store is not None else record.get("store", "ROS2_HUMBLE")

    if args.tilt_only:
        R, t, info = compute_tilt_only_frame(planes)
        print(f"building frame (tilt-only): floor=plane {info['floor_plane_id']}, "
              f"ceiling=plane {info['ceiling_plane_id']}, no wall-based yaw correction")
    else:
        R, t, info = compute_building_frame(planes)
        print(f"building frame: floor=plane {info['floor_plane_id']}, "
              f"ceiling=plane {info['ceiling_plane_id']}, "
              f"dominant wall axis={info['dominant_wall_axis']} "
              f"(planes {info['dominant_wall_plane_ids']})")

    xyz = load_merged_cloud(bag, topic, store)
    print(f"aligning {len(xyz)} points (whole cloud, not just plane inliers)")
    aligned = apply_rigid(xyz, R, t)

    aligned_planes = align_and_reclose_planes(planes, R, t)
    args.planes_aligned_out.write_text(json.dumps(
        {"planes_source": str(args.planes), "planes": aligned_planes}, indent=2))
    print(f"wrote {len(aligned_planes)} re-closed plane(s) in the aligned frame to "
          f"{args.planes_aligned_out}")

    labels = np.zeros(len(aligned), dtype=np.int64)  # no classification here (geometry only)
    if args.voxel_size is not None:
        grid = voxelize(aligned, labels, args.voxel_size)
        depth_out = -1  # sentinel: arbitrary metric size, not a power-of-two octree lattice
        print(f"voxelize (uniform grid) @ {grid.voxel_size:.4f} m: {len(grid)} occupied voxels")
    else:
        # paper's literal method: one root bin encompassing all points, recursively
        # subdivided into 8 occupied children per level.
        root = build_octree(aligned, max_depth=args.depth)
        level_counts_list = level_counts(root)
        print(f"octree levels (occupied node count, root first): {level_counts_list}")
        leaf_centers, leaf_size = leaf_voxels(root)

        grid = voxelize_octree(aligned, labels, args.depth)
        depth_out = args.depth
        print(f"voxelize_octree: {len(grid)} occupied voxels @ {grid.voxel_size:.4f} m "
              f"(octree leaves: {len(leaf_centers)}, size {leaf_size:.4f} m)")
        if len(leaf_centers) != len(grid):
            print("WARNING: octree leaf count and voxelize_octree count disagree "
                  "(expected to match, see octree/octree.py docstring)")

    ok, n_empty, n_binned = verify_nonempty(grid, len(aligned))
    if not ok:
        print(f"WARNING: voxel invariant broken (n_empty={n_empty}, "
              f"n_binned={n_binned}, expected {len(aligned)})")

    np.savez(
        args.voxels_out,
        centers=grid.centers,
        counts=grid.counts,
        voxel_size=grid.voxel_size,
        origin=grid.origin,
        depth=depth_out,
    )
    print(f"wrote {len(grid)} voxel(s) to {args.voxels_out}")

    R_inv = R.T
    t_inv = -(R.T @ t)
    transform = {
        "planes_file": str(args.planes),
        "bag": str(bag),
        "topic": topic,
        "store": store,
        "building_frame": info,
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "rotation_inv": R_inv.tolist(),
        "translation_inv": t_inv.tolist(),
        "convention": (
            "row-vector points (N,3): aligned = points @ rotation.T + translation; "
            "world = aligned @ rotation_inv.T + translation_inv"
        ),
    }
    args.transform_out.write_text(json.dumps(transform, indent=2))
    print(f"wrote alignment transform to {args.transform_out}")

    if args.refit_box_from_voxels:
        voxel_planes = refit_box_from_voxels(
            grid.centers, grid.voxel_size,
            distance_threshold=args.refit_distance_threshold,
            min_inliers=args.refit_min_inliers,
            max_tilt_deg=args.refit_max_tilt_deg)
        args.planes_from_voxels_out.write_text(json.dumps(
            {"voxels_source": str(args.voxels_out), "planes": voxel_planes}, indent=2))
        print(f"wrote {len(voxel_planes)} plane(s) refit from voxels to "
              f"{args.planes_from_voxels_out}")


if __name__ == "__main__":
    main()
