"""RANSAC plane fitting on the dense accumulated LiDAR point cloud -- walls
and roof only.

Standalone geometry step: loads /cloud_registered from a rosbag2, merges all
frames (points are already in world/map frame -- no transform needed), applies
an optional ROI crop + Statistical Outlier Removal + declutter (same
options/defaults as PointCloudElaboration/PointCloudView/view_pointcloud.py),
then iteratively RANSAC-fits planes with open3d (pcd.segment_plane). For each
accepted plane, its inlier points are trimmed to their largest connected 2D
patch (drops stray far-away inliers, e.g. points seen through a doorway/window
that happen to lie on the same plane) and fit with an oriented minimum-area
rectangle (cv2.minAreaRect) -- a simple rectangular patch per plane, not a
general polygon.

Fragments of the same physical surface get deduped down to one plane each
(on by default, see --no-dedupe) -- "one plane per wall". The floor is fit
like everything else (RANSAC doesn't know the difference) but dropped from
the output by default, since only walls + roof are wanted (--keep-floor to
keep it). Opposite-facing walls (a corridor's two side walls, say) are then
nudged to share one common orientation so they come out exactly parallel,
instead of each carrying its own few degrees of independent RANSAC noise
(--no-parallel-walls to skip this). On top of that, every plane is snapped
onto the building's own orthogonal frame (derived from the data, not
assumed to be the world axes) so walls come out exactly perpendicular to
the roof and to any non-parallel wall (e.g. an end wall) -- "always 90 deg
between them" (--no-orthogonal to skip). That snap also makes leftover
duplicate fragments of the same surface share an *exact* normal, so they
get collapsed to one plane per direction too (--no-collapse-sides to keep
every fragment). Nothing is stretched to close a watertight box by
default -- that only happens if you explicitly pass --close-geometry.

No thermal data, no OpenStudio -- pure LiDAR geometry.

--bag (raw rosbag2 /cloud_registered) and --pcd (a .pcd file, e.g.
LoopClosureRaw.m's loop_closed_map.pcd) are mutually exclusive input
sources. --pcd skips the rosbag pipeline entirely and is preferable when
you have one: raw /cloud_registered still carries FAST-LIO drift (roll/
pitch tilt, yaw), so corridors come out skewed and an axis-aligned --roi
box cuts across them at an angle; loop_closed_map.pcd is already gravity/
yaw-aligned and loop-closed, so corridors are Manhattan-straight and a
simple xmin/xmax/ymin/ymax box cleanly isolates one of them.

Usage:
    python fit_planes.py (--bag <rosbag2_folder> | --pcd <file.pcd>)
        [--topic /cloud_registered]
        [--roi xmin,xmax,ymin,ymax,zmin,zmax]
        [--sor] [--sor-k 16] [--sor-std 1.5]
        [--declutter] [--cluster-gap 0.30] [--min-cluster 0] [--cluster-dist 0.0]
        [--voxel-preview 0.05]
        [--distance-threshold 0.02] [--ransac-n 3] [--num-iterations 1000]
        [--min-inliers 500] [--max-planes 20] [--rect-cluster-gap 0.3]
        [--max-tilt-deg 15] [--max-attempts 0]
        [--no-dedupe] [--dedupe-normal-deg 10] [--dedupe-offset-m 0.15]
        [--keep-floor] [--no-parallel-walls] [--parallel-max-angle-deg 45]
        [--parallel-min-separation-m 0.3] [--no-orthogonal] [--no-collapse-sides]
        [--snap-axis] [--close-geometry]
        [--out planes.json]

--declutter is the "automatic ROI": instead of a manual bounding box, it
drops points not connected to the main body (points outside the hall), the
same logic view_pointcloud.py uses. --rect-cluster-gap does the equivalent
trim per-plane, so a handful of stray inliers can't stretch a plane's
rectangle far beyond its real extent. --max-tilt-deg rejects candidate
planes whose normal isn't close to vertical (floor/ceiling) or horizontal
(wall) -- i.e. not aligned with the real building structure, like a
diagonal plane fit through noise -- and discards those points as noise
instead of returning them as a plane (0 = accept any orientation).

Occlusion/clutter/gaps often make RANSAC fit one physical floor/wall/ceiling
as several disjoint fragments. Dedupe (on by default) groups planes with
near-parallel normals and near-identical offset -- the same physical surface
-- and keeps only the largest (by inlier_count), dropping the redundant
fragments. --no-dedupe keeps every fragment as its own plane.

--snap-axis forces every accepted plane's normal onto the nearest world axis
(+-X, +-Y, +-Z), so tilt_deg is exactly 0 and all rectangles come out
perfectly Manhattan-aligned, instead of each carrying its own few degrees of
RANSAC noise.

--close-geometry goes one step further: independently axis-aligned rectangles
still don't share edges (each stops wherever its own inliers ended), so the
result isn't watertight and can't be imported to OpenStudio. It keeps only
the single largest plane per side per axis (X/Y/Z) and stretches each kept
rectangle to the shared bounding box, producing one closed 6(-ish)-face box.

Venv: C:\\venvs\\planefit (Python 3.12 -- open3d has no cp313 wheel yet,
see requirements.txt).
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import cKDTree


def read_pointcloud2(msg):
    # x,y,z as float32 at offsets 0,4,8; slice by point_step to skip extra fields
    step = msg.point_step
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def load_pcd_cloud(path):
    """Load xyz straight from a .pcd file -- e.g. LoopClosureRaw.m's
    loop_closed_map.pcd, already gravity/yaw-aligned and loop-closed, so
    corridors come out Manhattan-straight instead of skewed by raw FAST-LIO
    drift the way /cloud_registered is. Bypasses the rosbag pipeline
    entirely (no --topic/--store, no per-frame merge)."""
    if not Path(path).exists():
        raise SystemExit(f"--pcd {path} does not exist (check the path -- "
                          f"open3d fails silently on a bad path otherwise)")
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if len(xyz) == 0:
        raise SystemExit(f"--pcd {path} loaded 0 points -- not a valid/readable .pcd file")
    print(f"{len(xyz)} points from {path}")
    return xyz


def load_merged_cloud(bag, topic, store):
    typestore = get_typestore(Stores[store])
    frames = []
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Topic {topic!r} not found. Available: {topics}")
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            frames.append(read_pointcloud2(msg))
    xyz = np.vstack(frames)
    print(f"{len(frames)} frame(s), {len(xyz)} points")
    return xyz


def crop_roi(xyz, roi):
    xmin, xmax, ymin, ymax, zmin, zmax = roi
    mask = (
        (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)
        & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
        & (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    )
    print(f"ROI crop: kept {int(mask.sum())} / {len(xyz)} points")
    return xyz[mask]


def statistical_outlier_removal(xyz, k=16, std_ratio=1.5):
    """Drop points whose mean distance to k neighbours is an outlier.
    Same approach as view_pointcloud.py's SOR (scipy branch)."""
    if len(xyz) <= k:
        return xyz
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz, k=k + 1, workers=-1)  # +1 = the point itself
    mean_d = dist[:, 1:].mean(axis=1)
    thresh = mean_d.mean() + std_ratio * mean_d.std()
    keep = mean_d < thresh
    print(f"SOR: removed {int((~keep).sum())} / {len(xyz)} points")
    return xyz[keep]


def cluster_labels(points, gap):
    """Label points by connectivity (works for 2D or 3D points): two points in
    touching grid cells of edge `gap` (full neighbourhood) belong to the same
    cluster. Same union-find approach as view_pointcloud.py's declutter."""
    keys = np.floor(points / gap).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    index = {tuple(v): i for i, v in enumerate(uniq)}
    parent = np.arange(len(uniq))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    dim = points.shape[1]
    neigh = [d for d in itertools.product((-1, 0, 1), repeat=dim) if any(d)]
    for i, v in enumerate(uniq):
        for d in neigh:
            j = index.get(tuple(v + d))
            if j is not None and j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(len(uniq))])
    _, comp = np.unique(roots, return_inverse=True)
    return comp[inv]


def largest_cluster_mask(points, gap):
    labels = cluster_labels(points, gap)
    counts = np.bincount(labels)
    main = int(counts.argmax())
    return labels == main


def declutter(xyz, gap, min_size=0, keep_dist=0.0):
    """Remove points not connected to the main body (points outside the
    hall) -- the "automatic ROI". Same semantics as view_pointcloud.py:
    keep_dist>0 also keeps clusters within that distance of the main one,
    min_size>0 (and keep_dist==0) keeps every cluster with >= min_size
    points, otherwise only the largest cluster survives."""
    labels = cluster_labels(xyz, gap)
    counts = np.bincount(labels)
    main = int(counts.argmax())
    n_clusters = len(counts)

    if keep_dist > 0:
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


def fit_oriented_rect(points, normal, centroid, rect_cluster_gap=0.0, align_to_structure=True):
    """Project inlier points into the plane's own 2D basis and fit a rectangle.

    align_to_structure=True (default): the 2D basis is locked to the
    building's real orientation instead of picked arbitrarily -- for a
    floor/ceiling (normal near vertical) the axes are world X/Y; for a wall
    (normal near horizontal) one axis is world-up (Z) and the other is
    horizontal along the wall. The rectangle is then the axis-aligned
    bounding box in that basis, so it can never come out diamond-rotated the
    way an unconstrained min-area rectangle can on a sparse/irregular patch.
    align_to_structure=False falls back to cv2.minAreaRect's true
    minimum-area (possibly rotated) box on an arbitrary basis.

    If rect_cluster_gap > 0, first trim to the largest connected 2D patch so
    a few stray inliers (e.g. seen through a doorway) can't stretch the
    rectangle far beyond the plane's real extent."""
    n = normal / np.linalg.norm(normal)

    if align_to_structure:
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(n[2]) > 0.5:  # floor/ceiling: axes = world X, Y (projected onto the plane)
            u = np.array([1.0, 0.0, 0.0]) - n[0] * n
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
        else:  # wall: one axis is world-up, the other horizontal along the wall
            u = np.cross(n, world_up)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-6:  # degenerate: normal ~parallel to world_up, shouldn't happen for a wall
                u = np.array([1.0, 0.0, 0.0]) - n[0] * n
                norm_u = np.linalg.norm(u)
            u /= norm_u
            v = world_up.copy()
    else:
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, arbitrary)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

    rel = points - centroid
    pts2d = np.stack([rel @ u, rel @ v], axis=1).astype(np.float32)

    if rect_cluster_gap > 0 and len(pts2d) > 10:
        mask = largest_cluster_mask(pts2d, rect_cluster_gap)
        dropped = len(pts2d) - int(mask.sum())
        if dropped:
            print(f"  rect trim: dropped {dropped} stray inlier(s) outside main patch")
        pts2d = pts2d[mask]

    if align_to_structure:
        xmin, ymin = pts2d.min(axis=0)
        xmax, ymax = pts2d.max(axis=0)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        w, h = float(xmax - xmin), float(ymax - ymin)
        angle = 0.0
        box2d = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
    else:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts2d)
        box2d = cv2.boxPoints(((cx, cy), (w, h), angle))  # 4x2, CCW in local basis

    box3d = centroid + box2d[:, 0:1] * u + box2d[:, 1:2] * v
    center3d = centroid + cx * u + cy * v

    return {
        "width_m": float(w),
        "height_m": float(h),
        "area_m2": float(w * h),
        "angle_deg": float(angle),
        "center_3d": center3d.tolist(),
        "corners_3d": box3d.tolist(),
        "basis_u": u.tolist(),
        "basis_v": v.tolist(),
    }


def tilt_from_structure_deg(normal):
    """Angle (deg) of `normal` from the nearest Manhattan-aligned orientation:
    0 deg = exactly vertical normal (floor/ceiling), 90 deg = exactly
    horizontal normal (wall). Assumes Z is up (world/map frame convention
    used throughout this script)."""
    n = normal / np.linalg.norm(normal)
    angle_to_vertical = np.degrees(np.arccos(np.clip(abs(n[2]), 0.0, 1.0)))
    return min(angle_to_vertical, abs(90.0 - angle_to_vertical))


_CANONICAL_AXES = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
])


def snap_normal_to_axis(normal):
    """Snap `normal` to the nearest of the 6 world axes (+-X, +-Y, +-Z), so
    the plane's tilt becomes exactly 0 deg. Assumes the structure really is
    Manhattan-aligned with the world frame (true here: walls already come
    out normal ~[+-1,0,0]/[0,+-1,0], floor/ceiling ~[0,0,+-1])."""
    n = normal / np.linalg.norm(normal)
    return _CANONICAL_AXES[np.argmax(_CANONICAL_AXES @ n)]


def segment_planes(xyz, distance_threshold, ransac_n, num_iterations, min_inliers, max_planes,
                    rect_cluster_gap=0.0, max_tilt_deg=0.0, max_attempts=0, align_to_structure=True,
                    snap_axis=False):
    """Iteratively RANSAC-fit planes. If max_tilt_deg > 0, a candidate plane
    whose normal isn't close to vertical (floor/ceiling) or horizontal (wall)
    -- i.e. doesn't match the building's real structure orientation -- is
    rejected: its inliers are discarded as noise (not returned as a plane)
    and fitting continues on what's left, up to max_attempts total tries.
    If snap_axis, every accepted plane's normal is snapped to the nearest
    world axis (tilt_deg forced to exactly 0) before the rectangle is fit."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if max_attempts <= 0:
        max_attempts = max_planes * 5

    planes = []
    remaining = pcd
    attempt = 0
    while len(planes) < max_planes and attempt < max_attempts:
        if len(remaining.points) < max(ransac_n, min_inliers):
            print(f"only {len(remaining.points)} points left, stopping")
            break

        model, inlier_idx = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        attempt += 1
        if len(inlier_idx) < min_inliers:
            print(f"best fit {len(inlier_idx)} inliers < --min-inliers {min_inliers}, stopping")
            break

        a, b, c, d = model
        normal = np.array([a, b, c])
        inlier_pts = np.asarray(remaining.points)[inlier_idx]

        # orientation, always computed (not just when --max-tilt-deg filters on it):
        # tilt_deg = degrees off the nearest Manhattan axis, orientation = which
        # structural face it matches (floor_ceiling: |normal_z|>0.5, else wall)
        tilt = tilt_from_structure_deg(normal)
        n_unit = normal / np.linalg.norm(normal)
        orientation = "floor_ceiling" if abs(n_unit[2]) > 0.5 else "wall"

        if max_tilt_deg > 0 and tilt > max_tilt_deg:
            print(f"reject: {len(inlier_idx)}-point plane tilted {tilt:.1f} deg off "
                  f"vertical/horizontal (> --max-tilt-deg {max_tilt_deg}), discarding as noise")
            remaining = remaining.select_by_index(inlier_idx, invert=True)
            continue

        centroid = inlier_pts.mean(axis=0)

        if snap_axis:
            normal_out = snap_normal_to_axis(normal)
            d_out = -float(np.dot(normal_out, centroid))  # plane still passes through the centroid
            tilt_out = 0.0
        else:
            normal_out = normal
            d_out = float(d)
            tilt_out = tilt

        rect = fit_oriented_rect(inlier_pts, normal_out, centroid, rect_cluster_gap=rect_cluster_gap,
                                  align_to_structure=align_to_structure)

        plane = {
            "id": len(planes),
            "normal": normal_out.tolist(),
            "d": d_out,
            "inlier_count": len(inlier_idx),
            "centroid_3d": centroid.tolist(),
            "orientation": orientation,
            "tilt_deg": tilt_out,
            **rect,
            "_inlier_pts": inlier_pts,  # kept in-memory only, stripped before JSON write --
                                         # lets later corrections (parallel/orthogonal) refit
                                         # the rectangle from real points instead of chaining
                                         # distortion through reprojected corners
        }
        planes.append(plane)
        print(f"plane {plane['id']}: {len(inlier_idx)} inliers, normal {normal_out.round(3)}, "
              f"orientation={orientation} tilt={tilt_out:.1f}deg (raw fit was {tilt:.1f}deg), "
              f"rect {rect['width_m']:.2f}x{rect['height_m']:.2f} m")

        remaining = remaining.select_by_index(inlier_idx, invert=True)

    print(f"{len(planes)} plane(s) extracted, {len(remaining.points)} points left unassigned")
    return planes


def canonical_normal_offset(normal, d):
    """Sign-canonicalize (normal, d) so two fits of the *same* physical plane
    compare equal regardless of which side segment_plane's normal happened to
    point to: flip both if the largest-magnitude normal component is negative."""
    n = np.array(normal, dtype=float)
    n = n / np.linalg.norm(n)
    if n[np.argmax(np.abs(n))] < 0:
        n, d = -n, -d
    return n, d


def _rect_source_points(p):
    """Points to re-fit a plane's rectangle from after its normal changes.
    Prefers the plane's own raw RANSAC inliers (kept in-memory as
    "_inlier_pts" by segment_planes) over its already-fit corners_3d --
    refitting from real points at every correction step avoids compounding
    distortion when a plane's normal is rotated more than a few degrees
    (chaining corner-only reprojections can skew a rectangle into a sliver).
    Falls back to corners_3d for planes that never carried inliers (e.g. a
    --close-geometry retry-fit plane)."""
    pts = p.get("_inlier_pts")
    return pts if pts is not None else np.array(p["corners_3d"])


def dedupe_planes(planes, normal_deg=10.0, offset_m=0.15):
    """A real corridor floor/wall/ceiling often gets RANSAC-fit as several
    disjoint fragments (clutter, occlusion, gaps) instead of one plane.
    Group planes that are effectively the same physical surface -- same
    orientation, near-parallel normals, near-coplanar -- and keep only the
    largest (by inlier_count) of each group, dropping the rest.

    Coplanarity is tested by point-to-plane distance of each fragment's
    *centroid* against the other fragment's fitted plane equation (the min
    of both directions), not by comparing their d values directly. Two
    fragments of the same long wall can differ in d by more than offset_m
    even when they're clearly the same surface, if the wall's fitted normal
    carries even a couple degrees of tilt: d = -normal.centroid drifts with
    how far along the wall each fragment's centroid sits. Point-to-plane
    distance doesn't have that problem -- it directly asks "does fragment
    j's centroid lie on fragment i's plane"."""
    order = sorted(range(len(planes)), key=lambda i: -planes[i]["inlier_count"])
    keyed = [canonical_normal_offset(planes[i]["normal"], planes[i]["d"]) for i in range(len(planes))]
    centroids = [np.array(planes[i]["centroid_3d"], dtype=float) for i in range(len(planes))]

    used = [False] * len(planes)
    kept = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        kept.append(i)
        ni, di = keyed[i]
        dropped = []
        for j in order:
            if used[j]:
                continue
            nj, dj = keyed[j]
            angle = np.degrees(np.arccos(np.clip(np.dot(ni, nj), -1.0, 1.0)))
            dist_j_on_i = abs(np.dot(ni, centroids[j]) + di)
            dist_i_on_j = abs(np.dot(nj, centroids[i]) + dj)
            coplanar = min(dist_j_on_i, dist_i_on_j) < offset_m
            if angle < normal_deg and coplanar:
                used[j] = True
                dropped.append(j)
        if dropped:
            print(f"dedupe: plane {planes[i]['id']} ({planes[i]['inlier_count']} inliers) absorbs "
                  f"{[planes[j]['id'] for j in dropped]} as the same surface")

    kept.sort()
    result = [planes[i] for i in kept]
    for new_id, p in enumerate(result):
        p["id"] = new_id
    print(f"dedupe: {len(planes)} -> {len(result)} plane(s)")
    return result


def classify_horizontal(planes, xyz):
    """Split the generic "floor_ceiling" orientation into "floor" and "roof",
    using which one sits closer to the point cloud's z-extremes -- so
    --keep-floor (and printing) can tell them apart. Walls are untouched."""
    if not len(xyz):
        return planes
    zlo, zhi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    for p in planes:
        if p["orientation"] != "floor_ceiling":
            continue
        z = p["centroid_3d"][2]
        p["orientation"] = "floor" if abs(z - zlo) <= abs(z - zhi) else "roof"
    return planes


def enforce_parallel_walls(planes, max_pair_angle_deg=45.0, min_separation_m=0.3):
    """Nudge each pair of facing walls (a corridor's two side walls, say)
    onto one shared axis, so they come out exactly parallel instead of each
    carrying its own few degrees of independent RANSAC noise, and give them
    opposite normal signs so they are unambiguously two distinct surfaces.
    Each wall's own position is untouched -- only its normal (and the
    offset/rectangle derived from it) is rotated in place about its own
    centroid; walls with no plausible partner are left as originally fit.

    Pairing compares the normals as undirected lines (|dot|), not as
    vectors. open3d's segment_plane picks each plane's normal sign
    arbitrarily -- nothing ties it to "outward" -- so a corridor's two
    facing walls very often come back with normals pointing the SAME way
    rather than opposite ways. Testing for anti-parallel normals therefore
    misses real facing pairs (it scores them near 180 deg instead of near
    0), which is how a whole wall could previously go unpaired, get snapped
    onto the same canonical axis as its opposite, and then be deleted by
    collapse_duplicate_sides as a "duplicate".

    Pairing is greedy: repeatedly take an unmatched wall and pair it with
    whichever remaining unmatched wall is most nearly parallel to it,
    provided that angle is within max_pair_angle_deg of parallel (walls
    perpendicular to each other, e.g. a corridor's end wall, are not a
    facing pair and must not be forced parallel) AND the two are at least
    min_separation_m apart along that shared axis (two nearly-coplanar
    fragments are one surface, not a facing pair -- forcing those to face
    opposite ways would invent a second wall where there is one)."""
    walls = [p for p in planes if p["orientation"] == "wall"]
    unmatched = list(walls)
    pairs = []
    while unmatched:
        a = unmatched.pop(0)
        na = np.array(a["normal"], dtype=float)
        na /= np.linalg.norm(na)
        best, best_angle, best_common = None, None, None
        for b in unmatched:
            nb = np.array(b["normal"], dtype=float)
            nb /= np.linalg.norm(nb)
            angle = np.degrees(np.arccos(np.clip(abs(np.dot(na, nb)), 0.0, 1.0)))
            if best_angle is None or angle < best_angle:
                aligned = nb if np.dot(na, nb) >= 0 else -nb  # sign-agnostic before averaging
                common = na + aligned
                best, best_angle = b, angle
                best_common = common / np.linalg.norm(common)

        if best is None:
            print(f"parallel: wall {a['id']} ({a['inlier_count']} inliers) has no other "
                  f"wall left to pair with, leaving as fit")
            continue
        if best_angle >= max_pair_angle_deg:
            print(f"parallel: wall {a['id']} ({a['inlier_count']} inliers) closest "
                  f"parallel candidate was wall {best['id']} at {best_angle:.1f} deg off "
                  f"parallel (> --parallel-max-angle-deg {max_pair_angle_deg}), leaving unpaired")
            continue
        sep = float(np.dot(best_common,
                           np.array(a["centroid_3d"]) - np.array(best["centroid_3d"])))
        if abs(sep) < min_separation_m:
            print(f"parallel: wall {a['id']} and wall {best['id']} are parallel but only "
                  f"{abs(sep):.2f} m apart (< --parallel-min-separation-m "
                  f"{min_separation_m}) -- fragments of one surface, not a facing pair, "
                  f"leaving both as fit")
            continue

        unmatched.remove(best)
        pairs.append((a, best, best_angle, best_common, sep))

    for a, b, angle, common, sep in pairs:
        # Outward-facing convention: each wall's normal points away from the space between
        # the pair. sep = (centroid_a - centroid_b) . common, so a sits on the +common side
        # when sep > 0. The two signs are always opposite, which is what keeps the pair from
        # later collapsing into one plane.
        sign_a = 1.0 if sep > 0 else -1.0
        for p, sign in ((a, sign_a), (b, -sign_a)):
            n_out = common * sign
            centroid = np.array(p["centroid_3d"])
            d_out = -float(np.dot(n_out, centroid))
            rect = fit_oriented_rect(_rect_source_points(p), n_out, centroid,
                                      rect_cluster_gap=0.0, align_to_structure=True)
            p["normal"] = n_out.tolist()
            p["d"] = d_out
            p["tilt_deg"] = tilt_from_structure_deg(n_out)
            p.update(rect)
        print(f"parallel: walls {a['id']} ({a['inlier_count']} inliers) & {b['id']} "
              f"({b['inlier_count']} inliers) were {angle:.1f} deg off parallel and are "
              f"{abs(sep):.2f} m apart -- aligned to one shared axis, facing opposite ways")

    return planes


def build_orthogonal_frame(planes):
    """Derive one consistent orthonormal building frame -- up, plus two
    horizontal directions 90 deg apart -- from the fitted planes themselves,
    instead of assuming the world X/Y/Z axes (that's what --snap-axis does).
    Robust even if the whole structure is yawed relative to the world frame.

    up: inlier-weighted average of roof/floor normals (flipped to +Z-ish).
    e1: the horizontal direction of the single largest wall, projected
    orthogonal to up. e2 = up x e1 -- the other horizontal direction (e.g.
    an end wall closing the corridor). Returns (up, None, None) if there's
    no wall to anchor e1/e2 on."""
    ups = [(np.array(p["normal"], dtype=float), p["inlier_count"])
           for p in planes if p["orientation"] in ("roof", "floor")]
    if ups:
        up = np.zeros(3)
        for n, w in ups:
            n = n / np.linalg.norm(n)
            if n[2] < 0:
                n = -n
            up += n * w
        up /= np.linalg.norm(up)
    else:
        up = np.array([0.0, 0.0, 1.0])

    walls = [p for p in planes if p["orientation"] == "wall"]
    if not walls:
        return up, None, None
    ref = max(walls, key=lambda p: p["inlier_count"])
    n_ref = np.array(ref["normal"], dtype=float)
    e1 = n_ref - np.dot(n_ref, up) * up
    if np.linalg.norm(e1) < 1e-6:
        return up, None, None
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    e2 /= np.linalg.norm(e2)
    return up, e1, e2


def enforce_orthogonal_planes(planes):
    """Snap every plane's normal onto the building's own orthogonal frame
    (see build_orthogonal_frame): walls onto +-e1/+-e2, roof/floor onto
    +-up. Two walls already forced parallel by enforce_parallel_walls land
    on the same axis and stay parallel; a wall on the other horizontal axis
    (e.g. the corridor's end wall) comes out exactly perpendicular to both
    of them, and every wall comes out exactly perpendicular to the roof --
    "always 90 deg between them", independent of the yaw of the fitted
    structure relative to the world frame. Only orientation changes; each
    plane's own centroid (position) is untouched.

    Roof/floor never pick their axis by dot product against +-up: open3d's
    segment_plane returns an arbitrary normal sign, so a floor and a roof
    can easily both come out with a raw normal pointing +Z-ish (nothing
    ties either one to "outward"), and nearest-axis-by-dot-product would
    then snap *both* onto the same +up direction -- collapse_duplicate_sides
    would fuse them and keep whichever has more inliers, which is not
    reliably the roof (a downward-looking scan can see more floor than
    ceiling). classify_horizontal's floor/roof split is by centroid height,
    not normal sign, and is reliable -- so roof is forced to +up and floor
    to -up directly from that label, guaranteeing they land on opposite
    canonical directions and are never merged as duplicates of each other."""
    up, e1, e2 = build_orthogonal_frame(planes)
    if e1 is None:
        print("orthogonal: no wall to anchor a horizontal direction on, skipping")
        return planes
    wall_axes = [e1, -e1, e2, -e2]
    for p in planes:
        if p["orientation"] == "roof":
            best = up
        elif p["orientation"] == "floor":
            best = -up
        else:
            n = np.array(p["normal"], dtype=float)
            n = n / np.linalg.norm(n)
            best = max(wall_axes, key=lambda a: np.dot(a, n))
        centroid = np.array(p["centroid_3d"], dtype=float)
        d_out = -float(np.dot(best, centroid))
        rect = fit_oriented_rect(_rect_source_points(p), best, centroid,
                                  rect_cluster_gap=0.0, align_to_structure=True)
        p["normal"] = best.tolist()
        p["d"] = d_out
        p["tilt_deg"] = tilt_from_structure_deg(best)
        p.update(rect)
    print(f"orthogonal: snapped {len(planes)} plane(s) to the building's own frame "
          f"(up={up.round(3).tolist()}, wall axes {e1.round(3).tolist()} / "
          f"{e2.round(3).tolist()}) -- exact 90 deg between walls and roof")
    return planes


def collapse_duplicate_sides(planes, offset_m=0.15):
    """Only meaningful right after enforce_orthogonal_planes: every plane's
    normal is then exactly one of the building's 6 canonical directions
    (+-up, +-e1, +-e2), so fragments of the same physical wall/roof that
    dedupe_planes still missed (e.g. two segments of a long tilted wall
    whose measured offsets drifted apart before snapping) now share the
    *exact* same normal and can be grouped losslessly. Keeps only the
    largest (by inlier_count) plane per surface, dropping the rest as
    redundant fragments -- no stretching, unlike --close-geometry; each
    kept plane's own rectangle is untouched.

    Sharing a canonical direction does NOT by itself make two planes
    duplicates: parallel surfaces at different offsets (a corridor's two
    side walls) are distinct walls, and they can legitimately land on the
    same direction when one of them was never paired up by
    enforce_parallel_walls. So planes are grouped by direction AND by
    offset, and only near-coplanar ones (|d| difference < offset_m, exact
    here since the group shares one normal) collapse together."""
    groups = {}
    for p in planes:
        key = tuple(np.round(p["normal"], 6))
        groups.setdefault(key, []).append(p)

    kept = []
    for key, group in groups.items():
        buckets = []
        for p in sorted(group, key=lambda q: -q["inlier_count"]):
            for bucket in buckets:
                if abs(p["d"] - bucket[0]["d"]) < offset_m:
                    bucket.append(p)
                    break
            else:
                buckets.append([p])
        if len(buckets) > 1:
            print(f"collapse: direction {[round(v, 2) for v in key]} holds "
                  f"{len(buckets)} distinct parallel surface(s) at offsets "
                  f"{[round(b[0]['d'], 2) for b in buckets]} -- keeping them separate")
        for bucket in buckets:
            best = bucket[0]  # bucket[0] is the largest: the group was sorted by inlier_count
            kept.append(best)
            dropped = [q["id"] for q in bucket[1:]]
            if dropped:
                print(f"collapse: direction {[round(v, 2) for v in key]} offset "
                      f"{best['d']:.2f} kept plane {best['id']} ({best['inlier_count']} "
                      f"inliers), dropped {dropped} as redundant fragments")

    kept.sort(key=lambda p: p["id"])
    for new_id, p in enumerate(kept):
        p["id"] = new_id
    print(f"collapse: {len(planes)} -> {len(kept)} plane(s)")
    return kept


def _retry_fit_closing_plane(xyz, axis, missing_side, distance_threshold, ransac_n, num_iterations,
                              retry_min_inliers, retry_margin, retry_min_density_ratio, density_ref,
                              rect_cluster_gap, align_to_structure, snap_axis, plane_id):
    """A big open face (e.g. a whole missing wall) usually means real data
    is there but it lost out to --min-inliers or just wasn't the largest fit
    on the first pass. Search a thin slab of the working cloud right at the
    boundary that would close this axis/side for a plane matching that
    orientation, using a relaxed inlier threshold. Returns a plane dict, or
    None if nothing convincing enough was found (in which case the caller
    leaves the face open, same as before).

    A rover/pedestrian entrance or corridor continuation can still produce a
    plane that technically fits open3d's RANSAC (scattered floor/ceiling
    edge points, glimpses through the opening) without a real wall being
    there. Its rectangle can even span an area as large as a genuine wall's,
    so area/coverage alone doesn't catch it -- what's actually different is
    that a real wall is a solid, densely-covered surface while an opening is
    thin, scattered points. So the candidate is only accepted if its point
    density (inliers / rectangle area) is at least retry_min_density_ratio
    of density_ref -- the typical density of the already-confirmed real
    walls/floor/ceiling found elsewhere in this same geometry."""
    if xyz is None or len(xyz) == 0:
        return None
    col = xyz[:, axis]
    if missing_side == "-":
        edge = float(col.min())
        mask = col <= edge + retry_margin
    else:
        edge = float(col.max())
        mask = col >= edge - retry_margin
    slab = xyz[mask]
    if len(slab) < max(ransac_n, retry_min_inliers):
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(slab)
    model, inlier_idx = pcd.segment_plane(distance_threshold=distance_threshold,
                                           ransac_n=ransac_n, num_iterations=num_iterations)
    if len(inlier_idx) < retry_min_inliers:
        return None

    a, b, c, d = model
    normal = np.array([a, b, c])
    if int(np.argmax(np.abs(normal))) != axis:
        return None  # the slab's best plane isn't even oriented the way we need

    inlier_pts = slab[inlier_idx]
    centroid = inlier_pts.mean(axis=0)
    tilt = tilt_from_structure_deg(normal)
    n_unit = normal / np.linalg.norm(normal)
    orientation = "floor_ceiling" if abs(n_unit[2]) > 0.5 else "wall"

    if snap_axis:
        normal = snap_normal_to_axis(normal)
        d = -float(np.dot(normal, centroid))
        tilt = 0.0

    rect = fit_oriented_rect(inlier_pts, normal, centroid, rect_cluster_gap=rect_cluster_gap,
                              align_to_structure=align_to_structure)

    density = len(inlier_idx) / rect["area_m2"] if rect["area_m2"] > 0 else 0.0
    if density_ref is not None and density < retry_min_density_ratio * density_ref:
        print(f"  retry candidate point density {density:.0f}/m2 is only "
              f"{density / density_ref:.0%} of confirmed walls' {density_ref:.0f}/m2 "
              f"(< --close-retry-min-density-ratio {retry_min_density_ratio:.0%}) -- "
              f"thin/scattered like a door or opening, not a solid wall, rejecting")
        return None

    return {
        "id": plane_id, "normal": normal.tolist(), "d": float(d),
        "inlier_count": len(inlier_idx), "centroid_3d": centroid.tolist(),
        "orientation": orientation, "tilt_deg": float(tilt), **rect,
    }


def close_geometry(planes, xyz=None, distance_threshold=0.02, ransac_n=3, num_iterations=1000,
                    retry_min_inliers=150, retry_margin=0.5, retry_min_density_ratio=0.4,
                    rect_cluster_gap=0.0, align_to_structure=True, snap_axis=False, roi=None,
                    cap_open_faces=False):
    """Reduce the plane set to a single watertight box, importable to
    OpenStudio as a shoebox Space: for each axis (X/Y/Z) and each side of the
    room's centroid along that axis, keep only the single largest (by
    inlier_count) plane -- dropping the other, redundant fragments on that
    side -- then stretch every kept plane's rectangle to the shared bounding
    box on its other two axes, so every face's edges meet its neighbors
    exactly instead of stopping wherever its own inliers happened to end.

    If a whole side of an axis has no plane at all (a big open face, e.g. a
    missing wall) and `xyz` (the working point cloud) is given, retries a
    targeted RANSAC fit in a thin slab at that boundary before giving up --
    the open face is only kept if that retry also fails to find anything.
    A point-density check rejects thin/scattered fits (a corridor
    continuation or doorway can span as much area as a real wall without
    being solidly covered), but that alone isn't reliable when the open
    side sits exactly on a manually-cropped --roi boundary -- that boundary
    is where *we* cut the data, not a real edge, so if `roi` is given
    (xmin,xmax,ymin,ymax,zmin,zmax) any open side that coincides with it is
    left open without even attempting a retry."""
    if not planes:
        return planes

    total_w = sum(p["inlier_count"] for p in planes) or 1
    center = np.zeros(3)
    for p in planes:
        center += np.array(p["centroid_3d"]) * p["inlier_count"]
    center /= total_w

    def axis_of(p):
        return int(np.argmax(np.abs(np.array(p["normal"]))))

    axis_groups = {0: [], 1: [], 2: []}
    for p in planes:
        axis_groups[axis_of(p)].append(p)

    # pass 1: keep the largest plane per non-empty side, note which (axis, side) are missing
    kept = []
    missing = []
    for axis in (0, 1, 2):
        group = axis_groups[axis]
        neg = [p for p in group if p["centroid_3d"][axis] < center[axis]]
        pos = [p for p in group if p["centroid_3d"][axis] >= center[axis]]
        for side_name, side in (("-", neg), ("+", pos)):
            if side:
                best = max(side, key=lambda p: p["inlier_count"])
                kept.append(best)
                dropped = [p["id"] for p in side if p is not best]
                if dropped:
                    print(f"close-geometry: axis {'XYZ'[axis]}{side_name} kept plane {best['id']} "
                          f"({best['inlier_count']} inliers), dropped {dropped} as redundant")
            elif group:  # some plane exists on this axis, just not this side -- worth a retry
                missing.append((axis, side_name))

    # a real wall's point density (inliers/area) from the confirmed (non-retry) planes --
    # the yardstick a retry candidate must live up to, since a door/opening can span a
    # similarly large area without being anywhere near as densely covered
    densities = [p["inlier_count"] / p["area_m2"] for p in kept if p.get("area_m2", 0) > 0]
    density_ref = float(np.median(densities)) if densities else None

    # pass 2: retry the missing sides now that density_ref reflects the *real* walls only
    next_id = max((p["id"] for p in planes), default=-1) + 1
    for axis, side_name in missing:
        if roi is not None:
            roi_bound = roi[2 * axis] if side_name == "-" else roi[2 * axis + 1]
            edge = float(xyz[:, axis].min() if side_name == "-" else xyz[:, axis].max()) if xyz is not None else None
            if edge is not None and abs(edge - roi_bound) < 0.05:
                print(f"close-geometry: axis {'XYZ'[axis]}{side_name} sits right on the --roi crop "
                      f"boundary ({roi_bound}) -- that's where we cut the data, not a real edge, "
                      f"leaving it open without retrying")
                continue
        print(f"close-geometry: axis {'XYZ'[axis]}{side_name} is a big open face, "
              f"retrying a targeted fit to close it...")
        found = _retry_fit_closing_plane(
            xyz, axis, side_name, distance_threshold, ransac_n, num_iterations,
            retry_min_inliers, retry_margin, retry_min_density_ratio, density_ref,
            rect_cluster_gap, align_to_structure, snap_axis, next_id)
        if found is not None:
            print(f"close-geometry: retry fit plane {next_id} ({found['inlier_count']} inliers) "
                  f"closes axis {'XYZ'[axis]}{side_name}")
            kept.append(found)
            next_id += 1
        else:
            print(f"close-geometry: retry found nothing convincing for axis {'XYZ'[axis]}{side_name}, "
                  f"leaving it open")

    if len(kept) < 3:
        print(f"close-geometry: only {len(kept)} independent face(s) found -- "
              f"not enough to close a box, keeping planes as-is")
        return planes

    # Shared box bounds per axis. Every face stretches to these same numbers -- using a
    # per-plane fallback here instead would leave faces on a half-open axis (e.g. the
    # ceiling) stopping at their own inlier extent, so they'd never meet the neighbours.
    def data_extent(axis):
        if xyz is not None and len(xyz):
            return float(xyz[:, axis].min()), float(xyz[:, axis].max())
        allc = np.array([c for p in kept for c in p["corners_3d"]])
        return float(allc[:, axis].min()), float(allc[:, axis].max())

    bounds = {}
    open_sides = []
    for axis in (0, 1, 2):
        vals = [p["centroid_3d"][axis] for p in kept if axis_of(p) == axis]
        lo_data, hi_data = data_extent(axis)
        if len(vals) >= 2 and max(vals) - min(vals) > 1e-9:
            bounds[axis] = (min(vals), max(vals))
        elif vals:
            # half-open: pin the side we actually measured, take the open side from the data
            known = vals[0]
            if abs(known - hi_data) < abs(known - lo_data):
                bounds[axis] = (lo_data, known)
                open_sides.append((axis, "-"))
            else:
                bounds[axis] = (known, hi_data)
                open_sides.append((axis, "+"))
            print(f"close-geometry: only one {'XYZ'[axis]}-side plane found -- that end is "
                  f"open (e.g. a corridor cut, not a real wall); using the data extent "
                  f"{bounds[axis][0]:.2f}..{bounds[axis][1]:.2f} so the other faces still line up")
        else:
            bounds[axis] = (lo_data, hi_data)
            open_sides.extend([(axis, "-"), (axis, "+")])

    if cap_open_faces:
        for axis, side_name in open_sides:
            fixed = bounds[axis][0] if side_name == "-" else bounds[axis][1]
            normal = [0.0, 0.0, 0.0]
            normal[axis] = 1.0
            centroid = [(bounds[a][0] + bounds[a][1]) / 2 for a in (0, 1, 2)]
            centroid[axis] = fixed
            kept.append({
                "id": next_id, "normal": normal, "d": -fixed,
                "inlier_count": 0, "centroid_3d": centroid,
                "orientation": "floor_ceiling" if axis == 2 else "wall",
                "tilt_deg": 0.0, "synthetic": True, "boundary_condition": "adiabatic",
                "width_m": 0.0, "height_m": 0.0, "area_m2": 0.0, "angle_deg": 0.0,
                "center_3d": centroid, "corners_3d": [centroid] * 4,
                "basis_u": [0.0, 0.0, 0.0], "basis_v": [0.0, 0.0, 0.0],
            })
            print(f"close-geometry: capping open face {'XYZ'[axis]}{side_name} at {fixed:.2f} with a "
                  f"synthetic adiabatic surface (plane {next_id}) -- no LiDAR evidence of a wall "
                  f"there, it exists only to make the solid watertight for OpenStudio")
            next_id += 1

    for p in kept:
        axis = axis_of(p)
        other = [a for a in (0, 1, 2) if a != axis]
        fixed = p["centroid_3d"][axis]
        ranges = [bounds[a] for a in other]
        (lo0, hi0), (lo1, hi1) = ranges
        a0, a1 = other

        corners3d = []
        for v0, v1 in [(lo0, lo1), (hi0, lo1), (hi0, hi1), (lo0, hi1)]:
            c = [0.0, 0.0, 0.0]
            c[axis], c[a0], c[a1] = fixed, v0, v1
            corners3d.append(c)
        p["corners_3d"] = corners3d

        center3d = [0.0, 0.0, 0.0]
        center3d[axis], center3d[a0], center3d[a1] = fixed, (lo0 + hi0) / 2, (lo1 + hi1) / 2
        p["center_3d"] = center3d

        p["width_m"] = float(hi0 - lo0)
        p["height_m"] = float(hi1 - lo1)
        p["area_m2"] = float(p["width_m"] * p["height_m"])

    kept.sort(key=lambda p: p["id"])
    for new_id, p in enumerate(kept):
        p["id"] = new_id
    print(f"close-geometry: {len(planes)} -> {len(kept)} plane(s), closed box "
          f"x={bounds.get(0)} y={bounds.get(1)} z={bounds.get(2)}")
    return kept


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, help="rosbag2 folder (raw /cloud_registered)")
    ap.add_argument("--pcd", type=Path,
                    help="load xyz straight from a .pcd instead of a bag -- e.g. "
                         "LoopClosureRaw.m's loop_closed_map.pcd, already gravity/"
                         "yaw-aligned and loop-closed so corridors are Manhattan-"
                         "straight instead of skewed by raw FAST-LIO drift. Mutually "
                         "exclusive with --bag; --topic/--store are ignored with --pcd")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--roi", help="xmin,xmax,ymin,ymax,zmin,zmax crop (metres)")
    ap.add_argument("--sor", action="store_true", help="apply Statistical Outlier Removal")
    ap.add_argument("--sor-k", type=int, default=16, help="neighbours per point for SOR")
    ap.add_argument("--sor-std", type=float, default=1.5,
                    help="SOR std multiplier (lower = more aggressive)")
    ap.add_argument("--declutter", action="store_true",
                    help="automatic ROI: drop points not connected to the main body "
                         "(points outside the hall), instead of a manual --roi box")
    ap.add_argument("--cluster-gap", type=float, default=0.30,
                    help="max gap (m) for points to count as the same cluster (--declutter)")
    ap.add_argument("--min-cluster", type=int, default=0,
                    help="keep every cluster with >= N points (instead of largest only)")
    ap.add_argument("--cluster-dist", type=float, default=0.0,
                    help="keep clusters within this distance (m) of the main cloud")
    ap.add_argument("--rect-cluster-gap", type=float, default=0.3,
                    help="per-plane rectangle trim: max gap (m) for a plane's inlier "
                         "points to count as the same 2D patch; only the largest patch "
                         "is fit (0 = off, use all inliers as-is)")
    ap.add_argument("--voxel-preview", type=float, default=0.0,
                    help="voxel size (m) for a QA occupied-voxel count only -- "
                         "does NOT affect RANSAC, which always runs full-resolution")
    ap.add_argument("--distance-threshold", type=float, default=0.02,
                    help="RANSAC inlier distance (m) -- LiDAR self-consistency, "
                         "tight on purpose (not the ~9cm cross-modal LiDAR<->camera "
                         "RMSE used elsewhere in this repo for thermal fusion)")
    ap.add_argument("--ransac-n", type=int, default=3)
    ap.add_argument("--num-iterations", type=int, default=1000)
    ap.add_argument("--min-inliers", type=int, default=500,
                    help="stop once the next best plane has fewer inliers than this")
    ap.add_argument("--max-planes", type=int, default=20, help="safety cap")
    ap.add_argument("--max-tilt-deg", type=float, default=15.0,
                    help="reject a candidate plane if its normal is more than this many "
                         "degrees off vertical (floor/ceiling) or horizontal (wall) -- "
                         "i.e. not aligned with the real building structure; its points "
                         "are discarded as noise, not returned as a plane (0 = accept any orientation)")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="total RANSAC tries (accepted + rejected); 0 = --max-planes * 5")
    ap.add_argument("--free-rotation", action="store_true",
                    help="let the rectangle rotate freely to the true cv2.minAreaRect "
                         "minimum-area box, instead of locking it to the building's real "
                         "orientation (world X/Y for floor/ceiling, up+horizontal for walls) -- "
                         "default is locked, since free rotation can come out diamond-shaped "
                         "on a sparse/irregular patch")
    ap.add_argument("--snap-axis", action="store_true",
                    help="force every accepted plane's normal onto the nearest world axis "
                         "(+-X/+-Y/+-Z) -- tilt_deg becomes exactly 0, rectangles come out "
                         "perfectly Manhattan-aligned instead of carrying a few degrees of RANSAC noise")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep every fragment as its own plane instead of merging "
                         "near-duplicate fragments of the same physical surface")
    ap.add_argument("--dedupe-normal-deg", type=float, default=10.0,
                    help="max angle (deg) between normals to count as the same surface")
    ap.add_argument("--dedupe-offset-m", type=float, default=0.15,
                    help="max offset difference (m) to count as the same surface")
    ap.add_argument("--keep-floor", action="store_true",
                    help="keep the floor plane in the output -- default is walls + "
                         "roof only; the floor is still fit like any other plane "
                         "during RANSAC (it just gets dropped from the final result)")
    ap.add_argument("--no-parallel-walls", action="store_true",
                    help="skip forcing opposite-facing walls onto a shared parallel "
                         "orientation -- each wall keeps its own raw RANSAC normal")
    ap.add_argument("--parallel-max-angle-deg", type=float, default=45.0,
                    help="max angle (deg) two walls' normals may be off parallel (compared "
                         "as undirected lines, since RANSAC's normal sign is arbitrary) to "
                         "still be treated as a facing pair and aligned onto one shared axis")
    ap.add_argument("--parallel-min-separation-m", type=float, default=0.3,
                    help="min distance (m) between two parallel walls for them to count as "
                         "a facing pair rather than two fragments of the same surface")
    ap.add_argument("--no-orthogonal", action="store_true",
                    help="skip snapping every plane onto the building's own orthogonal "
                         "frame (derived from the data, not the world axes) -- default "
                         "forces every wall exactly perpendicular to the roof and to any "
                         "non-parallel wall (e.g. an end wall), on top of the parallel "
                         "opposite-wall alignment")
    ap.add_argument("--no-collapse-sides", action="store_true",
                    help="with --orthogonal (default): after snapping, keep every "
                         "fragment sharing an exact canonical direction as its own plane "
                         "instead of collapsing to the single largest per direction")
    ap.add_argument("--close-geometry", action="store_true",
                    help="collapse to a single watertight box: keep only the largest plane "
                         "per side per axis and stretch each to the shared bounding box, so "
                         "edges meet and the result is importable to OpenStudio as a shoebox. "
                         "A whole missing side first gets a targeted retry fit (see "
                         "--close-retry-*) before being left open")
    ap.add_argument("--close-retry-margin", type=float, default=0.5,
                    help="--close-geometry: thickness (m) of the slab searched at an open "
                         "boundary when retrying a fit to close it")
    ap.add_argument("--close-retry-min-inliers", type=int, default=150,
                    help="--close-geometry: relaxed --min-inliers used only for the retry fit")
    ap.add_argument("--cap-open-faces", action="store_true",
                    help="--close-geometry: close any remaining open face with a synthetic "
                         "surface so the solid is watertight and importable to OpenStudio. "
                         "These carry synthetic=true / boundary_condition=adiabatic in the JSON "
                         "-- there is no LiDAR evidence of a wall there (it is a corridor cut or "
                         "the rover's entry), they exist only to seal the volume")
    ap.add_argument("--close-retry-min-density-ratio", type=float, default=0.4,
                    help="--close-geometry: reject a retry fit whose point density (inliers/area) "
                         "is below this fraction of the confirmed real walls' density -- a door/"
                         "entrance/corridor-continuation can span as much area as a real wall but "
                         "won't be nearly as densely covered (default 0.4)")
    ap.add_argument("--out", type=Path, default=Path("planes.json"))
    args = ap.parse_args()

    if bool(args.bag) == bool(args.pcd):
        raise SystemExit("pass exactly one of --bag or --pcd")

    if args.pcd:
        xyz = load_pcd_cloud(args.pcd)
        source = args.pcd
    else:
        xyz = load_merged_cloud(args.bag, args.topic, args.store)
        source = args.bag

    roi = None
    if args.roi:
        roi = tuple(float(x) for x in args.roi.split(","))
        if len(roi) != 6:
            raise SystemExit("--roi needs 6 comma-separated values: xmin,xmax,ymin,ymax,zmin,zmax")
        xyz = crop_roi(xyz, roi)

    if args.sor:
        xyz = statistical_outlier_removal(xyz, k=args.sor_k, std_ratio=args.sor_std)
        print(f"after SOR: {len(xyz)} points")

    if args.declutter:
        xyz = declutter(xyz, gap=args.cluster_gap,
                        min_size=args.min_cluster, keep_dist=args.cluster_dist)

    if args.voxel_preview > 0:
        keys = np.floor(xyz / args.voxel_preview).astype(np.int64)
        n_voxels = len(np.unique(keys, axis=0))
        print(f"voxel preview @ {args.voxel_preview}m: {n_voxels} occupied voxels "
              f"(QA count only; RANSAC still runs on all {len(xyz)} points)")

    print(f"RANSAC input: {len(xyz)} points")
    planes = segment_planes(xyz, args.distance_threshold, args.ransac_n,
                             args.num_iterations, args.min_inliers, args.max_planes,
                             rect_cluster_gap=args.rect_cluster_gap,
                             max_tilt_deg=args.max_tilt_deg, max_attempts=args.max_attempts,
                             align_to_structure=not args.free_rotation, snap_axis=args.snap_axis)

    if not args.no_dedupe:
        planes = dedupe_planes(planes, normal_deg=args.dedupe_normal_deg,
                                offset_m=args.dedupe_offset_m)

    planes = classify_horizontal(planes, xyz)

    if not args.no_parallel_walls:
        planes = enforce_parallel_walls(planes, max_pair_angle_deg=args.parallel_max_angle_deg,
                                         min_separation_m=args.parallel_min_separation_m)

    if not args.no_orthogonal:
        planes = enforce_orthogonal_planes(planes)
        if not args.no_collapse_sides:
            planes = collapse_duplicate_sides(planes, offset_m=args.dedupe_offset_m)

    if args.close_geometry:
        planes = close_geometry(planes, xyz=xyz, distance_threshold=args.distance_threshold,
                                 ransac_n=args.ransac_n, num_iterations=args.num_iterations,
                                 retry_min_inliers=args.close_retry_min_inliers,
                                 retry_margin=args.close_retry_margin,
                                 retry_min_density_ratio=args.close_retry_min_density_ratio,
                                 rect_cluster_gap=args.rect_cluster_gap,
                                 align_to_structure=not args.free_rotation,
                                 snap_axis=args.snap_axis, roi=roi,
                                 cap_open_faces=args.cap_open_faces)

    if not args.keep_floor:
        before = len(planes)
        planes = [p for p in planes if p["orientation"] != "floor"]
        if len(planes) != before:
            print(f"dropped {before - len(planes)} floor plane(s) (--keep-floor to keep)")
        for new_id, p in enumerate(planes):
            p["id"] = new_id

    for p in planes:
        p.pop("_inlier_pts", None)

    args.out.write_text(json.dumps(
        {"bag": str(source), "topic": args.topic if args.bag else None, "planes": planes}, indent=2))
    print(f"wrote {len(planes)} plane(s) to {args.out}")


if __name__ == "__main__":
    main()
