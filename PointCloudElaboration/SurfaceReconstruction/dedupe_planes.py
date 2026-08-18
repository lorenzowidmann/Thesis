"""Merge near-duplicate plane fragments in a PlaneExtraction .bvg/.ply, before
handing it to PolyFit.

Why this exists: PlaneExtraction's raw RANSAC output splits a single real
surface into several disjoint fragments whenever there's occlusion, clutter,
or (on a long SLAM corridor) accumulated drift bowing one end of a wall out
of the other end's plane -- e.g. 56 "planes" for a corridor that structurally
has a handful of walls plus floor/ceiling. PolyFit's docs say it "assumes the
model is closed and all necessary planes are provided" -- feeding it that
many near-duplicate fragments doesn't just look messy, it makes the
hypothesis+selection MIP dramatically bigger: reconstructing directly against
a 56-plane bag froze past two minutes and 790+ CPU-seconds with no result
(verified by hand); a deduped, few-plane input is what actually makes
PolyFit's optimization tractable, not just tidier.

Algorithm (ported from OpenStudioModel/fit_planes.py's dedupe_planes -- same
greedy largest-absorbs-smaller approach, adapted to work on an already
RANSAC-segmented point cloud instead of a list of fresh RANSAC fits):
    1. For each plane id (from "v:primitive_index"), PCA-fit its own (normal,
       offset) from its member points -- not the noisier per-point normals
       already on the cloud.
    2. Sort ids by point count, largest first. Each still-unclaimed plane
       absorbs every other unclaimed plane whose (normal, offset) is within
       --dedupe-normal-deg / --dedupe-offset-m of it (same physical surface).
    3. Optionally drop merged groups smaller than --min-plane-points
       (demoted to unsegmented, primitive_index=-1) -- stray fragments that
       survived RANSAC's own --min-support but are still too small to be a
       "necessary plane" PolyFit should reconstruct against.
    4. Relabel survivors to a contiguous 0..K-1 range and write "v:
       primitive_index" / "v:primitive_type" back onto the SAME cloud object
       (per-point, via the only property setter Easy3D's Python bindings
       expose -- see module docstring in extract_planes.py's sibling for the
       .vector()-is-a-copy gotcha this works around).

Usage:
    python dedupe_planes.py <bag>_planes.bvg [--output-dir <dir>]
        [--dedupe-normal-deg 10] [--dedupe-offset-m 0.15]
        [--min-plane-points 0]

Venv: C:\\venvs\\surfacereconstruction has easy3d installed already (same
GitHub-Releases-not-PyPI wheel as PlaneExtraction -- see that folder's
README if you need to set this venv up elsewhere).
"""
import argparse
from pathlib import Path

import easy3d
import numpy as np

PLANE = easy3d.PrimitivesRansac.PLANE      # 0
UNKNOWN = easy3d.PrimitivesRansac.UNKNOWN  # -1


def canonical_normal_offset(normal, centroid):
    """Sign-canonicalize a plane fit so two fits of the *same* physical
    plane compare equal regardless of which way PCA's eigenvector happened
    to point: flip both if the largest-magnitude normal component is
    negative. Same convention as fit_planes.py's canonical_normal_offset."""
    n = normal / np.linalg.norm(normal)
    d = -float(np.dot(n, centroid))
    if n[np.argmax(np.abs(n))] < 0:
        n, d = -n, -d
    return n, d


def fit_plane_pca(points):
    """Least-squares plane through `points` (Nx3): centroid + normal from
    the smallest-eigenvalue eigenvector of the covariance matrix."""
    centroid = points.mean(axis=0)
    cov = np.cov((points - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # smallest eigenvalue -> flattest direction
    return canonical_normal_offset(normal, centroid)


def dedupe(ids, counts, normals, offsets, normal_deg, offset_m, log):
    """Greedy largest-first absorption, same logic as fit_planes.py's
    dedupe_planes. Returns {old_id: group_leader_id}."""
    order = sorted(range(len(ids)), key=lambda i: -counts[i])
    used = [False] * len(ids)
    leader_of = {}
    for i in order:
        if used[i]:
            continue
        used[i] = True
        leader_of[ids[i]] = ids[i]
        absorbed = []
        for j in order:
            if used[j]:
                continue
            angle = np.degrees(np.arccos(np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0)))
            if angle < normal_deg and abs(offsets[i] - offsets[j]) < offset_m:
                used[j] = True
                leader_of[ids[j]] = ids[i]
                absorbed.append(ids[j])
        if absorbed:
            log(f"  plane {ids[i]} ({counts[i]} pts) absorbs {absorbed} as the same surface")
    return leader_of


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("planes_file", type=Path, help="*_planes.bvg or *_planes.ply from extract_planes.py")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: same folder as the input file")
    ap.add_argument("--dedupe-normal-deg", type=float, default=10.0,
                    help="max angle (deg) between normals to count as the same plane (default: 10)")
    ap.add_argument("--dedupe-offset-m", type=float, default=0.15,
                    help="max offset difference (m) to count as the same plane (default: 0.15)")
    ap.add_argument("--min-plane-points", type=int, default=0,
                    help="drop merged planes smaller than this (0 = keep all survivors)")
    args = ap.parse_args()

    if not args.planes_file.exists():
        raise SystemExit(f"{args.planes_file} not found")

    output_dir = args.output_dir or args.planes_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.planes_file.stem
    ply_out = output_dir / f"{stem}_dedup.ply"
    bvg_out = output_dir / f"{stem}_dedup.bvg"

    print(f"Loading {args.planes_file}...")
    cloud = easy3d.PointCloudIO.load(str(args.planes_file))
    if cloud is None or cloud.n_vertices() == 0:
        raise SystemExit(f"Easy3D failed to load {args.planes_file}")

    idx_prop = cloud.get_vertex_property("v:primitive_index", int)
    if idx_prop is None:
        raise SystemExit(
            f"{args.planes_file} has no 'v:primitive_index' property -- "
            "is this a *_planes.bvg/.ply from extract_planes.py?")
    type_prop = cloud.get_vertex_property("v:primitive_type", int)

    pts = cloud.to_numpy()
    idx = np.array(idx_prop.vector(), dtype=np.int64)

    ids = sorted(set(idx.tolist()) - {-1})
    if not ids:
        raise SystemExit("No segmented planes in this file (all points are unsegmented)")

    counts, normals, offsets = [], [], []
    for i in ids:
        seg = pts[idx == i]
        counts.append(len(seg))
        n, d = fit_plane_pca(seg)
        normals.append(n)
        offsets.append(d)

    print(f"{len(ids)} plane(s) before dedupe. Merging (normal < {args.dedupe_normal_deg} deg, "
          f"offset < {args.dedupe_offset_m} m)...")
    leader_of = dedupe(ids, counts, normals, offsets,
                        args.dedupe_normal_deg, args.dedupe_offset_m, print)

    # merged point counts per surviving leader
    leader_points = {}
    for i, c in zip(ids, counts):
        leader_points[leader_of[i]] = leader_points.get(leader_of[i], 0) + c

    survivors = [leader for leader, pts_n in leader_points.items() if pts_n >= args.min_plane_points]
    dropped = [leader for leader, pts_n in leader_points.items() if pts_n < args.min_plane_points]
    if dropped:
        print(f"Dropping {len(dropped)} merged plane(s) below --min-plane-points {args.min_plane_points}: "
              f"{[(d, leader_points[d]) for d in dropped]}")
    survivors.sort(key=lambda leader: -leader_points[leader])
    new_id_of = {leader: new_id for new_id, leader in enumerate(survivors)}

    # old plane id -> new (possibly merged, renumbered, or -1 if dropped) id
    remap = {}
    for old_id in ids:
        leader = leader_of[old_id]
        remap[old_id] = new_id_of.get(leader, -1)

    print(f"Writing {len(survivors)} merged plane(s) back onto the cloud...")
    Vertex = easy3d.PointCloud.Vertex
    n_changed = 0
    for i in range(cloud.n_vertices()):
        old = int(idx[i])
        if old == -1:
            continue
        new = remap[old]
        if new != old:
            v = Vertex(i)
            idx_prop[v] = new
            if type_prop is not None:
                type_prop[v] = PLANE if new != -1 else UNKNOWN
            n_changed += 1
    print(f"  {n_changed} point(s) relabeled")

    ok_ply = easy3d.PointCloudIO.save(str(ply_out), cloud)
    ok_bvg = easy3d.PointCloudIO.save(str(bvg_out), cloud)
    if not ok_ply:
        print(f"WARNING: failed to save {ply_out}")
    if not ok_bvg:
        print(f"WARNING: failed to save {bvg_out}")

    print(f"\n{len(ids)} -> {len(survivors)} plane(s)")
    for leader in survivors:
        print(f"  plane {new_id_of[leader]:>3}: {leader_points[leader]:>8} points")

    print(f"\nSaved:")
    print(f"  {ply_out}")
    print(f"  {bvg_out}  (feed this to reconstruct_mesh.py)")


if __name__ == "__main__":
    main()
