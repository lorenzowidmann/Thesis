"""Attach a majority-vote material label to each voxel of AlignedOctree's
voxel grid.

Categorical counterpart of TemperatureToVoxel/temperature_to_voxel.py --
same reprojection/z-buffer pipeline (reused: range prefilter,
project_lidar_to_camera, z_buffer_mask), but a mean doesn't make sense for a
material label, so this votes instead, at two levels:
  1. Per-point: across every pose that saw a point, tally which material its
     FLIR pixel (after the z-buffer) resolved to; the point's label is
     whichever material got the most votes (majority vote).
  2. Per-voxel: among a voxel's points' (already point-level-majority)
     labels, again take the majority -- same reasoning, material is still
     categorical at this level too.

Pipeline (mirrors temperature_to_voxel.py's structure):
  1. Load voxels.npz + transform.json (AlignedOctree's output; not
     regenerated here).
  2. Load the filtered bag's single merged point cloud ONCE -- raw,
     unaligned points, SLAM/map (camera_init) frame.
  3. Load sync_manifest.json's triplets.
  4. Build the material vocabulary: one pass over every triplet's
     segments.json (--material-map-dir), collecting the distinct
     `top_material` strings seen, in first-seen order -- fixes each
     material's integer id before the heavy per-pose loop, so votes can be a
     plain (n_points, n_materials) int array instead of a per-point dict.
  5. For each triplet, same range prefilter + project_lidar_to_camera +
     z_buffer_mask as temperature_to_voxel.py, then:
       a. Load that frame's segment_id.npy (EmissivityCalculation/
          project_to_flir.py's output, FLIR pixel grid, already nearest-fill
          complete -- see project_to_flir.py's docstring); sample it at the
          surviving points' pixel coordinates.
       b. Resolve each sampled segment id to a material via that SAME
          frame's segments.json (id -> top_material -> vocabulary index);
          drop pixels whose segment id has no record in this frame's
          segments.json (stale-pairing guard, excluded from this pose's
          contribution only, same "excluded, not fatal" pattern as
          temperature_to_voxel.py's NaN-drop).
       c. votes[point_index, material_id] += 1 for every point that survived.
  6. Per-point majority material = argmax over the material axis (ties break
     to the vocabulary's earliest-discovered material -- numpy argmax's
     first-max convention).
  7. Apply transform.json's rigid transform to the WHOLE raw cloud, same as
     temperature_to_voxel.py.
  8. Bin the aligned points into voxels.npz's EXISTING grid (same
     origin/voxel_size/depth -> integer voxel index convention as
     octree/voxelizer.py); each voxel's material = majority vote among its
     points' per-point majority labels; `material_confidence` = the winning
     label's share of that voxel's material-labeled points (same idea as
     material_map_consensus/segments.json's own per-segment "agreement"
     field, one level up: at the voxel instead of the FLIR-segment).
  9. Write voxels_material.npz.

--material-map-dir default (confirmed with the user, not inferable from
temperature_to_voxel.py -- it never reads segments.json at all):
<sync-dir>/material_map_consensus, EmissivityCalculation/voxel_consensus.py's
multi-view-voted materials. Chosen because temperature_to_voxel.py's own
corrected_temperature_consensus.npy was itself produced from these same
consensus materials (RadiometricCalibration/correct_session.py
--material-map-dir material_map_consensus), so material and temperature
voxels stay mutually consistent. NOT material_map/ (classify_session.py's
default SLIC run, 69 segments on the sample frame -- a different segment-id
space than segment_id.npy's 0-27 range, so it wouldn't even resolve) or
material_map_sam/ (same 28-segment SAM id space as material_map_consensus,
but each frame's raw single-view CLIP call, pre-vote).

Usage:
    python material_to_voxel.py
        [--bag <rosbag2_folder>] [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--sync-dir <fullrate_session_dir>] [--material-map-dir <material_map_consensus_dir>]
        [--voxels ../AlignedOctree/voxels.npz] [--transform ../AlignedOctree/transform.json]
        [--calibration rig_calibration.yaml]
        [--range-max-m 20] [--zbuffer-tol-m 0.08] [--min-depth-m 0.05]
        [--out voxels_material.npz]

Venv: C:\\venvs\\planefit (same as TemperatureToVoxel -- needs opencv-python
for projection.py's cv2.projectPoints, plus pyyaml for rig_calibration.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from projection import project_lidar_to_camera
from rig_calibration import load_rig_calibration

HERE = Path(__file__).resolve().parent

DEFAULT_BAG = Path(
    r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20"
    r"\rosbag2_2026_07_30-18_12_20_filtered"
)
DEFAULT_SYNC_DIR = Path(r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate")
DEFAULT_MATERIAL_MAP_DIR = DEFAULT_SYNC_DIR / "material_map_consensus"
DEFAULT_VOXELS = HERE.parent / "AlignedOctree" / "voxels.npz"
DEFAULT_TRANSFORM = HERE.parent / "AlignedOctree" / "transform.json"
DEFAULT_CALIBRATION = HERE / "rig_calibration.yaml"

# Same constants/values as temperature_to_voxel.py (Piano1_CorridoioLungo.m
# section 9's zBufferTol_m / rangeMax_m -- there's no material-specific
# reason to diverge, this is the same reprojection geometry).
MIN_DEPTH_M = 0.05
RANGE_MAX_M = 20.0
ZBUFFER_TOL_M = 0.08
SEGMENT_ID_NAME = "segment_id.npy"       # EmissivityCalculation/project_to_flir.py output
SEGMENTS_JSON_NAME = "segments.json"     # EmissivityCalculation/classify_session.py / voxel_consensus.py output


# ---------------------------------------------------------------------------
# Point cloud loading -- copied from temperature_to_voxel.py (itself copied
# from AlignedOctree/fit_closed_planes.py), same convention: raw, unaligned
# bag points in the SLAM/map frame. This folder is self-contained, no
# cross-import.
# ---------------------------------------------------------------------------

def read_pointcloud2(msg):
    step = msg.point_step
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


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
    return xyz.astype(np.float64)


# ---------------------------------------------------------------------------
# Z-buffer -- copied verbatim from temperature_to_voxel.py.
# ---------------------------------------------------------------------------

def z_buffer_mask(u, v, z, width, height, tol):
    """Nearest point per pixel wins, but any point within `tol` of that
    pixel's true minimum depth also survives (not a strict single winner)."""
    n = len(u)
    if n == 0:
        return np.zeros(0, dtype=bool)
    uu = np.clip(np.round(u).astype(np.int64), 0, width - 1)
    vv = np.clip(np.round(v).astype(np.int64), 0, height - 1)
    bin_idx = vv * width + uu

    min_z = np.full(width * height, np.inf)
    np.minimum.at(min_z, bin_idx, z)  # scatter-min, correct under duplicate indices
    return z <= min_z[bin_idx] + tol


# ---------------------------------------------------------------------------
# Material vocabulary + per-pose vote accumulation.
# ---------------------------------------------------------------------------

def build_material_vocabulary(triplets, material_map_dir, segments_json_name=SEGMENTS_JSON_NAME):
    """First pass: read every triplet's segments.json once, collecting the
    distinct `top_material` strings seen, in first-seen order -- fixes each
    material's integer id (list index) before the heavy reprojection loop, so
    accumulate_material_votes can use a plain (n_points, n_materials) int
    array instead of a per-point dict/Counter."""
    materials = []
    seen = set()
    n_missing = 0
    for tr in triplets:
        stem = Path(tr["flir"]["file"]).stem
        seg_path = material_map_dir / stem / segments_json_name
        if not seg_path.exists():
            n_missing += 1
            continue
        seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
        for seg in seg_data["segments"]:
            m = seg["top_material"]
            if m not in seen:
                seen.add(m)
                materials.append(m)
    print(f"material vocabulary: {len(materials)} material(s) across "
          f"{len(triplets) - n_missing}/{len(triplets)} triplet(s) with a "
          f"{segments_json_name}: {materials}")
    return materials


def accumulate_material_votes(points_world, triplets, cal, sync_dir, material_map_dir, material_to_id,
                               range_max_m=RANGE_MAX_M, zbuffer_tol_m=ZBUFFER_TOL_M,
                               min_depth_m=MIN_DEPTH_M, segment_id_name=SEGMENT_ID_NAME,
                               segments_json_name=SEGMENTS_JSON_NAME):
    """votes[i, j] = how many poses' surviving FLIR pixel resolved point i's
    segment to material j. Categorical counterpart of temperature_to_voxel.py's
    sum_temp/cnt_temp -- a mean doesn't make sense for a label, so this counts
    instead, one column per vocabulary material, indexed by the SAME original
    point index into `points_world` (not a per-pose subset index)."""
    n = len(points_world)
    n_materials = len(material_to_id)
    votes = np.zeros((n, n_materials), dtype=np.int32)

    emissivity_map_dir = sync_dir / "emissivity_map"
    n_skipped_no_segment_id = 0
    n_skipped_no_segments_json = 0
    n_used = 0

    for i, tr in enumerate(triplets):
        pos = np.asarray(tr["lidar"]["position"], dtype=np.float64)
        quat = np.asarray(tr["lidar"]["orientation"], dtype=np.float64)  # xyzw

        # coarse range prefilter, world frame, before any transform (speed only)
        d2 = np.sum((points_world - pos) ** 2, axis=1)
        near_idx = np.where(d2 <= range_max_m ** 2)[0]
        if len(near_idx) == 0:
            continue

        uv, depth, valid = project_lidar_to_camera(
            points_world[near_idx], pos, quat, cal.T_lidar_to_flir,
            cal.flir.K, cal.flir.dist, cal.flir.width, cal.flir.height)
        valid = valid & (depth > min_depth_m)
        if not valid.any():
            continue

        u, v, z = uv[valid, 0], uv[valid, 1], depth[valid]

        keep = z_buffer_mask(u, v, z, cal.flir.width, cal.flir.height, zbuffer_tol_m)
        if not keep.any():
            continue

        flir_stem = Path(tr["flir"]["file"]).stem
        seg_id_path = emissivity_map_dir / flir_stem / segment_id_name
        if not seg_id_path.exists():
            n_skipped_no_segment_id += 1
            continue
        seg_json_path = material_map_dir / flir_stem / segments_json_name
        if not seg_json_path.exists():
            n_skipped_no_segments_json += 1
            continue

        segment_id_map = np.load(seg_id_path)  # (H, W) int32, project_to_flir.py's output
        seg_data = json.loads(seg_json_path.read_text(encoding="utf-8"))

        # id -> material_id lookup array, sized to this frame's max segment
        # id (segments.json's own "id" field -- the SAM/SLIC label value,
        # not a running index into the segments list).
        max_id = max((s["id"] for s in seg_data["segments"]), default=-1)
        id_lookup = np.full(max_id + 1, -1, dtype=np.int64)
        for s in seg_data["segments"]:
            mat_id = material_to_id.get(s["top_material"])
            if mat_id is not None:
                id_lookup[s["id"]] = mat_id

        uf = np.clip(np.round(u[keep]).astype(np.int64), 0, cal.flir.width - 1)
        vf = np.clip(np.round(v[keep]).astype(np.int64), 0, cal.flir.height - 1)
        seg_ids = segment_id_map[vf, uf]

        # guard against a segment id project_to_flir.py wrote that this
        # frame's segments.json has no record for (stale pairing) --
        # dropped from this pose's contribution only, same as
        # temperature_to_voxel.py's NaN-drop on the temperature sample.
        in_range = (seg_ids >= 0) & (seg_ids <= max_id)
        material_ids = np.full(len(seg_ids), -1, dtype=np.int64)
        material_ids[in_range] = id_lookup[seg_ids[in_range]]
        ok = material_ids >= 0
        if not ok.any():
            continue

        global_idx = near_idx[valid][keep][ok]
        np.add.at(votes, (global_idx, material_ids[ok]), 1)
        n_used += 1

        if (i + 1) % 20 == 0 or i + 1 == len(triplets):
            covered = int((votes.sum(axis=1) > 0).sum())
            print(f"  pose {i + 1}/{len(triplets)}, points covered so far: {covered}")

    print(f"triplets used: {n_used}/{len(triplets)}, "
          f"skipped (no {segment_id_name}): {n_skipped_no_segment_id}/{len(triplets)}, "
          f"skipped (no {segments_json_name}): {n_skipped_no_segments_json}/{len(triplets)}")
    return votes


def per_point_majority_material(votes):
    """Majority vote per point across all poses. -1 where the point never
    got a single vote. Ties (equal top counts) break to whichever material
    was discovered earliest building the vocabulary -- numpy argmax's
    first-max convention; only matters on an exact tie."""
    total = votes.sum(axis=1)
    has_votes = total > 0
    point_material_id = np.full(len(votes), -1, dtype=np.int64)
    point_material_id[has_votes] = np.argmax(votes[has_votes], axis=1)
    print(f"points with >=1 material vote: {int(has_votes.sum())} / {len(votes)} "
          f"({100 * has_votes.mean():.1f}%)")
    return point_material_id


# ---------------------------------------------------------------------------
# Voxel binning -- same origin/voxel_size/depth -> integer index convention
# and pack/searchsorted matching trick as temperature_to_voxel.py, copied
# rather than imported (self-contained, no cross-import).
# ---------------------------------------------------------------------------

def voxel_index(points, origin, voxel_size, depth):
    """idx = floor((points - origin) / voxel_size), reproducing whichever of
    voxelizer.py's two lattices built voxels.npz (see
    temperature_to_voxel.py's voxel_index for the depth>=0 octree-clipping
    branch's rationale)."""
    idx = np.floor((points - origin) / voxel_size).astype(np.int64)
    if depth >= 0:
        n = 2 ** int(depth)
        np.clip(idx, 0, n - 1, out=idx)
    return idx


def voxel_centers_to_index(centers, origin, voxel_size):
    """Invert voxelizer.py's `center = (idx + 0.5) * voxel_size + origin`."""
    return np.round((centers - origin) / voxel_size - 0.5).astype(np.int64)


def bin_material_into_voxels(aligned_points, point_material_id, vox_centers, origin, voxel_size, depth, n_materials):
    """Per-voxel majority vote among ITS points' (already point-level-
    majority) material labels -- a second vote, one level up: material stays
    categorical here too, so a mean still doesn't apply.
    `material_confidence` = the winning label's share of that voxel's
    material-labeled points -- same idea as material_map_consensus/
    segments.json's own per-segment "agreement" field, just computed at the
    voxel level instead of the FLIR-segment level."""
    has_material = point_material_id >= 0
    pt_idx = voxel_index(aligned_points[has_material], origin, voxel_size, depth)
    vox_idx = voxel_centers_to_index(vox_centers, origin, voxel_size)

    combined_lo = np.minimum(pt_idx.min(axis=0) if len(pt_idx) else vox_idx.min(axis=0), vox_idx.min(axis=0))
    combined_hi = np.maximum(pt_idx.max(axis=0) if len(pt_idx) else vox_idx.max(axis=0), vox_idx.max(axis=0))
    ranges = combined_hi - combined_lo + 1

    def pack(idx):
        s = idx - combined_lo
        return (s[:, 0].astype(np.int64)
                + s[:, 1].astype(np.int64) * ranges[0]
                + s[:, 2].astype(np.int64) * ranges[0] * ranges[1])

    vox_key = pack(vox_idx)
    order = np.argsort(vox_key)
    sorted_key = vox_key[order]

    pt_key = pack(pt_idx)
    pos = np.searchsorted(sorted_key, pt_key)
    pos = np.clip(pos, 0, len(sorted_key) - 1)
    found = sorted_key[pos] == pt_key
    n_unmatched = int((~found).sum())
    if n_unmatched:
        print(f"WARNING: {n_unmatched} material-valid point(s) didn't land in any "
              f"existing voxels.npz voxel (expected 0 -- check --voxels/--transform "
              f"match the same alignment run as the point cloud used here)")

    voxel_row = order[pos[found]]
    mat_ids = point_material_id[has_material][found]

    m = len(vox_centers)
    histogram = np.zeros((m, n_materials), dtype=np.int64)
    np.add.at(histogram, (voxel_row, mat_ids), 1)

    n_per_voxel = histogram.sum(axis=1)
    has_mat = n_per_voxel > 0
    voxel_material_id = np.full(m, -1, dtype=np.int64)
    voxel_material_id[has_mat] = np.argmax(histogram[has_mat], axis=1)
    voxel_confidence = np.full(m, np.nan, dtype=np.float64)
    voxel_confidence[has_mat] = (histogram[has_mat, voxel_material_id[has_mat]]
                                  / n_per_voxel[has_mat])

    print(f"voxels with a valid majority material: {int(has_mat.sum())} / {m} "
          f"({100 * has_mat.mean():.1f}%)")
    return voxel_material_id, voxel_confidence, n_per_voxel


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=DEFAULT_BAG, help="rosbag2 folder")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--sync-dir", type=Path, default=DEFAULT_SYNC_DIR,
                    help="folder containing sync_manifest.json and emissivity_map/ "
                         "(segment_id.npy, EmissivityCalculation/project_to_flir.py's output)")
    ap.add_argument("--material-map-dir", type=Path, default=DEFAULT_MATERIAL_MAP_DIR,
                    help="folder containing <flir_stem>/segments.json (default: "
                         "<sync-dir>/material_map_consensus -- multi-view-voted materials, "
                         "consistent with temperature_to_voxel.py's "
                         "corrected_temperature_consensus.npy, itself built from these; "
                         "see the module docstring for why NOT material_map/ or material_map_sam/)")
    ap.add_argument("--voxels", type=Path, default=DEFAULT_VOXELS,
                    help="AlignedOctree's voxels.npz (not regenerated here)")
    ap.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM,
                    help="AlignedOctree's transform.json (not regenerated here)")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                    help="rig_calibration.yaml")
    ap.add_argument("--range-max-m", type=float, default=RANGE_MAX_M,
                    help="coarse world-frame prefilter radius around each pose (speed only)")
    ap.add_argument("--zbuffer-tol-m", type=float, default=ZBUFFER_TOL_M,
                    help="z-buffer tolerance, same semantics as temperature_to_voxel.py")
    ap.add_argument("--min-depth-m", type=float, default=MIN_DEPTH_M,
                    help="minimum camera-frame forward depth to accept a projection")
    ap.add_argument("--out", type=Path, default=Path("voxels_material.npz"))
    args = ap.parse_args()

    cal = load_rig_calibration(args.calibration)

    voxels = np.load(args.voxels)
    vox_centers = voxels["centers"]
    vox_counts = voxels["counts"]
    voxel_size = float(voxels["voxel_size"])
    origin = voxels["origin"]
    depth = int(voxels["depth"])
    print(f"loaded {args.voxels}: {len(vox_centers)} voxels, voxel_size={voxel_size:.4f} m")

    transform = json.loads(args.transform.read_text())
    R = np.array(transform["rotation"])
    t = np.array(transform["translation"])

    points_world = load_merged_cloud(args.bag, args.topic, args.store)

    sync_manifest = json.loads((args.sync_dir / "sync_manifest.json").read_text())
    triplets = sync_manifest["triplets"]
    print(f"loaded {len(triplets)} triplet(s) from {args.sync_dir / 'sync_manifest.json'}")

    materials = build_material_vocabulary(triplets, args.material_map_dir)
    if not materials:
        raise SystemExit(f"no material found under {args.material_map_dir} -- check --material-map-dir")
    material_to_id = {m: i for i, m in enumerate(materials)}

    print("--- accumulating per-point material votes over all poses ---")
    votes = accumulate_material_votes(
        points_world, triplets, cal, args.sync_dir, args.material_map_dir, material_to_id,
        range_max_m=args.range_max_m, zbuffer_tol_m=args.zbuffer_tol_m, min_depth_m=args.min_depth_m)
    point_material_id = per_point_majority_material(votes)

    # aligned = points @ R.T + t (same convention as aligned_octree.py's
    # apply_rigid), applied to the WHOLE cloud, not just points with a
    # valid material.
    aligned_points = points_world @ R.T + t

    voxel_material_id, voxel_confidence, n_material_votes = bin_material_into_voxels(
        aligned_points, point_material_id, vox_centers, origin, voxel_size, depth, len(materials))

    np.savez(
        args.out,
        centers=vox_centers,
        counts=vox_counts,
        material_id=voxel_material_id.astype(np.int32),
        material_confidence=voxel_confidence,
        n_material_votes=n_material_votes.astype(np.int32),
        materials=np.array(materials),
        voxel_size=voxel_size,
        origin=origin,
        depth=depth,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
