"""Attach a corrected mean temperature to each voxel of AlignedOctree's
voxel grid.

Ports the working logic in MATLAB_PointCloudVisualization/Piano1_CorridoioLungo.m
section 9 ("Colorazione per temperatura FLIR") to Python, feeding
AlignedOctree/voxels.npz's existing voxel grid instead of a fresh point
cloud + a fresh independent pooling voxel size.

Pipeline (mirrors the MATLAB script's structure):
  1. Load voxels.npz + transform.json (AlignedOctree's output; not
     regenerated here).
  2. Load the filtered bag's single merged point cloud ONCE -- these are
     RAW, unaligned points, still in the SLAM/map (camera_init) frame, same
     as MATLAB's `xyzFilt = pc.Location` (the accumulated map cloud, read
     once before the per-pose loop starts, not re-fetched per pose).
  3. Load sync_manifest.json's triplets.
  4. For each triplet, reproject that SAME loaded cloud (not a per-timestamp
     bag re-fetch, unlike EmissivityCalculation/project_to_flir.py):
       a. Coarse range prefilter (rangeMax_m, default 20 m): Euclidean
          distance in WORLD frame from the triplet's lidar position, applied
          BEFORE any transform -- a pure speed optimization, matches
          Piano1_CorridoioLungo.m's `d2 = sum((xyzFilt - t_wb').^2, 2)`.
       b. Project into the FLIR camera frame via projection.py's
          project_lidar_to_camera (world -> LiDAR-local -> FLIR-local ->
          pixels+distortion) -- this already implements the same pinhole +
          2-term-radial/2-term-tangential model as MATLAB's hand-rolled
          projectPinholeTemp (FLIR's k3 = 0, so cv2.projectPoints' 5-param
          model and MATLAB's 4-param formula are numerically identical), so
          no separate reimplementation was written; only the extra
          MIN_DEPTH_M = 0.05 m cutoff project_lidar_to_camera doesn't apply
          (it only checks depth > 0) is added here to match
          projectPinholeTemp's stricter `z > 0.05`.
       c. z_buffer_mask (ported fresh -- projection.py has no equivalent):
          nearest point per pixel wins, but any point within
          ZBUFFER_TOL_M (0.08 m, same as MATLAB's zBufferTol_m) of that
          pixel's true minimum depth also survives -- not a strict
          single-winner z-buffer. This is what keeps a point on an occluded
          surface from stealing a visible surface's pixel/temperature.
       d. Load that frame's corrected_temperature_consensus.npy (skip the
          triplet -- not an error -- if it doesn't exist for that frame,
          matching the MATLAB script's `continue`); sample it at the
          surviving points' pixel coordinates (nearest-neighbour, same as
          MATLAB's `round(u)`/`round(v)`); drop NaN samples (excluded from
          THIS pose's contribution only).
       e. Accumulate a running sum + count of temperature per RAW point
          index across all poses that validly observed it (sumTemp/cntTemp
          in the MATLAB script).
  5. Per-point temperature = sum / count (NaN where zero observations).
  6. Apply transform.json's rigid transform to the WHOLE raw cloud (not just
     the points with a valid temperature), same as aligned_octree.py.
  7. Bin the aligned points into voxels.npz's EXISTING grid (same
     origin/voxel_size/depth -> integer voxel index convention as
     octree/voxelizer.py); mean_temperature per voxel = mean of that
     voxel's points' per-point temperatures (points with no valid
     temperature don't contribute).
  8. Write voxels_temperature.npz.

No solar-artifact correction here -- corrected_temperature_consensus.npy is
already the radiometrically calibrated, multi-view-consensus-material
result, used as-is.

Usage:
    python temperature_to_voxel.py
        [--bag <rosbag2_folder>] [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--sync-dir <fullrate_session_dir>]
        [--voxels ../AlignedOctree/voxels.npz] [--transform ../AlignedOctree/transform.json]
        [--calibration rig_calibration.yaml]
        [--range-max-m 20] [--zbuffer-tol-m 0.08] [--min-depth-m 0.05]
        [--out voxels_temperature.npz]

Venv: C:\\venvs\\planefit (same as AlignedOctree -- needs opencv-python for
projection.py's cv2.projectPoints, plus pyyaml for rig_calibration.py).
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
DEFAULT_VOXELS = HERE.parent / "AlignedOctree" / "voxels.npz"
DEFAULT_TRANSFORM = HERE.parent / "AlignedOctree" / "transform.json"
DEFAULT_CALIBRATION = HERE / "rig_calibration.yaml"

# Piano1_CorridoioLungo.m section 9 constants (same names/values/comments).
MIN_DEPTH_M = 0.05      # projectPinholeTemp: z > 0.05
RANGE_MAX_M = 20.0      # rangeMax_m: coarse prefilter, speed only
ZBUFFER_TOL_M = 0.08    # zBufferTol_m
CORRECTED_NAME = "corrected_temperature_consensus.npy"


# ---------------------------------------------------------------------------
# Point cloud loading -- same load_merged_cloud/read_pointcloud2 as
# fit_closed_planes.py (AlignedOctree), copied here rather than imported
# (this folder is self-contained, no cross-import, same convention).
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
# Z-buffer: ports Piano1_CorridoioLungo.m's zBufferMaskTemp (no equivalent
# in projection.py).
# ---------------------------------------------------------------------------

def z_buffer_mask(u, v, z, width, height, tol):
    """Nearest point per pixel wins, but any point within `tol` of that
    pixel's true minimum depth also survives (not a strict single winner) --
    a thin/grazing surface can legitimately put several points a few cm
    apart in depth into the same pixel. Points elsewhere in depth (behind
    an occluder) are dropped for *this* pose only; they may still
    contribute correctly from a pose that sees them unoccluded.

    u, v, z are 1-D arrays of equal length (already pixel-valid, i.e. from
    project_lidar_to_camera's `valid` mask). Returns a bool mask of the same
    length.
    """
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
# Per-pose accumulation loop.
# ---------------------------------------------------------------------------

def accumulate_temperatures(points_world, triplets, cal, sync_dir,
                             range_max_m=RANGE_MAX_M, zbuffer_tol_m=ZBUFFER_TOL_M,
                             min_depth_m=MIN_DEPTH_M, corrected_name=CORRECTED_NAME):
    """sumTemp/cntTemp idiom from Piano1_CorridoioLungo.m, indexed by
    original index into `points_world` (not a per-pose subset index)."""
    n = len(points_world)
    sum_temp = np.zeros(n, dtype=np.float64)
    cnt_temp = np.zeros(n, dtype=np.int32)

    emissivity_map_dir = sync_dir / "emissivity_map"
    n_skipped_no_npy = 0
    n_used = 0

    for i, tr in enumerate(triplets):
        pos = np.asarray(tr["lidar"]["position"], dtype=np.float64)
        quat = np.asarray(tr["lidar"]["orientation"], dtype=np.float64)  # xyzw

        # a. coarse range prefilter, world frame, before any transform
        d2 = np.sum((points_world - pos) ** 2, axis=1)
        near_idx = np.where(d2 <= range_max_m ** 2)[0]
        if len(near_idx) == 0:
            continue

        # b. project into the FLIR camera frame (world -> lidar-local -> flir-local -> pixels)
        uv, depth, valid = project_lidar_to_camera(
            points_world[near_idx], pos, quat, cal.T_lidar_to_flir,
            cal.flir.K, cal.flir.dist, cal.flir.width, cal.flir.height)
        valid = valid & (depth > min_depth_m)  # projectPinholeTemp's stricter z>0.05
        if not valid.any():
            continue

        u, v, z = uv[valid, 0], uv[valid, 1], depth[valid]

        # c. z-buffer: nearest-within-tolerance per pixel
        keep = z_buffer_mask(u, v, z, cal.flir.width, cal.flir.height, zbuffer_tol_m)
        if not keep.any():
            continue

        # d. load + sample this frame's corrected temperature map
        flir_stem = Path(tr["flir"]["file"]).stem
        npy_path = emissivity_map_dir / flir_stem / corrected_name
        if not npy_path.exists():
            n_skipped_no_npy += 1
            continue
        temp_map = np.load(npy_path)  # (H, W) float32, NaN = no plausible material

        uf = np.clip(np.round(u[keep]).astype(np.int64), 0, cal.flir.width - 1)
        vf = np.clip(np.round(v[keep]).astype(np.int64), 0, cal.flir.height - 1)
        vals = temp_map[vf, uf].astype(np.float64)

        global_idx = near_idx[valid][keep]
        ok = np.isfinite(vals)  # exclude NaN from this pose's contribution only
        global_idx, vals = global_idx[ok], vals[ok]

        sum_temp[global_idx] += vals
        cnt_temp[global_idx] += 1
        n_used += 1

        if (i + 1) % 20 == 0 or i + 1 == len(triplets):
            covered = int((cnt_temp > 0).sum())
            print(f"  pose {i + 1}/{len(triplets)}, points covered so far: {covered}")

    print(f"triplets used: {n_used}/{len(triplets)}, "
          f"skipped (no {corrected_name}): {n_skipped_no_npy}/{len(triplets)}")
    return sum_temp, cnt_temp


def per_point_temperature(sum_temp, cnt_temp):
    has_obs = cnt_temp > 0
    temperature = np.full(len(sum_temp), np.nan, dtype=np.float64)
    temperature[has_obs] = sum_temp[has_obs] / cnt_temp[has_obs]
    print(f"points with valid temperature (>=1 pose): {int(has_obs.sum())} / "
          f"{len(sum_temp)} ({100 * has_obs.mean():.1f}%)")
    if has_obs.any():
        print(f"observations per covered point: mean={cnt_temp[has_obs].mean():.1f}  "
              f"max={int(cnt_temp.max())}")
    return temperature


# ---------------------------------------------------------------------------
# Voxel binning -- same origin/voxel_size/depth -> integer index convention
# as octree/voxelizer.py's voxelize()/voxelize_octree() (AlignedOctree),
# reimplemented directly here rather than importing the whole module: this
# script only ever needs to map points onto an *existing* grid, not build a
# new one, so voxelize()/voxelize_octree()/VoxelGrid/filter_by_count would
# be dead weight.
# ---------------------------------------------------------------------------

def voxel_index(points, origin, voxel_size, depth):
    """idx = floor((points - origin) / voxel_size), replicating whichever of
    voxelizer.py's two lattices built voxels.npz:
    - depth == -1 (sentinel, see aligned_octree.py --voxel-size): the plain
      uniform grid, voxelize() -- no clipping.
    - depth >= 0: the power-of-two octree lattice, voxelize_octree() -- far
      boundary points folded into the last cell via clip(idx, 0, 2**depth-1).
    Reproducing the exact same lattice (not just "a" lattice of the same
    voxel_size) is what makes a point here land in the SAME voxel a
    co-located point did when voxels.npz was built."""
    idx = np.floor((points - origin) / voxel_size).astype(np.int64)
    if depth >= 0:
        n = 2 ** int(depth)
        np.clip(idx, 0, n - 1, out=idx)
    return idx


def voxel_centers_to_index(centers, origin, voxel_size):
    """Invert voxelizer.py's `center = (idx + 0.5) * voxel_size + origin`."""
    return np.round((centers - origin) / voxel_size - 0.5).astype(np.int64)


def bin_temperatures_into_voxels(aligned_points, per_point_temp, vox_centers, origin, voxel_size, depth):
    """mean_temperature / n_temp_samples per voxel of the EXISTING grid
    (vox_centers, from voxels.npz) -- only points with a non-NaN
    per_point_temp contribute."""
    has_temp = np.isfinite(per_point_temp)
    pt_idx = voxel_index(aligned_points[has_temp], origin, voxel_size, depth)
    vox_idx = voxel_centers_to_index(vox_centers, origin, voxel_size)

    # Pack (i,j,k) into a single int64 key (same trick as
    # octree/voxelizer.py's _grid_from_index) so matching is a 1-D sort +
    # searchsorted instead of a per-point Python dict lookup -- an order of
    # magnitude faster with ~1e6 points.
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
        print(f"WARNING: {n_unmatched} temperature-valid point(s) didn't land in any "
              f"existing voxels.npz voxel (expected 0 -- check --voxels/--transform "
              f"match the same alignment run as the point cloud used here)")

    voxel_row = order[pos[found]]
    temps = per_point_temp[has_temp][found]

    m = len(vox_centers)
    sum_per_voxel = np.bincount(voxel_row, weights=temps, minlength=m)
    n_per_voxel = np.bincount(voxel_row, minlength=m)
    mean_temperature = np.full(m, np.nan, dtype=np.float64)
    has_mean = n_per_voxel > 0
    mean_temperature[has_mean] = sum_per_voxel[has_mean] / n_per_voxel[has_mean]

    print(f"voxels with valid mean_temperature: {int(has_mean.sum())} / {m} "
          f"({100 * has_mean.mean():.1f}%)")
    return mean_temperature, n_per_voxel


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=DEFAULT_BAG, help="rosbag2 folder")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--sync-dir", type=Path, default=DEFAULT_SYNC_DIR,
                    help="folder containing sync_manifest.json and emissivity_map/")
    ap.add_argument("--voxels", type=Path, default=DEFAULT_VOXELS,
                    help="AlignedOctree's voxels.npz (not regenerated here)")
    ap.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM,
                    help="AlignedOctree's transform.json (not regenerated here)")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                    help="rig_calibration.yaml")
    ap.add_argument("--range-max-m", type=float, default=RANGE_MAX_M,
                    help="coarse world-frame prefilter radius around each pose (speed only)")
    ap.add_argument("--zbuffer-tol-m", type=float, default=ZBUFFER_TOL_M,
                    help="z-buffer tolerance: points within this many metres of a "
                         "pixel's true minimum depth all survive (not a strict "
                         "single winner)")
    ap.add_argument("--min-depth-m", type=float, default=MIN_DEPTH_M,
                    help="minimum camera-frame forward depth to accept a projection")
    ap.add_argument("--out", type=Path, default=Path("voxels_temperature.npz"))
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

    print("--- accumulating per-point temperature over all poses ---")
    sum_temp, cnt_temp = accumulate_temperatures(
        points_world, triplets, cal, args.sync_dir,
        range_max_m=args.range_max_m, zbuffer_tol_m=args.zbuffer_tol_m,
        min_depth_m=args.min_depth_m)
    per_point_temp = per_point_temperature(sum_temp, cnt_temp)

    # aligned = points @ R.T + t (same convention as aligned_octree.py's
    # apply_rigid), applied to the WHOLE cloud, not just points with a
    # valid temperature.
    aligned_points = points_world @ R.T + t

    mean_temperature, n_temp_samples = bin_temperatures_into_voxels(
        aligned_points, per_point_temp, vox_centers, origin, voxel_size, depth)

    np.savez(
        args.out,
        centers=vox_centers,
        counts=vox_counts,
        mean_temperature=mean_temperature,
        n_temp_samples=n_temp_samples,
        voxel_size=voxel_size,
        origin=origin,
        depth=depth,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
