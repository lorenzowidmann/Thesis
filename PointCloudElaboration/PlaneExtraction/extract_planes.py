"""RANSAC plane extraction on a rosbag2 point cloud, via Easy3D.

Standalone geometry step: reads the single merged PointCloud2 message off a
rosbag2 (produced e.g. by PointCloudFilterGUI's "Save filtered bag..."),
writes it to an intermediate .ply, then runs Easy3D's PrimitivesRansac (PLANE
only) on it and saves the segmented result as .bvg (PolyFit-ready) and .ply
(for a quick look in CloudCompare/Open3D). The user only ever passes the bag
folder -- the .ply round-trip is internal plumbing, kept on disk under
intermediate/ for debugging/reuse, not something you drive by hand.

No polygon fitting here (that's PolyFit's job on the .bvg output, not yet
wired up) -- this script stops at plane segmentation.

Required, non-obvious step not in the CLI list below: Easy3D's RANSAC detector
refuses to do anything without per-point normals ("RANSAC Detector requires
point cloud normals", 0 primitives, no error raised) and -- worse -- saving
.bvg afterwards on a cloud that skipped normal estimation segfaults the
process outright (verified empirically; PointCloudIO_vg::save_bvg dereferences
the primitive-type property unconditionally, and detect() never created it
because it bailed out before touching the cloud). So normals are ALWAYS
estimated right after loading, via easy3d.PointCloudNormals.estimate() with
the library's own default k=16 (--normal-k to override) -- this is not
optional and always runs, unlike the five RANSAC parameters below.

dist_threshold and bitmap_resolution are fractions of the point cloud's
bounding box, NOT absolute metres (per the C++ doc comments: dist_threshold
relative to the box's max dimension, bitmap_resolution relative to its
width -- both effectively GenericBox::max_range(), the single largest axis
extent). This script prints the box's per-axis extent, max_range, AND
diagonal_length at startup, plus what your current --dist-threshold /
--bitmap-resolution fractions resolve to in metres, so you can sanity-check
before waiting on a run.

Usage:
    python extract_planes.py <bag_folder> --output-dir <dir>
        [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--min-support 1000] [--dist-threshold 0.005]
        [--bitmap-resolution 0.02] [--normal-threshold 0.8]
        [--overlook-probability 0.001] [--normal-k 16]

New pip dependencies -- READ BEFORE INSTALLING:
    rosbags   -- normal PyPI package, `pip install rosbags`.
    open3d    -- normal PyPI package, `pip install open3d`.
    easy3d    -- WARNING: `pip install easy3d` installs THE WRONG PACKAGE. PyPI's
                 "easy3d" (0.1.1) is an unrelated, unaffiliated 3.6 KB stub
                 ("A simple 3D utility package", camera-pose viewer only) --
                 it does NOT contain PointCloudIO/PrimitivesRansac and has
                 nothing to do with github.com/LiangliangNan/Easy3D (the
                 library this script actually needs, cloned into
                 ClaudeCode/Easy3D). Instead, download the matching wheel
                 from that repo's GitHub Releases page and pip-install the
                 .whl directly, e.g. for Python 3.12 / Windows:
                     https://github.com/LiangliangNan/Easy3D/releases
                     -> easy3d-2.6.1-cp312-cp312-win_amd64.whl
                     pip install easy3d-2.6.1-cp312-cp312-win_amd64.whl
                 Match the cpXXX tag to your venv's Python version.

Venv: C:\\venvs\\planeextraction has all three installed (Python 3.12) --
tested end-to-end against the reference bag (see README.md).
"""
import argparse
from pathlib import Path

import easy3d
import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


# --------------------------------------------------------------------------- #
# rosbag2 -> numpy                                                            #
# --------------------------------------------------------------------------- #
def read_pointcloud2(msg):
    # x,y,z as float32 at offsets 0,4,8; slice by point_step to skip any extra
    # fields (this script only ever needs xyz, same as the other PointCloud
    # Elaboration tools) -- matches PointCloudView/view_pointcloud.py.
    step = msg.point_step
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def read_bag(bag_path, topic, store_name):
    """Read+merge every frame on `topic`. Returns (xyz float32 Nx3, frame_id)."""
    typestore = get_typestore(Stores[store_name])
    frames = []
    frame_id = "map"
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Topic {topic!r} not found. Available: {topics}")
        n_msgs = 0
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if n_msgs == 0:
                frame_id = msg.header.frame_id or frame_id
            frames.append(read_pointcloud2(msg))
            n_msgs += 1
    if n_msgs != 1:
        print(f"Note: {n_msgs} messages on {topic} (expected 1 -- merging all of them)")
    xyz = np.vstack(frames) if frames else np.empty((0, 3), dtype=np.float32)
    return xyz, frame_id


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path, help="rosbag2 folder (metadata.yaml + .db3)")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="where to write intermediate/ and the .bvg/.ply outputs "
                         "(default: <bag_folder>_planes, next to the bag)")
    p.add_argument("--topic", default="/cloud_registered")
    p.add_argument("--store", default="ROS2_HUMBLE",
                    help="rosbags typestore, for bags without embedded type defs")

    # RANSAC parameters -- current Easy3D tutorial defaults (Tutorial_703_Cloud_
    # PlaneExtraction) as fallback, see easy3d::PrimitivesRansac::detect().
    p.add_argument("--min-support", type=int, default=1000,
                    help="minimum points required to accept a plane (default: 1000)")
    p.add_argument("--dist-threshold", type=float, default=0.005,
                    help="distance threshold, fraction of bbox max dimension (default: 0.005)")
    p.add_argument("--bitmap-resolution", type=float, default=0.02,
                    help="bitmap resolution, fraction of bbox width (default: 0.02)")
    p.add_argument("--normal-threshold", type=float, default=0.8,
                    help="cosine of the max normal deviation allowed (default: 0.8)")
    p.add_argument("--overlook-probability", type=float, default=0.001,
                    help="probability a primitive is overlooked (default: 0.001)")

    # Prerequisite for RANSAC to do anything at all -- not one of the five
    # tutorial RANSAC parameters, but exposed since it directly affects
    # detection quality. Default matches easy3d::PointCloudNormals::estimate's
    # own C++ default.
    p.add_argument("--normal-k", type=int, default=16,
                    help="neighbours used for per-point normal estimation, "
                         "required before RANSAC can run (default: 16)")
    return p


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = build_parser().parse_args(argv)

    bag_path = args.bag
    if not (bag_path / "metadata.yaml").exists():
        raise SystemExit(f"No metadata.yaml in {bag_path} -- pass the rosbag2 folder, not a file inside it")

    output_dir = args.output_dir or bag_path.parent / f"{bag_path.name}_planes"
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    stem = bag_path.name
    ply_in = intermediate_dir / f"{stem}.ply"
    ply_out = output_dir / f"{stem}_planes.ply"
    bvg_out = output_dir / f"{stem}_planes.bvg"

    # --- 1-2. read bag -> numpy -> intermediate .ply (via Open3D) ---
    print(f"Reading {bag_path} (topic={args.topic})...")
    xyz, frame_id = read_bag(bag_path, args.topic, args.store)
    print(f"{len(xyz)} points, frame_id={frame_id!r}")
    if len(xyz) == 0:
        raise SystemExit("No points read -- check --topic and the bag contents")

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    o3d.io.write_point_cloud(str(ply_in), pc)
    print(f"Wrote intermediate cloud -> {ply_in}")

    # --- 3. load into Easy3D ---
    cloud = easy3d.PointCloudIO.load(str(ply_in))
    if cloud is None or cloud.n_vertices() == 0:
        raise SystemExit(f"Easy3D failed to load {ply_in}")

    bbox = cloud.bounding_box()
    extent = [bbox.max_coord(a) - bbox.min_coord(a) for a in range(3)]
    max_dim = bbox.max_range()
    diag = bbox.diagonal_length()
    print(f"\nBounding box: X={extent[0]:.3f}m  Y={extent[1]:.3f}m  Z={extent[2]:.3f}m")
    print(f"  max dimension : {max_dim:.3f} m  (what --dist-threshold / --bitmap-resolution scale against)")
    print(f"  diagonal      : {diag:.3f} m")
    print(f"  --dist-threshold {args.dist_threshold} -> {args.dist_threshold * max_dim:.4f} m")
    print(f"  --bitmap-resolution {args.bitmap_resolution} -> {args.bitmap_resolution * max_dim:.4f} m\n")

    # --- normals (required -- see module docstring) ---
    print(f"Estimating normals (k={args.normal_k})...")
    if not easy3d.PointCloudNormals.estimate(cloud, args.normal_k, False):
        raise SystemExit("Normal estimation failed -- RANSAC cannot run without it")

    # --- 3-4. RANSAC plane detection ---
    ransac = easy3d.PrimitivesRansac()
    ransac.add_primitive_type(easy3d.PrimitivesRansac.PLANE)
    print("Running RANSAC plane detection...")
    num = ransac.detect(
        cloud,
        args.min_support,
        args.dist_threshold,
        args.bitmap_resolution,
        args.normal_threshold,
        args.overlook_probability,
    )

    # --- 5. save outputs (v:primitive_type / v:primitive_index already set on
    # `cloud` by detect(), regardless of whether any plane was found -- safe
    # to save either way as long as normals were estimated, see docstring) ---
    ok_ply = easy3d.PointCloudIO.save(str(ply_out), cloud)
    ok_bvg = easy3d.PointCloudIO.save(str(bvg_out), cloud)
    if not ok_ply:
        print(f"WARNING: failed to save {ply_out}")
    if not ok_bvg:
        print(f"WARNING: failed to save {bvg_out}")

    # --- 6. summary ---
    planes = sorted(ransac.get_planes(), key=lambda p: len(p.vertices), reverse=True)
    print(f"\n{num} plane(s) found:")
    for p in planes:
        n = p.normal
        print(f"  plane {p.primitive_index:>3}: {len(p.vertices):>8} points   "
              f"normal=({n.x:+.3f}, {n.y:+.3f}, {n.z:+.3f})")
    unsegmented = cloud.n_vertices() - sum(len(p.vertices) for p in planes)
    print(f"  {'(unsegmented)':>10}: {unsegmented:>8} points")

    print(f"\nSaved:")
    print(f"  {ply_out}  (v:primitive_type / v:primitive_index -- CloudCompare/Open3D)")
    print(f"  {bvg_out}  (PolyFit-ready)")


if __name__ == "__main__":
    main()
