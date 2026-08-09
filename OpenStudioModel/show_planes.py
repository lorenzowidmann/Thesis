"""QA viewer for fit_planes.py output: colors the cloud by fitted plane and
overlays each plane's rectangle, rendered with PyVista (same style as
PointCloudElaboration/PointCloudView/view_pointcloud.py). Off-screen by
default -- saves a PNG. --interactive opens a real, rotatable window instead
(blocks until you close it).

Usage:
    python show_planes.py --bag <rosbag2_folder> --planes planes.json
        [--topic /cloud_registered] [--roi xmin,xmax,ymin,ymax,zmin,zmax]
        [--sor] [--sor-k 16] [--sor-std 1.5]
        [--voxel-display 0.03] [--label-threshold 0.05]
        [--planes-only] [--interactive] [--out planes_view.png]

--planes-only hides the point cloud and shows just the fitted rectangles as
filled, colored patches (skips loading the bag entirely -- faster).

Venv: C:\\venvs\\planefit (same as fit_planes.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from fit_planes import crop_roi, declutter, load_merged_cloud, statistical_outlier_removal


def assign_labels(xyz, planes, threshold):
    """Label each point with the id of the nearest plane, if within
    `threshold` metres of it (else -1 = unassigned)."""
    labels = np.full(len(xyz), -1, dtype=np.int32)
    best_dist = np.full(len(xyz), np.inf)
    for p in planes:
        n = np.array(p["normal"])
        dist = np.abs(xyz @ n + p["d"]) / np.linalg.norm(n)
        mask = (dist < threshold) & (dist < best_dist)
        labels[mask] = p["id"]
        best_dist[mask] = dist[mask]
    return labels


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE")
    ap.add_argument("--roi", help="xmin,xmax,ymin,ymax,zmin,zmax crop (metres)")
    ap.add_argument("--sor", action="store_true")
    ap.add_argument("--sor-k", type=int, default=16)
    ap.add_argument("--sor-std", type=float, default=1.5)
    ap.add_argument("--declutter", action="store_true",
                    help="match fit_planes.py's --declutter so the displayed cloud "
                         "lines up with what RANSAC actually saw")
    ap.add_argument("--cluster-gap", type=float, default=0.30)
    ap.add_argument("--min-cluster", type=int, default=0)
    ap.add_argument("--cluster-dist", type=float, default=0.0)
    ap.add_argument("--planes", type=Path, default=Path("planes.json"))
    ap.add_argument("--label-threshold", type=float, default=0.05,
                    help="max distance (m) for a point to be colored by its nearest plane")
    ap.add_argument("--voxel-display", type=float, default=0.03,
                    help="downsample just for render speed (0 = off); does not affect planes.json")
    ap.add_argument("--planes-only", action="store_true",
                    help="hide the point cloud, show only the fitted rectangles as filled "
                         "colored patches (skips loading the bag)")
    ap.add_argument("--interactive", action="store_true",
                    help="open a real rotatable window instead of saving a PNG "
                         "(blocks until you close it)")
    ap.add_argument("--show-synthetic", action="store_true",
                    help="also draw the synthetic cap faces (--cap-open-faces in fit_planes.py). "
                         "Hidden by default: OpenStudio does not need a fully closed solid, and "
                         "these faces are not measured surfaces")
    ap.add_argument("--no-labels", action="store_true",
                    help="drop the per-plane text labels (cleaner for figures)")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="zoom factor applied after framing (>1 tighter, <1 wider)")
    ap.add_argument("--pad", type=float, default=0.2,
                    help="margin added around the frame, as a fraction of each axis's extent "
                         "(0.2 = 20%% wider on every side) -- use this to 'zoom out', not --zoom, "
                         "since it keeps the box centered instead of just rescaling")
    ap.add_argument("--frame-bounds",
                    help="xmin,xmax,ymin,ymax,zmin,zmax to frame the camera on, overriding the "
                         "default (plane bounds, padded by --pad) -- pass the bounds printed by "
                         "one render to another so both match exactly")
    ap.add_argument("--out", type=Path, default=Path("planes_view.png"))
    args = ap.parse_args()

    data = json.loads(args.planes.read_text())
    planes = data["planes"]
    if not args.show_synthetic:
        real = [pl for pl in planes if not pl.get("synthetic")]
        if len(real) != len(planes):
            print(f"hiding {len(planes) - len(real)} synthetic cap face(s) "
                  f"(--show-synthetic to draw them)")
        planes = real
    n_planes = max(len(planes), 1)

    p = pv.Plotter(off_screen=not args.interactive, window_size=(1600, 1000))

    if args.planes_only:
        print(f"planes-only: skipping bag load, rendering {len(planes)} plane(s)")
        pts, faces, scalars = [], [], []
        for i, plane in enumerate(planes):
            corners = np.array(plane["corners_3d"])
            pts.append(corners)
            faces.append([4, 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
            scalars.extend([plane["id"]] * 4)
        if planes:
            mesh = pv.PolyData(np.vstack(pts), np.hstack(faces))
            mesh["plane"] = scalars
            p.add_mesh(mesh, scalars="plane", cmap="tab20", clim=(0, max(n_planes - 1, 1)),
                       show_edges=True, edge_color="black", line_width=2,
                       opacity=0.85, show_scalar_bar=False)
    else:
        xyz = load_merged_cloud(args.bag, args.topic, args.store)
        if args.roi:
            roi = tuple(float(x) for x in args.roi.split(","))
            xyz = crop_roi(xyz, roi)
        if args.sor:
            xyz = statistical_outlier_removal(xyz, k=args.sor_k, std_ratio=args.sor_std)

        if args.declutter:
            xyz = declutter(xyz, gap=args.cluster_gap,
                            min_size=args.min_cluster, keep_dist=args.cluster_dist)

        if args.voxel_display > 0:
            keys = np.floor(xyz / args.voxel_display).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            xyz = xyz[idx]
            print(f"display downsample @ {args.voxel_display}m: {len(xyz)} points")

        labels = assign_labels(xyz, planes, threshold=args.label_threshold)
        print(f"labeled {int((labels >= 0).sum())} / {len(xyz)} points to {len(planes)} plane(s)")

        unassigned = xyz[labels < 0]
        if len(unassigned):
            p.add_mesh(pv.PolyData(unassigned), color="#cccccc", point_size=2,
                       render_points_as_spheres=False)

        assigned_mask = labels >= 0
        if assigned_mask.any():
            cloud = pv.PolyData(xyz[assigned_mask])
            cloud["plane"] = labels[assigned_mask]
            p.add_mesh(cloud, scalars="plane", cmap="tab20", clim=(0, max(n_planes - 1, 1)),
                       point_size=3, render_points_as_spheres=False, show_scalar_bar=False)

        for plane in planes:
            corners = np.array(plane["corners_3d"])
            loop = np.vstack([corners, corners[0]])
            p.add_lines(loop, color="black", width=3, connected=True)

    if planes and not args.no_labels:
        centers = np.array([pl["center_3d"] for pl in planes])
        tags = [f"{pl['id']}: {pl.get('orientation', '?')} {pl.get('tilt_deg', float('nan')):.0f}deg"
                + (" [synthetic]" if pl.get("synthetic") else "")
                for pl in planes]
        p.add_point_labels(centers, tags, point_size=1, font_size=14,
                           text_color="black", shape_opacity=0.6, always_visible=True)

    p.set_background("white")
    p.camera_position = "iso"
    # Frame explicitly instead of relying on "iso"'s auto-fit, which fits to whatever's in
    # the scene -- a stray, loosely-connected part of the building can survive --declutter
    # and blow the cloud's bounding box out far past the corridor, so framing on raw cloud
    # extent decenters the box rather than centering it. Plane bounds (padded by --pad) are
    # stable across --planes-only vs normal mode and keep the box centered either way.
    if args.frame_bounds:
        bounds = tuple(float(x) for x in args.frame_bounds.split(","))
    elif planes:
        allc = np.array([c for pl in planes for c in pl["corners_3d"]])
        lo, hi = allc.min(axis=0), allc.max(axis=0)
        margin = (hi - lo) * args.pad
        lo, hi = lo - margin, hi + margin
        bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    else:
        bounds = None
    if bounds is not None:
        print(f"camera framed on bounds {','.join(f'{b:.3f}' for b in bounds)}")
        p.reset_camera(bounds=bounds)
    if args.zoom != 1.0:
        p.camera.zoom(args.zoom)
    if args.interactive:
        print("opening interactive window -- close it to exit")
        p.show()
    else:
        p.show(screenshot=str(args.out))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
