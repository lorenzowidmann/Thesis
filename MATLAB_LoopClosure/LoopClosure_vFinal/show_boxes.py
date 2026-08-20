"""QA viewer for fit_planes.py output: colors the cloud by fitted plane and
overlays each plane's rectangle, rendered with PyVista (same style as
PointCloudElaboration/PointCloudView/view_pointcloud.py). Off-screen by
default -- saves a PNG. --interactive opens a real, rotatable window instead
(blocks until you close it).

--live opens that same interactive window but keeps it open and polls
--planes on disk (default every 1s): whenever fit_planes.py rewrites it
(with new RANSAC/dedupe/parallel-wall parameters), the rectangles, labels,
and per-point coloring are cleared and redrawn in place -- the point cloud
itself is loaded once, not on every poll, and the camera isn't reset -- so
tuning fit_planes.py's flags is: edit params, re-run fit_planes.py in
another terminal, watch this window update. Ctrl+C in the terminal (or
closing the window) exits.

Usage:
    python show_planes.py --bag <rosbag2_folder> --planes planes.json
        [--topic /cloud_registered] [--roi xmin,xmax,ymin,ymax,zmin,zmax]
        [--sor] [--sor-k 16] [--sor-std 1.5]
        [--voxel-display 0.03] [--label-threshold 0.05]
        [--planes-only] [--interactive] [--live] [--live-interval-ms 1000]
        [--out planes_view.png]

--planes-only hides the point cloud and shows just the fitted rectangles as
filled, colored patches (skips loading the bag entirely -- faster, and the
natural pairing with --live since there's no cloud to load up front).

Venv: C:\\venvs\\planefit (same as fit_planes.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from fit_planes import crop_roi, declutter, load_merged_cloud, load_pcd_cloud, statistical_outlier_removal


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


def load_planes(path, show_synthetic):
    data = json.loads(path.read_text())
    planes = data["planes"]
    if not show_synthetic:
        real = [pl for pl in planes if not pl.get("synthetic")]
        if len(real) != len(planes):
            print(f"hiding {len(planes) - len(real)} synthetic cap face(s) "
                  f"(--show-synthetic to draw them)")
        planes = real
    return planes


def draw_scene(p, planes, xyz, args):
    """(Re)draw everything that depends on `planes` (and, if a point cloud
    was loaded, the per-point coloring derived from it): plane rectangles,
    labels, north arrow, and the labeled/unlabeled point cloud. Removes
    every actor added by a previous call first, so this can be called
    repeatedly (--live) without piling up stale geometry or resetting the
    camera. Returns the list of actors added, for the next call to remove."""
    actors = []
    n_planes = max(len(planes), 1)

    if xyz is None:
        if planes:
            pts, faces, scalars = [], [], []
            for i, plane in enumerate(planes):
                corners = np.array(plane["corners_3d"])
                pts.append(corners)
                faces.append([4, 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
                scalars.extend([plane["id"]] * 4)
            mesh = pv.PolyData(np.vstack(pts), np.hstack(faces))
            mesh["plane"] = scalars
            actors.append(p.add_mesh(mesh, scalars="plane", cmap="tab20",
                                      clim=(0, max(n_planes - 1, 1)),
                                      show_edges=True, edge_color="black", line_width=2,
                                      opacity=0.85, show_scalar_bar=False))
    else:
        labels = assign_labels(xyz, planes, threshold=args.label_threshold)
        print(f"labeled {int((labels >= 0).sum())} / {len(xyz)} points to {len(planes)} plane(s)")

        unassigned = xyz[labels < 0]
        if len(unassigned):
            actors.append(p.add_mesh(pv.PolyData(unassigned), color="#cccccc",
                                      point_size=2, render_points_as_spheres=False))

        assigned_mask = labels >= 0
        if assigned_mask.any():
            cloud = pv.PolyData(xyz[assigned_mask])
            cloud["plane"] = labels[assigned_mask]
            actors.append(p.add_mesh(cloud, scalars="plane", cmap="tab20",
                                      clim=(0, max(n_planes - 1, 1)), point_size=3,
                                      render_points_as_spheres=False, show_scalar_bar=False))

        for plane in planes:
            corners = np.array(plane["corners_3d"])
            loop = np.vstack([corners, corners[0]])
            actors.append(p.add_lines(loop, color="black", width=3, connected=True))

    if args.north_offset_deg is not None and planes:
        allc = np.array([c for pl in planes for c in pl["corners_3d"]])
        lo, hi = allc.min(axis=0), allc.max(axis=0)
        diag = float(np.linalg.norm(hi - lo))
        theta = np.radians((-args.north_offset_deg) % 360.0)
        north_dir = np.array([np.sin(theta), np.cos(theta), 0.0])  # local bearing -> XY vector
        arrow_len = diag * 0.15
        origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + diag * 0.05])
        arrow = pv.Arrow(start=origin, direction=north_dir, scale=arrow_len)
        actors.append(p.add_mesh(arrow, color="red"))
        actors.append(p.add_point_labels([origin + north_dir * arrow_len * 1.2], ["N"],
                                          font_size=28, text_color="red", shape_opacity=0.0,
                                          always_visible=True, show_points=False))

    if planes and not args.no_labels:
        centers = np.array([pl["center_3d"] for pl in planes])
        tags = [f"{pl['id']}: {pl.get('orientation', '?')} {pl.get('tilt_deg', float('nan')):.0f}deg"
                + (" [synthetic]" if pl.get("synthetic") else "")
                for pl in planes]
        actors.append(p.add_point_labels(centers, tags, point_size=1, font_size=14,
                                          text_color="black", shape_opacity=0.6,
                                          always_visible=True))
    return actors


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, help="rosbag2 folder (raw /cloud_registered)")
    ap.add_argument("--pcd", type=Path,
                    help="load the displayed cloud from a .pcd instead of a bag -- e.g. "
                         "loop_closed_map.pcd, matching fit_planes.py --pcd. Mutually "
                         "exclusive with --bag; --topic/--store are ignored with --pcd")
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
    ap.add_argument("--live", action="store_true",
                    help="keep the window open and auto-redraw whenever --planes changes "
                         "on disk (poll every --live-interval-ms) -- for tuning "
                         "fit_planes.py's parameters without relaunching this viewer or "
                         "reloading the bag each time. Implies --interactive; ignores --out")
    ap.add_argument("--live-interval-ms", type=int, default=1000,
                    help="--live: poll interval in milliseconds")
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
    ap.add_argument("--north-offset-deg", type=float, default=None,
                    help="draw a red 'N' arrow above the scene, using sun_incidence.py's "
                         "convention: true compass bearing of the local +Y axis. Local north "
                         "direction = bearing (-north_offset_deg mod 360) in the local XY plane. "
                         "Purely a QA overlay to sanity-check the offset against a known map/photo "
                         "-- not written to any output.")
    ap.add_argument("--top-view", action="store_true",
                    help="camera straight down (plan view) instead of iso -- directly comparable "
                         "to a north-up satellite/map screenshot, use with --north-offset-deg")
    ap.add_argument("--north-up", action="store_true",
                    help="with --top-view and --north-offset-deg: rotate the camera so true "
                         "north (not local +Y) renders vertical -- the corridor then appears "
                         "tilted by the offset, exactly like it does in a north-up satellite photo")
    ap.add_argument("--out", type=Path, default=Path("planes_view.png"))
    args = ap.parse_args()

    if args.live:
        args.interactive = True
    if args.north_up and args.north_offset_deg is None:
        raise SystemExit("--north-up needs --north-offset-deg")

    planes = load_planes(args.planes, args.show_synthetic)

    p = pv.Plotter(off_screen=not args.interactive, window_size=(1600, 1000))

    if args.bag and args.pcd:
        raise SystemExit("pass at most one of --bag or --pcd")

    xyz = None
    if not args.planes_only:
        if args.pcd:
            xyz = load_pcd_cloud(args.pcd)
        elif args.bag:
            xyz = load_merged_cloud(args.bag, args.topic, args.store)
        else:
            raise SystemExit("--bag or --pcd is required unless --planes-only")
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
    else:
        print(f"planes-only: skipping bag load, rendering {len(planes)} plane(s)")

    actors = draw_scene(p, planes, xyz, args)

    p.set_background("white")
    # Orthographic instead of the pyvista default perspective camera: under perspective, a
    # plane viewed close to edge-on (unavoidable for some face in an elongated scene, see
    # below) can render as a thin sliver that appears to float away from where it actually
    # is -- parallel projection keeps every edge's screen position exactly proportional to
    # its real coordinates, standard practice for CAD/architectural elevations.
    p.enable_parallel_projection()

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

    if args.top_view:
        p.camera_position = "xy"
        if args.north_up:
            theta = np.radians((-args.north_offset_deg) % 360.0)
            north_dir = np.array([np.sin(theta), np.cos(theta), 0.0])
            p.camera.up = tuple(north_dir)  # true north vertical -- corridor renders tilted
        else:
            p.camera.up = (0.0, 1.0, 0.0)  # local +Y up, matching the north-arrow convention
    elif bounds is not None:
        # A plain "iso" camera treats X/Y/Z symmetrically -- fine for a roughly cubic
        # scene, but a long corridor (one axis several times longer than the other two)
        # then gets viewed almost end-on, foreshortening its longest wall into a sliver
        # that LOOKS non-perpendicular to the other faces even though it isn't (the
        # planes.json geometry is exactly orthogonal; this was purely a viewing-angle
        # artifact). Bias the view direction away from the longest axis so the camera
        # looks across the corridor instead of down its length.
        lo = np.array(bounds[0::2])
        hi = np.array(bounds[1::2])
        center = (lo + hi) / 2
        extent = hi - lo
        diag = float(np.linalg.norm(extent))
        long_axis = int(np.argmax(extent))
        weights = np.array([1.0, 1.0, 1.0])
        weights[long_axis] = 0.35
        direction = weights / np.linalg.norm(weights)
        up = np.array([0.0, 0.0, 1.0])
        if abs(direction[2]) > 0.9:  # looking near straight down/up -- pick a usable up vector
            up = np.array([0.0, 1.0, 0.0])
        cam_pos = center + direction * diag
        p.camera_position = [tuple(cam_pos), tuple(center), tuple(up)]
    else:
        p.camera_position = "iso"

    if bounds is not None:
        print(f"camera framed on bounds {','.join(f'{b:.3f}' for b in bounds)}")
        p.reset_camera(bounds=bounds)
    if args.zoom != 1.0:
        p.camera.zoom(args.zoom)

    if args.live:
        state = {"actors": actors, "mtime": args.planes.stat().st_mtime if args.planes.exists() else None}

        def refresh(*_):
            try:
                mtime = args.planes.stat().st_mtime
            except FileNotFoundError:
                return
            if mtime == state["mtime"]:
                return
            state["mtime"] = mtime
            try:
                new_planes = load_planes(args.planes, args.show_synthetic)
            except (json.JSONDecodeError, KeyError):
                print(f"{args.planes} mid-write or malformed, skipping this poll")
                return
            for a in state["actors"]:
                p.remove_actor(a, render=False)
            state["actors"] = draw_scene(p, new_planes, xyz, args)
            p.render()
            print(f"--live: {args.planes} changed, redrew {len(new_planes)} plane(s)")

        print(f"--live: watching {args.planes} every {args.live_interval_ms}ms -- "
              f"re-run fit_planes.py to update this window, close it to exit")
        p.add_timer_event(max_steps=10_000_000, duration=args.live_interval_ms, callback=refresh)
        p.show()
    elif args.interactive:
        print("opening interactive window -- close it to exit")
        p.show()
    else:
        p.show(screenshot=str(args.out))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
