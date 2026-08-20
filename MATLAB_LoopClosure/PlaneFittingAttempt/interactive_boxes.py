"""Interactive pyvista box-resize tool: draggable/rotatable 3D box widgets
directly over the point cloud, so you place regions by hand instead of
typing --roi numbers -- drag a handle to resize, drag an edge to rotate
(free rotation, NOT constrained to 90 deg -- a box can follow a diagonal
stretch of hallway), press a key to add as many more boxes as you want.

Because a rotated box can't be described by fit_planes.py's axis-aligned
--roi, closing the window crops the ORIGINAL point cloud to each box's
exact oriented volume (not just its axis-aligned bounding box) and saves
each one as its own <name>.pcd -- feed that straight into fit_planes.py
--pcd, no --roi needed. The axis-aligned bounding box is also printed for
reference (exact if you left a box unrotated, loose otherwise).

Usage:
    python interactive_boxes.py (--bag <rosbag2_folder> | --pcd <file.pcd>)
        [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--sor] [--declutter]
        --box "name:xmin,xmax,ymin,ymax,zmin,zmax" [--box ... repeatable]
        [--no-rotation] [--handle-size 0.006] [--point-size 2]
        [--vmin -1] [--vmax 3]
        [--out-prefix box]

--box seeds the starting boxes (e.g. paste the --roi lines auto_regions.py
printed, prefixed "name:"). You don't need any to start empty-handed --
press 'a' once the window is open to drop a first one.

While the window is open:
    drag a face/corner handle     resize that box
    drag an edge handle           rotate that box (disabled with
                                   --no-rotation, stays axis-aligned)
    drag the middle               move the whole box
    'a' key                       add a new box (default size, centered
                                   on the current camera focal point) --
                                   for a second, third, ... hallway branch
    'p' key                       print every box's current state without
                                   closing the window
    close the window               finalize: crop + save each box's
                                   oriented selection to <out-prefix><n>.pcd

Venv: C:\\venvs\\planefit (same as fit_planes.py; needs pyvista + open3d).
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista as pv

from fit_planes import crop_roi, declutter, load_merged_cloud, load_pcd_cloud, statistical_outlier_removal

PALETTE = ["red", "lime", "deepskyblue", "yellow", "magenta", "orange", "cyan", "white",
           "gold", "hotpink"]


def parse_boxes(box_args):
    boxes = {}
    for spec in box_args:
        name, coords = spec.split(":", 1)
        boxes[name] = tuple(float(v) for v in coords.split(","))
    return boxes


def obb_from_widget_poly(box_poly, z_range=None):
    """The widget's PolyData always carries corners 0-7 (a consistent
    corner order regardless of rotation: 0-3 one face, 4-7 the opposite
    face, edge 0->1/0->3/0->4 along the box's own local axes) plus the
    exact center as its 15th point -- so the box's orientation and extent
    can be read directly off it without touching GetTransform (which reads
    back as an unplaced identity matrix off-screen/right after creation,
    unreliable to depend on)."""
    pts = np.asarray(box_poly.points)
    center = pts[14].copy()
    ex, ey, ez = pts[1] - pts[0], pts[3] - pts[0], pts[4] - pts[0]
    lx, ly, lz = np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez)
    ux = ex / lx if lx > 1e-9 else ex
    uy = ey / ly if ly > 1e-9 else ey
    uz = ez / lz if lz > 1e-9 else ez

    if z_range is not None:
        # VTK's box widget has no clean way to lock just one axis mid-drag, so instead
        # the Z the user ends up with (if they dragged it at all -- the handles for that
        # are the least convenient ones to grab, so mostly they won't) is simply ignored
        # here: the box's "up" is pinned to world Z and its vertical extent to z_range,
        # keeping only the X/Y axes and footprint the widget actually reports. That's the
        # right modeling assumption too -- corridors are level, only the XY yaw matters.
        z0, z1 = z_range
        uz = np.array([0.0, 0.0, 1.0])
        lz = z1 - z0
        center[2] = (z0 + z1) / 2
    return center, (ux, uy, uz), (lx, ly, lz)


def obb_mask(xyz, box_poly, z_range=None):
    center, (ux, uy, uz), (lx, ly, lz) = obb_from_widget_poly(box_poly, z_range=z_range)
    rel = xyz - center
    px, py, pz = rel @ ux, rel @ uy, rel @ uz
    return (np.abs(px) <= lx / 2) & (np.abs(py) <= ly / 2) & (np.abs(pz) <= lz / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, help="rosbag2 folder (raw /cloud_registered)")
    ap.add_argument("--pcd", type=Path, help="load xyz from a .pcd instead of a bag")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE")
    ap.add_argument("--sor", action="store_true")
    ap.add_argument("--declutter", action="store_true")
    ap.add_argument("--box", action="append", default=[],
                    help="name:xmin,xmax,ymin,ymax,zmin,zmax -- repeatable, one per "
                         "starting box; optional, press 'a' in-window to add more")
    ap.add_argument("--no-rotation", action="store_true",
                    help="keep boxes strictly axis-aligned (old behaviour) instead of "
                         "letting them rotate freely to follow a diagonal hallway")
    ap.add_argument("--z-base", type=float, default=0.0,
                    help="fixed Z for every box's base -- boxes never move up/down, only "
                         "resize/rotate in X/Y (VTK's box widget can't cleanly lock just "
                         "one drag axis, so any Z you do drag it to is simply ignored: the "
                         "saved crop always uses --z-base/--z-top, not whatever the widget "
                         "shows)")
    ap.add_argument("--z-top", type=float, default=None,
                    help="fixed Z for every box's top (default: the cloud's own max Z, "
                         "i.e. floor to ceiling)")
    ap.add_argument("--handle-size", type=float, default=0.006,
                    help="widget handle size, as a fraction of that box's own diagonal "
                         "-- lower if the drag handles look oversized on a long thin box")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--out-prefix", default="box",
                    help="each box is saved as <out-prefix><n>.pcd, e.g. box0.pcd")
    args = ap.parse_args()

    if bool(args.bag) == bool(args.pcd):
        raise SystemExit("pass exactly one of --bag or --pcd")

    xyz = load_pcd_cloud(args.pcd) if args.pcd else load_merged_cloud(args.bag, args.topic, args.store)
    if args.sor:
        xyz = statistical_outlier_removal(xyz)
    if args.declutter:
        xyz = declutter(xyz, gap=0.30)

    z_top = args.z_top if args.z_top is not None else float(xyz[:, 2].max())
    z_range = (args.z_base, z_top)
    print(f"boxes locked to Z {z_range[0]:.2f}..{z_range[1]:.2f} (--z-base/--z-top)")

    p = pv.Plotter(off_screen=False, window_size=(1600, 1000))
    p.enable_parallel_projection()  # avoid the perspective flare seen on tall thin boxes

    cloud = pv.PolyData(xyz)
    cloud["z"] = xyz[:, 2]
    vmin = args.vmin if args.vmin is not None else float(xyz[:, 2].min())
    vmax = args.vmax if args.vmax is not None else float(xyz[:, 2].max())
    p.add_mesh(cloud, scalars="z", cmap="turbo", clim=(vmin, vmax),
               point_size=args.point_size, render_points_as_spheres=False,
               scalar_bar_args={"title": "Z (m)"})
    p.set_background("white")
    p.camera_position = "iso"

    current = {}   # name -> latest box_poly (pv.PolyData) from that widget's callback
    counter = [0]  # next box index, for auto-generated names/colors

    def make_callback(name):
        def callback(box_poly):
            current[name] = box_poly.copy()
        return callback

    def add_box(name=None, bounds=None):
        if name is None:
            name = f"{args.out_prefix}{counter[0]}"
        color = PALETTE[counter[0] % len(PALETTE)]
        counter[0] += 1
        if bounds is None:
            fx, fy, _ = p.camera.focal_point
            half = 3.0
            bounds = (fx - half, fx + half, fy - half, fy + half, z_range[0], z_range[1])
        widget = p.add_box_widget(callback=make_callback(name), bounds=bounds, factor=1.0,
                                   rotation_enabled=not args.no_rotation, color=color)
        widget.SetHandleSize(args.handle_size)
        # add_box_widget already fires the callback once immediately on creation (see its
        # source: `_the_callback(box_widget, None)` right after the observer is registered),
        # so `current[name]` is populated here with no extra nudge needed.
        p.add_point_labels([bounds[0::2]], [name], font_size=16, text_color=color,
                           shape_opacity=0.6, always_visible=True)
        print(f"added {name}")

    for name, bounds in parse_boxes(args.box).items():
        add_box(name=name, bounds=bounds)

    def print_current():
        print("\ncurrent box state (Z always shown locked to --z-base/--z-top, "
              f"{z_range[0]:.2f}..{z_range[1]:.2f}, regardless of the widget's own Z):")
        for name, box_poly in current.items():
            xmin, xmax, ymin, ymax, _, _ = box_poly.bounds
            print(f'  {name}: AABB --roi="{xmin:.2f},{xmax:.2f},{ymin:.2f},{ymax:.2f},'
                  f'{z_range[0]:.2f},{z_range[1]:.2f}" (X/Y loose if rotated -- close the '
                  f"window to save the exact oriented crop per box)")

    p.add_key_event("a", add_box)
    p.add_key_event("p", print_current)
    p.add_text("'a' add box -- drag handles to resize -- drag an edge to rotate -- "
               "'p' print -- close window to finish", font_size=9, color="black")

    print(f"{len(current)} starting box(es), opening window -- 'a' to add more, drag to "
          f"adjust/rotate, 'p' to print, close window to finish")
    p.show()

    if not current:
        print("no boxes were placed, nothing to save")
        return

    print("\nfinal boxes:")
    for name, box_poly in current.items():
        mask = obb_mask(xyz, box_poly, z_range=z_range)
        n = int(mask.sum())
        xmin, xmax, ymin, ymax, _, _ = box_poly.bounds
        print(f'  {name}: {n} points -- AABB --roi="{xmin:.2f},{xmax:.2f},{ymin:.2f},'
              f'{ymax:.2f},{z_range[0]:.2f},{z_range[1]:.2f}"')
        if n == 0:
            print(f"    (empty, skipping .pcd)")
            continue
        out_path = f"{name}.pcd"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[mask].astype(np.float64))
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"    saved {out_path} -- run: fit_planes.py --pcd {out_path} --close-geometry")


if __name__ == "__main__":
    main()
