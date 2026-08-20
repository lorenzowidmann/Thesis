"""Interactive 2D top-down zone marker: draggable, yaw-rotatable rectangles
over a high-resolution density map, for marking corridor/room regions that
are skewed by residual FAST-LIO drift (a straight hallway that isn't quite
axis-aligned) -- a plain axis-aligned --roi box cuts across those at an
angle, this lets you tilt the box to match instead.

2D on purpose, not the 3D pyvista box widget (see interactive_boxes.py):
only X/Y position, footprint size, and yaw (rotation around the vertical
axis) are meaningful here -- Z is fixed (--z-base/--z-top, floor to
ceiling by default) and there's no reason to let a box tilt out of level,
so a flat top-down view with 2D rectangle handles is both simpler and a
better match for the actual task than a full 3D widget.

Background is a FINE log-density heatmap (--cell-size, much smaller than
auto_regions.py's occupancy grid default) specifically so kinks/rotation
in a corridor's walls -- the visual signature of drift -- are visible
instead of smoothed away.

Usage:
    python mark_zones.py (--bag <rosbag2_folder> | --pcd <file.pcd>)
        [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--sor] [--declutter]
        [--cell-size 0.08] [--mode density|height] [--vmin] [--vmax]
        --box "name:cx,cy,w,h,theta_deg" [--box ... repeatable]
        --box "name:xmin,xmax,ymin,ymax" [--box ... repeatable]   (AABB form also accepted)
        [--z-base 0] [--z-top]
        [--out-prefix box]

While the window is open:
    drag the body              move the box
    drag a corner handle       resize (opposite corner stays put)
    drag the small circle      rotate around the box's own center
    'a' key                    add a new box at the view center
    'd' key                    delete the last-clicked box
    'p' key                    print every box's current state
    close the window           finalize: crop + save each box to <name>.pcd

Venv: C:\\venvs\\planefit (same as fit_planes.py; needs matplotlib + open3d).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Polygon

from fit_planes import crop_roi, declutter, load_merged_cloud, load_pcd_cloud, statistical_outlier_removal

PALETTE = ["red", "lime", "deepskyblue", "yellow", "magenta", "orange", "cyan",
           "gold", "hotpink", "chartreuse"]
HANDLE_PICK_PX = 10       # screen-pixel radius for grabbing a corner/rotate handle
ROTATE_OFFSET_FRAC = 0.25  # rotate handle distance above the top edge, as a fraction of h


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def corner_world(box, sx, sy):
    local = np.array([sx * box["w"] / 2, sy * box["h"] / 2])
    return rot(box["theta"]) @ local + np.array([box["cx"], box["cy"]])


def rotate_handle_world(box):
    off = box["h"] / 2 + max(0.3, ROTATE_OFFSET_FRAC * box["h"])
    return rot(box["theta"]) @ np.array([0.0, off]) + np.array([box["cx"], box["cy"]])


def polygon_points(box):
    return np.array([corner_world(box, sx, sy) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


def load_boxes_json(path):
    """Reload boxes saved by --out -- same {name: {cx,cy,w,h,theta}} shape,
    theta stored in radians (unlike --box's theta_deg, since this is the
    machine-written/read round-trip format, not the hand-typed one)."""
    data = json.loads(Path(path).read_text())
    return {name: dict(v) for name, v in data.items()}


def save_boxes_json(path, boxes):
    Path(path).write_text(json.dumps(boxes, indent=2))
    print(f"saved {path} ({len(boxes)} box(es)) -- reload with --load {path}")


def parse_boxes(box_args):
    """Accepts either 'name:cx,cy,w,h,theta_deg' (5 values) or
    'name:xmin,xmax,ymin,ymax' (4 values, an AABB -- e.g. pasted straight
    from auto_regions.py's printed --roi, dropping the Z pair) -- the AABB
    form seeds an unrotated box."""
    boxes = {}
    for spec in box_args:
        name, coords = spec.split(":", 1)
        vals = [float(v) for v in coords.split(",")]
        if len(vals) == 5:
            cx, cy, w, h, theta_deg = vals
            boxes[name] = {"cx": cx, "cy": cy, "w": w, "h": h, "theta": np.radians(theta_deg)}
        elif len(vals) in (4, 6):
            xmin, xmax, ymin, ymax = vals[:4]
            boxes[name] = {"cx": (xmin + xmax) / 2, "cy": (ymin + ymax) / 2,
                           "w": xmax - xmin, "h": ymax - ymin, "theta": 0.0}
        else:
            raise SystemExit(f"--box {spec!r}: expected 5 values (cx,cy,w,h,theta_deg) "
                              f"or 4/6 (xmin,xmax,ymin,ymax[,zmin,zmax])")
    return boxes


class ZoneMarker:
    def __init__(self, xyz, boxes, cell_size, mode, vmin, vmax, z_range, out_prefix, out_json=None):
        self.xyz = xyz
        self.boxes = boxes
        self.z_range = z_range
        self.out_prefix = out_prefix
        self.out_json = out_json
        self.next_idx = len(boxes)
        self.selected = next(iter(boxes), None)
        self.drag = None  # dict describing the active drag, or None

        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self._draw_background(cell_size, mode, vmin, vmax)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)")
        self.fig.suptitle("drag body=move, corner=resize, circle=rotate -- "
                           "'a' add, 'd' delete, 'p' print -- close window when done")

        self.artists = {}  # name -> dict(poly, corners, rothandle, label)
        for name, box in self.boxes.items():
            self._make_artists(name)
        self._redraw_all()

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _draw_background(self, cell_size, mode, vmin, vmax):
        xyz = self.xyz
        xbins = np.arange(xyz[:, 0].min(), xyz[:, 0].max() + cell_size, cell_size)
        ybins = np.arange(xyz[:, 1].min(), xyz[:, 1].max() + cell_size, cell_size)
        if mode == "density":
            self.ax.hist2d(xyz[:, 0], xyz[:, 1], bins=[xbins, ybins], cmap="gray_r", norm=LogNorm())
        else:
            sumZ, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=[xbins, ybins], weights=xyz[:, 2])
            cnt, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=[xbins, ybins])
            with np.errstate(invalid="ignore"):
                meanZ = np.ma.masked_invalid(sumZ / cnt)
            pcm = self.ax.pcolormesh(xbins, ybins, meanZ.T, cmap="turbo", shading="auto",
                                      vmin=vmin, vmax=vmax)
            self.fig.colorbar(pcm, ax=self.ax, label="mean Z (m)")
        print(f"background: {len(xbins)-1}x{len(ybins)-1} cells @ {cell_size}m")

    def _color(self, name):
        i = list(self.boxes).index(name) if name in self.boxes else 0
        return PALETTE[i % len(PALETTE)]

    def _make_artists(self, name):
        color = self._color(name)
        poly = Polygon(polygon_points(self.boxes[name]), closed=True, fill=False,
                        edgecolor=color, linewidth=2)
        self.ax.add_patch(poly)
        corners = [Circle((0, 0), 0, facecolor=color, edgecolor="black", zorder=5) for _ in range(4)]
        for c in corners:
            self.ax.add_patch(c)
        roth = Circle((0, 0), 0, facecolor="white", edgecolor=color, linewidth=2, zorder=5)
        self.ax.add_patch(roth)
        label = self.ax.text(0, 0, name, color=color, fontsize=12, weight="bold",
                              ha="center", va="center")
        self.artists[name] = {"poly": poly, "corners": corners, "roth": roth, "label": label}

    def _pixel_radius_to_data(self):
        # handle pick/draw radius in DATA units, from a fixed on-screen pixel size, so
        # handles stay a sane clickable size regardless of zoom level
        p0 = self.ax.transData.transform((0, 0))
        p1 = self.ax.transData.transform((1, 0))
        px_per_data = np.linalg.norm(p1 - p0) or 1.0
        return HANDLE_PICK_PX / px_per_data

    def _redraw_one(self, name):
        box = self.boxes[name]
        art = self.artists[name]
        r = self._pixel_radius_to_data() * 0.6
        art["poly"].set_xy(polygon_points(box))
        for c, (sx, sy) in zip(art["corners"], [(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            c.center = corner_world(box, sx, sy)
            c.set_radius(r)
        art["roth"].center = rotate_handle_world(box)
        art["roth"].set_radius(r)
        art["label"].set_position((box["cx"], box["cy"]))

    def _redraw_all(self):
        for name in self.boxes:
            self._redraw_one(name)
        self.fig.canvas.draw_idle()

    def _hit_test(self, x, y):
        """Returns (name, mode, extra) for whatever's under (x,y), checking
        handles before body so an edge/corner near the outline still picks
        the handle, not a move."""
        pick_r = self._pixel_radius_to_data()
        for name, box in self.boxes.items():
            if np.linalg.norm(rotate_handle_world(box) - [x, y]) < pick_r:
                return name, "rotate", None
            for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                if np.linalg.norm(corner_world(box, sx, sy) - [x, y]) < pick_r:
                    return name, "resize", (sx, sy)
        for name, box in self.boxes.items():
            local = rot(-box["theta"]) @ (np.array([x, y]) - [box["cx"], box["cy"]])
            if abs(local[0]) <= box["w"] / 2 and abs(local[1]) <= box["h"] / 2:
                return name, "move", None
        return None, None, None

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        name, mode, extra = self._hit_test(event.xdata, event.ydata)
        if name is None:
            return
        self.selected = name
        box = self.boxes[name]
        if mode == "move":
            self.drag = {"name": name, "mode": mode,
                         "press": np.array([event.xdata, event.ydata]),
                         "start": dict(box)}
        elif mode == "resize":
            sx, sy = extra
            anchor = corner_world(box, -sx, -sy)
            self.drag = {"name": name, "mode": mode, "sx": sx, "sy": sy,
                         "anchor": anchor, "theta0": box["theta"]}
        elif mode == "rotate":
            self.drag = {"name": name, "mode": mode}

    def _on_motion(self, event):
        if self.drag is None or event.inaxes != self.ax or event.xdata is None:
            return
        d = self.drag
        box = self.boxes[d["name"]]
        mouse = np.array([event.xdata, event.ydata])

        if d["mode"] == "move":
            delta = mouse - d["press"]
            box["cx"] = d["start"]["cx"] + delta[0]
            box["cy"] = d["start"]["cy"] + delta[1]
        elif d["mode"] == "resize":
            R0 = rot(d["theta0"])
            local_vec = R0.T @ (mouse - d["anchor"])
            new_w = max(0.3, abs(local_vec[0]))
            new_h = max(0.3, abs(local_vec[1]))
            new_center = d["anchor"] + R0 @ np.array([d["sx"] * new_w / 2, d["sy"] * new_h / 2])
            box["w"], box["h"], box["theta"] = new_w, new_h, d["theta0"]
            box["cx"], box["cy"] = new_center
        elif d["mode"] == "rotate":
            box["theta"] = np.arctan2(mouse[1] - box["cy"], mouse[0] - box["cx"]) - np.pi / 2

        self._redraw_one(d["name"])
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self.drag = None

    def _on_key(self, event):
        if event.key == "a":
            fx = np.mean(self.ax.get_xlim())
            fy = np.mean(self.ax.get_ylim())
            name = f"{self.out_prefix}{self.next_idx}"
            self.next_idx += 1
            self.boxes[name] = {"cx": fx, "cy": fy, "w": 4.0, "h": 2.0, "theta": 0.0}
            self._make_artists(name)
            self.selected = name
            self._redraw_all()
            print(f"added {name}")
        elif event.key == "d" and self.selected in self.boxes:
            name = self.selected
            for art in self.artists[name].values():
                if isinstance(art, list):
                    for a in art:
                        a.remove()
                else:
                    art.remove()
            del self.artists[name]
            del self.boxes[name]
            self.selected = next(iter(self.boxes), None)
            self.fig.canvas.draw_idle()
            print(f"deleted {name}")
        elif event.key == "p":
            self.print_state()
            if self.out_json:
                save_boxes_json(self.out_json, self.boxes)  # autosave, in case the
                                                              # window crashes/closes by accident

    def print_state(self):
        print("\ncurrent zones:")
        for name, box in self.boxes.items():
            print(f"  {name}: center=({box['cx']:.2f},{box['cy']:.2f}) "
                  f"w={box['w']:.2f} h={box['h']:.2f} theta={np.degrees(box['theta']):.1f}deg")

    def points_in_box(self, box):
        rel = self.xyz[:, :2] - [box["cx"], box["cy"]]
        local = rel @ rot(box["theta"])  # == rot(-theta) applied via right-multiply
        mask_xy = (np.abs(local[:, 0]) <= box["w"] / 2) & (np.abs(local[:, 1]) <= box["h"] / 2)
        mask_z = (self.xyz[:, 2] >= self.z_range[0]) & (self.xyz[:, 2] <= self.z_range[1])
        return mask_xy & mask_z

    def run_and_export(self):
        plt.show()
        if self.out_json:
            save_boxes_json(self.out_json, self.boxes)
        if not self.boxes:
            print("no zones marked, nothing to save")
            return
        print("\nfinal zones:")
        for name, box in self.boxes.items():
            mask = self.points_in_box(box)
            n = int(mask.sum())
            corners = polygon_points(box)
            xmin, ymin = corners.min(axis=0)
            xmax, ymax = corners.max(axis=0)
            print(f"  {name}: center=({box['cx']:.2f},{box['cy']:.2f}) w={box['w']:.2f} "
                  f"h={box['h']:.2f} theta={np.degrees(box['theta']):.1f}deg -- {n} points "
                  f'-- loose AABB --roi="{xmin:.2f},{xmax:.2f},{ymin:.2f},{ymax:.2f},'
                  f'{self.z_range[0]:.2f},{self.z_range[1]:.2f}"')
            if n == 0:
                print("    (empty, skipping .pcd)")
                continue
            out_path = f"{name}.pcd"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.xyz[mask].astype(np.float64))
            o3d.io.write_point_cloud(out_path, pcd)
            print(f"    saved {out_path} -- run: fit_planes.py --pcd {out_path} --close-geometry")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, help="rosbag2 folder (raw /cloud_registered)")
    ap.add_argument("--pcd", type=Path, help="load xyz from a .pcd instead of a bag")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE")
    ap.add_argument("--roi", help="xmin,xmax,ymin,ymax,zmin,zmax crop before marking (metres)")
    ap.add_argument("--sor", action="store_true")
    ap.add_argument("--declutter", action="store_true")
    ap.add_argument("--cell-size", type=float, default=0.08,
                    help="background heatmap cell size (m) -- fine on purpose, so a "
                         "corridor kinked/rotated by drift is visible, not smoothed away "
                         "like auto_regions.py's coarser occupancy grid")
    ap.add_argument("--mode", choices=["density", "height"], default="density")
    ap.add_argument("--vmin", type=float, default=None, help="--mode height colorbar min")
    ap.add_argument("--vmax", type=float, default=None, help="--mode height colorbar max")
    ap.add_argument("--box", action="append", default=[],
                    help="'name:cx,cy,w,h,theta_deg' or 'name:xmin,xmax,ymin,ymax' "
                         "(AABB, e.g. pasted from auto_regions.py) -- repeatable, optional, "
                         "press 'a' in-window to add more")
    ap.add_argument("--load", type=Path,
                    help="reload box positions/sizes/rotations from a --out file of a "
                         "previous run, to keep editing where you left off -- merges with "
                         "any --box also given (--load first, --box can add more/override "
                         "same-named ones)")
    ap.add_argument("--z-base", type=float, default=0.0, help="fixed Z base for every zone")
    ap.add_argument("--z-top", type=float, default=None,
                    help="fixed Z top for every zone (default: the cloud's own max Z)")
    ap.add_argument("--out-prefix", default="box", help="new boxes are named <out-prefix><n>")
    ap.add_argument("--out", type=Path, default=Path("zones.json"),
                    help="save every box's center/size/rotation here (JSON) when you close "
                         "the window (and on every 'p' press, as an autosave) -- this is "
                         "what --load reopens; the per-box .pcd crops are saved separately "
                         "and always, regardless of --out")
    args = ap.parse_args()

    if bool(args.bag) == bool(args.pcd):
        raise SystemExit("pass exactly one of --bag or --pcd")

    xyz = load_pcd_cloud(args.pcd) if args.pcd else load_merged_cloud(args.bag, args.topic, args.store)
    if args.roi:
        xyz = crop_roi(xyz, tuple(float(x) for x in args.roi.split(",")))
    if args.sor:
        xyz = statistical_outlier_removal(xyz)
    if args.declutter:
        xyz = declutter(xyz, gap=0.30)

    z_top = args.z_top if args.z_top is not None else float(xyz[:, 2].max())
    z_range = (args.z_base, z_top)
    print(f"zones locked to Z {z_range[0]:.2f}..{z_range[1]:.2f}")

    boxes = {}
    if args.load:
        boxes.update(load_boxes_json(args.load))
        print(f"loaded {len(boxes)} zone(s) from {args.load}")
    boxes.update(parse_boxes(args.box))
    marker = ZoneMarker(xyz, boxes, args.cell_size, args.mode, args.vmin, args.vmax,
                         z_range, args.out_prefix, out_json=args.out)
    print(f"{len(boxes)} starting zone(s), opening window -- 'a' add, 'd' delete, "
          f"'p' print, close window to finish")
    marker.run_and_export()


if __name__ == "__main__":
    main()
