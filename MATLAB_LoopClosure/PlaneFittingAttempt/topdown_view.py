"""Top-down (X-Y) diagnostic view of a point cloud, from a raw rosbag2 or a
.pcd -- for reading off topology and ROI boundaries before running
fit_planes.py, and for telling real slope from FAST-LIO roll/pitch drift.

Two modes (--mode):
    density  log-scaled point-count heatmap -- good for seeing the shape of
             the building (corridors, junctions, clutter) regardless of height.
    height   mean-Z per cell (default) -- a real corridor is flat (uniform
             color along its length); FAST-LIO drift that hasn't been
             gravity/yaw-corrected shows up as a smooth color GRADIENT along
             a branch (it "climbs" or "sinks") even though it's physically
             level. Compare a --bag run (raw /cloud_registered, still has
             drift) against a --pcd run on LoopClosureRaw.m's
             loop_closed_map.pcd (already corrected) to tell the two apart.

--region NAME:xmin,xmax,ymin,ymax (repeatable) prints point count + Z
range/median/std for that box, e.g. to sanity-check a candidate
fit_planes.py --roi before committing to it. With --pyvista, the same
--region boxes are also drawn as red wireframes over the cloud (full Z
span), so you can rotate around and see whether a candidate ROI actually
brackets the corridor before committing to it.

--pyvista opens a real rotatable 3D window (the cloud colored by height,
same colormap/--vmin/--vmax as --mode height) instead of writing the flat
matplotlib PNG -- --mode/--bins-x/--bins-y/--grid-step/--out are ignored
in this mode. Off-screen by default like show_planes.py isn't an option
here since the whole point is to look around; it always opens interactive
and blocks until you close the window.

Usage:
    python topdown_view.py (--bag <rosbag2_folder> | --pcd <file.pcd>)
        [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--roi xmin,xmax,ymin,ymax,zmin,zmax]
        [--sor] [--declutter]
        [--mode height|density] [--bins-x 400] [--bins-y 300]
        [--vmin -1] [--vmax 3] [--grid-step 5]
        [--region "A:−4,40,−2,3" --region "B:15,22,3,26"]
        [--cache cloud.npy]
        [--pyvista [--point-size 2] [--top-view]]
        [--out topdown.png]

--cache saves the loaded (and cropped/decluttered) xyz to a .npy the first
time, and loads straight from it on later runs if the file already exists --
skips re-reading a multi-million-point bag every time you tweak --mode/
--region/--vmin while iterating. Delete the cache (or point --cache
elsewhere) after changing --bag/--pcd/--roi/--declutter/--sor, since those
aren't re-applied when the cache is reused.

Venv: C:\\venvs\\planefit (same as fit_planes.py; needs matplotlib too).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from fit_planes import crop_roi, declutter, load_merged_cloud, load_pcd_cloud, statistical_outlier_removal


def load_xyz(args):
    if args.cache and Path(args.cache).exists():
        xyz = np.load(args.cache)
        print(f"loaded {len(xyz)} points from cache {args.cache}")
        return xyz

    if args.pcd:
        xyz = load_pcd_cloud(args.pcd)
    else:
        xyz = load_merged_cloud(args.bag, args.topic, args.store)

    if args.roi:
        roi = tuple(float(x) for x in args.roi.split(","))
        xyz = crop_roi(xyz, roi)
    if args.sor:
        xyz = statistical_outlier_removal(xyz)
    if args.declutter:
        xyz = declutter(xyz, gap=0.30)

    if args.cache:
        np.save(args.cache, xyz)
        print(f"cached to {args.cache}")
    return xyz


def parse_regions(region_args):
    """'NAME:xmin,xmax,ymin,ymax' strings -> [(name, xmin, xmax, ymin, ymax), ...]."""
    regions = []
    for spec in region_args:
        name, coords = spec.split(":", 1)
        xmin, xmax, ymin, ymax = (float(v) for v in coords.split(","))
        regions.append((name, xmin, xmax, ymin, ymax))
    return regions


def print_region_stats(xyz, regions):
    for name, xmin, xmax, ymin, ymax in regions:
        m = (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
        z = xyz[m, 2]
        if len(z) == 0:
            print(f"  {name}: 0 points")
            continue
        print(f"  {name} [{xmin},{xmax},{ymin},{ymax}]: {m.sum()} points, "
              f"Z {z.min():.2f}..{z.max():.2f} (median {np.median(z):.2f}, std {z.std():.2f})")


def show_pyvista(xyz, regions, args, title):
    """Open a real rotatable 3D window: the cloud colored by height, plus
    each --region as a red wireframe box (full Z span of the data) so you
    can visually check a candidate fit_planes.py --roi against the actual
    3D structure instead of just a flat top-down projection."""
    import pyvista as pv

    p = pv.Plotter(off_screen=False, window_size=(1600, 1000))
    # Orthographic instead of pyvista's default perspective camera: under perspective, a
    # tall wireframe box (full Z span) viewed from an oblique angle projects with its far
    # (high-Z) edge shifted laterally relative to its near (low-Z) edge, so the box reads
    # as flared/rotated way beyond the cloud's real XY footprint. Parallel projection keeps
    # every edge's screen position exactly proportional to its real coordinates.
    p.enable_parallel_projection()
    cloud = pv.PolyData(xyz)
    cloud["z"] = xyz[:, 2]
    vmin = args.vmin if args.vmin is not None else float(xyz[:, 2].min())
    vmax = args.vmax if args.vmax is not None else float(xyz[:, 2].max())
    p.add_mesh(cloud, scalars="z", cmap="turbo", clim=(vmin, vmax),
               point_size=args.point_size, render_points_as_spheres=False,
               scalar_bar_args={"title": "Z (m)"})

    zlo, zhi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for name, xmin, xmax, ymin, ymax in regions:
        corners = np.array([
            [xmin, ymin, zlo], [xmax, ymin, zlo], [xmax, ymax, zlo], [xmin, ymax, zlo],
            [xmin, ymin, zhi], [xmax, ymin, zhi], [xmax, ymax, zhi], [xmin, ymax, zhi],
        ])
        for a, b in edges:
            p.add_lines(corners[[a, b]], color="red", width=3)
        p.add_point_labels([corners.mean(axis=0)], [name], font_size=20,
                           text_color="red", shape_opacity=0.6, always_visible=True)

    p.set_background("white")
    if args.top_view:
        p.camera_position = "xy"
        p.camera.up = (0.0, 1.0, 0.0)
    else:
        p.camera_position = "iso"
    p.add_text(title, font_size=10, color="black")
    print("opening pyvista window -- close it to exit")
    p.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, help="rosbag2 folder (raw /cloud_registered)")
    ap.add_argument("--pcd", type=Path, help="load xyz from a .pcd instead of a bag")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE")
    ap.add_argument("--roi", help="xmin,xmax,ymin,ymax,zmin,zmax crop (metres)")
    ap.add_argument("--sor", action="store_true", help="apply Statistical Outlier Removal")
    ap.add_argument("--declutter", action="store_true",
                    help="drop points not connected to the main body (gap=0.30m)")
    ap.add_argument("--mode", choices=["height", "density"], default="height")
    ap.add_argument("--bins-x", type=int, default=400)
    ap.add_argument("--bins-y", type=int, default=300)
    ap.add_argument("--vmin", type=float, default=None, help="--mode height: colorbar min (m)")
    ap.add_argument("--vmax", type=float, default=None, help="--mode height: colorbar max (m)")
    ap.add_argument("--grid-step", type=float, default=5.0, help="gridline spacing (m), 0 = off")
    ap.add_argument("--region", action="append", default=[],
                    help='NAME:xmin,xmax,ymin,ymax -- repeatable, prints point count + Z '
                         'stats for that box (e.g. to check a candidate --roi)')
    ap.add_argument("--cache", type=Path,
                    help="load/save the (cropped/decluttered) cloud here as .npy, to skip "
                         "re-reading a big bag on repeat runs -- stale after changing "
                         "--bag/--pcd/--roi/--declutter/--sor, delete it then")
    ap.add_argument("--pyvista", action="store_true",
                    help="open a rotatable 3D window (cloud colored by height + --region "
                         "boxes as wireframes) instead of writing a flat top-down PNG")
    ap.add_argument("--point-size", type=float, default=2.0, help="--pyvista point size")
    ap.add_argument("--top-view", action="store_true",
                    help="--pyvista: start the camera straight down, matching the flat "
                         "top-down PNG, instead of the default iso angle")
    ap.add_argument("--out", type=Path, default=Path("topdown.png"))
    args = ap.parse_args()

    if bool(args.bag) == bool(args.pcd):
        raise SystemExit("pass exactly one of --bag or --pcd")

    xyz = load_xyz(args)
    print(f"X {xyz[:,0].min():.2f}..{xyz[:,0].max():.2f}  "
          f"Y {xyz[:,1].min():.2f}..{xyz[:,1].max():.2f}  "
          f"Z {xyz[:,2].min():.2f}..{xyz[:,2].max():.2f}")

    regions = parse_regions(args.region)
    if regions:
        print_region_stats(xyz, regions)

    source = args.pcd if args.pcd else args.bag
    if args.pyvista:
        show_pyvista(xyz, regions, args, title=source.name)
        return

    xbins = np.linspace(xyz[:, 0].min(), xyz[:, 0].max(), args.bins_x)
    ybins = np.linspace(xyz[:, 1].min(), xyz[:, 1].max(), args.bins_y)

    fig, ax = plt.subplots(figsize=(14, 10))
    if args.mode == "density":
        ax.hist2d(xyz[:, 0], xyz[:, 1], bins=[xbins, ybins], cmap="viridis", norm=LogNorm())
        title = "point density (log)"
    else:
        sumZ, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=[xbins, ybins], weights=xyz[:, 2])
        cnt, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=[xbins, ybins])
        with np.errstate(invalid="ignore"):
            meanZ = np.ma.masked_invalid(sumZ / cnt)
        pcm = ax.pcolormesh(xbins, ybins, meanZ.T, cmap="turbo", shading="auto",
                             vmin=args.vmin, vmax=args.vmax)
        fig.colorbar(pcm, ax=ax, label="mean Z (m)")
        title = "mean height Z"

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"{source.name} -- top-down, {title}")
    ax.set_aspect("equal")
    if args.grid_step > 0:
        ax.set_xticks(np.arange(np.floor(xyz[:,0].min()/args.grid_step)*args.grid_step,
                                 np.ceil(xyz[:,0].max()/args.grid_step)*args.grid_step + 1, args.grid_step))
        ax.set_yticks(np.arange(np.floor(xyz[:,1].min()/args.grid_step)*args.grid_step,
                                 np.ceil(xyz[:,1].max()/args.grid_step)*args.grid_step + 1, args.grid_step))
        ax.grid(True, alpha=0.3, color="white")
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
