"""PyVista viewer for temperature_to_voxel.py's voxel output.

Adapted from AlignedOctree/view_voxels.py: same min-count filter, declutter
toggle, and planes_aligned.json box overlay, but voxels are colored by
mean_temperature instead of raw point count -- this pipeline has real
per-voxel measurements now, not just a density proxy. Voxels with no valid
temperature sample (mean_temperature is NaN -- no pose ever saw that voxel
with a usable FLIR pixel) are dropped from the render entirely by default;
--show-no-temp brings them back as grey instead.

Also overlays, when available:
- the closed box from AlignedOctree's planes_aligned.json (already in the
  aligned frame -- see aligned_octree.align_and_reclose_planes), should hug
  the voxel walls at an exact 90 degrees if the alignment is right.
- the raw aligned points (re-loaded from the bag and re-transformed via
  AlignedOctree's transform.json, same as aligned_octree.py does) -- off by
  default, expensive at full resolution, so --points-stride subsamples for
  display only.

(A --filler flag -- patching small floor/ceiling coverage gaps with
synthetic voxels -- was tried and removed; see FILLER.md for the full
design, the debugging journey, and the removed code, kept for a future
attempt rather than lost.)

Color range: [--clim-low-percentile, --clim-high-percentile] of the voxels
with a valid mean_temperature (default 1st/99th), not the full min/max -- a
handful of misprojected/edge-case voxels can otherwise stretch the scale far
past the real temperature range in the scene.

In the interactive window, the min-count filter (--min-count) is live: the
scene rebuilds in place, no relaunch needed. min-count still filters by each
voxel's `counts` field (original geometric point count from voxels.npz, not
n_temp_samples) -- same mechanic as AlignedOctree's viewer, unchanged.
    ]     raise min-count by --min-count-step (default 1)
    [     lower min-count by --min-count-step
    0     reset min-count to 1 (show every voxel)
    d     toggle declutter (drop floating voxels, see drop_floating_voxels) --
          on by default; raising min-count can strand a voxel that used to
          touch a since-hidden neighbour, and declutter drops it
(Not bound to +/-/Up/Down: pyvista's own defaults already use those for
camera zoom and point size, so reusing them would fire both at once.)
The current threshold, declutter state, and voxel/point counts print to the
console (and show as on-screen text) on every change.

Usage:
    python view_voxels.py                              # interactive window
    python view_voxels.py --min-count 3                 # hide sparse voxels
    python view_voxels.py --show-no-temp                 # show no-observation voxels in grey
    python view_voxels.py --points --points-stride 20    # overlay raw points
    python view_voxels.py --no-planes                    # hide the box overlay
    python view_voxels.py --screenshot out.png            # headless render

Venv: C:\\venvs\\planefit (same as the rest of this folder; pyvista is
already in it, see requirements.txt).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Thermal-appropriate default (turbo: perceptually reasonable rainbow map,
# same family MATLAB_PointCloudVisualization/Piano1_CorridoioLungo.m uses
# for its own coloring). --colormap overrides.
DEFAULT_COLORMAP = "turbo"
NO_TEMP_COLOR = "grey"
POINT_COLOR = "black"
POINT_SIZE = 2.0
BOX_COLOR = "red"
BOX_LINE_WIDTH = 3


def load_voxels(path):
    d = np.load(path)
    return (d["centers"], d["counts"], d["mean_temperature"], d["n_temp_samples"],
            float(d["voxel_size"]), d["origin"], int(d["depth"]))


def drop_floating_voxels(centers, origin, voxel_size):
    """Keep only the largest 26-connected component of occupied voxels.

    Raising the min-count filter can strand a few voxels that used to touch
    a since-hidden neighbour: they survive the count threshold but no longer
    touch anything else shown, so they render as specks floating in empty
    space. This labels connected components on the actual voxel lattice
    (recovering each voxel's integer (i,j,k) cell index from its center,
    inverting voxelizer.py's `center = (idx + 0.5) * voxel_size + origin`)
    and drops every component except the largest -- the same "declutter"
    idea fit_closed_planes.py already applies to raw points (its
    cluster_labels/largest_cluster_mask), just done on the voxel grid
    instead. 26-connected (any shared face, edge, or corner counts as
    touching) matches how adjacent cubes visually read as "connected" in
    the render.

    Returns a boolean mask into `centers` (True = keep).
    """
    from scipy import ndimage

    if len(centers) <= 1:
        return np.ones(len(centers), dtype=bool)

    idx = np.round((centers - origin) / voxel_size - 0.5).astype(np.int64)
    lo = idx.min(axis=0)
    shape = tuple((idx.max(axis=0) - lo + 1).tolist())
    local = idx - lo

    grid = np.zeros(shape, dtype=bool)
    grid[local[:, 0], local[:, 1], local[:, 2]] = True

    labels, _n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=np.int8))
    comp_id = labels[local[:, 0], local[:, 1], local[:, 2]]
    sizes = np.bincount(comp_id)
    largest = int(np.argmax(sizes[1:]) + 1) if len(sizes) > 1 else 0
    return comp_id == largest


def _cube_glyphs(centers, size):
    import pyvista as pv

    pdata = pv.PolyData(centers)
    cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
    return pdata.glyph(geom=cube, scale=False, orient=False, factor=size)


def load_box_wireframe(planes_aligned_path):
    """The closed box's rectangle edges (AlignedOctree's planes_aligned.json
    corners_3d, already in the aligned frame) as a single pyvista
    line-segment mesh for overlay. See AlignedOctree/view_voxels.py for why
    this must be the re-closed (post-alignment) box, not planes.json's
    corners naively rotated."""
    import pyvista as pv

    planes = json.loads(Path(planes_aligned_path).read_text())["planes"]

    pts = []
    for p in planes:
        corners = np.asarray(p["corners_3d"], dtype=float)
        for i in range(4):
            pts.append(corners[i])
            pts.append(corners[(i + 1) % 4])
    pts = np.asarray(pts)

    mesh = pv.PolyData(pts)
    n_lines = len(pts) // 2
    cells = np.empty(n_lines * 3, dtype=np.int64)
    cells[0::3] = 2
    cells[1::3] = np.arange(0, len(pts), 2)
    cells[2::3] = np.arange(1, len(pts), 2)
    mesh.lines = cells
    return mesh


def load_aligned_points(transform_path, stride=1):
    """Re-load the raw bag (temperature_to_voxel.load_merged_cloud) and
    re-apply the same rigid transform (AlignedOctree's transform.json) --
    for display only, not recomputed from the voxels."""
    from temperature_to_voxel import load_merged_cloud

    t = json.loads(Path(transform_path).read_text())
    xyz = load_merged_cloud(Path(t["bag"]), t["topic"], t["store"])
    R = np.array(t["rotation"])
    translation = np.array(t["translation"])
    aligned = xyz @ R.T + translation
    return aligned[::stride] if stride > 1 else aligned


# Keysyms bound to raise/lower the live min-count threshold: ']' / '['.
# Deliberately NOT Up/Down/plus/minus -- pyvista's own defaults already bind
# those to camera zoom and point-size (BasePlotter.reset_key_events); reusing
# them would fire both behaviors on one keypress. Brackets are unclaimed by
# both pyvista's defaults and VTK's interactor-style bindings (w/s/r/3/e/f/p/j/t).
INCREASE_KEY = "bracketright"  # ']'
DECREASE_KEY = "bracketleft"   # '['
RESET_KEY = "0"
DECLUTTER_KEY = "d"  # toggle drop_floating_voxels; 'd' is unclaimed by pyvista/VTK defaults


def build_plotter(centers, counts, mean_temperature, voxel_size, min_count, colormap,
                   box_mesh, points, off_screen, clim_low_pct=1.0, clim_high_pct=99.0,
                   min_count_step=1, origin=None, declutter=True, show_no_temp=False):
    import pyvista as pv

    # Color range from the voxels that actually have a valid temperature,
    # clipped to [clim_low_pct, clim_high_pct] so a handful of misprojected/
    # edge-case voxels don't stretch the scale far past the scene's real
    # temperature range. Fixed once from the *full* set (not recomputed as
    # the live min-count filter changes), so raising min-count doesn't keep
    # rescaling what each color means underfoot.
    has_temp_all = np.isfinite(mean_temperature)
    clim = None
    if has_temp_all.any():
        valid_temps = mean_temperature[has_temp_all]
        lo = float(np.percentile(valid_temps, clim_low_pct))
        hi = float(np.percentile(valid_temps, clim_high_pct))
        if hi > lo:
            clim = (lo, hi)

    pl = pv.Plotter(window_size=(1400, 900), off_screen=off_screen)
    pl.set_background("white")

    grid_origin = origin if origin is not None else centers.min(axis=0)
    state = {"min_count": int(min_count), "declutter": bool(declutter)}

    def render_voxels():
        keep = counts >= state["min_count"]
        shown_centers = centers[keep]
        shown_temps = mean_temperature[keep]
        n_after_count = len(shown_centers)

        # No-temp voxels are dropped BEFORE declutter, not after: declutter's
        # connectivity check must see exactly what will actually be shown.
        # Checking it on the full (min-count-filtered) occupancy first would
        # count a voxel as "connected" through a no-temp neighbour that then
        # disappears from the render -- leaving it visibly floating anyway,
        # just past the point declutter could still catch it.
        if not show_no_temp:
            has_temp = np.isfinite(shown_temps)
            shown_centers, shown_temps = shown_centers[has_temp], shown_temps[has_temp]

        n_before_declutter = len(shown_centers)
        if state["declutter"] and n_before_declutter > 1:
            main_mask = drop_floating_voxels(shown_centers, grid_origin, voxel_size)
            shown_centers, shown_temps = shown_centers[main_mask], shown_temps[main_mask]

        n_dropped_floating = n_before_declutter - len(shown_centers)
        n_with_temp = int(np.isfinite(shown_temps).sum())
        print(f"voxels: {len(centers)} total, {n_after_count} pass min-count="
              f"{state['min_count']}, {n_dropped_floating} dropped as floating "
              f"(declutter={'on' if state['declutter'] else 'off'}), "
              f"{len(shown_centers)} shown ({n_with_temp} with valid temperature)")

        if len(shown_centers) == 0:
            if "voxels" in pl.actors:
                pl.remove_actor("voxels", render=False)
        else:
            glyphs = _cube_glyphs(shown_centers, voxel_size)
            glyphs["temperature"] = np.repeat(shown_temps, glyphs.n_cells // len(shown_centers))
            pl.add_mesh(glyphs, scalars="temperature", cmap=colormap, clim=clim,
                        nan_color=NO_TEMP_COLOR, show_scalar_bar=True,
                        scalar_bar_args={"title": "mean temperature (C)"}, name="voxels")

        pl.add_text(
            f"min-count: {state['min_count']}   "
            f"(right/left bracket key: change by {min_count_step}, 0: reset)\n"
            f"declutter (drop floating voxels): {'on' if state['declutter'] else 'off'}   "
            f"(d: toggle)",
            position="upper_left", font_size=10, name="mincount_label")
        pl.render()

    def bump(delta):
        state["min_count"] = max(1, state["min_count"] + delta)
        render_voxels()

    def reset():
        state["min_count"] = 1
        render_voxels()

    def toggle_declutter():
        state["declutter"] = not state["declutter"]
        render_voxels()

    pl.add_key_event(INCREASE_KEY, lambda: bump(min_count_step))
    pl.add_key_event(DECREASE_KEY, lambda: bump(-min_count_step))
    pl.add_key_event(RESET_KEY, reset)
    pl.add_key_event(DECLUTTER_KEY, toggle_declutter)

    render_voxels()

    if box_mesh is not None:
        pl.add_mesh(box_mesh, color=BOX_COLOR, line_width=BOX_LINE_WIDTH, name="closed_box")

    if points is not None:
        pl.add_points(points, color=POINT_COLOR, point_size=POINT_SIZE,
                      render_points_as_spheres=False, name="points")

    pl.add_axes()
    return pl


def make_orbit_gif(pl, path, n_frames=36, factor=2.5, viewup=(0.0, 0.0, 1.0)):
    """Write a 360-degree orbit around the current scene to an animated GIF
    (pyvista's generate_orbital_path/orbit_on_path -- a fixed circular path
    around the bounds already added to `pl`, not a live rotate-with-the-mouse
    session). `factor` scales the orbit radius relative to the scene bounds."""
    pl.open_gif(str(path))
    orbit_path = pl.generate_orbital_path(factor=factor, n_points=n_frames, viewup=list(viewup))
    pl.orbit_on_path(orbit_path, write_frames=True, viewup=list(viewup))
    pl.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxels", type=Path, default=HERE / "voxels_temperature.npz")
    ap.add_argument("--transform", type=Path,
                    default=HERE.parent / "AlignedOctree" / "transform.json")
    ap.add_argument("--planes-aligned", type=Path,
                    default=HERE.parent / "AlignedOctree" / "planes_aligned.json",
                    help="closed box overlay, output of AlignedOctree/aligned_octree.py "
                         "(already in the aligned frame)")
    ap.add_argument("--no-planes", action="store_true", help="don't overlay the closed box")
    ap.add_argument("--min-count", type=int, default=1,
                    help="hide voxels with fewer than this many points (starting "
                         "value -- live-adjustable in the interactive window, see "
                         "--min-count-step / key bindings in the module docstring). "
                         "Filters by voxels.npz's original point `counts`, not "
                         "n_temp_samples.")
    ap.add_argument("--min-count-step", type=int, default=1,
                    help="how much ]/[ change min-count by per press")
    ap.add_argument("--colormap", default=DEFAULT_COLORMAP)
    ap.add_argument("--clim-low-percentile", type=float, default=1.0,
                    help="low end of the color range, as a percentile of voxels with "
                         "a valid mean_temperature")
    ap.add_argument("--clim-high-percentile", type=float, default=99.0,
                    help="high end of the color range (percentile)")
    ap.add_argument("--show-no-temp", action="store_true",
                    help="show voxels with no valid mean_temperature in grey, "
                         "instead of dropping them from the render entirely")
    ap.add_argument("--points", action="store_true",
                    help="overlay the raw aligned points (re-reads the bag)")
    ap.add_argument("--points-stride", type=int, default=20,
                    help="keep every Nth raw point for display (--points only)")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="headless render to this image file instead of opening a window")
    ap.add_argument("--orbit-gif", type=Path, default=None,
                    help="headless: write a 360-degree orbit animation to this .gif "
                         "instead of opening a window")
    ap.add_argument("--orbit-frames", type=int, default=36)
    ap.add_argument("--no-declutter", action="store_true",
                    help="don't drop floating voxels (islands disconnected from the "
                         "largest connected component) -- on by default, live-togglable "
                         "with the 'd' key in the interactive window")
    args = ap.parse_args()

    centers, counts, mean_temperature, n_temp_samples, voxel_size, origin, depth = \
        load_voxels(args.voxels)
    depth_str = f"depth={depth}" if depth >= 0 else "depth=n/a (uniform grid, --voxel-size)"
    n_with_temp = int(np.isfinite(mean_temperature).sum())
    print(f"loaded {args.voxels}: {len(centers)} voxels, {depth_str}, "
          f"voxel_size={voxel_size:.4f} m, {n_with_temp} with valid mean_temperature "
          f"({100 * n_with_temp / max(len(centers), 1):.1f}%)")

    box_mesh = None
    if not args.no_planes and args.planes_aligned.exists():
        box_mesh = load_box_wireframe(args.planes_aligned)

    points = None
    if args.points:
        points = load_aligned_points(args.transform, stride=args.points_stride)
        print(f"overlaying {len(points)} raw points (stride {args.points_stride})")

    off_screen = args.screenshot is not None or args.orbit_gif is not None
    pl = build_plotter(
        centers, counts, mean_temperature, voxel_size, args.min_count, args.colormap,
        box_mesh, points, off_screen=off_screen,
        clim_low_pct=args.clim_low_percentile, clim_high_pct=args.clim_high_percentile,
        min_count_step=args.min_count_step, origin=origin, declutter=not args.no_declutter,
        show_no_temp=args.show_no_temp)

    if args.screenshot:
        pl.screenshot(str(args.screenshot))
        print(f"wrote {args.screenshot}")
    elif args.orbit_gif:
        make_orbit_gif(pl, args.orbit_gif, n_frames=args.orbit_frames)
        print(f"wrote {args.orbit_gif} ({args.orbit_frames} frames)")
    else:
        pl.show()


if __name__ == "__main__":
    main()
