"""PyVista viewer for material_to_voxel.py's voxel output.

Adapted from TemperatureToVoxel/view_voxels.py: same min-count filter,
declutter toggle, and planes_aligned.json box overlay, but voxels are
colored by `material_id` -- a categorical label, not a continuous field, so
this uses a discrete/qualitative colormap (matplotlib's "tab20", one swatch
per vocabulary material) with the color bar annotated by material name at
each integer tick, instead of TemperatureToVoxel's percentile-clipped
continuous range. Voxels with no valid material (material_id == -1 -- no
pose ever resolved that voxel's points to a material) are dropped from the
render entirely by default; --show-no-material brings them back as grey
instead (same idea as --show-no-temp).

Also overlays, when available (unchanged from TemperatureToVoxel/AlignedOctree):
- the closed box from AlignedOctree's planes_aligned.json.
- the raw aligned points (re-loaded from the bag and re-transformed via
  AlignedOctree's transform.json) -- off by default, --points-stride
  subsamples for display only.

In the interactive window, the min-count filter (--min-count) is live: the
scene rebuilds in place, no relaunch needed. min-count still filters by each
voxel's `counts` field (original geometric point count from voxels.npz, not
n_material_votes) -- same mechanic as the other viewers.
    ]     raise min-count by --min-count-step (default 1)
    [     lower min-count by --min-count-step
    0     reset min-count to 1 (show every voxel)
    d     toggle declutter (drop floating voxels) -- on by default; raising
          min-count can strand a voxel that used to touch a since-hidden
          neighbour, and declutter drops it
(Not bound to +/-/Up/Down: pyvista's own defaults already use those for
camera zoom and point size.)
The current threshold, declutter state, and voxel/point counts print to the
console (and show as on-screen text) on every change.

No separate legend widget: the color bar itself is annotated with one tick
per vocabulary material (VTK/pyvista `annotations`), so it doubles as the
legend -- avoids maintaining two things that could disagree.

Usage:
    python view_voxels.py                                # interactive window
    python view_voxels.py --min-count 3                   # hide sparse voxels
    python view_voxels.py --show-no-material               # show unlabeled voxels in grey
    python view_voxels.py --points --points-stride 20      # overlay raw points
    python view_voxels.py --no-planes                      # hide the box overlay
    python view_voxels.py --screenshot out.png              # headless render

Venv: C:\\venvs\\planefit (pyvista + matplotlib already in it, see
requirements.txt / TemperatureToVoxel's).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

NO_MATERIAL_COLOR = "grey"
POINT_COLOR = "black"
POINT_SIZE = 2.0
BOX_COLOR = "red"
BOX_LINE_WIDTH = 3
PALETTE = "tab20"  # matplotlib qualitative colormap, 20 distinct swatches


def load_voxels(path):
    d = np.load(path, allow_pickle=False)
    materials = [str(m) for m in d["materials"]]
    return (d["centers"], d["counts"], d["material_id"], d["material_confidence"],
            d["n_material_votes"], materials, float(d["voxel_size"]), d["origin"], int(d["depth"]))


def build_material_colormap(materials):
    """One fixed swatch per vocabulary material (tab20, cycling past 20) --
    fixed once from the FULL vocabulary, so a material's color never
    changes as the live min-count filter changes what's shown."""
    from matplotlib.colors import ListedColormap
    from matplotlib import colormaps

    base = colormaps[PALETTE]
    colors = [base(i % base.N)[:3] for i in range(max(len(materials), 1))]
    return ListedColormap(colors)


def drop_floating_voxels(centers, origin, voxel_size):
    """Keep only the largest 26-connected component of occupied voxels --
    copied from TemperatureToVoxel/view_voxels.py (same declutter idea)."""
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
    """Copied from TemperatureToVoxel/view_voxels.py -- see there for why
    this must be the re-closed (post-alignment) box."""
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
    """Re-load the raw bag and re-apply transform.json -- for display only,
    copied from TemperatureToVoxel/view_voxels.py, adapted to this folder's
    own load_merged_cloud (self-contained, no cross-import)."""
    from material_to_voxel import load_merged_cloud

    t = json.loads(Path(transform_path).read_text())
    xyz = load_merged_cloud(Path(t["bag"]), t["topic"], t["store"])
    R = np.array(t["rotation"])
    translation = np.array(t["translation"])
    aligned = xyz @ R.T + translation
    return aligned[::stride] if stride > 1 else aligned


INCREASE_KEY = "bracketright"  # ']'
DECREASE_KEY = "bracketleft"   # '['
RESET_KEY = "0"
DECLUTTER_KEY = "d"


def build_plotter(centers, counts, material_id, materials, voxel_size, min_count,
                   box_mesh, points, off_screen, min_count_step=1, origin=None,
                   declutter=True, show_no_material=False):
    import pyvista as pv

    cmap = build_material_colormap(materials)
    k = max(len(materials), 1)
    clim = (-0.5, k - 0.5)
    annotations = {float(i): name for i, name in enumerate(materials)}

    pl = pv.Plotter(window_size=(1400, 900), off_screen=off_screen)
    pl.set_background("white")

    grid_origin = origin if origin is not None else centers.min(axis=0)
    state = {"min_count": int(min_count), "declutter": bool(declutter)}

    def render_voxels():
        keep = counts >= state["min_count"]
        shown_centers = centers[keep]
        shown_material = material_id[keep]
        n_after_count = len(shown_centers)

        # No-material voxels dropped BEFORE declutter, not after -- same
        # reasoning as TemperatureToVoxel/view_voxels.py's no-temp filter:
        # declutter's connectivity check must see exactly what will be shown.
        if not show_no_material:
            has_material = shown_material >= 0
            shown_centers, shown_material = shown_centers[has_material], shown_material[has_material]

        n_before_declutter = len(shown_centers)
        if state["declutter"] and n_before_declutter > 1:
            main_mask = drop_floating_voxels(shown_centers, grid_origin, voxel_size)
            shown_centers, shown_material = shown_centers[main_mask], shown_material[main_mask]

        n_dropped_floating = n_before_declutter - len(shown_centers)
        n_with_material = int((shown_material >= 0).sum())
        print(f"voxels: {len(centers)} total, {n_after_count} pass min-count="
              f"{state['min_count']}, {n_dropped_floating} dropped as floating "
              f"(declutter={'on' if state['declutter'] else 'off'}), "
              f"{len(shown_centers)} shown ({n_with_material} with a material label)")

        if len(shown_centers) == 0:
            if "voxels" in pl.actors:
                pl.remove_actor("voxels", render=False)
        else:
            # -1 (no material) -> NaN so nan_color paints it grey, same
            # idiom as TemperatureToVoxel's mean_temperature NaN handling.
            scalars = shown_material.astype(np.float64)
            scalars[shown_material < 0] = np.nan

            glyphs = _cube_glyphs(shown_centers, voxel_size)
            glyphs["material"] = np.repeat(scalars, glyphs.n_cells // len(shown_centers))
            # Vertical, sized to fit one label per vocabulary material --
            # the default horizontal bar crowds/clips labels past ~6-8
            # categories (seen with the real 14-material vocabulary).
            pl.add_mesh(glyphs, scalars="material", cmap=cmap, clim=clim,
                        nan_color=NO_MATERIAL_COLOR, show_scalar_bar=True,
                        annotations=annotations,
                        scalar_bar_args={"title": "material", "n_labels": 0,
                                         "n_colors": k, "fmt": "%.0f",
                                         "vertical": True, "height": 0.7, "width": 0.07,
                                         "position_x": 0.88, "position_y": 0.15,
                                         "label_font_size": 12, "title_font_size": 14},
                        name="voxels")

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
    """Copied from TemperatureToVoxel/view_voxels.py."""
    pl.open_gif(str(path))
    orbit_path = pl.generate_orbital_path(factor=factor, n_points=n_frames, viewup=list(viewup))
    pl.orbit_on_path(orbit_path, write_frames=True, viewup=list(viewup))
    pl.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxels", type=Path, default=HERE / "voxels_material.npz")
    ap.add_argument("--transform", type=Path,
                    default=HERE.parent / "AlignedOctree" / "transform.json")
    ap.add_argument("--planes-aligned", type=Path,
                    default=HERE.parent / "AlignedOctree" / "planes_aligned.json",
                    help="closed box overlay, output of AlignedOctree/aligned_octree.py")
    ap.add_argument("--no-planes", action="store_true", help="don't overlay the closed box")
    ap.add_argument("--min-count", type=int, default=1,
                    help="hide voxels with fewer than this many points (starting "
                         "value -- live-adjustable, see key bindings in the module "
                         "docstring). Filters by voxels.npz's original point `counts`, "
                         "not n_material_votes.")
    ap.add_argument("--min-count-step", type=int, default=1,
                    help="how much ]/[ change min-count by per press")
    ap.add_argument("--show-no-material", action="store_true",
                    help="show voxels with no valid material label in grey, instead "
                         "of dropping them from the render entirely")
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
                    help="don't drop floating voxels -- on by default, live-togglable "
                         "with the 'd' key")
    args = ap.parse_args()

    centers, counts, material_id, material_confidence, n_material_votes, materials, \
        voxel_size, origin, depth = load_voxels(args.voxels)
    depth_str = f"depth={depth}" if depth >= 0 else "depth=n/a (uniform grid, --voxel-size)"
    n_with_material = int((material_id >= 0).sum())
    print(f"loaded {args.voxels}: {len(centers)} voxels, {depth_str}, "
          f"voxel_size={voxel_size:.4f} m, {n_with_material} with a material label "
          f"({100 * n_with_material / max(len(centers), 1):.1f}%), "
          f"{len(materials)} material(s) in vocabulary: {materials}")

    box_mesh = None
    if not args.no_planes and args.planes_aligned.exists():
        box_mesh = load_box_wireframe(args.planes_aligned)

    points = None
    if args.points:
        points = load_aligned_points(args.transform, stride=args.points_stride)
        print(f"overlaying {len(points)} raw points (stride {args.points_stride})")

    off_screen = args.screenshot is not None or args.orbit_gif is not None
    pl = build_plotter(
        centers, counts, material_id, materials, voxel_size, args.min_count,
        box_mesh, points, off_screen=off_screen,
        min_count_step=args.min_count_step, origin=origin, declutter=not args.no_declutter,
        show_no_material=args.show_no_material)

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
