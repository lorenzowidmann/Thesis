"""PyVista viewer for find_window_gaps.py's output.

Shows the whole building's voxels.npz as translucent grey context, the
closed box overlay (planes_aligned.json), and the synthetic window-fill
voxels (window_gaps.npz) highlighted -- one distinct color per window_id
(matplotlib's "tab20", same categorical-palette idea as
MaterialToVoxel/view_voxels.py) so it's visually obvious how many separate
windows were found and where. On-screen text lists each window's wall id,
cell count and area (from windows.json if present alongside the npz).

Live key:
    c     toggle the grey context building on/off (the fill voxels are
          small relative to the whole cloud -- easier to inspect alone)
(Not bound to +/-/Up/Down/other viewers' bracket keys: no live min-count
filter here, there's nothing to threshold -- window_gaps.npz is already a
small, fixed result.)

Usage:
    python view_window_gaps.py                          # interactive window
    python view_window_gaps.py --no-context               # fill voxels only, no grey building
    python view_window_gaps.py --points --points-stride 20  # overlay raw points
    python view_window_gaps.py --screenshot out.png        # headless render

Venv: C:\\venvs\\planefit (pyvista + matplotlib, same as MaterialToVoxel's
view_voxels.py; scipy not needed here, no declutter).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

CONTEXT_COLOR = "lightgrey"
CONTEXT_OPACITY = 0.20
BOX_COLOR = "red"
BOX_LINE_WIDTH = 3
POINT_COLOR = "black"
POINT_SIZE = 2.0
PALETTE = "tab20"
FILL_SCALE = 1.15  # slightly oversized so fill voxels visually pop through the translucent context


def _cube_glyphs(centers, size):
    import pyvista as pv

    pdata = pv.PolyData(centers)
    cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
    return pdata.glyph(geom=cube, scale=False, orient=False, factor=size)


def load_box_wireframe(planes_aligned_path):
    """Copied from AlignedOctree/TemperatureToVoxel/MaterialToVoxel's
    view_voxels.py (same self-contained convention)."""
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


def window_color_map(n_windows):
    from matplotlib import colormaps

    base = colormaps[PALETTE]
    return [base(i % base.N)[:3] for i in range(max(n_windows, 1))]


CONTEXT_KEY = "c"


def build_plotter(context_centers, context_voxel_size, fill_centers, window_id, wall_id,
                   fill_voxel_size, windows_meta, box_mesh, points, off_screen,
                   show_context=True):
    import pyvista as pv

    pl = pv.Plotter(window_size=(1400, 900), off_screen=off_screen)
    pl.set_background("white")

    state = {"context": bool(show_context)}

    def render_context():
        if not state["context"] or context_centers is None or len(context_centers) == 0:
            if "context" in pl.actors:
                pl.remove_actor("context", render=False)
        else:
            glyphs = _cube_glyphs(context_centers, context_voxel_size)
            pl.add_mesh(glyphs, color=CONTEXT_COLOR, opacity=CONTEXT_OPACITY, name="context")
        pl.add_text(
            f"context building: {'on' if state['context'] else 'off'}   (c: toggle)\n"
            f"{len(windows_meta)} window(s), {len(fill_centers)} fill voxel(s) total",
            position="upper_left", font_size=10, name="label")
        pl.render()

    def toggle_context():
        state["context"] = not state["context"]
        render_context()

    pl.add_key_event(CONTEXT_KEY, toggle_context)
    render_context()

    n_windows = int(window_id.max()) + 1 if len(window_id) else 0
    colors = window_color_map(n_windows)
    for w in range(n_windows):
        mask = window_id == w
        if not mask.any():
            continue
        glyphs = _cube_glyphs(fill_centers[mask], fill_voxel_size * FILL_SCALE)
        pl.add_mesh(glyphs, color=colors[w], opacity=1.0, name=f"window_{w}")

    if box_mesh is not None:
        pl.add_mesh(box_mesh, color=BOX_COLOR, line_width=BOX_LINE_WIDTH, name="closed_box")

    if points is not None:
        pl.add_points(points, color=POINT_COLOR, point_size=POINT_SIZE,
                      render_points_as_spheres=False, name="points")

    # Per-window legend text (id, wall, cells, area) -- separate from the
    # top-left toggle label so a long window list doesn't crowd it.
    if windows_meta:
        lines = [f"win {w['window_id']}: wall {w['wall_id']}, {w['n_cells']} cells, "
                 f"{w['area_m2']:.2f} m^2" for w in windows_meta]
        pl.add_text("\n".join(lines), position="lower_right", font_size=9, name="windows_legend")

    pl.add_axes()
    return pl


def load_aligned_points(transform_path, stride=1):
    """Re-load the raw bag and re-apply transform.json -- for display only.
    Reuses MaterialToVoxel's load_merged_cloud (no separate copy kept here:
    this folder has no bag-loading need of its own otherwise)."""
    import sys
    sys.path.insert(0, str(HERE.parent / "MaterialToVoxel"))
    from material_to_voxel import load_merged_cloud

    t = json.loads(Path(transform_path).read_text())
    xyz = load_merged_cloud(Path(t["bag"]), t["topic"], t["store"])
    R = np.array(t["rotation"])
    translation = np.array(t["translation"])
    aligned = xyz @ R.T + translation
    return aligned[::stride] if stride > 1 else aligned


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaps", type=Path, default=HERE / "window_gaps.npz",
                     help="find_window_gaps.py's output")
    ap.add_argument("--windows-json", type=Path, default=None,
                     help="default: same basename as --gaps with windows.json's "
                          "naming (find_window_gaps.py's --windows-out)")
    ap.add_argument("--voxels", type=Path,
                     default=HERE.parent / "AlignedOctree" / "voxels.npz",
                     help="context building (translucent) -- the SAME voxels.npz "
                          "--gaps was computed from, ideally")
    ap.add_argument("--planes-aligned", type=Path,
                     default=HERE.parent / "AlignedOctree" / "planes_aligned.json")
    ap.add_argument("--no-context", action="store_true", help="don't show the context building")
    ap.add_argument("--no-planes", action="store_true", help="don't overlay the closed box")
    ap.add_argument("--transform", type=Path,
                     default=HERE.parent / "AlignedOctree" / "transform.json")
    ap.add_argument("--points", action="store_true",
                     help="overlay the raw aligned points (re-reads the bag)")
    ap.add_argument("--points-stride", type=int, default=20)
    ap.add_argument("--screenshot", type=Path, default=None)
    ap.add_argument("--orbit-gif", type=Path, default=None)
    ap.add_argument("--orbit-frames", type=int, default=36)
    args = ap.parse_args()

    gaps = np.load(args.gaps)
    fill_centers = gaps["centers"]
    window_id = gaps["window_id"]
    wall_id = gaps["wall_id"]
    fill_voxel_size = float(gaps["voxel_size"])
    print(f"loaded {args.gaps}: {len(fill_centers)} fill voxel(s)")

    windows_json = args.windows_json
    if windows_json is None:
        candidate = args.gaps.with_name(args.gaps.stem.replace("window_gaps", "windows") + ".json")
        windows_json = candidate if candidate.exists() else args.gaps.parent / "windows.json"
    windows_meta = []
    if windows_json.exists():
        windows_meta = json.loads(windows_json.read_text())["windows"]
        print(f"loaded {windows_json}: {len(windows_meta)} window(s)")
    else:
        print(f"NOTE: {windows_json} not found -- per-window legend text will be empty")

    context_centers, context_voxel_size = None, fill_voxel_size
    if not args.no_context and args.voxels.exists():
        v = np.load(args.voxels)
        context_centers = v["centers"]
        context_voxel_size = float(v["voxel_size"])
        print(f"loaded {args.voxels}: {len(context_centers)} context voxel(s)")

    box_mesh = None
    if not args.no_planes and args.planes_aligned.exists():
        box_mesh = load_box_wireframe(args.planes_aligned)

    points = None
    if args.points:
        points = load_aligned_points(args.transform, stride=args.points_stride)
        print(f"overlaying {len(points)} raw points (stride {args.points_stride})")

    off_screen = args.screenshot is not None or args.orbit_gif is not None
    pl = build_plotter(
        context_centers, context_voxel_size, fill_centers, window_id, wall_id,
        fill_voxel_size, windows_meta, box_mesh, points, off_screen=off_screen,
        show_context=not args.no_context)

    if args.screenshot:
        pl.screenshot(str(args.screenshot))
        print(f"wrote {args.screenshot}")
    elif args.orbit_gif:
        pl.open_gif(str(args.orbit_gif))
        orbit_path = pl.generate_orbital_path(factor=2.5, n_points=args.orbit_frames, viewup=[0, 0, 1])
        pl.orbit_on_path(orbit_path, write_frames=True, viewup=[0, 0, 1])
        pl.close()
        print(f"wrote {args.orbit_gif} ({args.orbit_frames} frames)")
    else:
        pl.show()


if __name__ == "__main__":
    main()
