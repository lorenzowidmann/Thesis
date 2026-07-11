"""PyVista GUI showing how a point cloud is sampled into voxels.

The window shows the voxels at the current voxel size and, optionally, the raw
points overlaid. A slider changes the voxel size live (0.05-1.0 m), a checkbox
toggles the raw points, and a legend maps colors to semantic classes.

The interactive slider works in metric voxel size (metres) rather than octree
depth, so it can be limited to a real-world range; the octree hierarchy itself
(power-of-two depths) still drives `main.py --info` / `--selftest`.
"""

import numpy as np

from .classes import CLASSES, class_name, colorize
from .voxelizer import verify_nonempty, voxelize

# Metric limits of the voxel-size slider.
MIN_VOXEL_M = 0.05
MAX_VOXEL_M = 1.0
DEFAULT_VOXEL_M = 0.20

# Points drawn without subsampling up to this many (the default cloud is ~1M,
# so every voxel visibly contains its points; larger clouds subsample only for
# display speed — the non-empty check still runs on the full data).
MAX_DISPLAY_POINTS = 2_000_000

# Raw points get a single high-contrast color (not class colors) so they stand
# out against the class-colored voxels when toggled on.
POINT_COLOR = "black"
POINT_SIZE = 4.0

# When points are shown, the voxels are drawn as class-colored wireframe cages
# (same colors as the solid mode) so the opaque cube faces don't hide the black
# points inside them; when hidden, the voxels are solid and class-colored.
DEFAULT_POINTS_ON = False


def _add_voxels(pl, glyphs, points_on: bool):
    """Add the voxel mesh: solid class-colored, or class-colored wireframe when points show."""
    style = "wireframe" if points_on else "surface"
    return pl.add_mesh(
        glyphs, scalars="colors", rgb=True, style=style, line_width=1, name="voxels"
    )


def _cube_glyphs(centers: np.ndarray, colors: np.ndarray, size: float):
    """Build one cube of edge `size` at each center, carrying per-cube colors."""
    import pyvista as pv

    pdata = pv.PolyData(centers)
    pdata["colors"] = colors
    cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
    return pdata.glyph(geom=cube, scale=False, orient=False, factor=size)


def _build_voxels(points, labels, voxel_size):
    grid = voxelize(points, labels, voxel_size)
    glyphs = _cube_glyphs(grid.centers, colorize(grid.labels), grid.voxel_size)
    return glyphs, grid


def _present_class_legend(labels: np.ndarray):
    present = sorted(int(c) for c in np.unique(labels))
    return [[class_name(c), tuple(CLASSES.get(c, ("", (0.25, 0.25, 0.25)))[1])] for c in present]


def _add_points(pl, points, max_points):
    """Add the raw points in a single contrasting color, subsampled if huge."""
    import pyvista as pv

    pts = points
    if len(points) > max_points:
        sel = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        pts = points[sel]
    cloud = pv.PolyData(pts)
    return pl.add_mesh(
        cloud, color=POINT_COLOR, point_size=POINT_SIZE,
        render_points_as_spheres=True, name="points",
    )


def render_screenshot(
    points, labels, path, voxel_size=DEFAULT_VOXEL_M, max_points=MAX_DISPLAY_POINTS, show_points=False
):
    """Headless render of voxels (+ optional points) to an image file."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
    pl.set_background("white")
    if show_points:
        _add_points(pl, points, max_points)
    glyphs, grid = _build_voxels(points, labels, voxel_size)
    _add_voxels(pl, glyphs, show_points)
    ok, n_empty, _ = verify_nonempty(grid, len(points))
    status = "OK" if ok else f"FAIL: {n_empty} empty!"
    pl.add_text(
        f"voxel {grid.voxel_size:.2f} m   {len(grid):,} voxels   (>=1 pt/voxel: {status})",
        font_size=11, name="info",
    )
    pl.camera_position = "iso"
    pl.screenshot(path)
    pl.close()
    return grid


def launch(points, labels, voxel_size=DEFAULT_VOXEL_M, max_points=MAX_DISPLAY_POINTS):
    """Open the interactive viewer with a voxel-size slider and a points toggle."""
    import pyvista as pv

    pl = pv.Plotter(window_size=(1400, 900))
    pl.set_background("white")

    points_actor = _add_points(pl, points, max_points)
    points_actor.SetVisibility(DEFAULT_POINTS_ON)
    state = {"voxel": float(voxel_size), "points_on": DEFAULT_POINTS_ON}

    def show_voxels(size):
        glyphs, grid = _build_voxels(points, labels, size)
        _add_voxels(pl, glyphs, state["points_on"])

        # Verify the invariant after every voxel-size change: no empty voxel,
        # all points binned. Report it on-screen and in the console.
        ok, n_empty, n_binned = verify_nonempty(grid, len(points))
        status = "OK" if ok else f"FAIL: {n_empty} empty!"
        print(
            f"[check] voxel {grid.voxel_size:.3f} m: {len(grid):,} voxels, "
            f"min {grid.counts.min()} pt/voxel, "
            f"{n_binned:,}/{len(points):,} points binned -> {status}"
        )
        pl.add_text(
            f"voxel {grid.voxel_size:.2f} m   {len(grid):,} voxels"
            f"   (>=1 pt/voxel: {status})",
            font_size=10, position="upper_left", name="info",
        )

    def on_size(value):
        v = float(value)
        if abs(v - state["voxel"]) > 1e-6:
            state["voxel"] = v
            show_voxels(v)

    def on_toggle_points(flag):
        state["points_on"] = bool(flag)
        points_actor.SetVisibility(bool(flag))
        show_voxels(state["voxel"])  # solid <-> wireframe so points stay visible

    show_voxels(voxel_size)
    # Voxel-size slider along the bottom edge, clear of the legend.
    pl.add_slider_widget(
        on_size, [MIN_VOXEL_M, MAX_VOXEL_M], value=voxel_size,
        title="voxel size (m)", fmt="%.2f", style="modern",
        pointa=(0.30, 0.08), pointb=(0.70, 0.08),
        title_height=0.018, slider_width=0.02, tube_width=0.004,
        interaction_event="end",
    )
    pl.add_checkbox_button_widget(on_toggle_points, value=DEFAULT_POINTS_ON, size=26, position=(10, 10))
    pl.add_text("points on/off", font_size=9, position=(44, 12), name="toggle_label")
    # Compact class legend pinned to the lower-right corner (small font: the
    # row height drives the text size, so keep it tight).
    legend = _present_class_legend(labels)
    height = min(0.018 * len(legend) + 0.01, 0.28)
    pl.add_legend(
        legend, bcolor="white", size=(0.10, height), loc="lower right",
        face=None, border=True,
    )
    pl.camera_position = "iso"
    pl.show()
