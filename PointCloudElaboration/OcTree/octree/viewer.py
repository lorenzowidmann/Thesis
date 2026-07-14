"""PyVista GUI showing how a point cloud is sampled into voxels.

The window shows the voxels at the current voxel size and, optionally, the raw
points overlaid. A voxel-size slider changes the resolution live (0.05-1.0 m),
a minimum-points-per-voxel slider (1-10) combined with a filter toggle hides
sparse voxels (often disconnected scan noise), a points checkbox overlays the
raw points, and a legend maps colors to semantic classes.

The interactive slider works in metric voxel size (metres) rather than octree
depth, so it can be limited to a real-world range; the octree hierarchy itself
(power-of-two depths) still drives `main.py --info` / `--selftest`.
"""

import numpy as np

from .classes import CLASSES, class_name, colorize
from .smoothing import PlaneAnchor, merge_planar_surface, project_axis_aligned, smooth_surface
from .voxelizer import filter_by_count, verify_nonempty, voxelize

# Metric limits of the voxel-size slider.
MIN_VOXEL_M = 0.05
MAX_VOXEL_M = 1.0
DEFAULT_VOXEL_M = 0.20

# Limits of the minimum-points-per-voxel filter slider. When the filter toggle
# is on, voxels with fewer than this many points are hidden — a simple density
# filter for sparse, often-disconnected/noisy voxels (see filter_by_count in
# voxelizer.py for what it does and does not guarantee).
MIN_COUNT_FLOOR = 1
MIN_COUNT_CEIL = 10
DEFAULT_MIN_COUNT = 1
DEFAULT_FILTER_ON = False

# Points drawn without subsampling up to this many (the default cloud is ~1M,
# so every voxel visibly contains its points; larger clouds subsample only for
# display speed — the non-empty check still runs on the full data).
MAX_DISPLAY_POINTS = 2_000_000

# Surface smoothing (flatten voxels to a plane for OpenStudio). The plane is
# found by a RANSAC dominant-plane fit (offset_method 'ransac', see smoothing.py)
# so it follows the actual wall at any orientation. The checkbox toggles
# smoothing on/off and the small u/v/z selector chooses which detected plane to
# flatten: 'u' the dominant facade (default), 'v' the perpendicular facade, 'z'
# the roof/floor. tolerance comes from the CLI.
DEFAULT_SMOOTH_ON = False
DEFAULT_SMOOTH_AXIS = "u"
SMOOTH_AXIS_CHOICES = ("u", "v", "z")
DEFAULT_OFFSET_METHOD = "ransac"
DEFAULT_TOLERANCE = 3

# Anchor toggle (RANSAC path only): without it, every voxel-size change re-fits
# the plane from scratch, and a different voxel size can pick a slightly
# different plane (different inlier set -> different position/orientation) —
# the surface visibly drifts. With it on, the plane fitted at the moment the
# toggle is switched on (or the axis is changed) is captured as a PlaneAnchor
# and reused on every later voxel-size/filter change, so only the raster's
# resolution changes, not which plane or where on it the surface sits.
DEFAULT_ANCHOR_ON = False

# Axis-aligned re-projection toggle (RANSAC path only). When on, the smoothed
# surface is re-rastered on a world-axis-aligned grid (vertical columns on a
# facade, X/Y on a roof) instead of the diagonal PCA basis, and colour blobs
# smaller than DEFAULT_MIN_SIDE metres on both sides are dropped as noise
# (see smoothing.project_axis_aligned). Off by default — the diagonal PCA
# surface stays the default behaviour.
DEFAULT_AXIS_ALIGNED_ON = False
DEFAULT_MIN_SIDE = 1.0

# Adjacent-rectangle merging (opt-in, post-smoothing). Launch-time only (no
# live toggle, matching --tolerance/--offset-method/--ransac-iters, which
# also aren't exposed as GUI widgets) — see main.py's --merge-adjacent /
# --merge-gap-tolerance / --merge-fit-strategy and smoothing.merge_planar_surface.
DEFAULT_MERGE_ADJACENT_ON = False
DEFAULT_MERGE_GAP_TOLERANCE = 0.0
DEFAULT_MERGE_FIT_STRATEGY = "max_inscribed"
DEFAULT_MERGE_MIN_COVERAGE = 0.8

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


def _planar_mesh(surface):
    """Build a class-colored quad mesh from a PlanarSurface's sub-surface polygons."""
    import pyvista as pv

    verts, faces, colors = [], [], []
    n = 0
    for sub in surface.subsurfaces:
        rgb = colorize(np.array([sub.class_id]))[0]
        for poly in sub.polygons:
            verts.append(poly)
            faces.extend([4, n, n + 1, n + 2, n + 3])
            colors.append(rgb)
            n += 4
    if not verts:
        return None
    mesh = pv.PolyData(np.vstack(verts), np.array(faces))
    mesh.cell_data["colors"] = np.asarray(colors, float)
    return mesh


def _build_voxels(points, labels, voxel_size, min_count=1):
    """Voxelize, then optionally filter to voxels with >= min_count points.

    Returns (glyphs, full_grid, shown_grid): full_grid backs the non-empty
    check and the "total voxels" stat; shown_grid is what min_count kept
    (equal to full_grid when min_count <= 1) and is what gets rendered.
    """
    grid = voxelize(points, labels, voxel_size)
    shown = filter_by_count(grid, min_count)
    glyphs = _cube_glyphs(shown.centers, colorize(shown.labels), shown.voxel_size)
    return glyphs, grid, shown


def _present_class_legend(labels: np.ndarray):
    present = sorted(int(c) for c in np.unique(labels))
    return [[class_name(c), tuple(CLASSES.get(c, ("", (0.25, 0.25, 0.25)))[1])] for c in present]


def _info_text(grid_full, grid_shown, filter_on, min_count, status):
    text = f"voxel {grid_full.voxel_size:.2f} m   {len(grid_full):,} voxels"
    if filter_on and min_count > 1:
        pct = (len(grid_shown) / len(grid_full)) if len(grid_full) else 0.0
        text += f"   | filter >={min_count} pts: {len(grid_shown):,} shown ({pct:.0%})"
    return text + f"   (>=1 pt/voxel: {status})"


def _smooth_info_text(surface, anchored: bool = False):
    anchor_note = "  [anchored]" if anchored else ""
    if surface.n_polygons == 0:
        return (
            f"SMOOTH |{surface.axis}|  no plane found at this voxel size — "
            f"try another axis or a coarser voxel{anchor_note}"
        )
    if surface.normal is not None:
        nx, ny, nz = surface.normal
        plane_note = f" n=({nx:+.2f},{ny:+.2f},{nz:+.2f})"
        label = "RANSAC"
    else:
        plane_note = f" yaw={surface.rotation_deg:.1f}deg" if surface.axis in ("u", "v") else ""
        label = "SMOOTHED"
    fill_note = ""
    if surface.n_filled or surface.n_unknown:
        fill_note = f"   | filled {surface.n_filled:,} (+{surface.n_unknown:,} unknown)"
    return (
        f"{label} |{surface.axis}|{plane_note} plane {surface.plane_coord:+.2f} m   "
        f"{surface.n_inliers:,} in / {surface.n_deviations:,} dev   "
        f"{len(surface.subsurfaces)} sub-surfaces, {surface.n_polygons} polygons"
        f"{fill_note}{anchor_note}"
    )


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
    points, labels, path, voxel_size=DEFAULT_VOXEL_M, max_points=MAX_DISPLAY_POINTS,
    show_points=False, min_count=1, smooth=False, smooth_axis=DEFAULT_SMOOTH_AXIS,
    offset_method=DEFAULT_OFFSET_METHOD, tolerance=DEFAULT_TOLERANCE, rotation_deg=None,
    ransac_threshold=None, ransac_iters=500, seed=0,
    axis_aligned=DEFAULT_AXIS_ALIGNED_ON, min_side=DEFAULT_MIN_SIDE,
    merge_adjacent=DEFAULT_MERGE_ADJACENT_ON, merge_gap_tolerance=DEFAULT_MERGE_GAP_TOLERANCE,
    merge_fit_strategy=DEFAULT_MERGE_FIT_STRATEGY, merge_min_coverage=DEFAULT_MERGE_MIN_COVERAGE,
):
    """Headless render of voxels or a smoothed planar surface to an image file.

    min_count > 1 renders only voxels with at least that many points. smooth=True
    flattens the (filtered) grid onto a plane and renders the resulting planar
    sub-surfaces instead of the voxels. With the default offset_method 'ransac',
    smooth_axis selects which RANSAC-fitted plane to use ('u' dominant facade,
    'v' perpendicular facade, 'z' roof/floor); ransac_threshold/iters/seed tune
    the fit. rotation_deg only applies to the legacy 'mode'/'median'/'outer'
    methods on axis 'u'/'v'. merge_adjacent=True runs merge_planar_surface after
    smoothing (and after axis_aligned, if both are on) and prints its diagnostic
    report to the console.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
    pl.set_background("white")
    if show_points:
        _add_points(pl, points, max_points)
    glyphs, grid, shown = _build_voxels(points, labels, voxel_size, min_count)

    if smooth:
        surface = smooth_surface(
            shown, smooth_axis, offset_method, tolerance, rotation_deg=rotation_deg,
            ransac_threshold=ransac_threshold, ransac_iters=ransac_iters, seed=seed,
        )
        if axis_aligned and surface.normal is not None:
            surface = project_axis_aligned(
                shown, surface, min_side_m=min_side, tolerance_voxels=tolerance,
            )
        if merge_adjacent and surface.normal is None and smooth_axis in ("u", "v"):
            print(
                "[merge] skipped: needs offset_method='ransac' when smooth_axis is "
                "'u'/'v' (legacy PCA-yaw u/v can't be reversed to its raster grid)"
            )
        elif merge_adjacent:
            surface, merge_summary = merge_planar_surface(
                surface, gap_tolerance=merge_gap_tolerance, fit_strategy=merge_fit_strategy,
                min_coverage=merge_min_coverage,
            )
            merge_summary.print_report()
        mesh = _planar_mesh(surface)
        if mesh is not None:
            pl.add_mesh(mesh, scalars="colors", rgb=True, name="voxels", show_edges=True)
        pl.add_text(
            _smooth_info_text(surface), font_size=11, name="info",
        )
        result = surface
    else:
        _add_voxels(pl, glyphs, show_points)
        ok, n_empty, _ = verify_nonempty(grid, len(points))
        status = "OK" if ok else f"FAIL: {n_empty} empty!"
        pl.add_text(
            _info_text(grid, shown, min_count > 1, min_count, status),
            font_size=11, name="info",
        )
        result = shown

    pl.camera_position = "iso"
    pl.screenshot(path)
    pl.close()
    return result


def launch(
    points, labels, voxel_size=DEFAULT_VOXEL_M, max_points=MAX_DISPLAY_POINTS,
    min_count=DEFAULT_MIN_COUNT, filter_on=DEFAULT_FILTER_ON,
    smooth_on=DEFAULT_SMOOTH_ON, smooth_axis=DEFAULT_SMOOTH_AXIS,
    offset_method=DEFAULT_OFFSET_METHOD, tolerance=DEFAULT_TOLERANCE, rotation_deg=None,
    ransac_threshold=None, ransac_iters=500, seed=0, anchor_on=DEFAULT_ANCHOR_ON,
    axis_aligned_on=DEFAULT_AXIS_ALIGNED_ON, min_side=DEFAULT_MIN_SIDE,
    merge_adjacent_on=DEFAULT_MERGE_ADJACENT_ON, merge_gap_tolerance=DEFAULT_MERGE_GAP_TOLERANCE,
    merge_fit_strategy=DEFAULT_MERGE_FIT_STRATEGY, merge_min_coverage=DEFAULT_MERGE_MIN_COVERAGE,
):
    """Open the interactive viewer: voxel-size slider, min-points filter, points/filter/smooth toggles.

    Smoothing fits the plane with RANSAC (offset_method 'ransac'); the smooth
    axis (u/v/z) has its own live radio-style selector that chooses which
    detected plane to flatten (dominant facade / perpendicular facade /
    roof-floor), so the other wall can be inspected without relaunching. The
    "anchor plane" toggle (RANSAC only) freezes the currently fitted plane so
    later voxel-size/filter changes reproject onto it instead of re-fitting —
    see DEFAULT_ANCHOR_ON. merge_adjacent_on runs merge_planar_surface on every
    re-render (launch-time only, no live toggle — see DEFAULT_MERGE_ADJACENT_ON)
    and prints its diagnostic report to the console each time.
    """
    import pyvista as pv

    pl = pv.Plotter(window_size=(1400, 900))
    pl.set_background("white")

    points_actor = _add_points(pl, points, max_points)
    points_actor.SetVisibility(DEFAULT_POINTS_ON)
    state = {
        "voxel": float(voxel_size),
        "points_on": DEFAULT_POINTS_ON,
        "min_count": int(min_count),
        "filter_on": bool(filter_on),
        "smooth_on": bool(smooth_on),
        # Respected as-is even if it's a literal 'x'/'y' not on the live toggle
        # row below (which only offers u/v/z) — clicking one of those switches
        # away from it, but a startup choice of x/y isn't silently discarded.
        "smooth_axis": smooth_axis,
        "rotation_deg": rotation_deg,
        "anchor_on": bool(anchor_on) and offset_method == "ransac",
        "anchor_plane": None,  # captured (or reused) the first time anchor_on is True
        "axis_aligned_on": bool(axis_aligned_on) and offset_method == "ransac",
        "min_side": float(min_side),
        "merge_adjacent_on": bool(merge_adjacent_on),
        "merge_gap_tolerance": float(merge_gap_tolerance),
        "merge_fit_strategy": merge_fit_strategy,
        "merge_min_coverage": float(merge_min_coverage),
    }
    axis_widgets = {}  # filled after creation, used to keep the radio selection in sync

    def show_voxels():
        effective_min = state["min_count"] if state["filter_on"] else 1
        glyphs, grid, shown = _build_voxels(points, labels, state["voxel"], effective_min)

        if state["smooth_on"]:
            # Pipeline: voxelize -> (filter) -> smooth. Render planar surface.
            use_anchor = state["anchor_plane"] if state["anchor_on"] else None
            surface = smooth_surface(
                shown, state["smooth_axis"], offset_method, tolerance,
                rotation_deg=state["rotation_deg"], ransac_threshold=ransac_threshold,
                ransac_iters=ransac_iters, seed=seed, anchor=use_anchor,
            )
            if state["anchor_on"]:
                # Latch the plane just used (fresh fit the first time, or the
                # same anchor reused) so it carries forward unchanged. Capture it
                # from the RANSAC surface, before any axis-aligned re-projection.
                state["anchor_plane"] = PlaneAnchor.from_surface(surface) or state["anchor_plane"]
            if state["axis_aligned_on"] and surface.normal is not None:
                # Second, world-axis-aligned surface derived from the RANSAC one
                # (reuses its plane; drops sub-min_side colour blobs).
                surface = project_axis_aligned(
                    shown, surface, min_side_m=state["min_side"], tolerance_voxels=tolerance,
                )
            if state["merge_adjacent_on"] and surface.normal is None and state["smooth_axis"] in ("u", "v"):
                print(
                    "[merge] skipped: needs offset_method='ransac' when smooth_axis is "
                    "'u'/'v' (legacy PCA-yaw u/v can't be reversed to its raster grid)"
                )
            elif state["merge_adjacent_on"]:
                surface, merge_summary = merge_planar_surface(
                    surface, gap_tolerance=state["merge_gap_tolerance"],
                    fit_strategy=state["merge_fit_strategy"],
                    min_coverage=state["merge_min_coverage"],
                )
                merge_summary.print_report()
            mesh = _planar_mesh(surface)
            pl.remove_actor("voxels", reset_camera=False)
            if mesh is not None:
                pl.add_mesh(mesh, scalars="colors", rgb=True, name="voxels", show_edges=True)
            text = _smooth_info_text(surface, anchored=state["anchor_on"])
            print(f"[smooth] {text}")
            pl.add_text(text, font_size=10, position="upper_left", name="info")
            return

        _add_voxels(pl, glyphs, state["points_on"])

        # Verify the invariant on the full (unfiltered) grid after every
        # change: no empty voxel, all points binned. Report on-screen + console.
        ok, n_empty, n_binned = verify_nonempty(grid, len(points))
        status = "OK" if ok else f"FAIL: {n_empty} empty!"
        filter_note = (
            f" | filter >={effective_min} pts: {len(shown):,} shown"
            if state["filter_on"] else ""
        )
        print(
            f"[check] voxel {grid.voxel_size:.3f} m: {len(grid):,} voxels, "
            f"min {grid.counts.min()} pt/voxel, "
            f"{n_binned:,}/{len(points):,} points binned -> {status}{filter_note}"
        )
        pl.add_text(
            _info_text(grid, shown, state["filter_on"], effective_min, status),
            font_size=10, position="upper_left", name="info",
        )

    def on_size(value):
        v = float(value)
        if abs(v - state["voxel"]) > 1e-6:
            state["voxel"] = v
            show_voxels()

    def on_min_count(value):
        v = int(round(value))
        if v != state["min_count"]:
            state["min_count"] = v
            if state["filter_on"]:
                show_voxels()

    def on_toggle_points(flag):
        state["points_on"] = bool(flag)
        points_actor.SetVisibility(bool(flag))
        show_voxels()  # solid <-> wireframe so points stay visible

    def on_toggle_filter(flag):
        state["filter_on"] = bool(flag)
        show_voxels()

    def on_toggle_smooth(flag):
        state["smooth_on"] = bool(flag)
        show_voxels()

    def on_toggle_anchor(flag):
        state["anchor_on"] = bool(flag) and offset_method == "ransac"
        # Dropping the anchor means "forget where we were" — re-enabling later
        # re-latches fresh to whatever plane is current at that point, rather
        # than snapping back to a stale one.
        state["anchor_plane"] = None
        if state["smooth_on"]:
            show_voxels()

    def on_toggle_axis_aligned(flag):
        state["axis_aligned_on"] = bool(flag) and offset_method == "ransac"
        if state["smooth_on"]:
            show_voxels()

    def on_pick_axis(axis):
        def handler(flag):
            if not flag:
                # Clicking the already-active axis would otherwise uncheck it
                # with nothing selected; re-check it and ignore (radio button).
                axis_widgets[axis].GetRepresentation().SetState(1)
                return
            state["smooth_axis"] = axis
            # A different axis means a different physical plane — drop the old
            # anchor so it re-latches fresh to the newly selected facade/roof.
            state["anchor_plane"] = None
            for other, w in axis_widgets.items():
                w.GetRepresentation().SetState(1 if other == axis else 0)
            if state["smooth_on"]:
                show_voxels()
        return handler

    show_voxels()
    # Voxel-size slider along the bottom edge, clear of the legend.
    pl.add_slider_widget(
        on_size, [MIN_VOXEL_M, MAX_VOXEL_M], value=voxel_size,
        title="voxel size (m)", fmt="%.2f", style="modern",
        pointa=(0.30, 0.08), pointb=(0.70, 0.08),
        title_height=0.018, slider_width=0.02, tube_width=0.004,
        interaction_event="end",
    )
    # Minimum-points-per-voxel filter slider, stacked above the voxel-size one.
    pl.add_slider_widget(
        on_min_count, [MIN_COUNT_FLOOR, MIN_COUNT_CEIL], value=state["min_count"],
        title="min points/voxel", fmt="%.0f", style="modern",
        pointa=(0.30, 0.16), pointb=(0.70, 0.16),
        title_height=0.018, slider_width=0.02, tube_width=0.004,
        interaction_event="end",
    )
    pl.add_checkbox_button_widget(on_toggle_points, value=DEFAULT_POINTS_ON, size=26, position=(10, 10))
    pl.add_text("points on/off", font_size=9, position=(44, 12), name="toggle_label")
    pl.add_checkbox_button_widget(on_toggle_filter, value=state["filter_on"], size=26, position=(10, 46))
    pl.add_text("filter on/off", font_size=9, position=(44, 48), name="filter_toggle_label")
    pl.add_checkbox_button_widget(on_toggle_smooth, value=state["smooth_on"], size=26, position=(10, 82))
    pl.add_text("smooth on/off", font_size=9, position=(44, 84), name="smooth_toggle_label")
    pl.add_checkbox_button_widget(on_toggle_anchor, value=state["anchor_on"], size=26, position=(10, 118))
    pl.add_text("anchor plane on/off", font_size=9, position=(44, 120), name="anchor_toggle_label")
    pl.add_checkbox_button_widget(on_toggle_axis_aligned, value=state["axis_aligned_on"], size=26, position=(10, 154))
    pl.add_text("axis-aligned on/off", font_size=9, position=(44, 156), name="axis_aligned_toggle_label")
    # Live radio-style axis selector (u / v / z) for smoothing — lets the
    # other wall direction be inspected without relaunching the script.
    pl.add_text("axis:", font_size=9, position=(10, 191), name="axis_row_label")
    axis_x = {"u": 60, "v": 100, "z": 140}
    for a in SMOOTH_AXIS_CHOICES:
        axis_widgets[a] = pl.add_checkbox_button_widget(
            on_pick_axis(a), value=(a == state["smooth_axis"]), size=20, position=(axis_x[a], 190)
        )
        pl.add_text(a, font_size=9, position=(axis_x[a] + 24, 191), name=f"axis_label_{a}")
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
