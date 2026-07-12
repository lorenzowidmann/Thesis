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
from .planes import run_dominant_plane
from .smoothing import smooth_surface
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

# Surface smoothing (flatten voxels to a plane for OpenStudio). offset_method /
# tolerance come from the CLI; the checkbox toggles smoothing on/off and a
# small u/v/z selector switches axis live, like the other steps. 'u' (the
# PCA-detected dominant wall direction) is the default: real buildings are
# rarely aligned with world x/y, so literal-axis flattening usually cuts
# across the facade instead of following it.
DEFAULT_SMOOTH_ON = False
DEFAULT_SMOOTH_AXIS = "u"
SMOOTH_AXIS_CHOICES = ("u", "v", "z")
DEFAULT_OFFSET_METHOD = "mode"
DEFAULT_TOLERANCE = 3

# RANSAC dominant-plane -> wall raster heatmap. The "planes" checkbox runs the
# RANSAC pipeline (planes.run_dominant_plane) on the voxel centroids and draws
# the flattened per-cell temperature raster as a heatmap on the fitted plane,
# with the minimum-area wall rectangle outlined. Parameters come from the CLI.
DEFAULT_PLANES_ON = False
DEFAULT_RANSAC_THRESHOLD = 0.10
DEFAULT_RANSAC_ITERS = 1000
DEFAULT_RASTER_CELL = None  # None -> use the current voxel size
DEFAULT_GROUND_BAND = 0.5

# Temperature heatmap colormap (cool blue -> pale -> hot red), interpolated in
# numpy so no matplotlib dependency is needed (mirrors the rgb=True class path).
_HEAT_STOPS = np.array([
    [0.23, 0.30, 0.75],
    [0.40, 0.65, 0.95],
    [0.95, 0.95, 0.75],
    [0.98, 0.65, 0.30],
    [0.75, 0.12, 0.15],
])

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


def _scalar_to_rgb(vals, vmin, vmax):
    """Map scalar values to RGB via the numpy heatmap colormap (cool->hot)."""
    span = vmax - vmin
    t = np.clip((vals - vmin) / span, 0.0, 1.0) if span > 1e-9 else np.zeros_like(vals)
    x = t * (len(_HEAT_STOPS) - 1)
    i0 = np.clip(np.floor(x).astype(int), 0, len(_HEAT_STOPS) - 2)
    f = (x - i0)[:, None]
    return _HEAT_STOPS[i0] * (1.0 - f) + _HEAT_STOPS[i0 + 1] * f


def _wall_heatmap_mesh(wall):
    """Quad mesh of the wall's occupied raster cells, colored by temperature.

    Each occupied cell (u, v) is placed in 3-D at origin + u*e_u + v*e_v and
    colored by its mean temperature (heatmap). Empty cells are omitted.
    """
    import pyvista as pv

    r = wall.raster
    occ = np.argwhere(r.counts > 0)
    if len(occ) == 0:
        return None
    iu, iv = occ[:, 0], occ[:, 1]
    cs = r.cell_size
    u_lo = r.u0 + iu * cs
    v_lo = r.v0 + iv * cs

    def world(u, v):
        return (
            wall.origin[None, :]
            + u[:, None] * wall.e_u[None, :]
            + v[:, None] * wall.e_v[None, :]
        )

    c00 = world(u_lo, v_lo)
    c10 = world(u_lo + cs, v_lo)
    c11 = world(u_lo + cs, v_lo + cs)
    c01 = world(u_lo, v_lo + cs)

    n = len(occ)
    verts = np.empty((n * 4, 3))
    verts[0::4], verts[1::4], verts[2::4], verts[3::4] = c00, c10, c11, c01
    base = np.arange(n) * 4
    faces = np.column_stack([np.full(n, 4), base, base + 1, base + 2, base + 3])

    cell_vals = r.values[iu, iv]
    finite = cell_vals[np.isfinite(cell_vals)]
    if len(finite):
        vmin, vmax = np.percentile(finite, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    colors = _scalar_to_rgb(np.nan_to_num(cell_vals, nan=vmin), vmin, vmax)

    mesh = pv.PolyData(verts, faces.ravel())
    mesh.cell_data["colors"] = colors
    return mesh, (float(vmin), float(vmax))


def _wall_outline(wall):
    """Closed polyline (4 corners) of the wall rectangle."""
    import pyvista as pv

    pts = np.vstack([wall.rect_xyz, wall.rect_xyz[0]])
    return pv.lines_from_points(pts)


def _planes_info_text(wall, vrange):
    w, h = wall.rect_dims
    n = wall.normal
    qc = wall.offset_stats
    return (
        f"RANSAC PLANE {wall.rank}/{wall.n_candidates} | "
        f"n=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})  "
        f"{wall.n_inliers:,}/{wall.n_fitted:,} inliers "
        f"(ground -{wall.n_ground:,})\n"
        f"wall {w:.1f} x {h:.1f} m   raster {wall.raster.shape[0]}x"
        f"{wall.raster.shape[1]} @ {wall.cell_size:.2f} m   "
        f"T {vrange[0]:.1f}-{vrange[1]:.1f} C\n"
        f"QC offset d: mean {qc['d_mean_abs_m']*100:.1f} cm, "
        f"p95 {qc['d_p95_abs_m']*100:.1f} cm, "
        f"protrusions {qc['n_protrusions']:,} ({qc['protrusion_frac']:.1%})"
    )


def _set_face_on(pl, wall):
    """Point the camera straight at the wall plane so the raster reads as 2-D."""
    w, h = wall.rect_dims
    dist = 1.5 * max(w, h, 1.0)
    pl.camera_position = [
        tuple(wall.origin + wall.normal * dist),
        tuple(wall.origin),
        tuple(wall.e_v),
    ]


def _render_planes(pl, wall):
    """Draw the wall heatmap + rectangle outline into plotter `pl`; return vrange."""
    pl.remove_actor("voxels", reset_camera=False)
    result = _wall_heatmap_mesh(wall)
    vrange = (0.0, 1.0)
    if result is not None:
        mesh, vrange = result
        pl.add_mesh(mesh, scalars="colors", rgb=True, name="voxels")
    pl.add_mesh(_wall_outline(wall), color="black", line_width=3, name="wall_outline")
    return vrange


def _build_voxels(points, labels, voxel_size, min_count=1, values=None):
    """Voxelize, then optionally filter to voxels with >= min_count points.

    Returns (glyphs, full_grid, shown_grid): full_grid backs the non-empty
    check and the "total voxels" stat; shown_grid is what min_count kept
    (equal to full_grid when min_count <= 1) and is what gets rendered.
    `values` is an optional per-point scalar (temperature) averaged per voxel.
    """
    grid = voxelize(points, labels, voxel_size, values=values)
    shown = filter_by_count(grid, min_count)
    glyphs = _cube_glyphs(shown.centers, colorize(shown.labels), shown.voxel_size)
    return glyphs, grid, shown


def _compute_wall(shown, threshold, iters, seed, keep_ground, ground_band, raster_cell,
                  rank=1, target_normal=None, orientation="any"):
    """Run the RANSAC dominant-plane pipeline on the voxel centroids of `shown`.

    Uses the per-voxel mean temperature (shown.values) when available, else a
    synthetic field on the centroids so the heatmap is populated on today's
    (thermal-less) sample. raster_cell None -> the current voxel size. `rank`
    (1-based, by inlier count), `target_normal` and `orientation` select which
    detected plane to show (rank=2 -> the other facade).
    """
    from .planes import synthetic_temperature

    if len(shown) < 3:
        return None
    vals = shown.values if shown.values is not None else synthetic_temperature(shown.centers)
    cell = raster_cell if raster_cell else shown.voxel_size
    return run_dominant_plane(
        shown.centers, vals, threshold=threshold, iters=iters, seed=seed,
        keep_ground=keep_ground, ground_band=ground_band, raster_cell=cell,
        labels=shown.labels, rank=rank, target_normal=target_normal,
        orientation=orientation,
    )


def _present_class_legend(labels: np.ndarray):
    present = sorted(int(c) for c in np.unique(labels))
    return [[class_name(c), tuple(CLASSES.get(c, ("", (0.25, 0.25, 0.25)))[1])] for c in present]


def _info_text(grid_full, grid_shown, filter_on, min_count, status):
    text = f"voxel {grid_full.voxel_size:.2f} m   {len(grid_full):,} voxels"
    if filter_on and min_count > 1:
        pct = (len(grid_shown) / len(grid_full)) if len(grid_full) else 0.0
        text += f"   | filter >={min_count} pts: {len(grid_shown):,} shown ({pct:.0%})"
    return text + f"   (>=1 pt/voxel: {status})"


def _smooth_info_text(surface):
    yaw_note = f" yaw={surface.rotation_deg:.1f}deg" if surface.axis in ("u", "v") else ""
    return (
        f"SMOOTHED |{surface.axis}|{yaw_note} plane {surface.plane_coord:+.2f} m   "
        f"{surface.n_inliers:,} in / {surface.n_deviations:,} dev   "
        f"{len(surface.subsurfaces)} sub-surfaces, {surface.n_polygons} polygons"
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
    planes=False, temperature=None, ransac_threshold=DEFAULT_RANSAC_THRESHOLD,
    ransac_iters=DEFAULT_RANSAC_ITERS, raster_cell=DEFAULT_RASTER_CELL,
    keep_ground=False, ground_band=DEFAULT_GROUND_BAND, seed=0,
    plane_rank=1, target_normal=None, orientation="any",
):
    """Headless render of voxels, a smoothed planar surface, or a plane raster.

    min_count > 1 renders only voxels with at least that many points. smooth=True
    flattens the (filtered) grid onto a plane along smooth_axis ('x'/'y'/'z' or
    the PCA-auto-aligned 'u'/'v') and renders the resulting planar sub-surfaces.
    planes=True instead runs the RANSAC dominant-plane pipeline on the voxel
    centroids and renders the flattened per-cell temperature raster (heatmap)
    with the wall rectangle outlined, viewed face-on. rotation_deg optionally
    overrides the auto-detected yaw (only meaningful with smooth_axis 'u'/'v').
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
    pl.set_background("white")
    if show_points and not planes:
        _add_points(pl, points, max_points)
    glyphs, grid, shown = _build_voxels(points, labels, voxel_size, min_count, values=temperature)

    if planes:
        wall = _compute_wall(shown, ransac_threshold, ransac_iters, seed,
                             keep_ground, ground_band, raster_cell,
                             rank=plane_rank, target_normal=target_normal,
                             orientation=orientation)
        if wall is None:
            raise RuntimeError("not enough voxels for RANSAC plane detection")
        vrange = _render_planes(pl, wall)
        pl.add_text(_planes_info_text(wall, vrange), font_size=10, name="info")
        _set_face_on(pl, wall)
        pl.screenshot(path)
        pl.close()
        return wall

    if smooth:
        surface = smooth_surface(shown, smooth_axis, offset_method, tolerance, rotation_deg=rotation_deg)
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


def render_wall_screenshot(wall, path, window_size=(1400, 900)):
    """Headless render of an already-computed WallPlane (heatmap + rectangle).

    Used by the --export-wall path so the PNG matches the exported wall exactly
    (no recomputation, so raw-points vs. centroids stays consistent). The camera
    looks along the plane normal, so the raster reads as a flat 2-D heatmap.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background("white")
    vrange = _render_planes(pl, wall)
    pl.add_text(_planes_info_text(wall, vrange), font_size=10, name="info")
    _set_face_on(pl, wall)
    pl.screenshot(path)
    pl.close()
    return path


def launch(
    points, labels, voxel_size=DEFAULT_VOXEL_M, max_points=MAX_DISPLAY_POINTS,
    min_count=DEFAULT_MIN_COUNT, filter_on=DEFAULT_FILTER_ON,
    smooth_on=DEFAULT_SMOOTH_ON, smooth_axis=DEFAULT_SMOOTH_AXIS,
    offset_method=DEFAULT_OFFSET_METHOD, tolerance=DEFAULT_TOLERANCE, rotation_deg=None,
    planes_on=DEFAULT_PLANES_ON, temperature=None,
    ransac_threshold=DEFAULT_RANSAC_THRESHOLD, ransac_iters=DEFAULT_RANSAC_ITERS,
    raster_cell=DEFAULT_RASTER_CELL, keep_ground=False,
    ground_band=DEFAULT_GROUND_BAND, seed=0,
    plane_rank=1, target_normal=None, orientation="any",
):
    """Open the interactive viewer: voxel-size slider, min-points filter, points/filter/smooth/planes toggles.

    The smooth axis (u/v/z) has its own live radio-style selector so the other
    wall direction can be inspected without relaunching. The "planes" toggle
    runs the RANSAC dominant-plane pipeline and shows the flattened per-cell
    temperature raster (heatmap) with the wall rectangle outlined.
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
        "planes_on": bool(planes_on),
        "plane_rank": int(plane_rank),
        # Cycle through the detected planes (facades) live; None until the first
        # plane render tells us how many candidates there are.
        "n_candidates": None,
        # Recenter the camera face-on when the plane identity changes (toggle on
        # or rank change), but not on every voxel-size tweak, so orbiting sticks.
        "planes_recenter": True,
    }
    axis_widgets = {}  # filled after creation, used to keep the radio selection in sync

    def show_voxels():
        effective_min = state["min_count"] if state["filter_on"] else 1
        glyphs, grid, shown = _build_voxels(
            points, labels, state["voxel"], effective_min, values=temperature
        )

        # RANSAC dominant-plane raster takes priority over the other views.
        if state["planes_on"]:
            wall = _compute_wall(shown, ransac_threshold, ransac_iters, seed,
                                 keep_ground, ground_band, raster_cell,
                                 rank=state["plane_rank"], target_normal=target_normal,
                                 orientation=orientation)
            if wall is None:
                print("[planes] not enough voxels for RANSAC plane detection")
                return
            state["n_candidates"] = wall.n_candidates
            vrange = _render_planes(pl, wall)
            if state["planes_recenter"]:
                _set_face_on(pl, wall)
                state["planes_recenter"] = False
            print(f"[planes] {_planes_info_text(wall, vrange)}")
            pl.add_text(_planes_info_text(wall, vrange), font_size=10,
                        position="upper_left", name="info")
            return
        pl.remove_actor("wall_outline", reset_camera=False)

        if state["smooth_on"]:
            # Pipeline: voxelize -> (filter) -> smooth. Render planar surface.
            surface = smooth_surface(
                shown, state["smooth_axis"], offset_method, tolerance,
                rotation_deg=state["rotation_deg"],
            )
            mesh = _planar_mesh(surface)
            pl.remove_actor("voxels", reset_camera=False)
            if mesh is not None:
                pl.add_mesh(mesh, scalars="colors", rgb=True, name="voxels", show_edges=True)
            print(f"[smooth] {_smooth_info_text(surface)}")
            pl.add_text(_smooth_info_text(surface), font_size=10,
                        position="upper_left", name="info")
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

    def on_toggle_planes(flag):
        state["planes_on"] = bool(flag)
        state["planes_recenter"] = True  # face the wall when planes turns on
        show_voxels()

    def on_next_plane(flag):
        # Cycle to the next detected plane (facade) and recenter on it. This is
        # a momentary button: it re-renders, then unchecks itself.
        if not state["planes_on"] or not state.get("n_candidates"):
            next_plane_widget.GetRepresentation().SetState(0)
            return
        state["plane_rank"] = state["plane_rank"] % state["n_candidates"] + 1
        state["planes_recenter"] = True
        show_voxels()
        next_plane_widget.GetRepresentation().SetState(0)

    def on_pick_axis(axis):
        def handler(flag):
            if not flag:
                # Clicking the already-active axis would otherwise uncheck it
                # with nothing selected; re-check it and ignore (radio button).
                axis_widgets[axis].GetRepresentation().SetState(1)
                return
            state["smooth_axis"] = axis
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
    # RANSAC dominant-plane -> wall temperature raster (heatmap).
    pl.add_checkbox_button_widget(on_toggle_planes, value=state["planes_on"], size=26, position=(10, 118))
    pl.add_text("planes on/off", font_size=9, position=(44, 120), name="planes_toggle_label")
    # Momentary "next plane" button: cycle through the detected facades.
    next_plane_widget = pl.add_checkbox_button_widget(
        on_next_plane, value=False, size=20, position=(150, 118), color_on="tan"
    )
    pl.add_text("next plane", font_size=9, position=(174, 120), name="next_plane_label")
    # Live radio-style axis selector (u / v / z) for smoothing — lets the
    # other wall direction be inspected without relaunching the script.
    pl.add_text("axis:", font_size=9, position=(10, 155), name="axis_row_label")
    axis_x = {"u": 60, "v": 100, "z": 140}
    for a in SMOOTH_AXIS_CHOICES:
        axis_widgets[a] = pl.add_checkbox_button_widget(
            on_pick_axis(a), value=(a == state["smooth_axis"]), size=20, position=(axis_x[a], 154)
        )
        pl.add_text(a, font_size=9, position=(axis_x[a] + 24, 155), name=f"axis_label_{a}")
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
