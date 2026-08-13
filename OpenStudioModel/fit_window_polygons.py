"""Fit rectangular window polygons on a wall, from real + inferred glass
voxels, for import into OpenStudio as SubSurfaces.

to_openstudio.py currently makes one blank Surface per plane in planes.json
-- no openings. This script finds where the actual glazing is (measured, not
guessed) and writes rectangles to_openstudio.py can turn into SubSurfaces.

Pipeline
--------
1. Build the wall's occupied grid at --voxel-size (default 0.20 m, matching
   the voxel data) over its real footprint (corners_3d).
2. A cell is GLAZED if EITHER:
     - a real voxel there has material in --glass-materials (measured glass), OR
     - the cell has no data on ANY layer along the wall's thickness axis (a
       "hole" -- glass transmits or reflects the LiDAR beam away rather than
       returning it, so an unexplained gap on an otherwise well-covered wall
       is the same signature as a real glass hit; see voxel_solar_ns.py's
       fillGlassHoles for the reasoning and the same caveat: a hole could
       also be an occluding object, e.g. a radiator, not glass -- there is
       no way to tell the two apart from returns alone).
   A cell with real OPAQUE data (concrete, paint, ...) is never glazed, no
   matter how close to a glazed cell -- real data always wins over inference.
3. --opaque-x-range (repeatable) excludes a known solid section from the
   glazed candidate set even if it has holes -- established from the south
   wall's real glass/wall/glass pattern (x confirmed by the operator, not
   inferred from material: the sensor's own material calls next to real
   mullions/pillars there are ambiguous, see voxel_solar_ns.py's docstring).
4. Binary closing (--merge-gap cells) bridges thin non-glazed gaps -- window
   mullions between adjacent panes of what is structurally one window --
   before connected-component labelling, so mullions don't split one window
   into several tiny rectangles.
5. Each connected component with area >= --min-area cells becomes one
   rectangle: its BOUNDING BOX in the wall's plane, not its exact (possibly
   ragged) outline -- OpenStudio subsurfaces are conventionally simple
   rectangles, and a window's true shape is a rectangle anyway; the ragged
   edges in the voxel data are measurement noise, not the window's real
   boundary.

Output: JSON, one entry per window: {plane_id, x0,x1 (or y0,y1 for a plane
whose in-plane axis is y), z0, z1, area_m2, n_cells, source: "measured"|
"inferred"|"mixed"}, in the SAME world frame as planes.json -- consumed by
to_openstudio.py --windows.

Usage:
    py fit_window_polygons.py --in thermal_voxels_u.csv --planes planes.json \\
        --plane-ids 1,4 --opaque-x-range 4:12,25:34.6 \\
        --out windows.json --plot windows_check.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def find_plane(planes, plane_id):
    for p in planes["planes"]:
        if p["id"] == plane_id:
            return p
    raise SystemExit(f"no plane with id {plane_id}")


def plane_axes(normal):
    """Thickness axis (most aligned with the normal) and the two in-plane
    axes. Correct for axis-aligned walls (this rig); a tilted wall would need
    basis_u/basis_v projection instead."""
    const_axis = int(np.argmax(np.abs(normal)))
    in_plane = [a for a in range(3) if a != const_axis]
    return const_axis, in_plane[0], in_plane[1]


def outward_normal(plane, interior):
    n = np.asarray(plane["normal"], dtype=float)
    d = float(plane["d"])
    k = np.linalg.norm(n)
    n, d = n / k, d / k
    centre = np.asarray(plane["center_3d"], dtype=float)
    if float(n @ (centre - interior)) < 0.0:
        n, d = -n, -d
    return n, d


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="thermal_voxels(_u).csv -- needs x,y,z,material")
    ap.add_argument("--planes", type=Path, required=True)
    ap.add_argument("--plane-ids", default="1,4", metavar="ID,ID,...",
                    help="wall plane ids to fit windows on (default 1,4 -- "
                         "north,south)")
    ap.add_argument("--voxel-size", type=float, default=0.20, metavar="M")
    ap.add_argument("--plane-threshold", type=float, default=0.15, metavar="M",
                    help="max distance from the plane for a voxel to belong to it")
    ap.add_argument("--glass-materials", default="glass", metavar="LIST")
    ap.add_argument("--infer-holes", action=argparse.BooleanOptionalAction, default=False,
                    help="also treat an unexplained hole (no return on any layer) as glazing, "
                         "not just measured glass. OFF by default: measured on session 9, the "
                         "inference behaves oppositely on the two walls and neither result is "
                         "credible. South (56%% coverage) has real measured glass in 4 clearly "
                         "separated groups; enabling the inference drowns them in one "
                         "continuous 19 m band with no piers. North (78%% coverage) has almost "
                         "no measured glass (39 cells in 1440), so with it ON every window is "
                         "an assumption and with it OFF the wall reads as nearly blind. "
                         "--infer-holes therefore states an assumption rather than a "
                         "measurement -- turn it on deliberately, per wall, knowing which of "
                         "the two failure modes you are accepting.")
    ap.add_argument("--occluder-materials", default="painted_metal", metavar="LIST",
                    help="materials that are interior objects standing IN FRONT of the wall, "
                         "not the wall itself (default painted_metal = the column radiators "
                         "under every window). A cell where the beam hit one of these says "
                         "nothing about what is behind it -- it is UNKNOWN, not opaque. "
                         "Measured on this rig: painted_metal is 84-89%% of all wall returns "
                         "in the z=0.1-0.7 band and drops to 4-22%% above z=0.9, i.e. a "
                         "continuous radiator band at exactly radiator height. Treating those "
                         "as opaque chopped the bottom off every window at a different, "
                         "data-coverage-dependent height. Unknown cells are allowed to join a "
                         "window component but cannot form one on their own (see "
                         "--min-glazed-frac). Empty string disables.")
    ap.add_argument("--occluder-z-max", type=float, default=1.0, metavar="M",
                    help="only treat --occluder-materials as occluders BELOW this world z "
                         "(default 1.0). Above it, painted metal on a wall is more likely a "
                         "real painted-metal panel/frame than a radiator, and should stay "
                         "opaque.")
    ap.add_argument("--min-width", type=float, default=0.0, metavar="M",
                    help="drop windows narrower than this (default 0 = off). Area alone does "
                         "not catch slivers: a 0.4 m x 1.8 m edge artefact passes a 0.5 m2 "
                         "area gate while being obviously not a window.")
    ap.add_argument("--min-height", type=float, default=0.0, metavar="M",
                    help="drop windows shorter than this (default 0 = off)")
    ap.add_argument("--min-coverage", type=float, default=0.40, metavar="0-1",
                    help="refuse to fit windows on a wall whose grid is less than this "
                         "fraction covered by real data (default 0.40). The hole=glass "
                         "inference assumes a hole is unexplained ON AN OTHERWISE "
                         "WELL-SCANNED wall; where the scan simply never covered the wall, "
                         "every uncovered cell reads as glazing and the whole wall comes out "
                         "as one window. Measured: session 9 walls sit at 52-79% coverage and "
                         "behave; session 6's south wall is at 18% (5 measured glass cells in "
                         "840) and produced a fully-glazed wall. This is a hard stop, not a "
                         "tuning knob -- lower it only if you have verified the sparse wall "
                         "really is glazed. 0 disables the check.")
    ap.add_argument("--min-glazed-frac", type=float, default=0.15, metavar="0-1",
                    help="a component must be at least this fraction genuinely glazed "
                         "(measured glass or hole) to count as a window (default 0.15) -- "
                         "stops a blob of pure radiator/unknown cells becoming a window")
    ap.add_argument("--min-sill-z", type=float, default=None, metavar="M",
                    help="ignore holes (not measured glass -- real data always wins) below "
                         "this world z when inferring glazing (default: no floor). A hole near "
                         "the floor is more likely a radiator's occlusion shadow than glass -- "
                         "radiators typically sit right below a window. Only applied on walls "
                         "whose in-plane axes include z (this rig: all of them).")
    ap.add_argument("--opaque-x-range", action="append", default=[], metavar="PLANE:X0:X1",
                    help="exclude this x-range (world coords) from glazing candidates on "
                         "ONE plane, regardless of holes -- repeatable, e.g. "
                         "--opaque-x-range 4:12:25 excludes x in [12,25] on plane 4 only. "
                         "Operator-supplied geometry (confirmed glass/wall/glass pattern), "
                         "not inferred -- see module docstring.")
    ap.add_argument("--merge-gap", type=int, default=1, metavar="CELLS",
                    help="binary-closing radius, in grid cells, to bridge mullions "
                         "between panes of the same window before labelling "
                         "(default 1 = bridge a single-cell gap)")
    ap.add_argument("--min-area", type=float, default=0.5, metavar="M2",
                    help="drop connected components smaller than this (default 0.5 "
                         "m2) -- filters noise specks, not real small windows")
    ap.add_argument("--split-period", type=float, default=None, metavar="M",
                    help="if a connected component's width (world x) exceeds 1.5x this, "
                         "split it into round(width/period) equal-width sub-windows instead "
                         "of keeping it as one. For a component that merges several real bays "
                         "because the wall between them had too little opaque data to break "
                         "the clustering (--merge-gap bridges real intra-window gaps and "
                         "inter-window wall gaps alike when they're a similar size) -- use the "
                         "period measured independently (autocorrelation on the hole profile) "
                         "rather than trying to re-tune clustering parameters to separate them "
                         "on their own.")
    ap.add_argument("--room-bbox", action=argparse.BooleanOptionalAction, default=True,
                    help="drop voxels outside the x/y footprint of --floor-plane-id's floor "
                         "plane (expanded by --plane-threshold) before anything else -- same "
                         "through-glass LiDAR noise filter as voxel_solar_ns.py. Default on.")
    ap.add_argument("--floor-plane-id", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None,
                    help="optional PNG: occupied grid + fitted rectangles, for a "
                         "visual sanity check before trusting the output")
    args = ap.parse_args()

    opaque_ranges = {}   # plane_id -> [(x0,x1), ...]
    for spec in args.opaque_x_range:
        pid_s, x0, x1 = spec.split(":")
        opaque_ranges.setdefault(int(pid_s), []).append((float(x0), float(x1)))

    with open(args.inp, newline="", encoding="utf-8") as f:
        import csv
        rows = list(csv.DictReader(f))

    planes = json.loads(args.planes.read_text())
    if args.room_bbox:
        fp = find_plane(planes, args.floor_plane_id)
        corners = np.array(fp["corners_3d"])
        (rx0, ry0), (rx1, ry1) = corners[:, :2].min(0), corners[:, :2].max(0)
        rx0, ry0 = rx0 - args.plane_threshold, ry0 - args.plane_threshold
        rx1, ry1 = rx1 + args.plane_threshold, ry1 + args.plane_threshold
        n_all = len(rows)
        rows = [r for r in rows
                if rx0 <= float(r["x"]) <= rx1 and ry0 <= float(r["y"]) <= ry1]
        print(f"--room-bbox: dropped {n_all - len(rows)}/{n_all} voxel(s) outside the room "
              f"(through-glass LiDAR noise)")

    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    material = np.array([r.get("material", "") for r in rows])
    glass_materials = {m.strip() for m in args.glass_materials.split(",") if m.strip()}
    is_glass = np.isin(material, list(glass_materials))
    occluder_materials = {m.strip() for m in args.occluder_materials.split(",") if m.strip()}
    # An occluder only counts as one where it plausibly IS one (below
    # --occluder-z-max) -- see that flag's help.
    is_occluder = (np.isin(material, list(occluder_materials)) &
                   (xyz[:, 2] < args.occluder_z_max)) if occluder_materials \
                  else np.zeros(len(material), bool)

    interior = np.mean([p["center_3d"] for p in planes["planes"]], axis=0)

    windows = []
    plot_data = []
    for pid in [int(s) for s in args.plane_ids.split(",")]:
        plane = find_plane(planes, pid)
        n, d = outward_normal(plane, interior)
        dist = np.abs(xyz @ n + d) / np.linalg.norm(n)
        on_wall = dist < args.plane_threshold
        const_axis, ax1, ax2 = plane_axes(np.asarray(plane["normal"]))

        iv = np.round(xyz / args.voxel_size - 0.5).astype(int)
        wall_iv = iv[on_wall]
        wall_glass = is_glass[on_wall]
        wall_occl = is_occluder[on_wall]

        corners = np.array(plane["corners_3d"])
        a1_lo = int(np.floor(corners[:, ax1].min() / args.voxel_size - 0.5))
        a1_hi = int(np.ceil(corners[:, ax1].max() / args.voxel_size - 0.5))
        a2_lo = int(np.floor(corners[:, ax2].min() / args.voxel_size - 0.5))
        a2_hi = int(np.ceil(corners[:, ax2].max() / args.voxel_size - 0.5))
        n1, n2 = a1_hi - a1_lo + 1, a2_hi - a2_lo + 1

        # occupied[i1,i2] = True if ANY layer along the thickness axis has
        # real data there (any material) -- a False cell is a hole.
        # glass_hit[i1,i2] = True if any layer's real data is glass there.
        occupied = np.zeros((n1, n2), dtype=bool)
        glass_hit = np.zeros((n1, n2), dtype=bool)
        occl_hit = np.zeros((n1, n2), dtype=bool)
        i1 = wall_iv[:, ax1] - a1_lo
        i2 = wall_iv[:, ax2] - a2_lo
        valid = (i1 >= 0) & (i1 < n1) & (i2 >= 0) & (i2 < n2)
        occupied[i1[valid], i2[valid]] = True
        glass_hit[i1[valid][wall_glass[valid]], i2[valid][wall_glass[valid]]] = True
        occl_hit[i1[valid][wall_occl[valid]], i2[valid][wall_occl[valid]]] = True

        # The integer grid is rounded outward from the wall's real corners, so
        # the outermost row/column can lie almost entirely OUTSIDE the wall
        # (measured: the wall top is z=1.804 but the grid's last z row covers
        # 1.80-2.00, and its bottom row covers -0.40..-0.269). Those cells can
        # never have wall data, so "hole = glass" turned them into spurious
        # glazing along the whole wall. Keep only cells whose CENTRE falls
        # inside the plane's real extent.
        c1_centres = (np.arange(a1_lo, a1_hi + 1) + 0.5) * args.voxel_size
        c2_centres = (np.arange(a2_lo, a2_hi + 1) + 0.5) * args.voxel_size
        in_wall = ((c1_centres >= corners[:, ax1].min()) &
                   (c1_centres <= corners[:, ax1].max()))[:, None] & \
                  ((c2_centres >= corners[:, ax2].min()) &
                   (c2_centres <= corners[:, ax2].max()))[None, :]
        n_out = int((~in_wall).sum())
        if n_out:
            print(f"plane {pid}: {n_out} grid cell(s) lie outside the wall's real extent "
                  f"(grid rounded outward from corners_3d) -- excluded")

        coverage = float(occupied.sum()) / occupied.size
        # Only meaningful with --infer-holes: without it a hole contributes
        # nothing, so poor coverage costs recall, not correctness.
        if args.infer_holes and args.min_coverage > 0 and coverage < args.min_coverage:
            print(f"plane {pid}: SKIPPED -- only {100*coverage:.0f}% of the wall grid has real "
                  f"data (< --min-coverage {100*args.min_coverage:.0f}%). The hole=glass "
                  f"inference is not valid on a wall this sparsely scanned: every unscanned "
                  f"cell would read as glazing. {int(glass_hit.sum())} measured glass cell(s) "
                  f"here.")
            continue

        glazed = ((glass_hit | ~occupied) if args.infer_holes else glass_hit) & in_wall
        # Third state: a cell whose only real data is an occluder (radiator)
        # standing in front of the wall. Not glazed, but not opaque either --
        # we simply cannot see what is behind it. Never overrides measured
        # glass ("real data always wins").
        unknown = occl_hit & ~glass_hit & in_wall

        # A hole below --min-sill-z is more likely a radiator's occlusion
        # shadow than glass (radiators typically sit right under a window).
        # Only the HOLE component is filtered -- real measured glass always
        # wins, per the module docstring's "real data always wins" rule.
        if args.min_sill_z is not None and (ax1 == 2 or ax2 == 2):
            z_axis = ax1 if ax1 == 2 else ax2
            z_of = (np.arange(a1_lo, a1_hi + 1) + 0.5) * args.voxel_size if z_axis == ax1 \
                else (np.arange(a2_lo, a2_hi + 1) + 0.5) * args.voxel_size
            below_sill_1d = z_of < args.min_sill_z
            below_sill_grid = below_sill_1d[:, None] if z_axis == ax1 else below_sill_1d[None, :]
            below_sill_grid = np.broadcast_to(below_sill_grid, (n1, n2))
            hole_only = ~occupied & ~glass_hit
            glazed &= ~(hole_only & below_sill_grid)

        # Exclude known-opaque x-ranges regardless of holes/measured glass
        # noise there -- operator-supplied geometry, see module docstring.
        if ax1 == 0 or ax2 == 0:  # x is one of the in-plane axes on this wall
            x_axis = ax1 if ax1 == 0 else ax2
            x_of = (np.arange(a1_lo, a1_hi + 1) + 0.5) * args.voxel_size if x_axis == ax1 \
                else (np.arange(a2_lo, a2_hi + 1) + 0.5) * args.voxel_size
            for x0, x1_ in opaque_ranges.get(pid, []):
                in_range = (x_of >= x0) & (x_of <= x1_)
                if x_axis == ax1:
                    glazed[in_range, :] = False
                    unknown[in_range, :] = False
                else:
                    glazed[:, in_range] = False
                    unknown[:, in_range] = False

        # Absorb occluded cells into glazing ONLY DOWNWARD, never sideways.
        # The radiator band is continuous along the whole wall, so letting
        # components grow through it in any direction bridges every bay into
        # one strip (measured: the north wall collapsed to a single component
        # covering the full 23.5 m, i.e. a 100%-glazed wall with no piers --
        # clearly wrong). Growing only downward reproduces the real
        # arrangement instead: glazing continues behind the radiator that
        # stands in front of its lower part, but two windows separated by a
        # pier stay separate, because the pier is opaque ABOVE the radiator.
        grown = glazed.copy()
        if unknown.any() and (ax1 == 2 or ax2 == 2):
            z_axis = 0 if ax1 == 2 else 1   # which GRID axis is world z
            while True:
                # shift `grown` down by one cell along z: a cell inherits from
                # the one directly above it
                above = np.roll(grown, -1, axis=z_axis)
                if z_axis == 0:
                    above[-1, :] = False
                else:
                    above[:, -1] = False
                newly = unknown & above & ~grown
                if not newly.any():
                    break
                grown |= newly

        closed = ndimage.binary_closing(grown, structure=np.ones((2*args.merge_gap+1,)*2)) \
                if args.merge_gap > 0 else grown
        labels, n_comp = ndimage.label(closed, structure=np.ones((3, 3)))  # 8-connectivity

        cell_area = args.voxel_size ** 2
        kept = 0
        n_dropped_unglazed = 0
        for lbl in range(1, n_comp + 1):
            comp = labels == lbl
            n_cells = int(comp.sum())
            area = n_cells * cell_area
            if area < args.min_area:
                continue
            glazed_frac = float((comp & glazed).sum()) / max(1, n_cells)
            if glazed_frac < args.min_glazed_frac:
                n_dropped_unglazed += 1
                continue
            idx1, idx2 = np.where(comp)
            b1_lo, b1_hi = idx1.min() + a1_lo, idx1.max() + a1_lo + 1
            b2_lo, b2_hi = idx2.min() + a2_lo, idx2.max() + a2_lo + 1

            # A component whose world-x width is much more than one period is
            # probably several real bays merged by --merge-gap bridging a
            # too-thin opaque gap between them, not one wide window -- split
            # it into equal sub-widths at the independently-measured period
            # instead of re-tuning clustering to separate them unsupervised.
            x_axis = ax1 if ax1 == 0 else (ax2 if ax2 == 0 else None)
            sub_bounds = [(b1_lo, b1_hi, b2_lo, b2_hi)]
            if args.split_period and x_axis is not None:
                b_lo, b_hi = (b1_lo, b1_hi) if x_axis == ax1 else (b2_lo, b2_hi)
                width_m = (b_hi - b_lo) * args.voxel_size
                n_split = max(1, round(width_m / args.split_period))
                if width_m > 1.5 * args.split_period and n_split > 1:
                    edges = np.linspace(b_lo, b_hi, n_split + 1)
                    edges = np.round(edges).astype(int)
                    sub_bounds = []
                    for s in range(n_split):
                        s_lo, s_hi = edges[s], edges[s + 1]
                        if x_axis == ax1:
                            sub_bounds.append((s_lo, s_hi, b2_lo, b2_hi))
                        else:
                            sub_bounds.append((b1_lo, b1_hi, s_lo, s_hi))

            for s1_lo, s1_hi, s2_lo, s2_hi in sub_bounds:
                # Re-fit the slice's OWN bounding box to the component cells
                # actually inside it. Without this every slice inherits the
                # whole component's extent, so one bay that happens to reach
                # higher makes all its siblings equally tall -- measured: bay
                # x[31.4,34.4] came out 20 cm taller than its neighbours while
                # its own top row was entirely opaque.
                sl = np.zeros_like(comp)
                sl[s1_lo - a1_lo:s1_hi - a1_lo, s2_lo - a2_lo:s2_hi - a2_lo] = True
                sl &= comp
                if not sl.any():
                    continue
                j1, j2 = np.where(sl)
                s1_lo, s1_hi = j1.min() + a1_lo, j1.max() + a1_lo + 1
                s2_lo, s2_hi = j2.min() + a2_lo, j2.max() + a2_lo + 1

                v1_lo, v1_hi = s1_lo * args.voxel_size, s1_hi * args.voxel_size
                v2_lo, v2_hi = s2_lo * args.voxel_size, s2_hi * args.voxel_size

                # measured vs inferred, within this sub-rectangle (post-closing,
                # so this reports on the ORIGINAL glazed mask, not the closed one)
                gi1_lo, gi1_hi = s1_lo - a1_lo, s1_hi - a1_lo
                gi2_lo, gi2_hi = s2_lo - a2_lo, s2_hi - a2_lo
                box_glass = glass_hit[gi1_lo:gi1_hi, gi2_lo:gi2_hi]
                box_glazed = glazed[gi1_lo:gi1_hi, gi2_lo:gi2_hi]
                sub_area = box_glazed.size * cell_area
                if sub_area < args.min_area:
                    continue
                # Dimensional gates: which of the two in-plane axes is the
                # horizontal one depends on the wall's orientation.
                w_span = v1_hi - v1_lo if ax1 != 2 else v2_hi - v2_lo
                h_span = v2_hi - v2_lo if ax1 != 2 else v1_hi - v1_lo
                if w_span < args.min_width or h_span < args.min_height:
                    continue
                frac_measured = float(box_glass.sum()) / max(1, int(box_glazed.sum()))
                source = "measured" if frac_measured > 0.9 else \
                         "inferred" if frac_measured < 0.1 else "mixed"

                # From the plane EQUATION (n, d from planes.json), not the
                # median of nearby real voxels: real wall data clusters ~5-10
                # cm off the true plane (registration/LiDAR-thickness noise,
                # already measured earlier on this rig -- north wall real
                # data centres near y=-0.7 against a true plane at y=-0.815).
                # A SubSurface not exactly coplanar with its parent Surface is
                # invalid OpenStudio/EnergyPlus geometry -- confirmed: an
                # earlier version using the median put windows ~8.5 cm off
                # the wall plane and SketchUp rendered them wrong.
                const_val = -d / n[const_axis] if n[const_axis] != 0 else \
                           float(plane["center_3d"][const_axis])

                corner = np.zeros((4, 3))
                axis_lo_hi = [(v1_lo, v1_hi), (v2_lo, v2_hi)]
                for k, (u, v) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
                    pt = np.zeros(3)
                    pt[ax1] = axis_lo_hi[0][u]
                    pt[ax2] = axis_lo_hi[1][v]
                    pt[const_axis] = const_val
                    corner[k] = pt

                windows.append({
                    "plane_id": pid,
                    "corners_3d": corner.tolist(),
                    "area_m2": round(sub_area, 3),
                    "n_cells": int(box_glazed.sum()),
                    "frac_measured": round(frac_measured, 2),
                    "source": source,
                })
                kept += 1
        print(f"plane {pid}: grid {n1}x{n2} cells ({args.voxel_size*100:.0f} cm), "
              f"{int(glazed.sum())} glazed ({int(glass_hit.sum())} measured + "
              f"{int((~occupied).sum())} holes, overlap possible), "
              f"{int(unknown.sum())} occluded/unknown -> {n_comp} component(s), "
              f"{kept} kept (>= {args.min_area} m2)"
              + (f", {n_dropped_unglazed} dropped (< {args.min_glazed_frac} glazed)"
                 if n_dropped_unglazed else ""))

        if args.plot:
            plot_data.append((pid, ax1, ax2, a1_lo, a2_lo, glazed, unknown, labels,
                              [w for w in windows if w["plane_id"] == pid]))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"windows": windows}, f, indent=2)
    print(f"\n{len(windows)} window(s) total -> {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(plot_data), 1, figsize=(14, 3.5 * len(plot_data)),
                                 squeeze=False)
        for row, (pid, ax1, ax2, a1_lo, a2_lo, glazed, unknown, labels, wins) in enumerate(plot_data):
            ax = axes[row, 0]
            vs = args.voxel_size
            ext = [a1_lo*vs, (a1_lo+glazed.shape[0])*vs,
                   a2_lo*vs, (a2_lo+glazed.shape[1])*vs]
            # blu = vetro (misurato o buco), grigio = occluso da radiatore
            ax.imshow(np.ma.masked_where(~unknown.T, unknown.T), origin="lower",
                      cmap="Greys", alpha=0.45, extent=ext, vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(~glazed.T, glazed.T), origin="lower",
                      cmap="Blues", alpha=0.75, extent=ext, vmin=0, vmax=1)
            for w in wins:
                xs = [c[ax1] for c in w["corners_3d"]]
                ys = [c[ax2] for c in w["corners_3d"]]
                xs.append(xs[0]); ys.append(ys[0])
                col = {"measured": "green", "inferred": "orange", "mixed": "red"}[w["source"]]
                ax.plot(xs, ys, color=col, lw=2)
            ax.set_title(f"plane {pid} (blu=vetro, grigio=occluso da radiatore; "
                         f"riquadri verde=misurato/arancio=inferito/rosso=misto)")
            ax.set_xlabel("x mondo (m)")
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
