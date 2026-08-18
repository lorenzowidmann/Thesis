# `--filler` (removed for now) — floor/ceiling gap filling

Removed from `view_voxels.py` because it never fully converged on correct
behavior after several iterations; kept here so the working parts, the dead
ends, and the exact code can be picked back up later instead of re-derived.

## What it was supposed to do

Small gaps in the floor/ceiling voxel coverage (LiDAR dropout spots, thin
coverage-edge margins) get a synthetic voxel: colored by the mean
temperature of its occupied neighbours, outlined in a thick contrasting
edge so it's never mistaken for a real measurement. Floor/ceiling only,
never walls.

## The debugging journey (what was tried, in order)

1. **`scipy.ndimage.binary_fill_holes`, strict enclosed-only.** Only fills a
   background region that never touches the 2D layer's own bounding-box
   edge. Real result: filled almost nothing (5 of 184 floor background
   cells) — most real gaps on a real (non-rectangular) scan footprint touch
   the ragged coverage edge somewhere, so this rule excluded them.
   **User feedback: "it does not fill the full gap."**

2. **Pure size-based selection**, no enclosure test: any 4-connected
   background component <= `max_gap_cells`, border-touching or not. Real
   result: filled all 369 background cells across both layers (all 6
   components were <= 200 cells). **User feedback: "not the sides"** — this
   filled long, thin coverage-margin strips along the corridor's side
   walls too, which read as "the sides," not "a hole."

3. **Morphological closing before flood-fill** (`binary_closing` then
   `binary_fill_holes`): meant to bridge a 1-2 cell ragged-edge nick so a
   real interior hole grazing the boundary still counts as enclosed, while
   leaving the true, large exterior alone. Real result: **no change at
   all** vs. approach 1 (still 5/184) — the "sides" components (98, 70, 11
   cells) turned out to be long, thin strips running almost the full
   corridor length (bbox e.g. 56x3 cells) that substantively touch the
   edge, not a 1-2 cell nick a small closing kernel could bridge.

4. **Reverted to approach 2** (pure size-based) after asking the user
   directly — confirmed they *did* want those edge-margin strips filled
   after all, despite calling them "the sides" earlier. Result: back to
   369 filler voxels, all 6 background components across the 2 dominant
   Z-layers.

5. **User's screenshot showed still-unfilled thin 1-voxel-wide slits.**
   Diagnosis: the exhaustive scan had already accounted for every
   background cell on the *single dominant* Z-layer per side — the slits
   were sitting on a *different*, adjacent Z-layer (a real floor/ceiling
   isn't perfectly flat at 15cm resolution; a slight dip/step is a whole
   extra Z-layer that `find_gap_cells` never even looked at, since it only
   ever scanned the one fixed `floor_z_idx`/`ceiling_z_idx`).

6. **Multi-layer classification**: include a neighbouring Z-layer as
   "still floor/ceiling" if it's within a few layers of the dominant one
   AND has at least some fraction (tried 0.15) of the dominant layer's
   occupied-cell count — meant to exclude wall-only layers (thin perimeter
   ring, low count) while including real floor/ceiling unevenness (dense
   interior coverage, high count). Real result: worked for the ceiling
   (correctly pulled in z=15, which had its own 91/72-cell gaps, while
   still excluding z=12-14) but **the same ratio threshold did not work
   for the floor**: the floor's occupancy ratio measured a roughly *flat*
   ~0.24-0.29 across 5 layers going up from the dominant floor layer (z=3
   through z=7) — that's just the wall's normal, roughly-constant
   cross-section density, not floor unevenness with any clean cutoff.
   **User feedback (with screenshot): "why are you doing it on the walls."**
   The filler had started coloring/outlining voxels a good half-metre up
   the wall as if they were "floor."

7. **Reverted to single dominant layer per side** (undoing step 6) — the
   safe, zero-wall-risk baseline. Confirmed via the same count-ratio data:
   there is no reliable signal in occupied-cell-count-ratio alone to tell
   "real floor/ceiling unevenness one layer up" apart from "just the
   wall's own normal density" for this dataset. Missing an occasional real
   off-layer gap was judged the safer failure mode than bleeding into
   walls.

8. **Same screenshot also showed three large openings** (pillars visible
   through them) that stayed unfilled even after step 7. Asked the user to
   confirm whether those were actually on a wall (visually, they looked
   like it — face-on wall view with interior structure visible behind).
   **Confirmed: yes, on the wall.** User's final call: leave walls alone,
   accept those three openings stay open. This is *expected*, not a bug —
   `--filler` was always scoped to floor/ceiling only, per the original
   request.

## Where it landed before removal (state at removal time)

Single dominant Z-layer per side (`find_floor_ceiling_z_layers`, despite
the plural name, returns single-element lists — see below), pure
size-based gap selection (`find_gap_cells`, no enclosure/closing test),
iterative ring-growing fill for gaps too wide to reach real data in one
averaging pass (`compute_filler_voxels`). This is a defensible, safe
baseline: zero confirmed wall bleed, real coverage gain (5 -> 369 filler
voxels vs. the strict-enclosed starting point), but known to still miss
gaps that sit on a Z-layer other than the single dominant one (step 5/6's
unresolved problem) — the ceiling z=15 gaps from step 5 are, once again,
not covered by this final state.

## What to try next, if picked back up

The real open problem is #6's unsolved half: reliably telling "a
real floor/ceiling Z-layer one step away from the dominant one" apart
from "a wall's own normal cross-section density at that height," when a
simple occupied-cell-count ratio doesn't cleanly separate them (works for
this dataset's ceiling, not for its floor). Ideas not yet tried:

- Use the *dominant layer's own occupied footprint* as a mask: a
  neighbouring layer's cells only count as "floor/ceiling" if they're
  spatially close to (e.g. within a couple cells of) that footprint's
  *interior*, not just anywhere the neighbouring layer happens to have
  cells -- a wall's cells at that height sit at the footprint's own
  perimeter, which is exactly where a real floor/ceiling's interior does
  *not* extend, so this might separate them where a pure count ratio can't.
- Explicit distance-to-wall-plane check per voxel using `planes_aligned.json`'s
  wall plane equations (project each candidate voxel onto the nearest wall
  plane's normal and threshold the distance), instead of an indirect,
  purely-Z/count-based heuristic.
- Ask the user for a manual per-run Z-layer list/tolerance instead of
  trying to auto-detect it -- simpler, no heuristic to get wrong, at the
  cost of needing to be re-tuned if the corridor geometry changes.

## Full code at time of removal

Everything below lived in `view_voxels.py`. To restore: paste these back
into their corresponding sections in the current `view_voxels.py`
(module docstring's `--filler` paragraph, `FILLER_EDGE_COLOR`/
`FILLER_EDGE_WIDTH` constants, the four functions/constant below,
`FILLER_KEY`, `build_plotter`'s `filler_*` parameters/`render_filler`/
`toggle_filler`/key-binding/label text, and `main()`'s `--filler`/
`--filler-max-gap-cells` args + `compute_filler_voxels` call +
`filler_centers`/`filler_temps`/`filler_on` passthrough to `build_plotter`).

### Constants

```python
FILLER_EDGE_COLOR = "black"
FILLER_EDGE_WIDTH = 3

DEFAULT_FILLER_MAX_GAP_CELLS = 200

# in the live-key section:
FILLER_KEY = "g"     # toggle the floor/ceiling gap filler; VTK's own 'f' is FlyTo, avoid it
```

### `find_floor_ceiling_z_layers`

```python
def find_floor_ceiling_z_layers(centers, origin, voxel_size, planes_aligned_path):
    """Which voxel Z-layer is the floor and which is the ceiling.

    Ties this to the actual RANSAC-fit floor/ceiling planes (planes_aligned.json,
    orientation == "floor_ceiling", the lower-centroid one is the floor,
    higher is the ceiling -- same convention aligned_octree.compute_building_frame
    uses) rather than the min/max occupied Z-layer of the voxel grid: a
    below-floor/above-ceiling clutter voxel can be 26-connected to the main
    structure (declutter wouldn't separate it), so that heuristic can pick
    the wrong layer.

    Single dominant layer per side, deliberately -- an earlier version also
    included neighbouring Z-layers with a substantial fraction of the
    dominant layer's occupied-cell count, meant to catch real floor/ceiling
    unevenness a layer off from the dominant one. In practice there's no
    reliable count-ratio signal to tell that apart from a wall's own,
    roughly constant, cross-section density: e.g. on one real dataset the
    floor's "occupancy ratio" measured ~0.24-0.29 fairly flatly from one
    layer above the floor up through five layers above it -- clearly just
    the wall's normal density, not floor unevenness with a clean cutoff.
    Risking wall bleed for an unreliable heuristic isn't worth it; missing
    an occasional real off-layer gap is the safer failure mode here.

    Returns (floor_z_idx, ceiling_z_idx) as the nearest *occupied* Z-index
    to each plane's reference height.
    """
    planes = json.loads(Path(planes_aligned_path).read_text())["planes"]
    fc = [p for p in planes if p["orientation"] == "floor_ceiling"]
    if len(fc) < 2:
        raise ValueError("need both a floor and a ceiling plane in planes_aligned.json")
    fc_z = sorted(p["centroid_3d"][2] for p in fc)
    floor_ref, ceiling_ref = fc_z[0], fc_z[-1]

    idx_z = np.round((centers[:, 2] - origin[2]) / voxel_size - 0.5).astype(np.int64)
    occupied_z = np.unique(idx_z)
    z_centers = (occupied_z + 0.5) * voxel_size + origin[2]
    floor_z_idx = int(occupied_z[np.argmin(np.abs(z_centers - floor_ref))])
    ceiling_z_idx = int(occupied_z[np.argmin(np.abs(z_centers - ceiling_ref))])
    return [floor_z_idx], [ceiling_z_idx]
```

### `find_gap_cells`

```python
def find_gap_cells(centers, target_z_idx, origin, voxel_size, max_gap_cells):
    """(x_idx, y_idx) of small gaps within the occupied footprint of
    Z-layer `target_z_idx`: any 4-connected background component with at
    most `max_gap_cells` missing cells, whether or not it touches the
    layer's bounding-box edge.

    scipy.ndimage.binary_fill_holes alone only fills a background region
    that never touches the 2D grid's border -- the textbook "enclosed hole"
    definition. That's too strict here: on a real (not rectangular) scan
    footprint the coverage boundary itself is jagged, and real gaps worth
    filling are often long, thin margins running along an edge rather than
    compact interior pockets -- e.g. a strip a few cells wide but running
    most of a wall's length, because that whole strip's edge of coverage
    simply falls a bit short of the fitted wall plane. binary_fill_holes
    would refuse ALL of that (touches the border everywhere along its
    length). Cell count, not border-touching, is what actually distinguishes
    a real gap (however elongated) from a genuinely large opening (no
    ceiling coverage at all over some sizeable area) -- so components are
    selected purely by size."""
    from scipy import ndimage

    idx = np.round((centers - origin) / voxel_size - 0.5).astype(np.int64)
    layer_idx = idx[idx[:, 2] == target_z_idx][:, :2]
    if len(layer_idx) == 0:
        return np.empty((0, 2), dtype=np.int64)

    lo = layer_idx.min(axis=0)
    shape = tuple((layer_idx.max(axis=0) - lo + 1).tolist())
    grid = np.zeros(shape, dtype=bool)
    local = layer_idx - lo
    grid[local[:, 0], local[:, 1]] = True

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labels, n = ndimage.label(~grid, structure=cross)
    sizes = np.bincount(labels.ravel())
    small = sizes <= max_gap_cells
    small[0] = False  # label 0 is the occupied (non-background) region itself
    gap_local = np.argwhere(small[labels])
    return gap_local + lo  # back to (x_idx, y_idx)
```

### `compute_filler_voxels`

```python
def compute_filler_voxels(centers, mean_temperature, origin, voxel_size, planes_aligned_path,
                           max_gap_cells=DEFAULT_FILLER_MAX_GAP_CELLS):
    """Synthetic voxels for small floor/ceiling gaps (find_gap_cells). Only
    floor and ceiling layers are considered, never walls.

    A gap wide enough that its middle cell isn't 26-connected-adjacent to
    any *real* data can't be filled in one pass -- there's nothing to
    average there yet. So this grows inward in rings instead: each pass,
    every still-empty gap cell touching at least one real-or-already-filled
    neighbour gets filled from those neighbours' mean; repeat until no gap
    cell gains a value in a whole pass (a gap cell with no path at all to
    real data -- e.g. every neighbour is itself NaN-temperature real data --
    stays unfilled and is counted as skipped).

    Returns (filler_centers, filler_temps), both possibly empty.
    """
    floor_layers, ceiling_layers = find_floor_ceiling_z_layers(
        centers, origin, voxel_size, planes_aligned_path)
    print(f"filler: floor Z-layer(s) {floor_layers}, ceiling Z-layer(s) {ceiling_layers}")

    idx_all = np.round((centers - origin) / voxel_size - 0.5).astype(np.int64)
    # dict lookup: occupied cell index -> row into centers/mean_temperature
    lookup = {tuple(row): i for i, row in enumerate(idx_all)}

    filler_centers, filler_temps = [], []
    n_skipped = 0
    for z in floor_layers + ceiling_layers:
        gaps = find_gap_cells(centers, z, origin, voxel_size, max_gap_cells)
        pending = {(int(gx), int(gy)) for gx, gy in gaps}
        filled = {}  # (x_idx, y_idx) -> temperature, this layer only

        progress = True
        while pending and progress:
            progress = False
            newly_filled = {}
            for gx, gy in pending:
                neighbor_vals = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            zz = z + dz
                            row = lookup.get((gx + dx, gy + dy, zz))
                            if row is not None and np.isfinite(mean_temperature[row]):
                                neighbor_vals.append(mean_temperature[row])
                            elif zz == z and (gx + dx, gy + dy) in filled:
                                neighbor_vals.append(filled[(gx + dx, gy + dy)])
                if neighbor_vals:
                    newly_filled[(gx, gy)] = float(np.mean(neighbor_vals))
            if newly_filled:
                filled.update(newly_filled)
                pending -= newly_filled.keys()
                progress = True

        n_skipped += len(pending)
        for (gx, gy), val in filled.items():
            cell_center = (np.array([gx, gy, z]) + 0.5) * voxel_size + origin
            filler_centers.append(cell_center)
            filler_temps.append(val)

    print(f"filler: {len(filler_centers)} gap voxel(s) added (floor+ceiling), "
          f"{n_skipped} gap(s) skipped (no path to any valid-temperature data)")
    return np.asarray(filler_centers).reshape(-1, 3), np.asarray(filler_temps)
```

### `build_plotter` additions

Parameters added to the signature:
```python
def build_plotter(centers, counts, mean_temperature, voxel_size, min_count, colormap,
                   box_mesh, points, off_screen, clim_low_pct=1.0, clim_high_pct=99.0,
                   min_count_step=1, origin=None, declutter=True, show_no_temp=False,
                   filler_centers=None, filler_temps=None, filler_on=False):
```

State + render function (goes alongside `render_voxels`):
```python
    state = {"min_count": int(min_count), "declutter": bool(declutter), "filler": bool(filler_on)}
    has_filler = filler_centers is not None and len(filler_centers) > 0

    def render_filler():
        if not (state["filler"] and has_filler):
            if "filler" in pl.actors:
                pl.remove_actor("filler", render=False)
            return
        glyphs = _cube_glyphs(filler_centers, voxel_size)
        glyphs["temperature"] = np.repeat(filler_temps, glyphs.n_cells // len(filler_centers))
        # Same colormap/clim as the real voxels (so the fill color still
        # reads as a temperature on the same scale), but a thick contrasting
        # edge -- the "recognisible color edge" -- so a filled gap is never
        # mistaken for an actual measurement.
        pl.add_mesh(glyphs, scalars="temperature", cmap=colormap, clim=clim,
                    show_scalar_bar=False, show_edges=True,
                    edge_color=FILLER_EDGE_COLOR, line_width=FILLER_EDGE_WIDTH,
                    name="filler")
```

Inside `render_voxels`, the label text gained a filler line:
```python
        filler_line = ""
        if has_filler:
            filler_line = (f"\nfiller (fill floor/ceiling gaps): "
                            f"{'on' if state['filler'] else 'off'}   (g: toggle)")
        pl.add_text(
            f"min-count: {state['min_count']}   "
            f"(right/left bracket key: change by {min_count_step}, 0: reset)\n"
            f"declutter (drop floating voxels): {'on' if state['declutter'] else 'off'}   "
            f"(d: toggle)" + filler_line,
            position="upper_left", font_size=10, name="mincount_label")
```

Toggle + key binding + initial render calls:
```python
    def toggle_filler():
        state["filler"] = not state["filler"]
        render_filler()
        render_voxels()  # cheap; just to refresh the on-screen label text

    if has_filler:
        pl.add_key_event(FILLER_KEY, toggle_filler)

    render_voxels()
    render_filler()
```
(`render_voxels()` / `render_filler()` were the two initial calls at the
end of setup, right before the `box_mesh`/`points` overlay additions.)

### `main()` additions

CLI args:
```python
    ap.add_argument("--filler", action="store_true",
                    help="fill small gaps within the floor/ceiling voxel layers "
                         "(e.g. LiDAR dropout spots) with synthetic voxels colored by "
                         "the mean temperature of their neighbours (grown inward ring "
                         "by ring for gaps too wide to reach real data in one step), "
                         "outlined in a contrasting edge so they read as synthetic. "
                         "Floor/ceiling only, never walls. Off by default, "
                         "live-togglable with the 'g' key. Requires --planes-aligned "
                         "(needed to tell floor/ceiling apart from other layers).")
    ap.add_argument("--filler-max-gap-cells", type=int, default=DEFAULT_FILLER_MAX_GAP_CELLS,
                    help="only fill a gap (a connected run of missing cells) with at "
                         "most this many cells -- keeps a genuinely large opening "
                         "(no ceiling coverage at all over some area) from being "
                         "filled in as if it were a small dropout spot")
```

Computing and passing the filler data through:
```python
    filler_centers, filler_temps = None, None
    if args.filler:
        if not args.planes_aligned.exists():
            print(f"WARNING: --filler needs --planes-aligned ({args.planes_aligned} not "
                  f"found) to tell floor/ceiling apart -- disabling filler")
        else:
            filler_centers, filler_temps = compute_filler_voxels(
                centers, mean_temperature, origin, voxel_size, args.planes_aligned,
                max_gap_cells=args.filler_max_gap_cells)
```

`build_plotter(...)` call additionally passed:
```python
        show_no_temp=args.show_no_temp, filler_centers=filler_centers,
        filler_temps=filler_temps, filler_on=args.filler)
```
