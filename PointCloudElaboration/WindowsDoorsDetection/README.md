# WindowsDoorsDetection — doors and windows from a ZED session

Classifies every SAM mask in a recorded ZED session as **door / window / other**,
then pools multi-view votes per 3-D voxel so a physical opening ends up with
whichever class the majority of the views that actually saw it agree on.

This is `EmissivityCalculation`'s pipeline with the material question swapped
for the opening question. The segmentation, the geometric zone prior, the
calibration loader and the rosbag reader are **reused unchanged** — only the
class table, the classifier wrapper and the two driver scripts are new.

Downstream of this, and deliberately **out of scope** here: fitting planes and
polygons to the consensus voxels, and the OpenStudio `.osm` export. That step
consumes `door_window_voxels.csv`.

## Layout

```
WindowsDoorsDetection/
  classify_openings.py         stage 1 — per-frame SAM + CLIP
  opening_voxel_consensus.py   stage 2 — multi-view voxel vote + 3-D map
  opening_table.csv            the taxonomy: class, prompt, notes
  openings/
    table.py                   OpeningTable (stdlib csv, no pandas)
    classifier.py              OpeningClassifier (CLIP zero-shot)
    zone_prior.py              floor/ceiling cannot be an opening
```

## Running it

Two venvs, same split as `EmissivityCalculation` (stage 1 needs CLIP, stage 2
reads a rosbag):

```
:: stage 1 — CLIP venv
C:\venvs\emissivity\Scripts\python.exe classify_openings.py ^
    --session-dir ...\ZED\20260730_161223\fullrate --limit 3 --overlay

:: stage 2 — rosbags venv
C:\venvs\sensorfusion\Scripts\python.exe opening_voxel_consensus.py ^
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20
```

Outputs:

| path | what |
|---|---|
| `<session>/opening_map/<stem>/labels.npy` | int32 HxW SAM mask-id raster, full ZED grid |
| `<session>/opening_map/<stem>/segments.json` | schema `opening_map/v1` — id, bbox, centroid_px, area_px, top_class, confidence, top_k, zone |
| `<session>/opening_map/<stem>/overlay.png` | `--overlay` only; boxes drawn on openings, not on `other` |
| `<session>/opening_map_consensus/<stem>/segments.json` | same, with the consensus class substituted and a `consensus` block per segment |
| `<session>/opening_map_consensus/door_window_voxels.csv` | **the deliverable**: one row per opening voxel, within `--max-range` |
| `<session>/opening_map_consensus/door_window_voxels.ply` | same, coloured door=red / window=blue |

`door_window_voxels.csv` columns: `x,y,z,opening_class,agreement,n_observations,n_votes,w_door,w_other,w_window`.
Every class's pooled weight is written, not just the winner's, so the polygon
step can apply its own threshold without re-running the vote. `n_observations`
is the number of distinct frames that saw the voxel; `n_votes` counts LiDAR
points and therefore grows with how densely the scan happened to hit the
surface — it is a sampling statistic, not a view count.

## Decisions, and what they cost

**Three classes, from a CSV.** `opening_table.csv` has `class,prompt,notes`,
mirroring `emissivity_table.csv`, so splitting `door` into `door_open` /
`door_closed` or adding `window_shutter` later is a file edit, not a code edit.
Class names must be unique — there is no multi-prompt-per-class pooling, one
row is one class is one prompt.

**`other` competes in the vote on equal terms.** That is the whole defence
against false positives: a wall voxel resolves definitively to `other`, and a
door voxel has to beat the wall votes to win, so one confident bad view cannot
invent an opening. Pooling only door/window votes would leave nothing voting
*against* a false positive.

**Vote mass is pooled, not per-voxel winners.** Summing each voxel's hard
winner is a second argmax on top of the per-voxel one, and it systematically
amplifies whichever class is already most common. Here that class is `other`
by a wide margin, so doing it the naive way would erase openings outright.
This is the same fix `voxel_consensus.py` documents (it cost that pipeline
`painted_metal +276, concrete −178` before it was found).

**Voxel 0.20 m, and the consensus thresholds, are inherited, not re-derived.**
`--voxel 0.20`, `--min-vote-confidence 0.5`, `--min-agreement 0.0`,
`--depth-power 0.0` are `voxel_consensus.py`'s measured values. 0.20 m is
floored by the ~9 cm registration error of composing the two LiDAR↔camera
extrinsics, and it is also the resolution the polygon fit inherits — a 0.9 m
door leaf is ~5 voxels wide, so going much coarser blurs an opening's outline
into the wall around it.

**No FLIR-FOV crop.** `classify_session.py` crops to the ~16% of the ZED frame
the FLIR can see, because `project_to_flir.py` discards the rest anyway. Doors
and windows are not a thermal question and can be anywhere in the frame, so the
whole frame is always segmented. Cost: expect roughly the 3.9 h / 107-frame
figure `classify_session.py` quotes for its own uncropped mode.

## Stage 1B — merge + geometric plausibility

Runs inside `classify_openings.py`, after classification, before the write.
`--no-geometry-filter` turns it off and writes the raw per-SAM-segment result.

1. **Merge** touching same-class segments by connected components on the class
   raster, with a 2 px dilation to bridge the hairline seam a mullion leaves
   between two SAM fragments of the same window. The dilation decides
   connectivity only — a merge never absorbs another class's pixels. The merged
   raster is written back to `labels.npy` under new ids; `merged_from` records
   which SAM ids each detection absorbed.
2. **Window rule** — a merged window that stands on the floor but is shorter
   than `--glass-wall-h-ratio` (0.60) of the frame is rejected.
3. **Door rules**, in order: bay-edge veto → minimum width → floor contact
   required → must not clear the glass-wall height → re-merge survivors.

Rejected detections keep their pixels and carry a `rejected` block naming the
rule. Nothing is silent, and `segments.json` is normalised against the raster
at the end, so every id in `labels.npy` has exactly one record and the areas
sum to the labelled pixel count.

### Everything here is mask shape, never bbox

The bbox is derived and reported; it never decides, draws or exports. The
overlay draws `cv2.findContours` outlines of the merged masks. This is not
cosmetic — it is what made the original QA read correct.

### Three things the measurements overturned

Measured on session 9 frames 1-3 (19 opening detections), not eyeballed:

| assumption | what the data said |
|---|---|
| `touches_floor` = `y1 >= H-3` | **0 of 19** detections pass. SAM segments the floor as one region reaching y=H, so a door's bottom edge is at the wall/floor junction mid-frame, never at y=1079. Replaced by mask-adjacency to the floor-zoned segment, tolerance 120 px (measured gaps: 1/31/40/48/101/114 standing vs 370/462 floating). |
| containment veto, >50% overlap | **Never fires** — real door↔window mask overlap is 0.1–6.3%. `labels.npy` is a *partition*, so a frame fragment is never inside a window mask, only beside it. It looked contained on bounding boxes. Replaced by one-sided edge adjacency. |
| glass wall at h/H ≥ 0.75–0.85 | The **actual** glass wall measures 0.68 and is the largest opening in the session; 0.75 rejects everything. Lowered to 0.60. Thin evidence — 0.08 of margin on one session. |

### No LiDAR here, on purpose

Glazing returns no LiDAR. In frames 1-3, **11 of 19** detections got zero
points, including the 315×735 px glass wall at conf 0.98–0.99 (zero points in
all three frames). Where depth did exist, `pixel_height × median_depth / fy`
gave 2.4–3.2 m for ordinary openings, because a mask spans foreground mullion
and background-through-glass. A metric gate at stage 1 would fail hardest on
the class it is meant to validate, so the metric check belongs at stage 2, on
consensus voxels. **Note that a pure glass pane contributes no voxels either**
— the stage-2 metric gate has to run on the frame/reveal voxels around an
opening, not on the glazing.

### The edge-veto tests against ALL merged windows

Including ones rule 2 goes on to reject. The question is "is this strip at the
edge of a glazed bay", and the bay is there regardless of whether the region
cleared a height test. Measured: testing only survivors let 2 of 3 bay-edge
strips through, because the window beside them had just been rejected by rule 2.

**Known risk:** a real door standing immediately beside a window in the same
wall scores high on one side and is falsely vetoed. Stage 2's multi-view
consensus is the backstop; `--no-geometry-filter` disables 1B entirely.

### Measured effect, session 9 frames 1-3

Raw SAM `window=30, door=7` → after 1B `window=8, door=1`. Visual check on the
overlay confirms every rejected door was a vertical window mullion between
glazed bays. One true miss, present before and after 1B: the real door at the
far end of the corridor is small, dark and distant, and CLIP calls it `other`.

## Range limit: `--max-range` (stage 2, default 8 m)

A surface classified from 20 m away is a few pixels of an oblique, blurred
region. Worse, SAM masks a continuous corridor run of glazing as touching
pieces that are all correctly `window`, so stage 1B merges them into one
detection spanning a huge depth range — session 9 frame 2 has a single merged
window region built from 8 SAM segments whose sampled depth runs
**4.8 m (p5) → 17.5 m (p95)**.

The gate lives at the **vote**, not in stage 1B, and that placement is the
whole point:

* it uses **exact per-point depth**, so no densification is involved. Filling
  the sparse LiDAR by nearest neighbour and trimming per pixel was measured and
  does not work: the fill hands the far glass the depth of the near mullions,
  so that 4.8–17.5 m region reports a *median* of 4.4 m and keeps 96% of itself
  at a 12 m cutoff, 80% even at 6 m.
* there is **no fail-open case**. A region-level depth gate has to invent an
  answer for glazing, which returns no LiDAR at all — and the two largest near
  windows in these frames have exactly zero points. At the vote, a point that
  does not exist simply does not vote, which is already the correct behaviour.
* it needs **no new dependency**. `rosbags` is not in the emissivity venv and
  `scipy` is not in the sensorfusion venv, so a depth-aware stage 1B would have
  required adding one.

Measured on session 9, first 3 frames:

| | votes | voxels | opening voxels |
|---|---|---|---|
| `--max-range 0` (off) | 19035 | 2443 | 102 (window 100, door 2) |
| `--max-range 8` (default) | 10469 | 497 | 34 (window 32, door 2) |

**This changes the 3-D product only.** Stage 1's per-frame `overlay.png` still
draws the full-length merged regions — the long blue outline running down the
corridor is expected and is not what gets exported. Judge the result on
`door_window_voxels.csv`/`.ply`.

Note also why the opening-voxel counts are small: the glazing itself returns no
LiDAR, so a window's voxels come from its mullions, frame and reveal. Anything
downstream that fits a polygon to these voxels is fitting the **frame**, not
the glass.

## The ceiling rule eats high windows

`zone_of()` is reused unchanged, and its ceiling rule is
`cy < 0.30 and bw > 0.8 * bh`. **That 0.8 was tuned on FLIR-FOV-cropped
frames**, where the crop keeps ~39% of the width but ~68% of the height, so a
real ceiling patch loses most of the "wide" shape it has in a full frame. This
module runs uncropped, i.e. outside the regime that threshold was measured in.

Consequence: a wide window high in the frame — a clerestory, the top of a tall
window bay, a skylight — satisfies `cy<0.30 and bw>0.8*bh`, is called `ceiling`,
and is forced to `other`. It is undetectable by construction.

This was accepted deliberately (the floor/ceiling force is the spec), but it is
instrumented rather than hidden: `classify_openings.py` prints the forced count
per zone and warns explicitly when any of them were `ceiling`. To measure the
real loss on a site, run the same session with `--no-zone-constraint` and diff
the window counts. If it is material, retune the rule — do not silently trust
the window count.

Note also that stage 2's `--respect-zones` re-applies the prior to the
*per-segment rewrite* only. It cannot re-apply it to the voxels themselves: a
zone is a 2-D property of a segment, and a voxel pools segments from different
zones. This is not a gap in practice — stage 1 has already forced floor/ceiling
segments to `other` before they ever vote, so a floor segment never casts a
`window` vote in the first place.

## Weak points, honestly

* **`other` carries one prompt for a very heterogeneous class** — walls, floors,
  ceilings, radiators, pillars, people, clutter. It is the least well-posed of
  the three prompts and the most likely source of both false positives and
  false negatives. If per-frame precision looks bad, this is the first thing to
  look at.
* **A door seen edge-on or half-occluded** is a legitimately hard single-view
  call. That is exactly what the multi-view vote exists to fix, so judge the
  pipeline on `door_window_voxels.csv`, not on per-frame `segments.json`.
* **Glazed door panels and window bays** compete for the same evidence. The
  taxonomy has no way to express "a door that is mostly glass".

## Dependency note: `SensorFusionLoader`, not `Calibration`

Both scripts import the calibration loader from `Thesis-final-wt2/SensorFusionLoader/`
(found by searching upwards, see `_find_root`), which holds `rig_calibration.py`, `rig_calibration.yaml` and
`projection.py`.

`EmissivityCalculation`'s own scripts (`classify_session.py`,
`voxel_consensus.py`, `project_to_flir.py`) still hardcode
`Path(__file__).resolve().parent.parent / "Calibration"`, a directory that does
not exist in this repo — **they are broken here as-is.** That is a pre-existing
gap this module does not fix.

It does have to work *around* it, in one place worth knowing about:
`opening_voxel_consensus.py` reuses `project_to_flir.nearest_clouds_for_targets`
for the one-pass bag read, and importing that module executes its broken
`sys.path.insert`. It loads cleanly only because `SensorFusionLoader` is
imported *first*, putting `rig_calibration` and `projection` into `sys.modules`
before `project_to_flir` asks for them. **Do not reorder those imports.**
