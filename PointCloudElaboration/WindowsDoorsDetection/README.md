# WindowsDoorsDetection — doors and windows from a ZED session

Classifies every region of a recorded ZED session as **door / window / other**,
then pools multi-view votes per 3-D voxel so a physical opening ends up with
whichever class the majority of the views that actually saw it agree on.

Stage 1 is one **Mask2Former** forward pass on the **ADE20K-150** taxonomy,
which already contains `windowpane` and `door`. It replaced SAM-everything +
CLIP-zero-shot; see "Why Mask2Former replaced SAM + CLIP" below for the
measurements that motivated the swap.

Downstream of this, and deliberately **out of scope** here: fitting planes and
polygons to the consensus voxels, and the OpenStudio `.osm` export. That step
consumes `door_window_voxels.csv`.

## Layout

```
WindowsDoorsDetection/
  classify_openings.py         stage 1 — Mask2Former + stage 1B geometry
  opening_voxel_consensus.py   stage 2 — multi-view voxel vote + 3-D map
  opening_table.csv            the taxonomy: class, ade, prompt, notes
  openings/
    segmentation_m2f.py        Mask2Former -> regions, confidence, zones
    lidar_metrics.py           metric size of a masked region, from the bag
    geometry.py                stage 1B: merge + plausibility rules
    table.py                   OpeningTable (stdlib csv, no pandas)
    zone_prior.py              LEGACY — superseded by zone_from_ade
    classifier.py              LEGACY — the CLIP zero-shot path, unused
```

## Running it

```
:: stage 1 — CLIP-free now, but still the torch venv. --bag turns on the
:: metric door check and additionally needs rosbags.
C:\venvs\emissivity\Scripts\python.exe classify_openings.py ^
    --session-dir ...\ZED\20260730_161223\fullrate ^
    --bag ...\rosbag2_2026_07_30-18_12_20 --limit 5 --overlay

:: stage 2 — rosbags venv, unchanged
C:\venvs\sensorfusion\Scripts\python.exe opening_voxel_consensus.py ^
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20
```

Outputs:

| path | what |
|---|---|
| `<session>/opening_map/<stem>/labels.npy` | int32 HxW region-id raster, full ZED grid, **-1 where no component reached `--min-area`** |
| `<session>/opening_map/<stem>/segments.json` | schema `opening_map/v1` — id, bbox, centroid_px, area_px, top_class, confidence, top_k, zone, `ade` |
| `<session>/opening_map/<stem>/overlay.png` | `--overlay` only; kept openings in colour, **rejected candidates in yellow with the rule that killed them** |
| `<session>/opening_map_consensus/<stem>/segments.json` | same, with the consensus class substituted and a `consensus` block per segment |
| `<session>/opening_map_consensus/door_window_voxels.csv` | **the deliverable**: one row per opening voxel, within `--max-range` |
| `<session>/opening_map_consensus/door_window_voxels.ply` | same, coloured door=red / window=blue |

`door_window_voxels.csv` columns: `x,y,z,opening_class,agreement,n_observations,n_votes,w_door,w_other,w_window`.
Every class's pooled weight is written, not just the winner's, so the polygon
step can apply its own threshold without re-running the vote. `n_observations`
is the number of distinct frames that saw the voxel; `n_votes` counts LiDAR
points and therefore grows with how densely the scan happened to hit the
surface — it is a sampling statistic, not a view count.

## Why Mask2Former replaced SAM + CLIP

Measured on session 9, same frames, same CPU:

| | SAM vit-base + CLIP | Mask2Former swin-large ADE |
|---|---|---|
| runtime / frame | ~27 s | **4.7–7.2 s** |
| boundary resolution | masks decoded at 256×256, `INTER_NEAREST` to 1920×1080 → ~7 px staircase | decoder logits upsampled bilinear at full res |
| unlabelled pixels | `_fill_gaps` painted them with the nearest id, **inventing** outlines across the floor and radiators | left at -1, they simply do not vote |
| `other` | one CLIP prompt covering walls, floors, ceilings, radiators, pillars, people, clutter | the other 148 ADE classes, each keeping its own region |
| corridor-end door | **missed** — CLIP called it `other` | found, conf 0.90 |

Two structural wins came free with the taxonomy:

* **The floor is a real mask.** Stage 1B's floor-contact test used to run
  against `zone_of()`'s bbox guess; it now runs against ADE `floor`.
* **The ceiling rule that ate high windows is gone.** `zone_prior.py` documents
  `zone_of()` calling a wide, high window `ceiling` — `cy < 0.30 and
  bw > 0.8 * bh`, a threshold tuned on FLIR-FOV-cropped frames and used here on
  uncropped ones — and forcing it to `other`, i.e. undetectable by
  construction. Zones now come from the ADE class
  (`segmentation_m2f.zone_from_ade`), so a clerestory stays a windowpane.
  `zone_prior.py` and `classifier.py` are kept on disk but nothing imports them.

Per-pixel **confidence** is derived rather than taken from the model, because
`post_process_semantic_segmentation` returns an argmax and stage 2 gates on
`--min-vote-confidence 0.5`. It is the winner's share of Mask2Former's semantic
map, `seg.max(c) / seg.sum(c)`. Well-defined, in [0, 1], and **not calibrated** —
treat it as a ranking score. Observed: 0.90 on the corridor door, 0.61–0.84 on
the glazed bays, 0.43–0.45 on the persistent false positive.

## Stage 1B — merge + geometric plausibility

Runs inside `classify_openings.py`, after segmentation, before the write.
`--no-geometry-filter` turns it off and writes the raw per-ADE-region result.

1. **Merge** touching same-class regions by connected components, with a 2 px
   dilation to bridge the hairline seam a mullion leaves between two fragments
   of the same window. The dilation decides connectivity only — a merge never
   absorbs another class's pixels. `merged_from` records what each detection
   absorbed.
2. **Window rule** — OFF by default (`--window-filter` to enable). Windows are
   merged and kept; only doors are filtered. The rule rejected a real bay at
   h_ratio 0.577 against a 0.60 threshold, and unlike a door a window has no
   metric check to arbitrate, because glazing returns no LiDAR.
3. **Door rules**, in order: metric veto → bay-edge veto → the three size rules
   → floor contact.

Rejected detections keep their pixels and carry a `rejected` block naming the
rule. Nothing is silent, and `segments.json` is normalised against the raster at
the end, so every id in `labels.npy` has exactly one record.

### Everything here is mask shape, never bbox

The bbox is derived and reported; it never decides, draws or exports. The
overlay draws `cv2.findContours` outlines of the merged masks. This is not
cosmetic — it is what made the original QA read correct.

### The LiDAR now arbitrates door size, and only door size

`--bag` measures every door candidate in metres before any size rule may
discard it (`openings/lidar_metrics.py`). Three outcomes:

* **ok** — rescues a candidate a size rule wanted to kill. Recorded as
  `geometry.rescued_by_lidar` and counted under `rescued` in the report.
* **bad** — rejects a candidate the size rules would have kept, rule
  `door_metric_dims`. Fires *before* the pixel rules: a real measurement beats
  every proxy below it.
* **unknown** — no measurement. The pixel rules decide alone. This is not a
  soft `bad`, and conflating the two would make every distant or glazed
  candidate fail for lack of evidence.

Rescuable rules are exactly the depth-dependent ones —
`door_below_min_area`, `door_below_min_width`, `door_taller_than_glass_wall`.
Floor contact and bay-edge adjacency are **never** rescued: they are statements
about where a region sits, which no metric measurement answers.

Size is `extent_px * median_depth / focal` — the mask's full pixel extent, with
only the scale borrowed from the LiDAR. Measuring the 3-D extent of the returns
instead would under-report systematically: LiDAR covers a door leaf densely but
stops short of the top edge and the reveal.

An abstention always says **why**, `few_points` or `multi_depth`, and the
distinction is the actionable part: `few_points` means the surface returned
nothing, `multi_depth` means the mask is not one surface and the fix is
upstream in what got merged into it.

Windows are untouched by any of this. Glazing returns nothing — see the
measurement in the next section — so a metric window gate would fail hardest on
the class it is meant to validate.

### Three things the measurements overturned (SAM era, still true)

| assumption | what the data said |
|---|---|
| `touches_floor` = `y1 >= H-3` | **0 of 19** detections pass. The floor is segmented as one region reaching y=H, so a door's bottom edge is at the wall/floor junction mid-frame. Replaced by mask-adjacency to the floor region, tolerance 120 px (measured gaps: 1/31/40/48/101/114 standing vs 370/462 floating). |
| containment veto, >50% overlap | **Never fires** — real door↔window mask overlap is 0.1–6.3%. Replaced by one-sided edge adjacency. |
| glass wall at h/H ≥ 0.75–0.85 | The **actual** glass wall measures 0.68; 0.75 rejects everything. Lowered to 0.60. Thin evidence — and see "Known problems" below, where 0.60 is now rejecting a real bay at 0.577. |

## The cloud both stages read: `--cloud-source`

Default **`raw`**. Both stages call `lidar_metrics.load_clouds`, so they cannot
silently read different clouds — stage 1's door measurements and stage 2's
votes have to agree about what the sensor saw.

`registered` (FAST-LIO's `/cloud_registered`) is kept for comparison and is
measurably broken for this purpose. `raw` rebuilds world clouds from
`/livox/lidar` + `/Odometry` through `../LivoxLidarOdometryLoader`. The poses
are FAST-LIO's either way — this changes which *points* are carried, not where
the rig thought it was.

Session 9, 5 frames, everything else identical:

| | `registered` | `raw` |
|---|---|---|
| points / scan | 6 349–6 465 | **79 765–84 196** |
| footprint in frame | u 666..1309 (8% of area) | u 0..1920, v 275..898 |
| votes | 16 699 | **320 286** |
| voxels | 513 | 1 326 |
| **opening voxels** | **0** | **82** (window 82, door 0) |
| agreement | — | median 1.00, 100% ≥ 0.5 |
| observations/voxel | — | median 4 |

The 82 voxels land as two strips down opposite sides of the corridor —
y ∈ [0.5, 1.7] and y ∈ [−1.3, −1.1], x 0.9–4.9 m — which is the two glazed
walls. Two things to read off them before trusting the polygon fit:

* **z spans only −0.1 to 1.3 m**, concentrated at 0.1–0.5. The HAP's 25°
  vertical against the ZED's 54° means the laser reaches roughly the middle of
  the frame height, so a floor-to-ceiling bay is captured over its lower ~1.4 m
  and no more. Not fixable by re-running FAST-LIO; it is the sensor.
* **Side A is 1.2 m thick, side B is one voxel thick.** Side B (21 of 24
  voxels at y = −1.1) is a clean plane. Side A is smeared because ADE
  `windowpane` swallows the radiators standing in front of the glass, so their
  surfaces vote `window` too.

`door = 0` at the default range: the only door found is the corridor-end
opening at 34.4 m, and `--max-range 8` drops its votes. At `--max-range 40` it
yields 474 door voxels — but `n_observations` falls from a median of 4 to 2 and
474 voxels is far more than a 2.5 × 2.7 m opening should fill, so those are
smeared along the ray, not a measurement. The honest fix is to use frames where
the rover is closer to that door, not a wider gate.

## Range limit: `--max-range` (stage 2, default 8 m)

A surface classified from 20 m away is a few pixels of an oblique, blurred
region. The gate lives at the **vote**, not in stage 1B, and that placement is
the whole point:

* it uses **exact per-point depth**, so no densification is involved. Filling
  the sparse LiDAR by nearest neighbour and trimming per pixel was measured and
  does not work: the fill hands the far glass the depth of the near mullions.
* there is **no fail-open case**. At the vote, a point that does not exist
  simply does not vote, which is already the correct behaviour.

**This changes the 3-D product only.** Stage 1's `overlay.png` still draws the
full-length regions. Judge the result on `door_window_voxels.csv`/`.ply`.

## Known problems, honestly

Measured on session 9 frames 1–5 with `--bag`, after the switch.

* **~~Stage 2 starves~~ — FIXED, see "The cloud both stages read" below.**
  `/cloud_registered` is a heavily cropped version of what the sensor
  recorded. Measured on session 9, first triplet:

  | | raw `/livox/lidar` | FAST-LIO `/cloud_registered` |
  |---|---|---|
  | points / message | 83 096 | **6 465** (−92%) |
  | azimuth | −59.5° … +60.8° | **−17.2° … +17.4°** |
  | elevation | −13.4° … +13.2° | −4.3° … +8.7° |
  | min range | **1.02 m** | **4.05 m** |

  Projected into the ZED frame that is a patch spanning u ∈ [666, 1309],
  v ∈ [369, 628] — about **8% of the frame area**, and *every one* of the 6465
  points already lands inside the image, so the camera FOV clips nothing. Both
  window bays fall entirely outside it: **0 returns even inside their bounding
  boxes**, not just outside their masks.

  The 4.05 m floor looks like FAST-LIO's `preprocess/blind`, and the ±17°
  azimuth cone like `mapping/fov_degree`; the sensor is a Livox HAP (120° × 25°),
  not a Mid-360. Re-running FAST-LIO with those relaxed, or registering
  `/livox/lidar` against `/Odometry` directly, is the fix. It is not a
  Mask2Former problem and not a glass problem.

  **`../LivoxLidarOdometryLoader/` rebuilds the cloud from the raw topic** and
  recovers it without re-running anything. Measured, frame `20250906_233144_R`:

  | | in frame | footprint | window id=14 | window id=15 |
  |---|---|---|---|---|
  | `/cloud_registered` | 6 465 | u 666..1309 | **0** | **0** |
  | rebuilt raw | 77 894 | u 0..1920 | **5 833** | **5 897** |

  **This disproves a claim this README made for a long time.** "Glazing returns
  no LiDAR — 11 of 19 opening detections got ZERO points, including the
  315×735 px glass wall at conf 0.98–0.99" was measured on the cropped cloud,
  which cannot tell glass apart from out-of-footprint. On the rebuilt cloud
  those same bays return **thousands of points at 1.6–2.5 m**.

  That premise is the stated reason the window rules are pixel-only, the reason
  the metric gate was put at stage 2 rather than stage 1B, and the reason
  `--max-range` gates at the vote. All three need re-deriving on a full cloud.
  Nothing in this module is *wrong* as code; its justifications are.
* **ADE `windowpane` is the glazed bay, occluders included.** A radiator
  standing in front of glass is inside the window mask. Arguably right for a
  polygon fit of the frame; not what the class name says.
* **Windows have no filter at all now.** `--window-filter` is off, so a
  Mask2Former `windowpane` false positive has nothing standing between it and
  the vote. Nothing in these 5 frames exercises that, which means it is
  untested rather than safe.
* **`--door-h-m` / `--door-w-m` are wide enough to admit a passage** (3.00 ×
  2.80 m), which is deliberate — the corridor-end opening is 2.77 × 2.55 m and
  is real — but it means the band no longer discriminates a door from a
  garage-sized hole. The lower bound is doing the work.

### Fixed, and what the fix was worth

Three thresholds were measured wrong on the first pass. All were guesses that
sat just inside the data.

| was | is | why |
|---|---|---|
| `MAX_DEPTH_RATIO 1.35` | **1.50** | 4 of 6 door candidates abstained `multi_depth` at 1.36/1.40/1.41/1.42 — the check was switching itself off on almost every real candidate, and the same object flipped `unknown`↔`bad` between frames on scan jitter. Now `unknown=0`. |
| `MAX_DOOR_W_M 2.20` | **2.80** | The corridor-end opening measures 2.36–2.55 m wide, consistently, and is a real passage. 2.20 rejected it as `door_metric_dims`. `MAX_DOOR_H_M` 2.80 → 3.00 for the same reason. |
| window rule ON | **`--window-filter` off** | It rejected a real bay at h_ratio 0.577 against a 0.60 threshold whose own comment warned it separated by 0.08. |

Effect on session 9 frames 1–5, doors only:

| | before | after |
|---|---|---|
| metric verdicts | ok=0, bad=2, unknown=4 | **ok=3, bad=3, unknown=0** |
| corridor-end door (2.67–2.74 × 2.36–2.52 m) | rejected / abstained | **kept, 3/3 frames, conf 0.87–0.90** |
| radiator-recess FP (0.37–0.82 × 0.40–0.43 m) | kept, 3/3 frames | **rejected, 3/3 frames** |
| windows kept | 11 of 15 | **15 of 15** |

The false positive is the case the metric check was built for: it is *stable*
across frames, so multi-view voting would never have removed it, and it is
0.4 m wide, so one look through the LiDAR settles it.

## Dependency note: `SensorFusionLoader`, not `Calibration`

Both scripts import the calibration loader from `Thesis-final-wt2/SensorFusionLoader/`
(found by searching upwards, see `_find_root`), which holds `rig_calibration.py`,
`rig_calibration.yaml` and `projection.py`.

`EmissivityCalculation`'s own scripts (`classify_session.py`,
`voxel_consensus.py`, `project_to_flir.py`) still hardcode
`Path(__file__).resolve().parent.parent / "Calibration"`, a directory that does
not exist in this repo — **they are broken here as-is.** That is a pre-existing
gap this module does not fix.

It does have to work *around* it, in two places now:
`opening_voxel_consensus.py` and `classify_openings.py::_load_lidar_stack` both
reuse `project_to_flir.nearest_clouds_for_targets` for the one-pass bag read,
and importing that module executes its broken `sys.path.insert`. It loads
cleanly only because `SensorFusionLoader` is imported *first*, putting
`rig_calibration` and `projection` into `sys.modules` before `project_to_flir`
asks for them. **Do not reorder those imports.**
