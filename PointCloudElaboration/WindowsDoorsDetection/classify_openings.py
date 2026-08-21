"""Door / window / other per SAM mask, for every frame of a recorded ZED
session (driven by SensorFusion/sync_manifest.py's sync_manifest.json).

Stage 1 of two. This is EmissivityCalculation/classify_session.py with the
material question swapped for the opening question: same session-dir
convention, same SAM segmentation (reused unchanged), same geometric zone
prior, same per-frame output shape -- but each mask is scored against
opening_table.csv's three classes instead of the emissivity table's twenty
materials, and there is no emissivity value and no low-emissivity gate
(no opening class is catastrophic downstream the way a bare-metal call is).

Two deliberate differences from classify_session.py:

  * NO --crop-to-flir-fov. That crop exists because project_to_flir.py
    discards everything the FLIR cannot see, so classifying it was wasted
    CLIP time. Doors and windows are not a thermal question -- they can be
    anywhere in the ZED frame -- so the whole frame is always segmented.
    Cost: the ~16% FLIR footprint becomes 100% of the frame, so expect
    roughly the 3.9 h / 107-frame figure classify_session.py quotes for its
    own uncropped mode.

  * The zone prior FORCES, rather than reranks. A floor or ceiling segment
    cannot be a door or a window, so its ranking is restricted to {other};
    see openings/zone_prior.py, including the known failure mode where a
    wide window high in the frame is called "ceiling" and lost. The per-zone
    forced counts are printed at the end so that loss is measurable.

Output per frame, under <out-dir>/<flir_frame_stem>/:
    labels.npy    -- int32 HxW SAM mask-id raster on the full ZED pixel grid
    segments.json -- schema "opening_map/v1": per-segment id, bbox, centroid,
                     area, top_class, confidence, top_k, zone
    overlay.png   -- optional (--overlay), mask boundaries + class labels

Venv: the same one classify_session.py uses (torch + transformers + skimage).

Usage:
    py classify_openings.py --session-dir ...\\ZED\\20260730_161223\\fullrate --limit 3 --overlay
    py classify_openings.py --session-dir ...\\fullrate --every-n 5
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from openings import OpeningTable, OpeningClassifier, OTHER_CLASS
from openings.zone_prior import OPENING_ZONE_CANDIDATES, restrict_opening_ranking
from openings import geometry as geom

# --- sibling modules, imported by path -------------------------------------
# The Thesis-final-wt2 root is located by searching upwards, see _find_root.
# NOTE that the calibration loader lives in SensorFusionLoader/, NOT Calibration/ --
# EmissivityCalculation's own scripts still hardcode "../Calibration" and are
# broken in this repo as a result; that is a pre-existing gap, not something
# this module fixes.
def _find_root(start: Path) -> Path:
    """Walk up until the directory holding SensorFusionLoader/ is found.

    Searched rather than counted with a fixed number of .parent hops: this
    module has already been moved one level once (WindowsDetection/
    OpeningClassification/ -> WindowsDoorsDetection/), which silently broke a
    hardcoded count. Searching survives the next move too.
    """
    for d in [start, *start.parents]:
        if (d / "SensorFusionLoader").is_dir():
            return d
    raise RuntimeError(
        f"SensorFusionLoader/ not found in any parent of {start} -- it holds "
        "rig_calibration.py/.yaml and projection.py, which this script needs.")


_ROOT = _find_root(Path(__file__).resolve().parent)
# segmentation.py and zones.py are imported as bare modules rather than as
# emissivity.segmentation / emissivity.zones: the package __init__ pulls in
# table.py (pandas) and sources.py (the ZED SDK), neither of which this stage
# needs. Same trick voxel_consensus.py uses for zones.py.
sys.path.insert(0, str(_ROOT / "EmissivityCalculation" / "emissivity"))
from segmentation import sam_segments, segment_boxes, superpixel_segments  # noqa: E402
from zones import zone_of  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_zed_frames_dir(session_dir: Path) -> Path:
    """Same convention as SensorFusion/sync_manifest.py::load_zed_frames --
    frames live in metadata.json's recording.frames_dir (default "frames")."""
    meta_path = session_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return session_dir / (meta.get("recording", {}).get("frames_dir") or "frames")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-segment door/window/other classification for a synced ZED session")
    p.add_argument(
        "--session-dir", required=True, metavar="DIR",
        help="ZED session folder holding metadata.json + frames/ + sync_manifest.json "
             "(the output of SensorFusion/sync_manifest.py).")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="Output root (default: <session-dir>/opening_map).")
    # --- segmenter ---------------------------------------------------------
    p.add_argument("--segmenter", choices=("sam", "slic"), default="sam",
                   help="sam (default): Segment Anything masks follow real objects -- a door "
                        "leaf or a window bay comes out as one region -- at ~27 s/frame on CPU. "
                        "slic: ~0.4 s/frame, but it partitions the image into ~n_segments cells "
                        "regardless of content, which shreds a door into pieces that then "
                        "disagree with each other.")
    p.add_argument("--n-segments", type=int, default=100, metavar="N",
                   help="Target SLIC superpixel count per frame (default 100). slic only.")
    p.add_argument("--compactness", type=float, default=10.0,
                   help="SLIC compactness -- higher favors regular shapes over edge fidelity "
                        "(default 10.0). slic only.")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="SLIC pre-blur, segmentation decision only (default 0.0). slic only.")
    p.add_argument("--sam-model", default="facebook/sam-vit-base", metavar="REPO")
    p.add_argument("--sam-grid", type=int, default=16, metavar="N",
                   help="SAM point-prompt grid, N x N (default 16). Lower is faster.")
    p.add_argument("--sam-min-area", type=int, default=1500, metavar="PX",
                   help="Drop SAM masks smaller than this (default 1500).")
    p.add_argument("--sam-nms-iou", type=float, default=0.7, metavar="IOU",
                   help="Drop SAM masks overlapping a better one above this IoU (default 0.7).")
    # --- geometric prior ---------------------------------------------------
    p.add_argument("--zone-constraint", action=argparse.BooleanOptionalAction, default=True,
                   help="Force segments whose geometry says floor or ceiling to 'other' "
                        "(default on): a floor or ceiling patch is not an opening, and the "
                        "restriction reuses the same CLIP forward pass, so it is free. "
                        "--no-zone-constraint to disable -- worth doing once per site to "
                        "measure how many real windows the ceiling rule eats; see "
                        "openings/zone_prior.py.")
    # --- stage 1B: merge + geometric plausibility --------------------------
    p.add_argument("--geometry-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="Stage 1B (default on): merge touching same-class segments into one "
                        "detection, then reject the ones whose MASK SHAPE is implausible for a "
                        "door or window. Pixel-space only, no LiDAR -- glazing returns no LiDAR, "
                        "so a metric gate here would fail hardest on windows; the metric check "
                        "lives at stage 2. --no-geometry-filter writes the raw per-SAM-segment "
                        "classification instead.")
    p.add_argument("--merge-dilate-px", type=int, default=geom.DILATE_PX, metavar="PX",
                   help=f"Bridge gaps up to this width when merging same-class segments "
                        f"(default {geom.DILATE_PX}). Connectivity only -- a merge never "
                        "absorbs pixels of another class.")
    p.add_argument("--floor-tol-px", type=int, default=geom.FLOOR_TOL_PX, metavar="PX",
                   help=f"Mask-adjacency distance to the floor segment that still counts as "
                        f"standing on the floor (default {geom.FLOOR_TOL_PX}). NOT the frame "
                        "bottom: SAM segments the floor as one region reaching y=H, so "
                        "zones.py's y1>=H-3 test never fires for an opening. Measured door "
                        "gaps: 1/31/40/48/101/114 standing vs 370/462 floating.")
    p.add_argument("--glass-wall-h-ratio", type=float, default=geom.GLASS_WALL_H_RATIO,
                   metavar="F",
                   help=f"Merged-region height over frame height for a glazed wall (default "
                        f"{geom.GLASS_WALL_H_RATIO}). A floor-touching window below this is "
                        "rejected; a door at or above it is rejected. Session 9's real glass "
                        "wall measures 0.68, next candidates 0.57-0.60 -- THIN EVIDENCE, "
                        "re-check on a full session.")
    p.add_argument("--door-edge-window-frac", type=float, default=geom.DOOR_EDGE_WINDOW_FRAC,
                   metavar="F",
                   help=f"Reject a door candidate with a window abutting more than this "
                        f"fraction of one vertical side (default {geom.DOOR_EDGE_WINDOW_FRAC}) "
                        "-- it is a bay-edge frame strip. Replaces the containment veto, which "
                        "cannot fire: labels.npy is a partition, so measured door/window mask "
                        "overlap is 0.1-6.3%%, never the >50%% containment would need.")
    p.add_argument("--door-edge-band-px", type=int, default=geom.DOOR_EDGE_BAND_PX, metavar="PX",
                   help=f"Width of the band sampled beside a door candidate (default "
                        f"{geom.DOOR_EDGE_BAND_PX}).")
    p.add_argument("--min-door-width-px", type=int, default=geom.MIN_DOOR_WIDTH_PX, metavar="PX",
                   help=f"Sliver guard (default {geom.MIN_DOOR_WIDTH_PX}). Pixel width is "
                        "depth-dependent and therefore weak; the real width gate is metric, at "
                        "stage 2.")
    p.add_argument("--top-k", type=int, default=3, metavar="N",
                   help="How many (class, confidence) candidates to keep per segment (default 3).")
    p.add_argument("--table", default=None, metavar="CSV",
                   help="Path to a custom opening table (default: opening_table.csv next to "
                        "this script). Edit that file, not code, to change the taxonomy.")
    p.add_argument(
        "--clip-model", default=None, metavar="REPO",
        help="HF CLIP model (default laion/CLIP-ViT-H-14-laion2B-s32B-b79K, the same model "
             "classify_session.py measured as best on SAM masks -- its confidences are "
             "calibrated enough to be worth weighting the multi-view votes by). Use "
             "openai/clip-vit-large-patch14 for a faster, weaker run.")
    p.add_argument("--every-n", type=int, default=1, metavar="N",
                   help="Process every Nth triplet from sync_manifest.json (default 1 = all).")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Stop after N processed frames (for a quick test run).")
    p.add_argument("--overlay", action="store_true",
                   help="Also save a mask-boundary + class-label PNG per frame.")
    return p.parse_args()


def draw_overlay(image: np.ndarray, labels: np.ndarray, segments: list[dict]) -> np.ndarray:
    """Mask boundaries, with the OPENINGS drawn as their real mask outline.

    Deliberately NOT cv2.rectangle. A bounding box misrepresents what was
    classified: a thin frame fragment sitting beside a window bay looks
    contained in it on boxes and separate on masks, and it was exactly that
    illusion that made the first QA pass misread the false positives. The
    outline drawn here is the same mask stage 1B reasons about.

    Also does NOT label every segment: "other" is the great majority of masks
    in any frame, and writing it on all of them buries the handful of regions
    the overlay exists to show.
    """
    import cv2
    from skimage.segmentation import mark_boundaries

    colors = {"door": (0, 0, 255), "window": (255, 128, 0)}    # BGR
    bounded = (mark_boundaries(image, labels, color=(0, 1, 0)) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(bounded, cv2.COLOR_RGB2BGR)
    for seg in segments:
        if seg["top_class"] == OTHER_CLASS:
            continue
        color = colors.get(seg["top_class"], (0, 255, 255))
        mask = (labels == int(seg["id"])).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, color, 3)
        cx, cy = seg["centroid_px"]
        cv2.putText(bgr, f"{seg['top_class']} {seg['confidence']:.2f}",
                    (int(cx) - 40, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return bgr


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)

    manifest_path = session_dir / "sync_manifest.json"
    if not manifest_path.exists():
        print(f"No sync_manifest.json in {session_dir} -- run SensorFusion/sync_manifest.py first.",
              file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_dir = load_zed_frames_dir(session_dir)

    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "opening_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = OpeningTable(args.table) if args.table else OpeningTable()
    clip_kwargs = {"model_name": args.clip_model} if args.clip_model else {}
    classifier = OpeningClassifier(table, **clip_kwargs)

    # Loaded once: SAM's weights are ~375 MB and loading them per frame would
    # dominate the runtime.
    sam_model = None
    if args.segmenter == "sam":
        from transformers import SamModel
        print(f"Loading SAM ({args.sam_model}) ...")
        sam_model = SamModel.from_pretrained(args.sam_model).eval()

    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]
    seg_note = (f"SAM {args.sam_grid}x{args.sam_grid} prompts" if args.segmenter == "sam"
                else f"SLIC {args.n_segments} segments/frame target")
    print(f"Processing {len(triplets)} frame(s) from {manifest_path.name} ({seg_note}), "
          f"full frame (no FLIR-FOV crop)")
    if args.zone_constraint:
        forced_zones = sorted(z for z, c in OPENING_ZONE_CANDIDATES.items() if c == [OTHER_CLASS])
        print(f"Zone constraint: ON ({'/'.join(forced_zones)} segments forced to '{OTHER_CLASS}')")

    total_segments = 0
    total_forced = Counter()        # zone -> segments the prior pulled off an opening call
    total_class = Counter()         # per-SAM-segment classes, BEFORE stage 1B
    total_rejected = Counter()      # stage 1B rule -> merged detections it killed
    total_kept = Counter()          # class -> merged detections that survived 1B

    for i, triplet in enumerate(triplets):
        t0 = time.time()
        zed_file = triplet["zed"]["file"]
        flir_file = triplet["flir"]["file"]
        flir_stem = Path(flir_file).stem

        image = np.asarray(Image.open(frames_dir / zed_file).convert("RGB"))

        if args.segmenter == "sam":
            labels = sam_segments(image, model_name=args.sam_model, grid=args.sam_grid,
                                  min_area=args.sam_min_area, nms_iou=args.sam_nms_iou,
                                  model=sam_model)
        else:
            labels = superpixel_segments(image, n_segments=args.n_segments,
                                         compactness=args.compactness, sigma=args.sigma)
        boxes = segment_boxes(labels)
        ih, iw = image.shape[:2]
        zones = [zone_of(seg, ih, iw) for seg in boxes]

        crops = [image[y0:y1, x0:x1] for seg in boxes for (x0, y0, x1, y1) in [seg["bbox"]]]
        # Rank against every class, not just top-k: the zone prior renormalises
        # over a subset and so needs the full distribution. top_k is applied
        # afterwards, to the restricted ranking.
        results = classifier.classify_batch(crops, top_k=len(table.classes))

        n_forced = Counter()
        segments = []
        for seg, zone, ranked in zip(boxes, zones, results):
            zone_info = None
            if args.zone_constraint:
                restricted = restrict_opening_ranking(ranked, zone)
                if restricted is not ranked and ranked[0][0] != restricted[0][0]:
                    # Counted only when the prior actually changed the answer:
                    # a floor segment CLIP already called "other" lost nothing.
                    # The override stays in the record, so it is auditable and
                    # never silent -- same contract as classify_session.py's
                    # "gated" field.
                    zone_info = {
                        "overrode": ranked[0][0],
                        "overrode_confidence": ranked[0][1],
                        "reason": f"zone '{zone}' cannot be an opening",
                    }
                    n_forced[zone] += 1
                ranked = restricted
            top_class, confidence = ranked[0]
            record = {
                "id": seg["id"],
                # bbox is kept (classify_session.py drops it) because the
                # downstream polygon step works on opening outlines, and a
                # segment's box is the cheapest handle on one.
                "bbox": list(seg["bbox"]),
                "centroid_px": seg["centroid_px"],
                "area_px": seg["area_px"],
                "top_class": top_class,
                "confidence": confidence,
                "top_k": [(c, p) for c, p in ranked[:args.top_k]],
                # Read back by the consensus stage, which re-applies the prior
                # to the pooled votes -- a voxel pools segments from different
                # zones, so the constraint has to be enforced twice.
                "zone": zone,
            }
            if zone_info is not None:
                record["zone_forced"] = zone_info
            segments.append(record)
            total_class[top_class] += 1

        # --- STAGE 1B: merge + geometric plausibility -----------------------
        # Runs on the raster, so labels.npy is rewritten with the merged region
        # ids. Windows are resolved before doors: the door edge-veto tests
        # against surviving windows, so a door can never veto a window.
        geom_report = None
        if args.geometry_filter:
            labels, segments, geom_report = geom.apply_geometry_filter(
                labels, segments, zone_of,
                dilate_px=args.merge_dilate_px,
                floor_tol_px=args.floor_tol_px,
                glass_wall_h_ratio=args.glass_wall_h_ratio,
                door_edge_window_frac=args.door_edge_window_frac,
                door_edge_band_px=args.door_edge_band_px,
                min_door_width_px=args.min_door_width_px,
                min_region_area_px=args.sam_min_area)
            for rule, n in geom_report["rejected"].items():
                total_rejected[rule] += n
            for cls, n in geom_report["kept"].items():
                total_kept[cls] += n

        frame_dir = out_dir / flir_stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "labels.npy", labels.astype(np.int32))
        (frame_dir / "segments.json").write_text(json.dumps({
            "schema": "opening_map/v1",
            "generated_by": "classify_openings.py",
            "generated_utc": utc_now_iso(),
            "source_zed_frame": zed_file,
            "source_flir_frame": flir_file,
            "classes": table.classes,
            "clip_model": args.clip_model or "",
            "n_segments": len(segments),
            # Always null: this stage never crops. The key is kept so the file
            # stays readable by anything written against material_map/v1.
            "flir_fov_crop": None,
            "zone_constraint": {
                "enabled": args.zone_constraint,
                "forced_to_other": {z: n for z, n in sorted(n_forced.items())},
            },
            # Stage 1B. Segments here are MERGED detections, not raw SAM masks:
            # "merged_from" lists the SAM ids a detection absorbed, and a
            # rejected detection carries a "rejected" block naming the rule.
            "geometry_filter": None if geom_report is None else {
                "enabled": True,
                "merged": geom_report["merged"],
                "kept": geom_report["kept"],
                "rejected": geom_report["rejected"],
                "params": {
                    "merge_dilate_px": args.merge_dilate_px,
                    "floor_tol_px": args.floor_tol_px,
                    "glass_wall_h_ratio": args.glass_wall_h_ratio,
                    "door_edge_window_frac": args.door_edge_window_frac,
                    "door_edge_band_px": args.door_edge_band_px,
                    "min_door_width_px": args.min_door_width_px,
                },
            },
            "segments": segments,
        }, indent=2), encoding="utf-8")

        if args.overlay:
            import cv2
            cv2.imwrite(str(frame_dir / "overlay.png"), draw_overlay(image, labels, segments))

        total_segments += len(segments)
        total_forced += n_forced
        dt = time.time() - t0
        n_open = sum(1 for s in segments if s["top_class"] != OTHER_CLASS)
        forced_note = f", {sum(n_forced.values())} zone-forced" if n_forced else ""
        geom_note = ""
        if geom_report is not None:
            nrej = sum(geom_report["rejected"].values())
            geom_note = (f", 1B: {geom_report['merged'].get('window', 0)}W/"
                         f"{geom_report['merged'].get('door', 0)}D merged"
                         + (f", {nrej} rejected" if nrej else ""))
        print(f"[{i + 1}/{len(triplets)}] {flir_stem}: {len(segments)} segments, "
              f"{n_open} opening(s) in {dt:.1f}s "
              f"({dt / max(1, len(segments)) * 1000:.0f} ms/segment){forced_note}{geom_note}")

    if total_segments:
        print("\nper-frame class counts: " +
              ", ".join(f"{c}={n}" for c, n in total_class.most_common()))
        if args.zone_constraint:
            n_forced_total = sum(total_forced.values())
            print(f"zone prior forced {n_forced_total}/{total_segments} segments to "
                  f"'{OTHER_CLASS}' ({100.0 * n_forced_total / total_segments:.1f}%): " +
                  (", ".join(f"{z}={n}" for z, n in sorted(total_forced.items())) or "none"))
            if total_forced.get("ceiling"):
                print(f"  NOTE {total_forced['ceiling']} of those were 'ceiling'. zone_of()'s "
                      "ceiling rule was tuned on FLIR-FOV-cropped frames and over-fires on "
                      "full frames -- a wide window high in the frame lands there. Re-run "
                      "with --no-zone-constraint to see how many were real windows.")
        if args.geometry_filter:
            print("\nstage 1B kept: " +
                  (", ".join(f"{c}={n}" for c, n in total_kept.most_common()) or "nothing"))
            if total_rejected:
                print("stage 1B rejected, by rule:")
                for rule, n in total_rejected.most_common():
                    print(f"  {n:4d}  {rule}")
    print(f"Done. Output in {out_dir}")
    print("Next: opening_voxel_consensus.py --session-dir ... --bag ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
