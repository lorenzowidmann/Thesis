"""Material + emissivity per superpixel, for every frame of a recorded ZED
session (driven by SensorFusion/sync_manifest.py's sync_manifest.json).

This is the "smallest possible zone" version of main.py's 3x3 grid: each
frame is segmented into SLIC superpixels (respecting real visual edges
instead of a fixed grid), and every superpixel is classified independently
with the same CLIP zero-shot classifier + emissivity_table.csv lookup
already used elsewhere in this module.

Output per frame, under <out-dir>/<flir_frame_stem>/:
    labels.npy    -- int32 HxW superpixel-id raster, ZED pixel grid
    segments.json -- schema "material_map/v1": per-segment id, centroid,
                      area, top material, confidence, emissivity, top_k
    overlay.png   -- optional (--overlay), superpixel boundaries + labels

Usage:
    py classify_session.py --session-dir ...\\ZED\\20260730_161223\\fullrate --limit 3 --overlay
    py classify_session.py --session-dir ...\\fullrate --every-n 5
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from emissivity import EmissivityTable, MaterialClassifier
from emissivity.segmentation import superpixel_segments, segment_boxes


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
        description="Per-superpixel material/emissivity classification for a synced ZED session")
    p.add_argument(
        "--session-dir", required=True, metavar="DIR",
        help="ZED session folder holding metadata.json + frames/ + sync_manifest.json "
        "(the output of SensorFusion/sync_manifest.py).",
    )
    p.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Output root (default: <session-dir>/material_map).",
    )
    p.add_argument("--n-segments", type=int, default=100, metavar="N",
                    help="Target SLIC superpixel count per frame (default 100).")
    p.add_argument("--compactness", type=float, default=10.0,
                    help="SLIC compactness -- higher favors regular shapes over edge fidelity (default 10.0).")
    p.add_argument("--top-k", type=int, default=3, metavar="N",
                    help="How many (material, confidence) candidates to keep per segment (default 3).")
    p.add_argument("--table", default=None, help="Path to a custom emissivity CSV")
    p.add_argument(
        "--clip-model", default="openai/clip-vit-large-patch14",
        help="HF CLIP model (default matches main.py: openai/clip-vit-large-patch14).",
    )
    p.add_argument("--every-n", type=int, default=1, metavar="N",
                    help="Process every Nth triplet from sync_manifest.json (default 1 = all).")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Stop after N processed frames (for a quick test run).")
    p.add_argument("--overlay", action="store_true",
                    help="Also save a superpixel-boundary + material-label PNG per frame.")
    return p.parse_args()


def draw_overlay(image: np.ndarray, labels: np.ndarray, segments: list[dict]) -> np.ndarray:
    import cv2
    from skimage.segmentation import mark_boundaries

    bounded = (mark_boundaries(image, labels, color=(0, 1, 0)) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(bounded, cv2.COLOR_RGB2BGR)
    for seg in segments:
        cx, cy = seg["centroid_px"]
        label = f"{seg['top_material'][:10]} e={seg['emissivity']:.2f}"
        cv2.putText(bgr, label, (int(cx) - 25, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1)
    return bgr


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)

    manifest_path = session_dir / "sync_manifest.json"
    if not manifest_path.exists():
        print(f"No sync_manifest.json in {session_dir} -- run SensorFusion/sync_manifest.py first.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_dir = load_zed_frames_dir(session_dir)

    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "material_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = EmissivityTable(args.table) if args.table else EmissivityTable()
    classifier = MaterialClassifier(table, model_name=args.clip_model)

    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]
    print(f"Processing {len(triplets)} frame(s) from {manifest_path.name} "
          f"({args.n_segments} segments/frame target)")

    for i, triplet in enumerate(triplets):
        t0 = time.time()
        zed_file = triplet["zed"]["file"]
        flir_file = triplet["flir"]["file"]
        flir_stem = Path(flir_file).stem

        image = np.asarray(Image.open(frames_dir / zed_file).convert("RGB"))
        labels = superpixel_segments(image, n_segments=args.n_segments, compactness=args.compactness)
        boxes = segment_boxes(labels)

        crops = [image[y0:y1, x0:x1] for seg in boxes for (x0, y0, x1, y1) in [seg["bbox"]]]
        results = classifier.classify_batch(crops, top_k=args.top_k)

        segments = []
        for seg, res in zip(boxes, results):
            material, confidence = res[0]
            rec = table.lookup(material)
            segments.append({
                "id": seg["id"],
                "centroid_px": seg["centroid_px"],
                "area_px": seg["area_px"],
                "top_material": material,
                "confidence": confidence,
                "emissivity": rec.emissivity,
                "top_k": [(m, c) for m, c in res],
            })

        frame_dir = out_dir / flir_stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "labels.npy", labels.astype(np.int32))
        (frame_dir / "segments.json").write_text(json.dumps({
            "schema": "material_map/v1",
            "generated_by": "classify_session.py",
            "generated_utc": utc_now_iso(),
            "source_zed_frame": zed_file,
            "source_flir_frame": flir_file,
            "n_segments": len(segments),
            "segments": segments,
        }, indent=2), encoding="utf-8")

        if args.overlay:
            import cv2
            cv2.imwrite(str(frame_dir / "overlay.png"), draw_overlay(image, labels, segments))

        dt = time.time() - t0
        print(f"[{i + 1}/{len(triplets)}] {flir_stem}: {len(segments)} segments in {dt:.1f}s "
              f"({dt / max(1, len(segments)) * 1000:.0f} ms/segment)")

    print(f"Done. Output in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
