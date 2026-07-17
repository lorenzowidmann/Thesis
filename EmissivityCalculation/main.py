"""Estimate material emissivity from a camera image.

Grabs a frame (image file, webcam, or ZED 2i), tiles it into a grid of cells,
classifies the material in each cell with CLIP zero-shot, and prints the
tabulated emissivity for every cell (not just the frame center). --show draws
every cell's box + best-match label. --roi restricts the grid to an explicit
cx,cy,w,h sub-region instead of the whole frame. --grid-size sets how many
rows/cols the (whole frame or --roi) region is divided into.

Usage:
    py main.py --image test_images/brick.jpg
    py main.py --webcam --show
    py main.py --zed --show
    py main.py --zed-uvc --show
    py main.py --zed-uvc --show --live   # keep classifying frames until you press 'q'
    py main.py --image photo.jpg --roi 320,240,200,200
    py main.py --image photo.jpg --grid-size 4
"""

import argparse
import sys

import numpy as np

from emissivity import EmissivityTable, MaterialClassifier
from emissivity.sources import ImageSource, WebcamSource, ZedSource, ZedUvcSource


def parse_args():
    p = argparse.ArgumentParser(description="Material emissivity estimation")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to an image file")
    src.add_argument("--webcam", action="store_true", help="Use the default webcam")
    src.add_argument(
        "--zed", action="store_true",
        help="Use the ZED 2i camera via the ZED SDK (requires pyzed + NVIDIA GPU/CUDA)",
    )
    src.add_argument(
        "--zed-uvc", action="store_true",
        help="Use the ZED 2i as a plain UVC webcam (OpenCV only, no SDK/GPU needed; "
        "crops the left eye out of the side-by-side stereo frame)",
    )
    src.add_argument(
        "--shared", action="store_true",
        help="Read the right eye from a running CameraServer/camera_server.py "
        "instead of opening the camera directly -- lets the grid tool run "
        "alongside drive_view.py off one shared ZED (start camera_server.py first)",
    )
    p.add_argument(
        "--camera-index", default="0", metavar="N",
        help="Device index for --webcam / --zed-uvc (default 0). On Linux, also "
        "accepts a device path like /dev/video1 -- useful when OpenCV's numeric "
        "index doesn't line up with the actual node, which happens with "
        "multi-node UVC cameras like the ZED 2i (check with 'v4l2-ctl "
        "--list-devices')",
    )
    p.add_argument(
        "--roi",
        help="Restrict the grid to an explicit cx,cy,w,h (center + size in "
        "pixels) sub-region instead of tiling the whole frame.",
    )
    p.add_argument(
        "--grid-size", type=int, default=3, metavar="N",
        help="Tile the classified region into an NxN grid (default 3) and "
        "classify each cell independently instead of just the frame center.",
    )
    p.add_argument("--show", action="store_true", help="Display frame with overlay")
    p.add_argument(
        "--live", action="store_true",
        help="Keep grabbing and classifying frames (model loaded once) until you "
        "press 'q' in the window or Ctrl+C. Needs --show and a camera source "
        "(--webcam/--zed/--zed-uvc), not --image",
    )
    p.add_argument(
        "--classify-every", type=int, default=5, metavar="N",
        help="--live only: re-run CLIP every Nth frame (default 5); every frame "
        "is still shown, with the last result overlaid in between. CLIP "
        "inference (~300-400ms on CPU) is what makes --live feel laggy, not "
        "the camera, so this decouples the smooth preview from it. Use 1 to "
        "classify every frame",
    )
    p.add_argument("--table", default=None, help="Path to a custom emissivity CSV")
    p.add_argument(
        "--clip-model", default="openai/clip-vit-large-patch14",
        help="HF CLIP model to use (default: openai/clip-vit-large-patch14, "
        "more accurate than the original openai/clip-vit-base-patch32). Each "
        "model is cached separately -- switching does not evict the other's "
        "cache. First use of a new model needs network access to download it "
        "(unset HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE for that one run).",
    )
    return p.parse_args()


def roi_bounds(frame: np.ndarray, roi: str) -> tuple[int, int, int, int]:
    cx, cy, w, h = (int(v) for v in roi.split(","))
    x0 = max(cx - w // 2, 0)
    y0 = max(cy - h // 2, 0)
    x1 = min(x0 + w, frame.shape[1])
    y1 = min(y0 + h, frame.shape[0])
    return x0, y0, x1, y1


def crop_roi(frame: np.ndarray, roi: str) -> np.ndarray:
    x0, y0, x1, y1 = roi_bounds(frame, roi)
    return frame[y0:y1, x0:x1]


def parse_camera_index(value: str) -> int | str:
    """"0"/"1"/... -> int for a normal numeric index; anything else (e.g. a
    Linux device path like "/dev/video1") is passed through as-is."""
    try:
        return int(value)
    except ValueError:
        return value


def default_center_roi(frame: np.ndarray, fraction: float = 0.5) -> str:
    """Centered square box (fraction of the frame's shorter side) so a camera
    source can be aimed at one material instead of classifying the whole,
    likely mixed-content, frame."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * fraction)
    return f"{w // 2},{h // 2},{side},{side}"


def classify_frame(frame, classifier, table, top_k):
    """Classify one frame and print the match table. Returns the best EmissivityRecord."""
    results = classifier.classify(frame, top_k=top_k)

    print(f"\n{'Material':<22}{'Confidence':<12}{'Emissivity':<12}Range")
    print("-" * 60)
    for material, conf in results:
        rec = table.lookup(material)
        print(f"{material:<22}{conf:<12.1%}{rec.emissivity:<12.2f}{rec.emissivity_range}")

    best = table.lookup(results[0][0])
    print(f"\nBest estimate: {best.material} -> emissivity = {best.emissivity} ({best.notes})")
    return best, results[0][1]


def grid_boxes(width: int, height: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    """Tile a width x height region into rows*cols cells, each (x0, y0, x1, y1)
    in that region's local coordinates, row-major order."""
    xs = [round(c * width / cols) for c in range(cols + 1)]
    ys = [round(r * height / rows) for r in range(rows + 1)]
    return [
        (xs[c], ys[r], xs[c + 1], ys[r + 1])
        for r in range(rows)
        for c in range(cols)
    ]


def classify_grid(frame, classifier, table, rows, cols, roi=None):
    """Classify every cell of a rows x cols grid tiling `frame` (or the
    sub-region given by `roi`, a cx,cy,w,h string, if provided). Prints a
    compact grid table and returns a list of
    (row, col, x0, y0, x1, y1, EmissivityRecord, confidence) in absolute frame
    coordinates."""
    if roi:
        base_x0, base_y0, base_x1, base_y1 = roi_bounds(frame, roi)
    else:
        base_x0, base_y0 = 0, 0
        base_y1, base_x1 = frame.shape[:2]

    cells = []
    print(f"\n{'Row':<5}{'Col':<5}{'Material':<20}{'Confidence':<12}Emissivity")
    print("-" * 60)
    boxes = grid_boxes(base_x1 - base_x0, base_y1 - base_y0, rows, cols)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        r, c = divmod(i, cols)
        abs_x0, abs_y0 = base_x0 + x0, base_y0 + y0
        abs_x1, abs_y1 = base_x0 + x1, base_y0 + y1
        crop = frame[abs_y0:abs_y1, abs_x0:abs_x1]
        material, confidence = classifier.classify(crop, top_k=1)[0]
        rec = table.lookup(material)
        print(f"{r:<5}{c:<5}{material:<20}{confidence:<12.1%}{rec.emissivity:.2f}")
        cells.append((r, c, abs_x0, abs_y0, abs_x1, abs_y1, rec, confidence))

    best_r, best_c, *_, best_rec, best_conf = max(cells, key=lambda cell: cell[7])
    print(
        f"\nBest estimate: {best_rec.material} (row {best_r}, col {best_c}) "
        f"-> emissivity = {best_rec.emissivity} ({best_conf:.0%})"
    )
    return cells


def draw_grid_overlay(frame, cells):
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for _r, _c, x0, y0, x1, y1, rec, _confidence in cells:
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 1)
        label = f"{rec.material[:12]} e={rec.emissivity:.2f}"
        cv2.putText(
            bgr, label, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
    return bgr


def main():
    args = parse_args()

    if args.live and args.image:
        print("--live needs a camera source (--webcam/--zed/--zed-uvc), not --image.", file=sys.stderr)
        return 2
    if args.live and not args.show:
        print("--live needs --show (press 'q' in the window to stop).", file=sys.stderr)
        return 2

    table = EmissivityTable(args.table) if args.table else EmissivityTable()
    classifier = MaterialClassifier(table, model_name=args.clip_model)

    camera_index = parse_camera_index(args.camera_index)

    if args.image:
        source = ImageSource(args.image)
    elif args.shared:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CameraServer"))
        from shared_frame import SharedZedSource
        source = SharedZedSource(eye="right")
    elif args.webcam:
        source = WebcamSource(camera_index)
    elif args.zed_uvc:
        source = ZedUvcSource(camera_index, eye="right")
    else:
        source = ZedSource()

    rows = cols = args.grid_size

    with source:
        if args.live:
            import cv2

            classify_every = max(1, args.classify_every)
            print("Live mode -- press 'q' in the image window to stop.")
            cells = None
            frame_count = 0
            try:
                while True:
                    frame = source.grab()
                    if cells is None or frame_count % classify_every == 0:
                        cells = classify_grid(frame, classifier, table, rows, cols, args.roi)
                    frame_count += 1
                    cv2.imshow("Emissivity estimation", draw_grid_overlay(frame, cells))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except KeyboardInterrupt:
                pass
            finally:
                cv2.destroyAllWindows()
            return 0

        frame = source.grab()
        cells = classify_grid(frame, classifier, table, rows, cols, args.roi)

        if args.show:
            import cv2

            cv2.imshow("Emissivity estimation", draw_grid_overlay(frame, cells))
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
