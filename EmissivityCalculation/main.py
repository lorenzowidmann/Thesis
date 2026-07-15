"""Estimate material emissivity from a camera image.

Grabs a frame (image file, webcam, or ZED 2i), classifies the material with
CLIP zero-shot, and prints the tabulated emissivity of the top matches.

Usage:
    py main.py --image test_images/brick.jpg
    py main.py --webcam --show
    py main.py --zed --show
    py main.py --zed-uvc --show
    py main.py --zed-uvc --show --live   # keep classifying frames until you press 'q'
    py main.py --image photo.jpg --roi 320,240,200,200
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
    p.add_argument(
        "--camera-index", type=int, default=0, metavar="N",
        help="Device index for --webcam / --zed-uvc (default 0)",
    )
    p.add_argument(
        "--roi",
        help="Crop region cx,cy,w,h (center + size in pixels) before classification",
    )
    p.add_argument("--top-k", type=int, default=3, help="Number of matches to show")
    p.add_argument("--show", action="store_true", help="Display frame with overlay")
    p.add_argument(
        "--live", action="store_true",
        help="Keep grabbing and classifying frames (model loaded once) until you "
        "press 'q' in the window or Ctrl+C. Needs --show and a camera source "
        "(--webcam/--zed/--zed-uvc), not --image",
    )
    p.add_argument("--table", default=None, help="Path to a custom emissivity CSV")
    return p.parse_args()


def crop_roi(frame: np.ndarray, roi: str) -> np.ndarray:
    cx, cy, w, h = (int(v) for v in roi.split(","))
    x0 = max(cx - w // 2, 0)
    y0 = max(cy - h // 2, 0)
    return frame[y0 : y0 + h, x0 : x0 + w]


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


def draw_overlay(frame, best, confidence):
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    label = f"{best.material}: e={best.emissivity} ({confidence:.0%})"
    cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
    classifier = MaterialClassifier(table)

    if args.image:
        source = ImageSource(args.image)
    elif args.webcam:
        source = WebcamSource(args.camera_index)
    elif args.zed_uvc:
        source = ZedUvcSource(args.camera_index)
    else:
        source = ZedSource()

    with source:
        if args.live:
            import cv2

            print("Live mode -- press 'q' in the image window to stop.")
            try:
                while True:
                    frame = source.grab()
                    if args.roi:
                        frame = crop_roi(frame, args.roi)
                    best, confidence = classify_frame(frame, classifier, table, args.top_k)
                    cv2.imshow("Emissivity estimation", draw_overlay(frame, best, confidence))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except KeyboardInterrupt:
                pass
            finally:
                cv2.destroyAllWindows()
            return 0

        frame = source.grab()
        if args.roi:
            frame = crop_roi(frame, args.roi)

        best, confidence = classify_frame(frame, classifier, table, args.top_k)

        if args.show:
            import cv2

            cv2.imshow("Emissivity estimation", draw_overlay(frame, best, confidence))
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
