"""Estimate material emissivity from a camera image.

Grabs a frame (image file, webcam, or ZED 2i), classifies the material with
CLIP zero-shot, and prints the tabulated emissivity of the top matches.

Camera sources (--webcam/--zed/--zed-uvc) classify a centered box, not the
whole frame, and draw it as a green aiming rectangle in --show -- only what's
inside it is fed to CLIP. --image still classifies the whole picture. --roi
overrides the box (either way) with an explicit cx,cy,w,h region.

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
        "--camera-index", default="0", metavar="N",
        help="Device index for --webcam / --zed-uvc (default 0). On Linux, also "
        "accepts a device path like /dev/video1 -- useful when OpenCV's numeric "
        "index doesn't line up with the actual node, which happens with "
        "multi-node UVC cameras like the ZED 2i (check with 'v4l2-ctl "
        "--list-devices')",
    )
    p.add_argument(
        "--roi",
        help="Crop region cx,cy,w,h (center + size in pixels) before classification. "
        "For camera sources this defaults to a centered box (see "
        "default_center_roi()) so --show draws a real aiming box, not just a "
        "decorative marker -- only what's inside it is classified. --image "
        "still defaults to the whole picture",
    )
    p.add_argument("--top-k", type=int, default=3, help="Number of matches to show")
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


def draw_overlay(frame, best, confidence, roi=None):
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if roi:
        x0, y0, x1, y1 = roi_bounds(frame, roi)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
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
    classifier = MaterialClassifier(table, model_name=args.clip_model)

    camera_index = parse_camera_index(args.camera_index)

    if args.image:
        source = ImageSource(args.image)
    elif args.webcam:
        source = WebcamSource(camera_index)
    elif args.zed_uvc:
        source = ZedUvcSource(camera_index)
    else:
        source = ZedSource()

    is_camera = not args.image
    roi = args.roi  # camera sources fall back to a default center box, once the
                     # first frame's size is known -- see default_center_roi()

    with source:
        if args.live:
            import cv2

            classify_every = max(1, args.classify_every)
            print("Live mode -- press 'q' in the image window to stop.")
            best = confidence = None
            frame_count = 0
            try:
                while True:
                    frame = source.grab()
                    if roi is None and is_camera:
                        roi = default_center_roi(frame)
                    crop = crop_roi(frame, roi) if roi else frame
                    if best is None or frame_count % classify_every == 0:
                        best, confidence = classify_frame(crop, classifier, table, args.top_k)
                    frame_count += 1
                    cv2.imshow("Emissivity estimation", draw_overlay(frame, best, confidence, roi))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except KeyboardInterrupt:
                pass
            finally:
                cv2.destroyAllWindows()
            return 0

        frame = source.grab()
        if roi is None and is_camera:
            roi = default_center_roi(frame)
        crop = crop_roi(frame, roi) if roi else frame

        best, confidence = classify_frame(crop, classifier, table, args.top_k)

        if args.show:
            import cv2

            cv2.imshow("Emissivity estimation", draw_overlay(frame, best, confidence, roi))
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
