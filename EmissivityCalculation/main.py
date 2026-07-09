"""Estimate material emissivity from a camera image.

Grabs a frame (image file, webcam, or ZED 2i), classifies the material with
CLIP zero-shot, and prints the tabulated emissivity of the top matches.

Usage:
    py main.py --image test_images/brick.jpg
    py main.py --webcam --show
    py main.py --zed --show
    py main.py --image photo.jpg --roi 320,240,200,200
"""

import argparse
import sys

import numpy as np

from emissivity import EmissivityTable, MaterialClassifier
from emissivity.sources import ImageSource, WebcamSource, ZedSource


def parse_args():
    p = argparse.ArgumentParser(description="Material emissivity estimation")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to an image file")
    src.add_argument("--webcam", action="store_true", help="Use the default webcam")
    src.add_argument("--zed", action="store_true", help="Use the ZED 2i camera")
    p.add_argument(
        "--roi",
        help="Crop region cx,cy,w,h (center + size in pixels) before classification",
    )
    p.add_argument("--top-k", type=int, default=3, help="Number of matches to show")
    p.add_argument("--show", action="store_true", help="Display frame with overlay")
    p.add_argument("--table", default=None, help="Path to a custom emissivity CSV")
    return p.parse_args()


def crop_roi(frame: np.ndarray, roi: str) -> np.ndarray:
    cx, cy, w, h = (int(v) for v in roi.split(","))
    x0 = max(cx - w // 2, 0)
    y0 = max(cy - h // 2, 0)
    return frame[y0 : y0 + h, x0 : x0 + w]


def main():
    args = parse_args()

    table = EmissivityTable(args.table) if args.table else EmissivityTable()
    print("Loading CLIP model (first run downloads ~600 MB)...")
    classifier = MaterialClassifier(table)

    if args.image:
        source = ImageSource(args.image)
    elif args.webcam:
        source = WebcamSource()
    else:
        source = ZedSource()

    with source:
        frame = source.grab()
        if args.roi:
            frame = crop_roi(frame, args.roi)

        results = classifier.classify(frame, top_k=args.top_k)

        print(f"\n{'Material':<22}{'Confidence':<12}{'Emissivity':<12}Range")
        print("-" * 60)
        for material, conf in results:
            rec = table.lookup(material)
            print(
                f"{material:<22}{conf:<12.1%}{rec.emissivity:<12.2f}{rec.emissivity_range}"
            )

        best = table.lookup(results[0][0])
        print(
            f"\nBest estimate: {best.material} -> emissivity = {best.emissivity}"
            f" ({best.notes})"
        )

        if args.show:
            import cv2

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            label = f"{best.material}: e={best.emissivity} ({results[0][1]:.0%})"
            cv2.putText(
                bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.imshow("Emissivity estimation", bgr)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
