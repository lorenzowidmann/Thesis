"""Merge several fit_planes.py outputs (e.g. one per --roi/corridor) into a
single planes.json for a combined view in show_planes.py.

Usage:
    python merge_planes.py box_A.json box_B.json box_D.json --out combined.json
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path, nargs="+", help="planes.json files to merge")
    ap.add_argument("--out", type=Path, default=Path("planes_merged.json"))
    args = ap.parse_args()

    merged = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        for p in data["planes"]:
            p["id"] = len(merged)
            merged.append(p)
        print(f"{path}: {len(data['planes'])} plane(s)")

    args.out.write_text(json.dumps(
        {"bag": [str(p) for p in args.inputs], "topic": None, "planes": merged}, indent=2))
    print(f"wrote {len(merged)} plane(s) total to {args.out}")


if __name__ == "__main__":
    main()
