"""Extract a single .las point cloud from a TUM-FACADE .7z archive.

The dataset ships each building as a .7z containing a big .las (with per-point
semantic labels) plus a redundant .xyz. This pulls out only the .las into
data/ so the rest of the tool can read it.

Usage:
    py extract_sample.py                 # the default sample (DEBY_LOD2_4959459)
    py extract_sample.py --id DEBY_LOD2_4959322 --category annotatedLocalCRS
"""

import argparse
import subprocess
import sys
from pathlib import Path

SEVENZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
TUM_ROOT = Path(r"C:\Users\loren\Desktop\tum-facade\tum-facade\pointclouds")
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_ID = "DEBY_LOD2_4959459"
DEFAULT_CATEGORY = "annotatedLocalCRS"


def parse_args():
    p = argparse.ArgumentParser(description="Extract a TUM-FACADE .las from its .7z")
    p.add_argument("--id", default=DEFAULT_ID, help="Building id, e.g. DEBY_LOD2_4959459")
    p.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help="annotatedLocalCRS | annotatedGlobalCRS | nonannotatedLocalCRS | nonannotatedGlobalCRS",
    )
    p.add_argument("--archive", default=None, help="Explicit path to a .7z (overrides id/category)")
    p.add_argument("--root", default=str(TUM_ROOT), help="TUM-FACADE pointclouds root")
    return p.parse_args()


def main():
    args = parse_args()

    if not SEVENZIP.exists():
        sys.exit(f"7-Zip not found at {SEVENZIP}. Install it or edit SEVENZIP in this script.")

    archive = Path(args.archive) if args.archive else Path(args.root) / args.category / f"{args.id}.7z"
    if not archive.exists():
        sys.exit(f"Archive not found: {archive}")

    las_name = f"{args.id}.las"
    DATA_DIR.mkdir(exist_ok=True)
    out_las = DATA_DIR / las_name

    if out_las.exists():
        print(f"Already extracted: {out_las}")
        return

    print(f"Extracting {las_name} from {archive.name} ...")
    # 'e' = extract flat; name pattern limits extraction to just the .las.
    result = subprocess.run(
        [str(SEVENZIP), "e", str(archive), las_name, f"-o{DATA_DIR}", "-y"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out_las.exists():
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(f"Extraction failed (exit {result.returncode}).")

    size_mb = out_las.stat().st_size / 1_048_576
    print(f"Wrote {out_las} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
