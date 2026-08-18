"""Polygonal surface reconstruction from a deduped .bvg, via PolyFit.

Hypothesis-and-selection reconstruction (Nan & Wonka, ICCV 2017): PolyFit
intersects the input planes into candidate faces, then solves a mixed-integer
program to pick the subset that best explains the point cloud, balancing
three weights (--weight-fitting / --weight-coverage / --weight-complexity).
Uses the bundled SCIP solver (Gurobi is faster but needs a separate license;
not wired up here -- see PolyFit's own README if you have one).

Feed this the OUTPUT OF dedupe_planes.py, not extract_planes.py's raw
*_planes.bvg directly -- PolyFit's own docs say it "assumes the model is
closed and all necessary planes are provided", and the hypothesis+selection
MIP scales with plane count: reconstructing straight from 56 raw (largely
near-duplicate) planes was left running past two minutes and 790+ CPU-seconds
with no result (verified by hand); deduped down to 30 real planes it still
takes a couple of minutes -- budget accordingly, and dedupe more aggressively
(looser --dedupe-normal-deg/--dedupe-offset-m, or a --min-plane-points floor)
if it's still too slow.

Usage:
    python reconstruct_mesh.py <bag>_planes_dedup.bvg [--output <model.obj>]
        [--weight-fitting 0.43] [--weight-coverage 0.27]
        [--weight-complexity 0.3]

Venv: C:\\venvs\\surfacereconstruction has polyfit installed already.
WARNING: `pip install polyfit` installs an unrelated PyPI stub (same trap as
`easy3d` -- see PlaneExtraction/README.md). Get the real one from
https://github.com/LiangliangNan/PolyFit/releases -- e.g. for Python 3.12 /
Windows: polyfit-1.6.0-cp312-cp312-win_amd64.whl.
"""
import argparse
import time
from pathlib import Path

import polyfit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bvg", type=Path, help="deduped .bvg from dedupe_planes.py")
    ap.add_argument("--output", type=Path, default=None,
                    help="default: <bvg_stem>.obj next to the input")
    ap.add_argument("--weight-fitting", type=float, default=0.43,
                    help="data-fitting weight (default: 0.43, PolyFit's own example default)")
    ap.add_argument("--weight-coverage", type=float, default=0.27,
                    help="model-coverage weight (default: 0.27)")
    ap.add_argument("--weight-complexity", type=float, default=0.3,
                    help="model-complexity weight (default: 0.3)")
    args = ap.parse_args()

    if not args.bvg.exists():
        raise SystemExit(f"{args.bvg} not found")
    output = args.output or args.bvg.with_suffix(".obj")

    polyfit.initialize()

    print(f"Loading {args.bvg}...")
    point_set = polyfit.read_point_set(str(args.bvg))
    if not point_set:
        raise SystemExit(f"Failed to load point set from {args.bvg}")

    print(f"Reconstructing (fitting={args.weight_fitting}, coverage={args.weight_coverage}, "
          f"complexity={args.weight_complexity}, solver=SCIP)... this can take a while.")
    t0 = time.time()
    mesh = polyfit.reconstruct(
        point_set, polyfit.SCIP,
        args.weight_fitting, args.weight_coverage, args.weight_complexity,
    )
    elapsed = time.time() - t0

    if not mesh:
        raise SystemExit(f"Reconstruction failed after {elapsed:.1f}s -- try deduping more "
                          "aggressively (looser --dedupe-normal-deg/--dedupe-offset-m or a "
                          "--min-plane-points floor) to hand PolyFit fewer, cleaner planes")

    print(f"Reconstructed {mesh.size_of_facets()} face(s) in {elapsed:.1f}s")
    ok = polyfit.save_mesh(str(output), mesh)
    if not ok:
        raise SystemExit(f"Reconstruction succeeded but saving to {output} failed")
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
