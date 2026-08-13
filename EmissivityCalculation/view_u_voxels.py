"""Interactive PyVista viewer for voxel_u_value.py's output CSV, with
flag_reason-based filtering.

Default: hides solar_suspected voxels (material-driven, formula can't model
them anyway) and shows everything else -- plausible (in-range) + implausible
(out-of-range, no solar explanation) together, colored by U-value, so you can
see whether the unexplained ones cluster spatially or scatter randomly.

Usage:
    python view_u_voxels.py --in thermal_voxels_u.csv
    python view_u_voxels.py --in thermal_voxels_u.csv --exclude solar_suspected,implausible
    python view_u_voxels.py --in thermal_voxels_u.csv --color-by flag_reason

Venv: C:\\venvs\\planefit (has pyvista; sensorfusion venv does not).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pyvista as pv


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="thermal_voxels_u.csv from voxel_u_value.py")
    ap.add_argument("--exclude", default="solar_suspected",
                    help="comma-separated flag_reason values to hide (default: solar_suspected). "
                         "Empty string shows everything.")
    ap.add_argument("--color-by", default="u_value_w_m2k",
                    choices=["u_value_w_m2k", "t_mean_c", "material", "flag_reason"],
                    help="default: u_value_w_m2k")
    ap.add_argument("--point-size", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None,
                    help="save a screenshot PNG instead of opening an interactive window")
    args = ap.parse_args()

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    kept = [r for r in rows if r.get("flag_reason", "") not in exclude]
    if not kept:
        raise SystemExit(f"--exclude {sorted(exclude)} removed all {len(rows)} voxel(s)")
    print(f"{len(kept)}/{len(rows)} voxel(s) shown (excluded flag_reason in {sorted(exclude)})")

    pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in kept])
    cloud = pv.PolyData(pts)

    p = pv.Plotter(off_screen=bool(args.out))

    if args.color_by in ("u_value_w_m2k", "t_mean_c"):
        scalars = np.array([float(r[args.color_by]) for r in kept])
        cloud[args.color_by] = scalars
        p.add_mesh(cloud, scalars=args.color_by, cmap="coolwarm",
                   point_size=args.point_size, render_points_as_spheres=True)
    else:
        cats = [r.get(args.color_by, "") or "(none)" for r in kept]
        uniq = sorted(set(cats))
        idx = np.array([uniq.index(c) for c in cats])
        cloud["cat"] = idx
        p.add_mesh(cloud, scalars="cat", cmap="tab20", point_size=args.point_size,
                   render_points_as_spheres=True,
                   annotations={i: name for i, name in enumerate(uniq)})

    p.add_axes()
    p.show_grid()
    if args.out:
        p.show(screenshot=str(args.out))
        print(f"wrote {args.out}")
    else:
        p.show(title=f"{args.inp.name} -- colored by {args.color_by}")


if __name__ == "__main__":
    main()
