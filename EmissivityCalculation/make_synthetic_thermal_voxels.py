"""SYNTHETIC pipeline-validation data -- NOT measured, do not treat as real.

Takes a real thermal_voxels.csv (real geometry, real material/solar_absorptance
from an actual session) and replaces t_mean_c with a value computed BACKWARDS
from a chosen ground-truth U-value and winter-like Tint/Text, so the round
trip through the *unmodified* voxel_u_value.py can be checked against a known
answer -- independent of whether any real session's data is good enough.

Only the no-solar formula is inverted (this is meant to emulate a
good-condition, big-DeltaT session with no solar anomaly, not to also fake a
solar term):

    Tsurf = Tint - U_true * (Tint - Text) / hsi

then Gaussian sensor noise is added to Tsurf, same idea as
RadiometricCalibration/make_demo_data.py's synthetic apparent-temperature map.

This validates the CODE (does voxel_u_value.py recover U_true from Tsurf it
didn't see the formula for), not the building's real thermal performance --
label it as synthetic everywhere it's used (filenames, plots, thesis text).

Usage:
    python make_synthetic_thermal_voxels.py --in thermal_voxels_wall_subset.csv \\
        --u-true 1.2 --tint 20 --text 2 --hsi 7.7 --noise-std 0.1 \\
        --out demo_data/thermal_voxels_wall_synthetic.csv

Then verify with the real script, same --tint/--text/--hsi:
    python voxel_u_value.py --in demo_data/thermal_voxels_wall_synthetic.csv \\
        --tint 20 --text 2 --hsi 7.7
    # -> printed U mean should land close to --u-true

No LiDAR/rosbags -- just numpy. Any venv with numpy works.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="a real thermal_voxels.csv (or a surface_type subset of one) -- "
                         "only its x,y,z,material,solar_absorptance columns are reused; "
                         "t_mean_c is overwritten with the synthetic value")
    ap.add_argument("--u-true", type=float, required=True, metavar="W_M2K",
                    help="ground-truth U-value to synthesize Tsurf from -- what "
                         "voxel_u_value.py should recover from the output")
    ap.add_argument("--tint", type=float, required=True, help="indoor air temperature, deg C")
    ap.add_argument("--text", type=float, required=True, help="outdoor air temperature, deg C")
    ap.add_argument("--hsi", type=float, required=True,
                    help="internal surface heat transfer coefficient, W/m2K -- use the same "
                         "value you'll pass to voxel_u_value.py when verifying")
    ap.add_argument("--noise-std", type=float, default=0.1, metavar="DEG_C",
                    help="Gaussian sensor noise added to synthetic Tsurf (default 0.1 deg C, "
                         "matching RadiometricCalibration/make_demo_data.py's convention)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed, for reproducibility")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: demo_data/<in stem>_synthetic.csv")
    args = ap.parse_args()

    if abs(args.tint - args.text) < 1e-9:
        raise SystemExit("--tint and --text are equal -- Tsurf inversion is undefined")

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")
    for col in ("x", "y", "z", "material", "solar_absorptance"):
        if col not in rows[0]:
            raise SystemExit(f"{args.inp} has no '{col}' column -- need a real "
                             f"thermal_voxels.csv-shaped file for the geometry/material")

    n = len(rows)
    tsurf_true = args.tint - args.u_true * (args.tint - args.text) / args.hsi
    rng = np.random.default_rng(args.seed)
    noise = rng.normal(0.0, args.noise_std, n)
    tsurf_noisy = tsurf_true + noise

    for r, t in zip(rows, tsurf_noisy):
        r["t_mean_c"] = round(float(t), 3)
        r["t_std_c"] = args.noise_std
        r["n_obs"] = r.get("n_obs") or 1
        # keep material/solar_absorptance as-is (real); everything else in the
        # row that isn't part of the geometry/material contract is dropped --
        # this is a fresh synthetic measurement, not a patched real one

    out = args.out or args.inp.with_name("demo_data").joinpath(args.inp.stem + "_synthetic.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["x", "y", "z", "t_mean_c", "t_std_c", "n_obs", "material", "solar_absorptance"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    print("*** SYNTHETIC DATA -- pipeline validation only, not a real measurement ***")
    print(f"{n} voxel(s), geometry/material from {args.inp}")
    print(f"Tsurf synthesized from U_true={args.u_true} W/m2K, Tint={args.tint}C, "
          f"Text={args.text}C, hsi={args.hsi} W/m2K -> noiseless Tsurf={tsurf_true:.3f}C, "
          f"+ N(0,{args.noise_std}) sensor noise")
    print(f"wrote {out}")
    print()
    print("verify with the real (unmodified) script:")
    print(f"  python voxel_u_value.py --in {out} --tint {args.tint} --text {args.text} "
          f"--hsi {args.hsi}")
    print(f"  -> U mean should land close to {args.u_true}")


if __name__ == "__main__":
    main()
