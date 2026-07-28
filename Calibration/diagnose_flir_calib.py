"""Diagnose FLIR Vue Pro R intrinsic distortion: is k3 real curvature or overfit?

The production tool (flir_intrinsic_calib.py) fits the full k1,k2,p1,p2,k3 model
and reports one K per run. On this camera (336x256, 8 mm lens, ~17 um microbolometer
pitch) a strong radial distortion is physically expected, so a large k1/k2 is not a
red flag. The open question is the highest-order term: with only 15 corners per view
(5x3 board), k3 is the least statistically constrained coefficient and the first to
absorb residual noise instead of real optical curvature.

This script does NOT re-solve the production problem. It answers one diagnostic
question -- "is k3 stable?" -- two complementary ways:

    1. Model comparison. Refit the same views under several distortion models
       (full / reduced k1,k2 / minimal k1). If dropping k3 and the tangential
       terms barely moves the reprojection error and leaves fx/fy/cx/cy plausible,
       the extra freedom was buying almost nothing -- k3 was fitting noise.

    2. Stability under resampling. Recalibrate the FULL model on many image
       subsets (leave-one-out, then k-fold) and report the mean and standard
       deviation of fx,fy,cx,cy,k1,k2,k3 across folds. A std(k3) large relative to
       |mean(k3)| is the direct signature of an unstable, overfit coefficient;
       a physically real distortion reproduces from subset to subset.

It reuses flir_intrinsic_calib.py's detection pipeline verbatim (read_thermal,
_to_uint8, detect_all, checkerboard_object_points) so the corners fed here are
byte-for-byte the same the production tool would use. No calibration output is
written for downstream use -- run the final calibration with the chosen model in
flir_intrinsic_calib.py; this tool only informs that choice. An optional --output
dumps the full diagnostic (all models + fold statistics) as JSON for the thesis
report's traceability.

Usage:
    py diagnose_flir_calib.py --image-dir thermal/ --checkerboard-size 5 3 --square-size 0.045
    py diagnose_flir_calib.py --image-dir thermal/ --checkerboard-size 5 3 --square-size 0.045 \
        --clahe --output flir_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Reuse the production detection pipeline verbatim -- same corners in, so the
# diagnosis is about the model/fit, never about a divergent detector.
from flir_intrinsic_calib import (
    checkerboard_object_points,
    detect_all,
    find_images,
)

# Theoretical pinhole focal length for this camera, in pixels:
# f_px = f_mm / pixel_pitch_mm = 8.0 / 0.017 ~= 470. Used only as the initial
# guess for the reduced models (CALIB_USE_INTRINSIC_GUESS); the full model is
# solved free, exactly like the production tool.
_FOCAL_GUESS_PX = 470.0

# Same sub-pixel/solver expectations as the production tool.
_MIN_VIEWS = 3

# Distortion models to compare. Each is (name, description, cv2 flags, use_guess).
# The flags fix the excluded coefficients to their initial value (0 here), so the
# printed coefficient reads exactly 0 for anything the model doesn't estimate.
_MODELS: list[tuple[str, str, int, bool]] = [
    (
        "full_k1k2p1p2k3",
        "k1 k2 p1 p2 k3 (baseline, = production tool)",
        0,
        False,
    ),
    (
        "reduced_k1k2",
        "k1 k2 only (fix k3, zero tangential)",
        cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        True,
    ),
    (
        "minimal_k1",
        "k1 only (fix k2, fix k3, zero tangential)",
        cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2 | cv2.CALIB_ZERO_TANGENT_DIST,
        True,
    ),
]

# Coefficient labels in OpenCV order, for uniform reporting across models.
_DIST_LABELS = ("k1", "k2", "p1", "p2", "k3")


# --------------------------------------------------------------------------- #
# One calibration fit                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Fit:
    """One (K, dist) solve under one distortion model, with its fit scores."""
    name: str
    description: str
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray                 # (5,) k1 k2 p1 p2 k3, 0 where fixed
    mean_px: float
    rms_px: float
    worst_px: float
    worst_name: str
    n_views: int

    @property
    def principal_point_in_frame(self) -> bool:
        return 0.0 <= self.cx <= self._w and 0.0 <= self.cy <= self._h

    _w: int = 0
    _h: int = 0


def _initial_guess(image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """K0/dist0 for the reduced models: f from the 8 mm/17 um optics, centre at
    the image midpoint. Only meaningful with CALIB_USE_INTRINSIC_GUESS."""
    w, h = image_size
    K0 = np.array(
        [[_FOCAL_GUESS_PX, 0.0, w / 2.0],
         [0.0, _FOCAL_GUESS_PX, h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K0, np.zeros(5, dtype=np.float64)


def _per_image_errors(
    used: list[str],
    obj_points: list[np.ndarray],
    corners_list: list[np.ndarray],
    rvecs,
    tvecs,
    K: np.ndarray,
    dist: np.ndarray,
) -> dict[str, float]:
    """RMS reprojection residual (px) per image -- same definition as the
    production tool's calibrate(), so numbers are directly comparable."""
    per: dict[str, float] = {}
    for name, objp, imgp, rvec, tvec in zip(used, obj_points, corners_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        d = imgp.reshape(-1, 2).astype(np.float64) - projected.reshape(-1, 2)
        per[name] = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
    return per


def _solve(
    obj_points: list[np.ndarray],
    corners_list: list[np.ndarray],
    image_size: tuple[int, int],
    flags: int,
    use_guess: bool,
) -> tuple[np.ndarray, np.ndarray, float, list, list]:
    """Thin wrapper over cv2.calibrateCamera returning (K, dist(5,), rms, rvecs, tvecs)."""
    if use_guess:
        K0, dist0 = _initial_guess(image_size)
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, corners_list, image_size, K0.copy(), dist0.copy(),
            flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS,
        )
    else:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, corners_list, image_size, None, None, flags=flags,
        )
    return K, dist.ravel()[:5], float(rms), rvecs, tvecs


def fit_model(
    name: str,
    description: str,
    flags: int,
    use_guess: bool,
    used: list[str],
    corners_list: list[np.ndarray],
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size: float,
) -> Fit:
    """Calibrate all views under one distortion model and score the fit."""
    obj = checkerboard_object_points(pattern_size, square_size)
    obj_points = [obj] * len(corners_list)
    K, dist, rms, rvecs, tvecs = _solve(
        obj_points, corners_list, image_size, flags, use_guess)
    per = _per_image_errors(used, obj_points, corners_list, rvecs, tvecs, K, dist)
    worst_name = max(per, key=per.get)
    w, h = image_size
    return Fit(
        name=name,
        description=description,
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        dist=dist,
        mean_px=float(np.mean(list(per.values()))),
        rms_px=rms,
        worst_px=per[worst_name],
        worst_name=worst_name,
        n_views=len(corners_list),
        _w=w, _h=h,
    )


# --------------------------------------------------------------------------- #
# Stability under resampling (full model only -- that's where k3 lives)        #
# --------------------------------------------------------------------------- #
@dataclass
class FoldStats:
    """Mean/std of each intrinsic across a set of subset recalibrations."""
    scheme: str                      # "leave-one-out" | "5-fold"
    n_fits: int
    mean: dict[str, float]
    std: dict[str, float]

    def ratio(self, key: str) -> float:
        """std/|mean| for one parameter; the instability signal when large."""
        m = abs(self.mean.get(key, 0.0))
        return self.std[key] / m if m > 1e-12 else float("inf")


def _params_of(K: np.ndarray, dist: np.ndarray) -> dict[str, float]:
    d = dist.ravel()
    return {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "k1": float(d[0]), "k2": float(d[1]), "k3": float(d[4]),
    }


def _fit_subset(
    indices: list[int],
    corners_list: list[np.ndarray],
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size: float,
) -> dict[str, float] | None:
    """Full-model calibration on a subset of views; None if too few for a solve."""
    if len(indices) < _MIN_VIEWS:
        return None
    obj = checkerboard_object_points(pattern_size, square_size)
    subset = [corners_list[i] for i in indices]
    obj_points = [obj] * len(subset)
    K, dist, _rms, _r, _t = _solve(obj_points, subset, image_size, 0, False)
    return _params_of(K, dist)


def _aggregate(scheme: str, rows: list[dict[str, float]]) -> FoldStats:
    keys = ("fx", "fy", "cx", "cy", "k1", "k2", "k3")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}
    return FoldStats(scheme=scheme, n_fits=len(rows), mean=mean, std=std)


def stability(
    corners_list: list[np.ndarray],
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size: float,
    kfolds: int,
) -> list[FoldStats]:
    """Resample the view set and refit the full model, two schemes:

    - leave-one-out: N fits, each dropping exactly one view. Fine-grained; the
      per-fit change is small, so a k3 that still wanders here is genuinely soft.
    - k-fold: partition views into k contiguous groups, fit on the other k-1
      (~(k-1)/k of the data). Removes a larger chunk, stressing the fit harder.

    Contiguous (not shuffled) partitions keep the run deterministic and
    reproducible for the thesis report.
    """
    n = len(corners_list)
    out: list[FoldStats] = []

    # Leave-one-out
    loo_rows: list[dict[str, float]] = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        p = _fit_subset(idx, corners_list, image_size, pattern_size, square_size)
        if p is not None:
            loo_rows.append(p)
    if loo_rows:
        out.append(_aggregate("leave-one-out", loo_rows))

    # k-fold (train on the complement of each fold)
    k = max(2, min(kfolds, n))
    bounds = np.linspace(0, n, k + 1, dtype=int)
    kf_rows: list[dict[str, float]] = []
    for f in range(k):
        test = set(range(bounds[f], bounds[f + 1]))
        train = [j for j in range(n) if j not in test]
        p = _fit_subset(train, corners_list, image_size, pattern_size, square_size)
        if p is not None:
            kf_rows.append(p)
    if kf_rows:
        out.append(_aggregate(f"{k}-fold", kf_rows))

    return out


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _fmt_dist(dist: np.ndarray) -> str:
    return "  ".join(f"{lab}={v:+.6f}" for lab, v in zip(_DIST_LABELS, dist.ravel()))


def print_models(fits: list[Fit], image_size: tuple[int, int]) -> None:
    w, h = image_size
    print("\n=== Distortion model comparison ===")
    for fit in fits:
        pp = "OK" if fit.principal_point_in_frame else "OUT OF FRAME"
        print(f"\n[{fit.name}] {fit.description}")
        print(f"  fx={fit.fx:.3f}  fy={fit.fy:.3f}  cx={fit.cx:.3f}  cy={fit.cy:.3f}"
              f"   principal point: {pp} (frame 0..{w}, 0..{h})")
        print(f"  dist: {_fmt_dist(fit.dist)}")
        print(f"  reproj: mean {fit.mean_px:.4f} px  rms {fit.rms_px:.4f} px  "
              f"worst {fit.worst_px:.4f} px ({fit.worst_name})")


def print_stability(folds: list[FoldStats]) -> None:
    if not folds:
        return
    print("\n=== Stability under resampling (full model) ===")
    for fs in folds:
        print(f"\n[{fs.scheme}] {fs.n_fits} fits")
        for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3"):
            m, s = fs.mean[k], fs.std[k]
            tag = f"   std/|mean| = {fs.ratio(k):.3f}" if k in ("k1", "k2", "k3") else ""
            print(f"  {k:>3}: mean {m:+.4f}  std {s:.4f}{tag}")


def _recommend(fits: list[Fit], folds: list[FoldStats]) -> tuple[str, list[str]]:
    """Heuristic pick + explicit reasoning for the thesis. Not a hard rule:
    prefers the simplest model whose reproj error is within ~0.02 px of the best,
    whose principal point sits in-frame, and -- if stability ran -- whose k3 is
    reproducible (std/|mean| < 0.5)."""
    reasons: list[str] = []
    best_err = min(f.mean_px for f in fits)

    # k3 stability verdict from leave-one-out if available.
    k3_unstable = None
    loo = next((f for f in folds if f.scheme == "leave-one-out"), None)
    if loo is not None:
        r = loo.ratio("k3")
        k3_unstable = r > 0.5
        reasons.append(
            f"k3 across leave-one-out: mean {loo.mean['k3']:+.3f}, std {loo.std['k3']:.3f} "
            f"(std/|mean| = {r:.2f}) -> {'UNSTABLE, absorbing noise' if k3_unstable else 'stable, reproducible'}"
        )

    # Simplest acceptable model (models are ordered full -> reduced -> minimal;
    # walk from simplest up and take the first that holds error and keeps pp in frame).
    ordered_simple_first = list(reversed(fits))
    pick = fits[0]  # default: full
    for f in ordered_simple_first:
        if f.principal_point_in_frame and (f.mean_px - best_err) <= 0.02:
            pick = f
            break

    if k3_unstable and pick.name == "full_k1k2p1p2k3":
        # If k3 is unstable but the full model still won on error, flag the tension.
        reasons.append(
            "full model has lowest error but its k3 is unstable -- the reduced_k1k2 "
            "model is the safer choice if its error is comparable"
        )
    reasons.append(
        f"reproj error range across models: {best_err:.4f}..{max(f.mean_px for f in fits):.4f} px "
        "(if the spread is tiny, the extra coefficients are not earning their place)"
    )
    return pick.name, reasons


def print_summary(
    fits: list[Fit],
    folds: list[FoldStats],
    image_dir: Path,
    n_used: int,
    n_found: int,
) -> None:
    pick, reasons = _recommend(fits, folds)
    print("\n=== Summary ===")
    print(f"  image-dir: {image_dir.resolve()}")
    print(f"  frames: {n_used} used / {n_found} found")
    for r in reasons:
        print(f"  - {r}")
    print(f"  recommended distortion model: {pick}")
    print("  (run the final calibration with this model in flir_intrinsic_calib.py)")


# --------------------------------------------------------------------------- #
# JSON (optional, for thesis traceability)                                     #
# --------------------------------------------------------------------------- #
def save_json(
    path: Path,
    fits: list[Fit],
    folds: list[FoldStats],
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size: float,
    image_dir: Path,
    used: list[str],
    n_found: int,
    provenance: dict,
) -> None:
    pick, reasons = _recommend(fits, folds)
    payload = {
        "camera": "FLIR Vue Pro R",
        "purpose": "distortion-model / k3-stability diagnostic (not a calibration output)",
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "checkerboard": {"inner_corners": list(pattern_size), "square_size_m": square_size},
        "models": [
            {
                "name": f.name,
                "description": f.description,
                "fx": f.fx, "fy": f.fy, "cx": f.cx, "cy": f.cy,
                "dist_coeffs": {lab: float(v) for lab, v in zip(_DIST_LABELS, f.dist.ravel())},
                "principal_point_in_frame": f.principal_point_in_frame,
                "reprojection_error": {
                    "mean_px": f.mean_px, "rms_px": f.rms_px,
                    "worst_px": f.worst_px, "worst_image": f.worst_name,
                },
            }
            for f in fits
        ],
        "stability": [
            {
                "scheme": fs.scheme,
                "n_fits": fs.n_fits,
                "mean": fs.mean,
                "std": fs.std,
                "std_over_abs_mean": {k: fs.ratio(k) for k in ("k1", "k2", "k3")},
            }
            for fs in folds
        ],
        "recommendation": {"model": pick, "reasons": reasons},
        "images": {"n_used": len(used), "n_found": n_found, "used": used},
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose FLIR Vue Pro R distortion: compare distortion models "
                    "and test k3 stability under resampling (does NOT produce a "
                    "calibration -- use flir_intrinsic_calib.py for that)."
    )
    p.add_argument(
        "--image-dir", required=True, metavar="DIR",
        help="Folder of thermal checkerboard frames (RJPG, or plain PNG/TIFF)",
    )
    p.add_argument(
        "--checkerboard-size", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"),
        help="INNER corner count, not squares: a board of 6x4 squares is 5 3",
    )
    p.add_argument(
        "--square-size", type=float, required=True, metavar="M",
        help="Checkerboard square edge in metres (e.g. 0.045 for 45 mm)",
    )
    p.add_argument(
        "--models", nargs="+", default=None, metavar="NAME",
        choices=[m[0] for m in _MODELS],
        help="Subset of distortion models to compare "
             f"(default: all -- {', '.join(m[0] for m in _MODELS)})",
    )
    p.add_argument(
        "--kfolds", type=int, default=5, metavar="K",
        help="k for the k-fold stability test (default 5); leave-one-out always runs",
    )
    p.add_argument(
        "--no-stability", action="store_true",
        help="Skip the resampling stability test, compare models only",
    )
    p.add_argument(
        "--clahe", action="store_true",
        help="Apply CLAHE local contrast equalisation before detection (same as the "
             "production tool) -- helps when a hot/cold background crushed board contrast",
    )
    p.add_argument(
        "--output", default=None, metavar="PATH",
        help="Also write the full diagnostic (all models + fold stats) as JSON here",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="List every image as it is accepted, not just the skipped ones",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pattern = (args.checkerboard_size[0], args.checkerboard_size[1])
    if min(pattern) < 2:
        raise SystemExit(f"--checkerboard-size must be >= 2 in both axes (got {pattern!r})")
    if args.square_size <= 0:
        raise SystemExit(f"--square-size must be positive (got {args.square_size!r})")

    selected = args.models or [m[0] for m in _MODELS]
    models = [m for m in _MODELS if m[0] in selected]

    image_dir = Path(args.image_dir)
    try:
        paths = find_images(image_dir)
        print(f"Found {len(paths)} frames in {image_dir}, detecting "
              f"{pattern[0]}x{pattern[1]} inner corners...")
        used, corners_list, image_size, _skipped = detect_all(
            paths, pattern, args.clahe, args.verbose)
        if len(corners_list) < _MIN_VIEWS:
            raise RuntimeError(
                f"only {len(corners_list)} usable view(s), need >= {_MIN_VIEWS}")
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from None

    fits = [
        fit_model(name, desc, flags, guess,
                  used, corners_list, image_size, pattern, args.square_size)
        for (name, desc, flags, guess) in models
    ]

    folds: list[FoldStats] = []
    if not args.no_stability:
        folds = stability(corners_list, image_size, pattern, args.square_size, args.kfolds)

    print_models(fits, image_size)
    print_stability(folds)
    print_summary(fits, folds, image_dir, len(used), len(paths))

    if args.output:
        provenance = {
            "tool": Path(__file__).name,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "opencv_version": cv2.__version__,
            "python_version": sys.version.split()[0],
            "image_dir": str(image_dir.resolve()),
            "command": " ".join(sys.argv),
        }
        out = Path(args.output)
        save_json(out, fits, folds, image_size, pattern, args.square_size,
                  image_dir, used, len(paths), provenance)
        print(f"\nSaved diagnostic JSON -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
