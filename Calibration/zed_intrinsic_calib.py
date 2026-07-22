"""Intrinsic calibration of the ZED 2i's right eye from checkerboard images.

Only the right eye is calibrated: it is the lens the rest of the pipeline
consumes, so no stereo baseline / rectification is solved here -- this is a
plain monocular cv2.calibrateCamera problem over a folder of right-eye frames.
OpenCV only, no ZED SDK, no imports from the other modules.

Pipeline per the standard OpenCV flow:
    findChessboardCorners -> cornerSubPix -> calibrateCamera -> reprojection error

Images where the full board isn't found are skipped and reported (filename +
reason), never fatal -- a calibration set almost always has a few frames where
the board is cut off, blurred or too oblique.

Two outputs:
- JSON (--output): our canonical record. Intrinsics plus provenance -- every
  input filename, per-image error, timestamp, OpenCV version, the exact CLI
  arguments -- so a result can be traced back to the images that produced it.
- LVT2Calib YAML (--lvt2calib-export): compatibility export only, for
  https://github.com/Clothooo/lvt2calib. See save_lvt2calib_yaml for why this
  is YAML and not the flat "K:/D:" text some of that repo's files use.

Usage:
    py zed_intrinsic_calib.py --image-dir data/right_eye --checkerboard-size 9 6 --square-size 0.025
    py zed_intrinsic_calib.py --image-dir imgs --checkerboard-size 9 6 --square-size 0.025 \
        --output right_intrinsics.json --lvt2calib-export intrinsic.yaml
    py zed_intrinsic_calib.py --image-dir imgs --checkerboard-size 9 6 --square-size 0.025 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Sub-pixel refinement termination: OpenCV's standard recommendation, and the
# ~0.1 px refinement target follows from the 1e-3 epsilon / 30 iteration cap.
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# calibrateCamera is underdetermined below this. It is a floor, not a target:
# 12-20 views spanning large tilts about both board axes is what actually
# pins down f and the distortion terms.
_MIN_VIEWS = 3
_RECOMMENDED_VIEWS = 10


# --------------------------------------------------------------------------- #
# Results                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class SkippedImage:
    """One input image that never reached calibrateCamera, and why."""
    filename: str
    reason: str


@dataclass
class CalibrationResult:
    """Solved intrinsics plus everything needed to reproduce/trace them."""
    camera_matrix: np.ndarray          # (3, 3) K
    dist_coeffs: np.ndarray            # (5,) k1 k2 p1 p2 k3
    image_size: tuple[int, int]        # (width, height)
    mean_reprojection_error: float     # px, over all used views
    rms_reprojection_error: float      # px, as returned by calibrateCamera
    per_image_error: dict[str, float]  # filename -> px
    images_used: list[str] = field(default_factory=list)
    images_skipped: list[SkippedImage] = field(default_factory=list)
    checkerboard_size: tuple[int, int] = (0, 0)  # inner corners (cols, rows)
    square_size: float = 0.0                     # metres

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


# --------------------------------------------------------------------------- #
# Detection                                                                    #
# --------------------------------------------------------------------------- #
def checkerboard_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    """3-D checkerboard corner coordinates in the board's own frame.

    (cols*rows, 3) float32 with z = 0, row-major in the same order
    findChessboardCorners reports the 2-D corners. `square_size` sets the world
    scale: it fixes the board poses (rvecs/tvecs) in metres but leaves K and the
    distortion coefficients unchanged, since those are scale-invariant.
    """
    cols, rows = pattern_size
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size)
    return obj


def _subpix_half_window(corners: np.ndarray, pattern_size: tuple[int, int]) -> int:
    """Sub-pixel refinement half-window, in pixels, from the corner spacing.

    cornerSubPix is normally called with a fixed (11, 11) window, which assumes
    the board is large in frame. The ZED's right eye is half of an already
    modest UVC frame (672x376 for a 1344x376 stereo pair), so a square is often
    only ~15 px across and an 11 px half-window spans past the neighbouring
    saddle points, dragging corners toward them.

    Measured on synthetic boards rendered through a known K (fx = fy = 520):
    forcing (11, 11) gave rms 2.87 px and fx 373; sizing the window to the
    detected spacing (this function, picking 4-7 px) gave rms 0.04 px and
    fx 519.8. Hence: keep the window strictly inside one square.
    """
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    spacing = min(
        float(np.linalg.norm(np.diff(grid, axis=1), axis=2).min()),  # along rows
        float(np.linalg.norm(np.diff(grid, axis=0), axis=2).min()),  # along columns
    )
    return int(np.clip(spacing / 2 - 1, 2, 11))


def detect_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Sub-pixel checkerboard inner corners of one grayscale image.

    Returns (cols*rows, 1, 2) float32 in OpenCV's corner order, or None if the
    full board isn't found. Partial detections are rejected rather than padded:
    calibrateCamera needs every corner of every view it is given.
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None
    half = _subpix_half_window(corners, pattern_size)
    return cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), _SUBPIX_CRITERIA)


def find_images(image_dir: Path) -> list[Path]:
    """Every image file in `image_dir`, sorted by name."""
    if not image_dir.is_dir():
        raise NotADirectoryError(f"--image-dir is not a directory: {image_dir}")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(
            f"no images in {image_dir} (looked for {', '.join(_IMAGE_EXTS)})"
        )
    return paths


def detect_all(
    paths: list[Path], pattern_size: tuple[int, int], verbose: bool = False
) -> tuple[list[str], list[np.ndarray], tuple[int, int], list[SkippedImage]]:
    """Detect the board in every image, collecting failures instead of raising.

    An image is skipped (with a reason) when it can't be decoded, when the full
    board isn't found, or when its resolution differs from the first accepted
    image -- calibrateCamera solves one K for one image size, so a stray frame
    at another resolution would silently corrupt the fit.

    Returns (used_filenames, corner_arrays, image_size, skipped).
    """
    used: list[str] = []
    corners_list: list[np.ndarray] = []
    skipped: list[SkippedImage] = []
    image_size: tuple[int, int] | None = None

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append(SkippedImage(path.name, "unreadable (cv2.imread returned None)"))
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif (w, h) != image_size:
            skipped.append(SkippedImage(
                path.name,
                f"image size {w}x{h} differs from {image_size[0]}x{image_size[1]}",
            ))
            continue

        corners = detect_corners(gray, pattern_size)
        if corners is None:
            skipped.append(SkippedImage(
                path.name,
                f"checkerboard {pattern_size[0]}x{pattern_size[1]} inner corners not found",
            ))
            continue

        used.append(path.name)
        corners_list.append(corners)
        if verbose:
            print(f"  [ok]   {path.name}")

    for s in skipped:
        print(f"  [skip] {s.filename}: {s.reason}", file=sys.stderr)

    if image_size is None:
        raise RuntimeError("no image could be decoded")
    return used, corners_list, image_size, skipped


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #
def calibrate(
    used: list[str],
    corners_list: list[np.ndarray],
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size: float,
    skipped: list[SkippedImage] | None = None,
) -> CalibrationResult:
    """Solve K and the distortion coefficients, and score the fit per image.

    Per-image error is the RMS reprojection residual in pixels, the same
    definition as calibrateCamera's overall return value, so the two are
    directly comparable: the mean says whether the fit is good, the worst image
    says which frame to drop and recapture. Computed with numpy because
    cv2.norm rejects the (N,1,2) detected/projected pair as a channel-count
    mismatch on OpenCV 5.
    """
    if len(corners_list) < _MIN_VIEWS:
        raise RuntimeError(
            f"only {len(corners_list)} usable view(s), need >= {_MIN_VIEWS} "
            "(check --checkerboard-size counts INNER corners, and that the "
            "board is fully visible and in focus)"
        )

    obj = checkerboard_object_points(pattern_size, square_size)
    obj_points = [obj] * len(corners_list)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, corners_list, image_size, None, None
    )

    per_image: dict[str, float] = {}
    for name, objp, imgp, rvec, tvec in zip(used, obj_points, corners_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        d = imgp.reshape(-1, 2).astype(np.float64) - projected.reshape(-1, 2)
        per_image[name] = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))

    return CalibrationResult(
        camera_matrix=K,
        dist_coeffs=dist.ravel(),
        image_size=image_size,
        mean_reprojection_error=float(np.mean(list(per_image.values()))),
        rms_reprojection_error=float(rms),
        per_image_error=per_image,
        images_used=list(used),
        images_skipped=list(skipped or []),
        checkerboard_size=pattern_size,
        square_size=float(square_size),
    )


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #
def save_json(result: CalibrationResult, path: Path, provenance: dict) -> None:
    """Write the canonical, provenance-tracked record.

    This is our own format and the one to keep: it carries not just K/D but
    which images produced them, which were rejected and why, and when/with what
    the run happened. The LVT2Calib export below is derived from it and is
    lossy by comparison.
    """
    payload = {
        "camera": "ZED 2i",
        "eye": "right",
        "camera_matrix": result.camera_matrix.tolist(),
        "dist_coeffs": result.dist_coeffs.tolist(),
        "dist_coeff_order": ["k1", "k2", "p1", "p2", "k3"],
        "fx": result.fx, "fy": result.fy, "cx": result.cx, "cy": result.cy,
        "image_size": {"width": result.image_size[0], "height": result.image_size[1]},
        "reprojection_error": {
            "mean_px": result.mean_reprojection_error,
            "rms_px": result.rms_reprojection_error,
            "worst_px": max(result.per_image_error.values()),
            "per_image_px": result.per_image_error,
        },
        "checkerboard": {
            "inner_corners": list(result.checkerboard_size),
            "square_size_m": result.square_size,
        },
        "images": {
            "n_used": len(result.images_used),
            "n_skipped": len(result.images_skipped),
            "used": result.images_used,
            "skipped": [{"filename": s.filename, "reason": s.reason}
                        for s in result.images_skipped],
        },
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, indent=2))


def save_lvt2calib_yaml(result: CalibrationResult, path: Path) -> None:
    """Compatibility export for LVT2Calib (github.com/Clothooo/lvt2calib).

    Format confirmed from that repo (branch ros_noetic) rather than guessed --
    its README only shows the layout as an image. src/camera/cam_pattern.cpp
    loads the file with:

        cv::FileStorage fs_reader(camera_info_dir_, cv::FileStorage::READ);
        fs_reader["CameraMat"] >> cam_intrinsic;
        fs_reader["DistCoeff"] >> cam_distcoeff;
        fs_reader["ImageSize"] >> img_size;

    so it is an OpenCV FileStorage document, matching their shipped
    data/camera_info/intrinsic.yaml -- NOT the flat "K: ... / D: ..." text in
    their data/camera_info/rgb_camParam.txt, whose parser (ReadMatFromTxt) sits
    commented out directly above the lines quoted here.

    ImageSize is required, not decorative: it is passed straight into
    initUndistortRectifyMap. It must stay a plain 2-element sequence so the C++
    side can read it as a cv::Size; CameraMat/DistCoeff carry the
    !!opencv-matrix tag. Written as text for exactly that reason -- Python's
    cv2.FileStorage would emit ImageSize as an opencv-matrix too.

    Drop the result in `(lvt2calib)/data/camera_info/` and pass its filename as
    the tool's cam_info_filename argument.
    """
    K = np.asarray(result.camera_matrix, dtype=np.float64)
    d = np.asarray(result.dist_coeffs, dtype=np.float64).ravel()
    w, h = result.image_size

    def _row(values) -> str:
        return ", ".join(f"{v:.15e}" for v in values)

    text = (
        "%YAML:1.0\n"
        "---\n"
        "CameraMat: !!opencv-matrix\n"
        "   rows: 3\n"
        "   cols: 3\n"
        "   dt: d\n"
        f"   data: [ {_row(K[0])},\n"
        f"           {_row(K[1])},\n"
        f"           {_row(K[2])} ]\n"
        "DistCoeff: !!opencv-matrix\n"
        "   rows: 1\n"
        f"   cols: {len(d)}\n"
        "   dt: d\n"
        f"   data: [ {_row(d)} ]\n"
        f"ImageSize: [ {w}, {h} ]\n"
    )
    path.write_text(text)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="ZED 2i right-eye intrinsic calibration from checkerboard images"
    )
    p.add_argument(
        "--image-dir", required=True, metavar="DIR",
        help="Folder of right-eye checkerboard images (one board pose per image)",
    )
    p.add_argument(
        "--checkerboard-size", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"),
        help="INNER corner count, not squares: a board of 10x7 squares is 9 6",
    )
    p.add_argument(
        "--square-size", type=float, required=True, metavar="M",
        help="Checkerboard square edge in metres (e.g. 0.025 for 25 mm)",
    )
    p.add_argument(
        "--output", default="zed_right_intrinsics.json", metavar="PATH",
        help="Canonical JSON output with provenance (default zed_right_intrinsics.json)",
    )
    p.add_argument(
        "--lvt2calib-export", default=None, metavar="PATH",
        help="Also write an LVT2Calib-compatible intrinsic YAML here "
             "(e.g. intrinsic.yaml, to drop in lvt2calib/data/camera_info/)",
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

    image_dir = Path(args.image_dir)
    # Expected, user-fixable failures (bad folder, wrong --checkerboard-size,
    # too few usable views) report as a one-line message, not a traceback.
    try:
        paths = find_images(image_dir)
        print(f"Found {len(paths)} images in {image_dir}, detecting "
              f"{pattern[0]}x{pattern[1]} inner corners...")
        used, corners_list, image_size, skipped = detect_all(paths, pattern, args.verbose)
        result = calibrate(used, corners_list, image_size, pattern, args.square_size, skipped)
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from None

    provenance = {
        "tool": Path(__file__).name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "opencv_version": cv2.__version__,
        "python_version": sys.version.split()[0],
        "image_dir": str(image_dir.resolve()),
        "command": " ".join(sys.argv),
    }

    out = Path(args.output)
    save_json(result, out, provenance)

    print(f"\nRight eye {image_size[0]}x{image_size[1]}, "
          f"{len(result.images_used)}/{len(paths)} images used")
    print(f"  fx={result.fx:.3f}  fy={result.fy:.3f}  "
          f"cx={result.cx:.3f}  cy={result.cy:.3f}")
    print("  dist (k1 k2 p1 p2 k3) = "
          f"{[round(v, 6) for v in result.dist_coeffs.tolist()]}")
    worst = max(result.per_image_error, key=result.per_image_error.get)
    print(f"  reprojection error: mean {result.mean_reprojection_error:.4f} px, "
          f"rms {result.rms_reprojection_error:.4f} px, "
          f"worst {result.per_image_error[worst]:.4f} px ({worst})")
    if len(result.images_used) < _RECOMMENDED_VIEWS:
        print(f"  note: under {_RECOMMENDED_VIEWS} views -- add poses with large "
              "tilts about both board axes and varied distances for a stable K")
    print(f"Saved JSON -> {out}")

    if args.lvt2calib_export:
        lvt = Path(args.lvt2calib_export)
        save_lvt2calib_yaml(result, lvt)
        print(f"Saved LVT2Calib intrinsics -> {lvt} "
              "(copy into lvt2calib/data/camera_info/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
