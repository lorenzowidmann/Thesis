"""Intrinsic calibration of the FLIR Vue Pro R from the 4-hole heated board.

A printed checkerboard is thermally uniform and invisible in LWIR, so this uses
the same four-circular-hole board as the LVT2Calib extrinsic step: the board is
heated during capture and each through-hole reads as a blob against it.

Structure, CLI conventions and JSON/provenance format mirror
zed_intrinsic_calib.py in this folder. The pipeline differs where the target
forces it:

    read RJPG (flyr) -> threshold by thermal contrast -> 4 blob centroids
    -> correspondence to the known hole coordinates -> solve K

WHAT THIS METHOD CANNOT DO -- read before trusting the output:

- **Distortion is not estimated.** Four points per view is barely enough to fix
  a homography and nothing is left over for radial/tangential terms. The output
  reports zeros, explicitly flagged, in the JSON and in the LVT2Calib export.
  This is a real loss: LVT2Calib's own shipped front_thermal_intrinsic.yaml for
  a 640x512 thermal camera carries k1 = -0.359, i.e. a strongly distorting lens
  that a zero vector does not correct at all. Treat K from here as a starting
  point, not a final calibration.
- **Precision depends heavily on how the set was captured.** Simulated with
  this camera's geometry (640x512, fx ~ 765), a 0.5 m board and 0.3 px centroid
  noise: capturing at 5-20 m gave fx +-59 px (+-7.8%) over 20 views, because the
  board spans only ~19 px at 20 m. Capturing at 2-4 m with 15-60 degree tilts
  gave +-4 px (+-0.5%). Shoot close and tilted; distance buys nothing here.

Two solvers, both genuinely estimating K (--method):

- `homography` -- Zhang's method: a homography per view, stacked constraints on
  omega = K^-T K^-1, closed-form K, then Levenberg-Marquardt refinement.
- `pnp` -- bundle adjustment: K plus a pose per view refined together by LM
  over all reprojection residuals, in pure numpy.

Note both differ from the OpenCV calls sometimes suggested for this:
cv2.decomposeHomographyMat(H, K) and cv2.solvePnP(..., cameraMatrix, ...) each
take K as a required *input* and recover only pose, so neither can calibrate a
camera on its own.

Usage:
    py flir_intrinsic_calib.py --image-dir thermal/ --method homography
    py flir_intrinsic_calib.py --image-dir thermal/ --method pnp --lvt2calib-export thermal_intrinsic.yaml
    py flir_intrinsic_calib.py --image-dir thermal/ --polarity hot --debug-overlay debug/
    py flir_intrinsic_calib.py --image-dir thermal/ --board-config board.json
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

# The board that ships with the rover: four holes on a 0.5 m square, board
# plane z = 0. Overridable (--board-coords / --board-config) because the
# geometry is a property of the physical target, not of this code.
DEFAULT_BOARD = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.5, 0.5, 0.0],
]

_RJPG_EXTS = (".jpg", ".jpeg", ".rjpg")
_PLAIN_EXTS = (".png", ".tif", ".tiff")
_IMAGE_EXTS = _RJPG_EXTS + _PLAIN_EXTS

_N_HOLES = 4

# Zhang needs >= 3 views to pin all four intrinsic parameters; more importantly
# the views must differ in orientation, which _warn_on_geometry checks for.
_MIN_VIEWS = 3
_RECOMMENDED_VIEWS = 20

# Below this the board is too small in frame for 4 centroids to constrain
# anything -- 0.5 m at 20 m on this camera is ~19 px across.
_MIN_BOARD_SPAN_PX = 60.0


# --------------------------------------------------------------------------- #
# Results                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class SkippedImage:
    """One input image that never reached the solver, and why."""
    filename: str
    reason: str


@dataclass
class ViewPose:
    """Solved board pose for one accepted view."""
    filename: str
    rvec: list[float]              # Rodrigues, board -> camera
    tvec: list[float]              # metres
    reprojection_error: float      # px, RMS over the 4 holes
    board_span_px: float           # largest hole-to-hole distance in the image


@dataclass
class CalibrationResult:
    """Solved K plus everything needed to reproduce and judge it."""
    camera_matrix: np.ndarray
    image_size: tuple[int, int]
    mean_reprojection_error: float
    rms_reprojection_error: float
    views: list[ViewPose] = field(default_factory=list)
    images_skipped: list[SkippedImage] = field(default_factory=list)
    board_coords: list[list[float]] = field(default_factory=list)
    method: str = ""

    # Fixed, not solved: 4 points per view leave nothing over for distortion.
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    distortion_estimated: bool = False

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
# Reading thermal frames                                                       #
# --------------------------------------------------------------------------- #
def read_thermal(path: Path) -> np.ndarray:
    """2-D float array of one thermal frame, in whatever unit the file carries.

    RJPGs are unpacked with flyr (already a RadiometricCalibration dependency)
    via `.celsius`. That is acceptable *here* specifically because hole
    detection only needs relative contrast across the frame -- unlike the
    radiometric pipeline, no absolute temperature accuracy is claimed or used.

    Plain .png/.tif are read as-is so exported or synthetic frames work too.
    """
    if path.suffix.lower() in _RJPG_EXTS:
        try:
            import flyr
        except ImportError:
            raise RuntimeError(
                "flyr is required to read radiometric JPEGs -- `pip install flyr` "
                "(see Calibration/requirements.txt)"
            ) from None
        return np.asarray(flyr.unpack(str(path)).celsius, dtype=np.float64)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unreadable (cv2.imread returned None)")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)


def _to_uint8(thermal: np.ndarray) -> np.ndarray:
    """Full-range normalisation to 8-bit, for the morphology/threshold ops."""
    lo, hi = float(thermal.min()), float(thermal.max())
    if hi - lo < 1e-9:
        return np.zeros(thermal.shape, np.uint8)
    return (((thermal - lo) / (hi - lo)) * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Hole detection                                                               #
# --------------------------------------------------------------------------- #
def detect_holes(
    thermal: np.ndarray,
    polarity: str = "cold",
    threshold: str = "otsu",
    percentile: float = 5.0,
    min_area: int = 6,
    max_area: int = 100_000,
    min_circularity: float = 0.5,
) -> tuple[np.ndarray | None, str, np.ndarray]:
    """Sub-pixel centroids of the four holes, or a reason they weren't found.

    A through-hole shows whatever is behind the board, so its sign against the
    heated panel depends on the scene: cooler against sky or a cold wall,
    warmer if something hot is behind. `polarity` picks which tail to segment
    ("cold" = holes darker than the board, the usual case outdoors).

    Centroids are intensity-weighted by each blob's contrast against the
    threshold rather than taken as the pixel-count centre: with only four
    points per view, centroid noise maps directly into K, and at these blob
    sizes a plain binary centroid quantises to ~0.5 px.

    Returns (centroids Nx2 float64 or None, reason, binary mask for debugging).
    """
    if polarity not in ("cold", "hot"):
        raise ValueError(f"polarity must be 'cold' or 'hot' (got {polarity!r})")

    gray = _to_uint8(thermal)
    # Work with "holes are bright" internally, whichever way round they are.
    work = cv2.bitwise_not(gray) if polarity == "cold" else gray
    work = cv2.GaussianBlur(work, (3, 3), 0)

    if threshold == "otsu":
        _t, mask = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold == "percentile":
        cut = float(np.percentile(work, 100.0 - percentile))
        _t, mask = cv2.threshold(work, cut, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"threshold must be 'otsu' or 'percentile' (got {threshold!r})")

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, n_labels):  # 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        # A circular hole fills ~pi/4 of its bounding box; this rejects the
        # long thin blobs that thresholding a warm background produces.
        if w == 0 or h == 0 or area / float(w * h) < min_circularity * 0.785:
            continue
        candidates.append((area, label))

    if len(candidates) != _N_HOLES:
        return None, (
            f"found {len(candidates)} blob(s) passing the filters, expected {_N_HOLES}"
        ), mask

    # Largest-first is irrelevant to correspondence (that is handled later),
    # but keeps the debug overlay deterministic.
    candidates.sort(reverse=True)
    weights = work.astype(np.float64)
    centroids = []
    for _area, label in candidates:
        ys, xs = np.nonzero(labels == label)
        w_blob = weights[ys, xs]
        w_sum = w_blob.sum()
        if w_sum <= 0:
            centroids.append([xs.mean(), ys.mean()])
        else:
            centroids.append([(xs * w_blob).sum() / w_sum, (ys * w_blob).sum() / w_sum])
    return np.asarray(centroids, dtype=np.float64), "", mask


def _order_ccw(points: np.ndarray) -> np.ndarray:
    """Indices ordering `points` counter-clockwise about their centroid.

    Fixes the winding, which is what actually matters: a mirrored assignment
    is not a rigid motion and would corrupt the solve, whereas a rotation of
    the labels is absorbed by that view's own R (see assign_correspondence).
    """
    c = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
    return np.argsort(ang)


def canonical_board(board: np.ndarray) -> np.ndarray:
    """The board's holes wound counter-clockwise about their centroid.

    Every stage downstream -- correspondence, both solvers, the reported poses
    -- must index the holes the same way, so the board is put in this canonical
    order once and that array is what gets passed around. (Ordering the
    detections to match a *differently* ordered board silently swaps two holes
    and yields a non-physical K.)
    """
    return board[_order_ccw(board[:, :2])]


def assign_correspondence(
    detected: np.ndarray, board: np.ndarray, K: np.ndarray | None = None
) -> np.ndarray:
    """Order `detected` to match `board`, which must already be canonical.

    Winding both sets counter-clockwise fixes the labelling up to a cyclic
    shift. For a board with 4-fold symmetry -- the default 0.5 m square is
    exactly this -- every shift is the same physical target rotated about its
    normal, and each view solves its own R, so the choice cannot affect K and
    shift 0 is taken.

    For an asymmetric board the shift is real, and once a provisional K exists
    the caller passes it here: each shift is scored by the reprojection error
    of the pose it implies, and the best wins. That is why calibration runs a
    second pass (see solve()).

    Note the image y axis points down, so a counter-clockwise winding in pixel
    coordinates is clockwise in the board frame. That consistent flip is
    equivalent to viewing the board from its far side and is absorbed by each
    view's R; it does not affect K (verified against synthetic ground truth,
    both windings recovering fx within 0.3%).
    """
    det_ccw = detected[_order_ccw(detected)]
    shifts = [np.roll(np.arange(_N_HOLES), -s) for s in range(_N_HOLES)]

    # Is the board invariant under a cyclic shift? Then no shift is "wrong".
    d0 = board[:, :2]
    symmetric = all(_quad_shape_matches(d0, d0[s]) for s in shifts[1:])
    if symmetric or K is None:
        return det_ccw

    best, best_err = det_ccw, np.inf
    for s in shifts:
        cand = det_ccw[s]
        ok, rvec, tvec = cv2.solvePnP(
            board.astype(np.float64), cand.astype(np.float64),
            K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            continue
        proj, _ = cv2.projectPoints(board, rvec, tvec, K, np.zeros(5))
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - cand) ** 2, axis=1))))
        if err < best_err:
            best, best_err = cand, err
    return best


def _quad_shape_matches(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether two 4-point sets have the same edge-length sequence."""
    ea = [np.linalg.norm(a[(i + 1) % 4] - a[i]) for i in range(4)]
    eb = [np.linalg.norm(b[(i + 1) % 4] - b[i]) for i in range(4)]
    return bool(np.allclose(ea, eb, atol=tol))


# --------------------------------------------------------------------------- #
# Solver 1: Zhang (homography)                                                 #
# --------------------------------------------------------------------------- #
def _v_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
    """Zhang's v_ij row relating a homography column pair to omega."""
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j],
    ])


def solve_zhang(board: np.ndarray, views: list[np.ndarray]) -> np.ndarray:
    """Closed-form K from per-view homographies (Zhang 2000), zero skew.

    Each view contributes two constraints on omega = K^-T K^-1: the two board
    axes are orthogonal and equal in scale, so h1' omega h2 = 0 and
    h1' omega h1 = h2' omega h2. Stacking those over views and taking the null
    space gives omega, from which K follows in closed form.
    """
    src = board[:, :2].astype(np.float64)
    rows = []
    for pts in views:
        H, _ = cv2.findHomography(src, pts.astype(np.float64), method=0)
        if H is None:
            continue
        H = H / H[2, 2]
        rows.append(_v_ij(H, 0, 1))
        rows.append(_v_ij(H, 0, 0) - _v_ij(H, 1, 1))
    if len(rows) < 4:
        raise RuntimeError("too few usable homographies for the closed-form solve")

    _u, _s, vt = np.linalg.svd(np.asarray(rows))
    b11, b12, b22, b13, b23, b33 = vt[-1]

    denom = b11 * b22 - b12 * b12
    if abs(denom) < 1e-18:
        raise RuntimeError(
            "degenerate closed-form solve -- views are too similar in "
            "orientation (tilt the board more between captures)"
        )
    cy = (b12 * b13 - b11 * b23) / denom
    lam = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11
    if lam / b11 <= 0 or lam * b11 / denom <= 0:
        raise RuntimeError(
            "closed-form solve produced a non-physical K -- check hole "
            "correspondence and that the board is tilted between views"
        )
    fx = np.sqrt(lam / b11)
    fy = np.sqrt(lam * b11 / denom)
    cx = -b13 * fx * fx / lam
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# Solver 2: bundle adjustment (pnp)                                            #
# --------------------------------------------------------------------------- #
def _pack(K: np.ndarray, poses: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    p = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    for rvec, tvec in poses:
        p.extend(np.asarray(rvec).ravel())
        p.extend(np.asarray(tvec).ravel())
    return np.asarray(p, dtype=np.float64)


def _unpack(p: np.ndarray, n_views: int):
    K = np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
    poses = []
    for i in range(n_views):
        base = 4 + 6 * i
        poses.append((p[base:base + 3].copy(), p[base + 3:base + 6].copy()))
    return K, poses


def _residuals_and_jacobian(p, board, views, want_jac=True):
    """Stacked reprojection residuals and, optionally, the analytic Jacobian.

    cv2.projectPoints returns the derivative block directly -- columns 0-2
    rvec, 3-5 tvec, 6-7 (fx, fy), 8-9 (cx, cy) -- so no finite differences are
    needed. The Jacobian is block-sparse: a view's pose only touches its own
    residuals, and only the four intrinsic columns are shared.
    """
    n_views = len(views)
    K, poses = _unpack(p, n_views)
    dist = np.zeros(5)
    m = _N_HOLES * 2

    res = np.zeros(n_views * m)
    J = np.zeros((n_views * m, 4 + 6 * n_views)) if want_jac else None

    for i, (obs, (rvec, tvec)) in enumerate(zip(views, poses)):
        proj, jac = cv2.projectPoints(board, rvec, tvec, K, dist)
        r0 = i * m
        res[r0:r0 + m] = (proj.reshape(-1, 2) - obs).ravel()
        if want_jac:
            J[r0:r0 + m, 0:2] = jac[:, 6:8]    # fx, fy
            J[r0:r0 + m, 2:4] = jac[:, 8:10]   # cx, cy
            J[r0:r0 + m, 4 + 6 * i:4 + 6 * i + 6] = jac[:, 0:6]  # rvec, tvec
    return (res, J) if want_jac else res


def solve_bundle(
    board: np.ndarray, views: list[np.ndarray], K0: np.ndarray, max_iter: int = 60
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Refine K and every board pose together by Levenberg-Marquardt.

    Pure numpy, matching this repo's convention of not pulling in scipy for a
    solve this size (see PointCloudElaboration/OcTree/octree/smoothing.py).
    Poses are seeded per view by IPPE, which is exact for a planar target.
    """
    poses = []
    for obs in views:
        ok, rvec, tvec = cv2.solvePnP(
            board, obs.astype(np.float64), K0, np.zeros(5), flags=cv2.SOLVEPNP_IPPE
        )
        if not ok:
            raise RuntimeError("pose initialisation failed for a view")
        poses.append((rvec.ravel(), tvec.ravel()))

    p = _pack(K0, poses)
    res, J = _residuals_and_jacobian(p, board, views)
    cost = float(res @ res)
    lam = 1e-3

    for _ in range(max_iter):
        JtJ = J.T @ J
        g = J.T @ res
        try:
            step = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), -g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue

        cand = p + step
        cand_res, cand_J = _residuals_and_jacobian(cand, board, views)
        cand_cost = float(cand_res @ cand_res)
        if cand_cost < cost:
            improvement = (cost - cand_cost) / max(cost, 1e-30)
            p, res, J, cost = cand, cand_res, cand_J, cand_cost
            lam = max(lam * 0.3, 1e-12)
            if improvement < 1e-10:
                break
        else:
            lam *= 10
            if lam > 1e12:
                break

    K, poses = _unpack(p, len(views))
    return K, poses


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def solve(
    board: np.ndarray,
    filenames: list[str],
    detections: list[np.ndarray],
    image_size: tuple[int, int],
    method: str,
    skipped: list[SkippedImage],
) -> CalibrationResult:
    """Assign correspondences, solve K, then re-assign against it and re-solve.

    The second pass exists because correspondence and K are coupled: for an
    asymmetric board the cyclic-shift choice needs a K to be scored against
    (see assign_correspondence), and the first pass is what provides one.
    """
    if len(detections) < _MIN_VIEWS:
        raise RuntimeError(
            f"only {len(detections)} usable view(s), need >= {_MIN_VIEWS} "
            "(check --polarity and the blob filters against --debug-overlay)"
        )

    board = canonical_board(board)
    views = [assign_correspondence(d, board) for d in detections]
    K = solve_zhang(board, views)

    views = [assign_correspondence(d, board, K) for d in detections]
    if method == "homography":
        K = solve_zhang(board, views)
        # LM refinement of the closed-form estimate, distortion pinned at zero.
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO * 0 |
                 cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
                 cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
        obj = [board.astype(np.float32)] * len(views)
        img = [v.astype(np.float32).reshape(-1, 1, 2) for v in views]
        _rms, K, _d, rvecs, tvecs = cv2.calibrateCamera(
            obj, img, image_size, K.copy(), np.zeros(5), flags=flags
        )
        poses = [(r.ravel(), t.ravel()) for r, t in zip(rvecs, tvecs)]
    elif method == "pnp":
        K, poses = solve_bundle(board, views, K)
    else:
        raise ValueError(f"method must be 'homography' or 'pnp' (got {method!r})")

    per_view: list[ViewPose] = []
    for name, obs, (rvec, tvec) in zip(filenames, views, poses):
        proj, _ = cv2.projectPoints(board, rvec, tvec, K, np.zeros(5))
        d = proj.reshape(-1, 2) - obs
        err = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
        span = float(max(
            np.linalg.norm(obs[i] - obs[j])
            for i in range(_N_HOLES) for j in range(i + 1, _N_HOLES)
        ))
        per_view.append(ViewPose(name, list(map(float, rvec)), list(map(float, tvec)),
                                 err, span))

    errors = [v.reprojection_error for v in per_view]
    return CalibrationResult(
        camera_matrix=K,
        image_size=image_size,
        mean_reprojection_error=float(np.mean(errors)),
        rms_reprojection_error=float(np.sqrt(np.mean(np.square(errors)))),
        views=per_view,
        images_skipped=skipped,
        board_coords=board.tolist(),
        method=method,
    )


def _warn_on_geometry(result: CalibrationResult) -> list[str]:
    """Capture-quality warnings that a low reprojection error would hide.

    Four exactly-determined points fit their own homography perfectly, so a
    small residual says nothing about whether the set constrains K. These
    checks look at the capture geometry instead.
    """
    warnings = []
    spans = [v.board_span_px for v in result.views]
    small = [v for v in result.views if v.board_span_px < _MIN_BOARD_SPAN_PX]
    if small:
        warnings.append(
            f"{len(small)}/{len(result.views)} view(s) have the board spanning "
            f"< {_MIN_BOARD_SPAN_PX:.0f} px (min {min(spans):.0f} px) -- too far away "
            "to constrain K; recapture at 2-4 m"
        )
    if len(result.views) < _RECOMMENDED_VIEWS:
        warnings.append(
            f"only {len(result.views)} views -- with 4 points each, aim for "
            f">= {_RECOMMENDED_VIEWS}"
        )

    # Zhang is degenerate when every board plane shares an orientation.
    normals = []
    for v in result.views:
        R, _ = cv2.Rodrigues(np.asarray(v.rvec))
        normals.append(R[:, 2])
    if len(normals) >= 2:
        spread = np.degrees(np.arccos(np.clip(
            [abs(float(a @ b)) for i, a in enumerate(normals) for b in normals[i + 1:]],
            -1.0, 1.0)))
        if spread.size and spread.max() < 20.0:
            warnings.append(
                f"board orientation varies by only {spread.max():.1f} deg across views "
                "-- near-parallel planes leave K poorly constrained; tilt 15-60 deg"
            )
    return warnings


# --------------------------------------------------------------------------- #
# Board geometry                                                               #
# --------------------------------------------------------------------------- #
def load_board(coords: list[float] | None, config: str | None) -> np.ndarray:
    """Board hole coordinates from --board-coords, --board-config, or default."""
    if coords and config:
        raise ValueError("pass --board-coords or --board-config, not both")
    if config:
        data = json.loads(Path(config).read_text())
        pts = data["holes"] if isinstance(data, dict) else data
    elif coords:
        if len(coords) % 3 != 0:
            raise ValueError(
                f"--board-coords needs X Y Z per hole ({len(coords)} values given)"
            )
        pts = [coords[i:i + 3] for i in range(0, len(coords), 3)]
    else:
        pts = DEFAULT_BOARD

    board = np.asarray(pts, dtype=np.float64)
    if board.shape != (_N_HOLES, 3):
        raise ValueError(f"board must be {_N_HOLES} XYZ points (got {board.shape})")
    if not np.allclose(board[:, 2], board[0, 2]):
        raise ValueError("board holes must be coplanar (equal Z)")
    return board


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #
def save_json(result: CalibrationResult, path: Path, provenance: dict) -> None:
    """Canonical, provenance-tracked record (same shape as zed_intrinsic_calib)."""
    payload = {
        "camera": "FLIR Vue Pro R",
        "target": "4-hole heated board",
        "method": result.method,
        "camera_matrix": result.camera_matrix.tolist(),
        "dist_coeffs": result.dist_coeffs.tolist(),
        "dist_coeff_order": ["k1", "k2", "p1", "p2", "k3"],
        "distortion_estimated": result.distortion_estimated,
        "distortion_note": (
            "NOT estimated by this method: 4 points per view fix a homography "
            "and leave no redundancy for radial/tangential terms. Values are "
            "zeros, not a measurement of a distortion-free lens."
        ),
        "fx": result.fx, "fy": result.fy, "cx": result.cx, "cy": result.cy,
        "image_size": {"width": result.image_size[0], "height": result.image_size[1]},
        "reprojection_error": {
            "mean_px": result.mean_reprojection_error,
            "rms_px": result.rms_reprojection_error,
            "worst_px": max((v.reprojection_error for v in result.views), default=0.0),
            "note": (
                "4 points exactly determine a homography, so a low residual "
                "does not by itself indicate a well-constrained K -- see "
                "capture_warnings"
            ),
        },
        "board": {"holes_xyz": result.board_coords},
        "views": {
            "n_used": len(result.views),
            "n_skipped": len(result.images_skipped),
            "per_view": [
                {
                    "filename": v.filename,
                    "rvec": v.rvec,
                    "tvec": v.tvec,
                    "reprojection_error_px": v.reprojection_error,
                    "board_span_px": v.board_span_px,
                }
                for v in result.views
            ],
            "skipped": [{"filename": s.filename, "reason": s.reason}
                        for s in result.images_skipped],
        },
        "capture_warnings": _warn_on_geometry(result),
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, indent=2))


def save_lvt2calib_yaml(result: CalibrationResult, path: Path) -> None:
    """LVT2Calib intrinsics export (github.com/Clothooo/lvt2calib).

    Identical layout to the ZED export -- their thermal camera is only a
    different row in the sensor table (README row 14, "TC", launched via
    thermal_cam_pattern.launch), not a different file format: their shipped
    data/camera_info/front_thermal_intrinsic.yaml carries the same
    CameraMat/DistCoeff/ImageSize fields that src/camera/cam_pattern.cpp reads
    with cv::FileStorage.

    The zeroed DistCoeff is called out in a leading comment (verified to parse:
    cv::FileStorage accepts '#' comment lines). That matters here -- their own
    thermal example has k1 = -0.359, so a reader seeing zeros should know this
    is "not estimated", not "measured as distortion-free".
    """
    K = np.asarray(result.camera_matrix, dtype=np.float64)
    d = np.asarray(result.dist_coeffs, dtype=np.float64).ravel()
    w, h = result.image_size

    def _row(values) -> str:
        return ", ".join(f"{v:.15e}" for v in values)

    text = (
        "%YAML:1.0\n"
        "---\n"
        f"# FLIR Vue Pro R intrinsics, 4-hole board, method={result.method}.\n"
        "# DistCoeff is NOT estimated by this method (4 points per view leave no\n"
        "# redundancy for distortion) -- the zeros below are a placeholder, not a\n"
        "# measurement. Undistortion with them is a no-op.\n"
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


def write_debug_overlay(
    thermal: np.ndarray, mask: np.ndarray, centroids: np.ndarray | None, path: Path
) -> None:
    """Normalised frame with the segmentation and detected centroids drawn.

    The one practical way to tune --polarity and the blob filters without
    sample images to calibrate the defaults against.
    """
    vis = cv2.cvtColor(_to_uint8(thermal), cv2.COLOR_GRAY2BGR)
    vis[mask > 0] = (0.5 * vis[mask > 0] + np.array([0, 0, 128])).astype(np.uint8)
    if centroids is not None:
        for i, (x, y) in enumerate(centroids):
            cv2.drawMarker(vis, (int(round(x)), int(round(y))), (0, 255, 0),
                           cv2.MARKER_CROSS, 12, 1)
            cv2.putText(vis, str(i), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(str(path), vis)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="FLIR Vue Pro R intrinsic calibration from the 4-hole heated board"
    )
    p.add_argument(
        "--image-dir", required=True, metavar="DIR",
        help="Folder of thermal frames of the heated board (RJPG, or plain PNG/TIFF)",
    )
    p.add_argument(
        "--method", choices=("homography", "pnp"), default="homography",
        help="homography: Zhang closed-form + LM refinement. "
             "pnp: bundle-adjust K and all poses together (default homography)",
    )
    p.add_argument(
        "--board-coords", type=float, nargs="*", default=None, metavar="V",
        help="Hole coordinates as X Y Z per hole, 12 values "
             f"(default {DEFAULT_BOARD})",
    )
    p.add_argument(
        "--board-config", default=None, metavar="PATH",
        help='JSON with the hole coordinates: {"holes": [[x,y,z], ...]}',
    )
    p.add_argument(
        "--polarity", choices=("cold", "hot"), default="cold",
        help="Whether the holes read cooler (default) or warmer than the heated "
             "board -- a through-hole shows the scene behind it",
    )
    p.add_argument(
        "--threshold", choices=("otsu", "percentile"), default="otsu",
        help="Segmentation strategy for the holes (default otsu)",
    )
    p.add_argument(
        "--percentile", type=float, default=5.0, metavar="P",
        help="With --threshold percentile: keep the most extreme P%% (default 5)",
    )
    p.add_argument(
        "--min-area", type=int, default=6, metavar="PX",
        help="Smallest blob accepted as a hole, in pixels (default 6)",
    )
    p.add_argument(
        "--max-area", type=int, default=100_000, metavar="PX",
        help="Largest blob accepted as a hole, in pixels (default 100000)",
    )
    p.add_argument(
        "--min-circularity", type=float, default=0.5, metavar="F",
        help="Blob fill ratio against a perfect disc, 0-1 (default 0.5)",
    )
    p.add_argument(
        "--debug-overlay", default=None, metavar="DIR",
        help="Write per-image segmentation/centroid overlays here (tuning aid)",
    )
    p.add_argument(
        "--output", default="flir_intrinsics.json", metavar="PATH",
        help="Canonical JSON output with provenance (default flir_intrinsics.json)",
    )
    p.add_argument(
        "--lvt2calib-export", default=None, metavar="PATH",
        help="Also write an LVT2Calib-compatible intrinsic YAML here",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="List every image as it is accepted, not just the skipped ones",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        board = load_board(args.board_coords, args.board_config)
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from None

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise SystemExit(f"error: --image-dir is not a directory: {image_dir}")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        raise SystemExit(
            f"error: no images in {image_dir} (looked for {', '.join(_IMAGE_EXTS)})"
        )

    debug_dir = Path(args.debug_overlay) if args.debug_overlay else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(paths)} frames in {image_dir}, detecting {_N_HOLES} holes "
          f"({args.polarity} against the board, {args.threshold} threshold)...")

    filenames: list[str] = []
    detections: list[np.ndarray] = []
    skipped: list[SkippedImage] = []
    image_size: tuple[int, int] | None = None

    for path in paths:
        try:
            thermal = read_thermal(path)
        except (ValueError, RuntimeError) as exc:
            skipped.append(SkippedImage(path.name, str(exc)))
            continue

        h, w = thermal.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif (w, h) != image_size:
            skipped.append(SkippedImage(
                path.name, f"image size {w}x{h} differs from {image_size[0]}x{image_size[1]}"))
            continue

        centroids, reason, mask = detect_holes(
            thermal, args.polarity, args.threshold, args.percentile,
            args.min_area, args.max_area, args.min_circularity,
        )
        if debug_dir:
            write_debug_overlay(thermal, mask, centroids, debug_dir / f"{path.stem}.png")
        if centroids is None:
            skipped.append(SkippedImage(path.name, reason))
            continue

        filenames.append(path.name)
        detections.append(centroids)
        if args.verbose:
            print(f"  [ok]   {path.name}")

    for s in skipped:
        print(f"  [skip] {s.filename}: {s.reason}", file=sys.stderr)

    if image_size is None:
        raise SystemExit("error: no frame could be read")

    try:
        result = solve(board, filenames, detections, image_size, args.method, skipped)
    except (RuntimeError, ValueError) as exc:
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

    print(f"\nFLIR {image_size[0]}x{image_size[1]}, {len(result.views)}/{len(paths)} "
          f"views used, method={result.method}")
    print(f"  fx={result.fx:.3f}  fy={result.fy:.3f}  "
          f"cx={result.cx:.3f}  cy={result.cy:.3f}")
    print("  dist (k1 k2 p1 p2 k3) = [0, 0, 0, 0, 0]  <- NOT ESTIMATED by this method")
    worst = max(result.views, key=lambda v: v.reprojection_error, default=None)
    if worst is not None:
        print(f"  reprojection error: mean {result.mean_reprojection_error:.4f} px, "
              f"rms {result.rms_reprojection_error:.4f} px, "
              f"worst {worst.reprojection_error:.4f} px ({worst.filename})")
    for warning in _warn_on_geometry(result):
        print(f"  warning: {warning}")
    print(f"Saved JSON -> {out}")

    if args.lvt2calib_export:
        lvt = Path(args.lvt2calib_export)
        save_lvt2calib_yaml(result, lvt)
        print(f"Saved LVT2Calib intrinsics -> {lvt} "
              "(copy into lvt2calib/data/camera_info/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
