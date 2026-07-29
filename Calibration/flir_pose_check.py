"""
For each detected board pose that is long enough (>= --min-frames frames, 60 by
default), check whether the four-hole calibration board is COMPLETELY INSIDE the
FLIR frame (not clipped by the image borders).

Pipeline
--------
1. Segment the poses exactly like detect_board_poses.py (frame differencing +
   debounced transitions). Reuses that module so the pose boundaries match.
2. Keep only the poses with >= --min-frames frames.
3. For a subsample of each pose's frames, detect the board and test whether its
   rotated bounding rectangle stays inside the image with a margin.

Board detection (thermal, single frame)
---------------------------------------
In LWIR the board panel reads at almost the SAME level as the wall behind it, so
intensity thresholding cannot isolate it. The only board-specific feature is the
four circular HOLES, which show the (colder/warmer) background through the panel.
So we:
  - Black-hat morphology -> the four dark holes pop out regardless of the panel's
    absolute brightness. Otsu + circularity -> the 4 largest circular blobs are
    the holes. Their centroid is the board CENTRE.
  - From the centre we scan outward along +/-x and +/-y until the panel intensity
    drops to the darker background: that distance is the board half-extent (the
    scan rays pass BETWEEN the hole rows/columns, so they never hit a hole).
  - The hole pattern is symmetric, so one side of an axis is enough: if the wall
    behind the board is bright on (say) the top side and no edge is found there,
    the opposite (bottom) edge, mirrored about the centre, gives the extent.
    This resolves the bright-wall side and also the upright-vs-upside-down
    (100x70 vs 70x100) orientation without any flag.
  - "Completely inside" = centre +/- measured half-extent stays inside the image
    with a --margin-px border on every side.

Usage
-----
    py flir_pose_check.py <flir_dir> [<flir_dir> ...]
    py flir_pose_check.py <flir_dir> --min-frames 60 --margin-px 4 --step 3
    py flir_pose_check.py <flir_dir> --debug-dir out_debug   # dump annotated frames
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python required:  py -m pip install opencv-python")

# reuse the pose-detection building blocks (no side effects on import)
from detect_board_poses import (
    collect_images,
    frame_difference,
    build_time_index,
    fmt_clock,
)

# physical board: 1.00 x 0.70 m, four holes at (+/-0.15, +/-0.15) m


def segment_poses(files, threshold, min_move_frames, min_pose_frames, downscale):
    """Replicate detect_board_poses segmentation.

    Returns (poses, diffs) where:
      diffs[k] = (file_index, Path, diff_value)
      poses    = list of (a, b) index ranges into diffs (stable segments)
    """
    diffs = []
    prev = None
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if downscale > 1:
            img = cv2.resize(img, (img.shape[1] // downscale,
                                   img.shape[0] // downscale))
        if prev is not None:
            diffs.append((i, f, frame_difference(prev, img)))
        prev = img

    if not diffs:
        return [], []

    values = np.array([d[2] for d in diffs])
    if threshold is None:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = median + 6.0 * (mad if mad > 1e-6 else 1.0)

    moving = values > threshold
    n = len(moving)

    transitions = []
    i = 0
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            if j - i >= min_move_frames:
                transitions.append((i, j))
            i = j
        else:
            i += 1

    poses = []
    prev_end = 0
    for a, b in transitions:
        if a - prev_end >= min_pose_frames:
            poses.append((prev_end, a))
        prev_end = b
    if n - prev_end >= min_pose_frames:
        poses.append((prev_end, n))

    return poses, diffs


def detect_holes(gray, hole_kernel, min_area, max_area, min_circ, rect_tol):
    """Detect the four board holes. Returns (pts(4,2), centre(2,)) or None."""
    gb = cv2.GaussianBlur(gray, (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_kernel, hole_kernel))
    bh = cv2.morphologyEx(gb, cv2.MORPH_BLACKHAT, k)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        if area / (np.pi * r * r) < min_circ:  # circularity
            continue
        cand.append((area, x, y))
    return select_holes(cand, rect_tol)


def _rect_score(p):
    """Lower = the 4 points look more like corners of a centred rectangle.

    Corners of a rectangle are equidistant from the centroid (equal half-
    diagonals) and come in opposite pairs (v and -v). A far spurious blob breaks
    both, so it scores high and is rejected.
    """
    m = p.mean(axis=0)
    v = p - m
    r = np.linalg.norm(v, axis=1)
    rmean = float(r.mean())
    if rmean < 1e-3:
        return 1e9
    rad_err = float(r.std()) / rmean           # radii should be equal
    sym_err = 0.0
    for i in range(4):
        d = np.linalg.norm(v + v[i], axis=1)   # distance from -v[i] to each v[j]
        d[i] = np.inf
        sym_err += float(d.min())
    sym_err /= 4.0 * rmean                      # each v should have an opposite
    return rad_err + sym_err


def select_holes(cand, rect_tol):
    """Pick the 4 candidate blobs that best form a centred rectangle (the holes).
    Returns (pts(4,2), centre(2,)) or None if no plausible set is found."""
    import itertools
    if len(cand) < 4:
        return None
    cand = sorted(cand, key=lambda t: -t[0])[:8]  # keep the 8 strongest
    areas = np.array([a for a, _, _ in cand], dtype=float)
    pts_all = np.array([[x, y] for _, x, y in cand], dtype=np.float32)

    best = None  # (score, pts)
    for idx in itertools.combinations(range(len(cand)), 4):
        ii = list(idx)
        aa = areas[ii]
        if aa.max() / max(aa.min(), 1e-6) > 3.0:   # holes are similar-sized
            continue
        p = pts_all[ii]
        if p.std(axis=0).max() < 4.0:              # too tightly clustered = noise
            continue
        score = _rect_score(p)
        if best is None or score < best[0]:
            best = (score, p)

    if best is None or best[0] > rect_tol:
        return None
    p = best[1]
    return p, p.mean(axis=0)


def _scan_edge(gray, cx, cy, dx, dy, i0, drop, run=3):
    """Walk from (cx,cy) along (dx,dy). Return distance to the board edge
    (intensity drop below i0-drop for `run` px) or None if the ray reaches the
    image border while still on the panel."""
    h_img, w_img = gray.shape[:2]
    cnt = 0
    t = 1
    while True:
        x = int(round(cx + dx * t))
        y = int(round(cy + dy * t))
        if x < 0 or x >= w_img or y < 0 or y >= h_img:
            return None  # ran off the frame still on the panel
        if gray[y, x] < i0 - drop:
            cnt += 1
            if cnt >= run:
                return t - run + 1
        else:
            cnt = 0
        t += 1


def board_extent(gray, ctr, drop):
    """Measure board half-extents (hx, hy) about the hole centre using the
    symmetric edge scan. Returns dict with hx, hy (or None), i0."""
    cx, cy = float(ctr[0]), float(ctr[1])
    x0, y0 = int(round(cx)), int(round(cy))
    patch = gray[max(0, y0 - 2):y0 + 3, max(0, x0 - 2):x0 + 3]
    i0 = float(np.median(patch)) if patch.size else float(gray[y0, x0])

    eL = _scan_edge(gray, cx, cy, -1, 0, i0, drop)
    eR = _scan_edge(gray, cx, cy, 1, 0, i0, drop)
    eU = _scan_edge(gray, cx, cy, 0, -1, i0, drop)
    eD = _scan_edge(gray, cx, cy, 0, 1, i0, drop)

    def half(a, b):
        vals = [e for e in (a, b) if e is not None]
        return sum(vals) / len(vals) if vals else None

    return {"i0": i0, "hx": half(eL, eR), "hy": half(eU, eD),
            "edges": (eL, eR, eU, eD)}


def board_inside(gray, ctr, ext, margin):
    """Return (inside_bool_or_None, clipped_sides list). None = inconclusive
    (an axis had no measurable edge on either side)."""
    h_img, w_img = gray.shape[:2]
    cx, cy = float(ctr[0]), float(ctr[1])
    hx, hy = ext["hx"], ext["hy"]
    if hx is None or hy is None:
        return None, []
    sides = []
    if cx - hx < margin:
        sides.append("L")
    if cx + hx > w_img - 1 - margin:
        sides.append("R")
    if cy - hy < margin:
        sides.append("U")
    if cy + hy > h_img - 1 - margin:
        sides.append("D")
    return (len(sides) == 0), sides


def main():
    ap = argparse.ArgumentParser(
        description="Check whether the four-hole board is fully inside the frame "
                    "in each long pose."
    )
    ap.add_argument("dirs", nargs="+", help="One or more FLIR image folders (in order)")
    # --- pose segmentation (same knobs as detect_board_poses) ---
    ap.add_argument("--threshold", type=float, default=None,
                    help="Frame-diff movement threshold (auto if omitted).")
    ap.add_argument("--min-move-frames", type=int, default=3,
                    help="Min consecutive above-threshold frames for a real move.")
    ap.add_argument("--min-pose-frames", type=int, default=15,
                    help="Min stable frames to call a segment a pose (segmentation).")
    ap.add_argument("--downscale", type=int, default=2,
                    help="Downscale factor for the frame-diff step (default 2).")
    # --- the actual check ---
    ap.add_argument("--min-frames", type=int, default=60,
                    help="Only check poses with at least this many frames (default 60).")
    ap.add_argument("--step", type=int, default=3,
                    help="Analyze every Nth frame within a pose (default 3).")
    ap.add_argument("--margin-px", type=int, default=3,
                    help="Required clear border, in pixels, on every image edge.")
    ap.add_argument("--inside-frac", type=float, default=0.9,
                    help="Pose passes if >= this fraction of DETECTED frames are "
                         "fully inside (default 0.90).")
    ap.add_argument("--detect-frac", type=float, default=0.5,
                    help="Pose is inconclusive if the board is detected in fewer "
                         "than this fraction of analyzed frames (default 0.50).")
    # --- board / hole detection tuning ---
    ap.add_argument("--hole-kernel", type=int, default=41,
                    help="Black-hat kernel (px), must exceed the hole diameter "
                         "(default 41).")
    ap.add_argument("--hole-min-area", type=float, default=100.0,
                    help="Min hole blob area in px (default 100).")
    ap.add_argument("--hole-max-area", type=float, default=3000.0,
                    help="Max hole blob area in px (default 3000).")
    ap.add_argument("--hole-min-circ", type=float, default=0.6,
                    help="Min hole circularity (default 0.6).")
    ap.add_argument("--rect-tol", type=float, default=0.35,
                    help="Max 'rectangle score' for the 4 holes: lower rejects "
                         "spurious/asymmetric sets more aggressively (default 0.35).")
    ap.add_argument("--edge-drop", type=float, default=20.0,
                    help="Intensity drop (0-255) from panel to background that "
                         "marks a board edge in the outward scan (default 20).")
    ap.add_argument("--debug-dir", default=None,
                    help="If set, save one annotated frame per checked pose here.")
    args = ap.parse_args()

    files = collect_images(args.dirs)
    if len(files) < 2:
        sys.exit("At least 2 images are required.")

    offsets, abs_start, source = build_time_index(files, args.dirs, fps=1.0)
    print(f"Images found: {len(files)}   Time source: {source}")

    poses, diffs = segment_poses(files, args.threshold, args.min_move_frames,
                                 args.min_pose_frames, args.downscale)
    if not poses:
        sys.exit("No poses detected.")

    long_poses = [(a, b) for (a, b) in poses if (b - a) >= args.min_frames]
    print(f"Poses detected: {len(poses)}   "
          f"with >= {args.min_frames} frames: {len(long_poses)}")
    print()

    debug_dir = None
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"{'#':>3}  {'start':<10} {'end':<10} {'frames':>6}  "
          f"{'det':>4} {'inside':>6}  verdict")
    print("-" * 90)

    results = []
    for k, (a, b) in enumerate(long_poses, start=1):
        pose_files = [diffs[idx][1] for idx in range(a, b)]
        sample = pose_files[::args.step] or pose_files

        n_analyzed = len(sample)
        n_detected = 0
        n_inside = 0
        clip_tally = {}
        first_annot = None

        for f in sample:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            hole = detect_holes(gray, args.hole_kernel, args.hole_min_area,
                                args.hole_max_area, args.hole_min_circ,
                                args.rect_tol)
            if hole is None:
                continue
            n_detected += 1
            pts, ctr = hole
            ext = board_extent(gray, ctr, args.edge_drop)
            inside, sides = board_inside(gray, ctr, ext, args.margin_px)
            if inside:
                n_inside += 1
            for s in sides:
                clip_tally[s] = clip_tally.get(s, 0) + 1
            if debug_dir is not None and first_annot is None:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for (px, py) in pts.astype(int):
                    cv2.circle(vis, (px, py), 4, (0, 165, 255), -1)
                col = (0, 200, 0) if inside else (0, 0, 255)
                if ext["hx"] is not None and ext["hy"] is not None:
                    cx, cy = int(ctr[0]), int(ctr[1])
                    hx, hy = int(ext["hx"]), int(ext["hy"])
                    cv2.rectangle(vis, (cx - hx, cy - hy), (cx + hx, cy + hy), col, 2)
                first_annot = (f.stem, vis)

        # verdict
        if n_analyzed == 0 or n_detected < args.detect_frac * n_analyzed:
            verdict = "INCONCLUSIVE (holes rarely detected)"
        else:
            frac_inside = n_inside / n_detected
            if frac_inside >= args.inside_frac:
                verdict = "INSIDE"
            else:
                worst = "".join(sorted(clip_tally, key=lambda s: -clip_tally[s]))
                verdict = f"CLIPPED [{worst}]"

        off_a = offsets[pose_files[0].name]
        off_b = offsets[pose_files[-1].name]
        print(f"{k:>3}  {fmt_clock(abs_start, off_a):<10} "
              f"{fmt_clock(abs_start, off_b):<10} {b - a:>6}  "
              f"{n_detected:>2}/{n_analyzed:<2} {n_inside:>3}/{max(n_detected,1):<2}  "
              f"{verdict}")

        if debug_dir is not None and first_annot is not None:
            out = debug_dir / f"pose{k:02d}_{first_annot[0]}.png"
            cv2.imwrite(str(out), first_annot[1])

        results.append((k, verdict))

    print("=" * 90)
    inside_ok = [k for k, v in results if v == "INSIDE"]
    print(f"Poses fully inside: {len(inside_ok)} / {len(long_poses)}  "
          f"-> {inside_ok}")
    print()
    print("NOTES:")
    print("  - 'det'    = frames where the 4 holes were detected / frames analyzed.")
    print("  - 'inside' = detected frames whose board rectangle is fully in-frame.")
    print("  - CLIPPED [sides]: L/R/U/D = which image edge(s) the board crosses.")
    print("  - Tune with --edge-drop (panel<->background contrast) and the")
    print("    --hole-* flags; verify with --debug-dir (orange = holes, green box =")
    print("    inside, red box = clipped).")
    print("  - Orientation (upright 100x70 vs upside-down 70x100) needs no flag:")
    print("    the extent is measured from the board centre by symmetric edge scan.")


if __name__ == "__main__":
    main()
