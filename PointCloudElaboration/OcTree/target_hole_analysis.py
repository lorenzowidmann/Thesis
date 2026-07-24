"""Four-circle calibration target: extract a Livox CustomMsg bag and assess
whether the LiDAR resolves each hole edge well enough for a reliable RANSAC
circle fit (L2V2T2Calib / lvt2calib board).

Two phases (see --help):

PHASE 1  extraction
    Reuses octree.rosbag_loader.load_db3 (pure stdlib+numpy CDR decoder, no ROS
    install) to read every livox_ros_driver2/msg/CustomMsg message on --topic
    and merge all scans into one cloud (stationary capture -> accumulation only
    raises density, no drift). Saves .npy + .csv and prints message/point
    counts. Gates on a suspiciously low point count before phase 2 (the
    near-zero "no-return" points a Livox emits are dropped and reported).

PHASE 2  hole-edge resolution
    1. Isolate the target plane by RANSAC (reusing octree.smoothing.
       fit_plane_ransac / extract_planes) -- among the fitted planes, pick the
       one that actually contains the four holes.
    2. Project onto the plane's PCA basis (octree.smoothing.plane_basis /
       project_to_plane).
    3. Auto-detect the four holes as enclosed voids, then register the detected
       centres onto the nominal centres (2D rigid fit) so points land in the
       board-local frame. Nominal diameter/centres are CLI flags.
    4. Flying-/mixed-pixel rejection: a candidate edge point whose perpendicular
       distance from the board plane exceeds --plane-tol is a ghost point at an
       intermediate range between the board and the background behind it -- it
       is excluded and counted, not treated as an edge point.
    5. Per hole: count surviving points in the r +/- --annulus-tol edge band,
       their angular distribution about the centre, and the local k-NN point
       spacing; emit a sufficient / insufficient verdict with numbers.

Pure numpy (+ optional matplotlib only for --plot), matching the OcTree module.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from octree.rosbag_loader import load_db3, _SUPPORTED_TYPES
from octree.smoothing import (
    extract_planes, plane_basis, project_to_plane, _label_components,
)

import sqlite3


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _parse_centres(text: str) -> np.ndarray:
    """'x0,y0;x1,y1;...' metres -> (N, 2) float array."""
    pts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = (float(t) for t in chunk.split(","))
        pts.append((x, y))
    return np.asarray(pts, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract a Livox CustomMsg bag and assess four-hole target resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O -------------------------------------------------------------------- #
    p.add_argument("--db3", default=None, help="Path to the rosbag2 .db3 file (phase 1)")
    p.add_argument("--topic", default=None,
                   help="LiDAR topic (auto-detected if the bag has a single supported one)")
    p.add_argument("--out-dir", default=".",
                   help="Directory for the extracted cloud (.npy / .csv) and the plot")
    p.add_argument("--stem", default="target_cloud", help="Basename for the extracted cloud files")
    p.add_argument("--from-npy", default=None,
                   help="Skip extraction; load an already-saved (N,3) .npy cloud instead")
    p.add_argument("--extract-only", action="store_true", help="Run phase 1 (extraction) and stop")

    # phase 1 sanity --------------------------------------------------------- #
    p.add_argument("--min-range", type=float, default=0.1,
                   help="Drop points closer than this (Livox no-return points sit near 0 m)")
    p.add_argument("--min-points", type=int, default=50_000,
                   help="Abort before phase 2 if fewer valid points than this were extracted")

    # target geometry (nominal, from CAD / lvt2calib) ------------------------ #
    p.add_argument("--hole-diameter", type=float, default=0.13,
                   help="Nominal hole diameter in metres")
    p.add_argument("--hole-centres", default="-0.15,0.15;0.15,0.15;0.15,-0.15;-0.15,-0.15",
                   help="Nominal hole centres 'x,y;...' in the board-local frame (metres), "
                        "board centre at (0,0)")
    p.add_argument("--board-size", default="1.00,0.70",
                   help="Nominal board width,height in metres (for cropping / the plot)")

    # plane isolation -------------------------------------------------------- #
    p.add_argument("--range-min", type=float, default=None,
                   help="Optional pass-through: keep only points with range >= this (metres)")
    p.add_argument("--range-max", type=float, default=None,
                   help="Optional pass-through: keep only points with range <= this (metres)")
    p.add_argument("--ransac-threshold", type=float, default=0.02,
                   help="Plane-fit inlier distance in metres")
    p.add_argument("--ransac-iters", type=int, default=800, help="RANSAC hypotheses per plane")
    p.add_argument("--max-planes", type=int, default=6, help="Candidate planes to fit and test")
    p.add_argument("--fit-cap", type=int, default=40_000,
                   help="Max points used for RANSAC hypotheses (inliers recomputed on all)")
    p.add_argument("--seed", type=int, default=0, help="RANSAC RNG seed (deterministic)")

    # flying-pixel / edge counting ------------------------------------------ #
    p.add_argument("--plane-tol", type=float, default=0.02,
                   help="Board-surface half-thickness (metres): a candidate farther than this "
                        "from the plane is a flying/mixed pixel and is excluded")
    p.add_argument("--fly-max", type=float, default=0.35,
                   help="Max |perpendicular offset| (metres) still associated with a hole "
                        "(beyond this a point is plain background, not a hole ghost)")
    p.add_argument("--annulus-tol", type=float, default=0.01,
                   help="Edge-band half-width around the nominal hole radius (metres)")
    p.add_argument("--min-edge-points", type=int, default=8,
                   help="Verdict threshold: minimum edge points per hole")
    p.add_argument("--min-sectors", type=int, default=6,
                   help="Verdict threshold: minimum occupied 30-deg angular sectors (of 12)")
    p.add_argument("--knn-k", type=int, default=2, help="k for the local k-NN spacing estimate")
    p.add_argument("--local-radius", type=float, default=0.15,
                   help="Radius (metres) of the board patch around each hole used for k-NN spacing")

    p.add_argument("--plot", default=None,
                   help="Write a 2D projected-cloud plot with holes highlighted to this path (PNG)")
    return p


# --------------------------------------------------------------------------- #
# Phase 1                                                                      #
# --------------------------------------------------------------------------- #
def _count_messages(path: str, topic: str | None) -> tuple[int, str]:
    """(#messages, resolved topic) for the chosen supported LiDAR topic."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, type FROM topics WHERE type IN (?, ?)", _SUPPORTED_TYPES)
        cands = cur.fetchall()
        if topic is not None:
            cands = [c for c in cands if c[1] == topic]
        if len(cands) != 1:
            names = ", ".join(f"{c[1]} [{c[2]}]" for c in cands) or "none"
            raise SystemExit(f"Need exactly one LiDAR topic (got: {names}); pass --topic")
        topic_id, name, _ = cands[0]
        cur.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (topic_id,))
        return int(cur.fetchone()[0]), name
    finally:
        conn.close()


def phase1_extract(args) -> np.ndarray:
    if not args.db3:
        raise SystemExit("Phase 1 needs --db3 PATH (or use --from-npy to skip extraction)")
    n_msgs, topic = _count_messages(args.db3, args.topic)
    print(f"[extract] bag: {args.db3}")
    print(f"[extract] topic: {topic}  ({n_msgs} CustomMsg messages)")

    pc = load_db3(args.db3, topic=topic)
    pts = np.asarray(pc.points, dtype=np.float64)
    rng = np.linalg.norm(pts, axis=1)
    near_zero = int((rng < args.min_range).sum())
    valid = pts[rng >= args.min_range]
    print(f"[extract] merged points: {len(pts):,}")
    print(f"[extract] dropped near-zero (< {args.min_range} m, no-return): {near_zero:,}")
    print(f"[extract] valid points: {len(valid):,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"{args.stem}.npy"
    csv = out_dir / f"{args.stem}.csv"
    np.save(npy, valid)
    np.savetxt(csv, valid, fmt="%.6f", delimiter=",", header="x,y,z", comments="")
    print(f"[extract] saved: {npy}")
    print(f"[extract] saved: {csv}")

    if len(valid) < args.min_points:
        raise SystemExit(
            f"[extract] ABORT: only {len(valid):,} valid points (< --min-points "
            f"{args.min_points:,}). Suspiciously low for a stationary capture -- check the "
            f"topic/decoder before trusting phase 2."
        )
    return valid


# --------------------------------------------------------------------------- #
# Phase 2 helpers                                                              #
# --------------------------------------------------------------------------- #
def detect_holes(u, v, cell, radius_nom, occ_min=2):
    """Find enclosed voids on the projected plane that look like holes.

    Returns a list of (u, v, est_radius, n_cells), largest first. A hole shows
    up as an empty region fully enclosed by occupied board cells, with an
    equivalent radius near the nominal one.
    """
    iu = np.floor((u - u.min()) / cell).astype(int)
    iv = np.floor((v - v.min()) / cell).astype(int)
    grid = np.zeros((iv.max() + 1, iu.max() + 1), int)
    np.add.at(grid, (iv, iu), 1)
    occ = grid >= occ_min
    labels, ncomp = _label_components(~occ)

    holes = []
    for comp in range(ncomp):
        cells = np.argwhere(labels == comp)
        rows, cols = cells[:, 0], cells[:, 1]
        if (rows == 0).any() or (rows == grid.shape[0] - 1).any() \
                or (cols == 0).any() or (cols == grid.shape[1] - 1).any():
            continue  # touches the border -> open, not an enclosed hole
        est_r = np.sqrt(len(cells) * cell * cell / np.pi)
        if 0.4 * radius_nom < est_r < 1.6 * radius_nom:
            cu = u.min() + (cols.mean() + 0.5) * cell
            cv = v.min() + (rows.mean() + 0.5) * cell
            holes.append((cu, cv, est_r, len(cells)))
    holes.sort(key=lambda t: -t[3])
    return holes


def rigid_fit_2d(src, dst):
    """Best 2D rotation+translation mapping src -> dst (no scaling). Returns (R, t)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:            # reflection guard
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cd - R @ cs
    return R, t


def match_centres(detected, nominal):
    """Order `detected` (M,2) to correspond to `nominal` (4,2) by angle about the
    respective centroids, choosing the rotation offset with least residual."""
    det = np.asarray(detected, float)
    nom = np.asarray(nominal, float)
    dc, nc = det.mean(0), nom.mean(0)
    da = np.argsort(np.arctan2(det[:, 1] - dc[1], det[:, 0] - dc[0]))
    na = np.argsort(np.arctan2(nom[:, 1] - nc[1], nom[:, 0] - nc[0]))
    det_o = det[da]
    nom_o = nom[na]
    best = None
    for shift in range(len(nom_o)):
        cand = np.roll(det_o, shift, axis=0)
        R, t = rigid_fit_2d(cand, nom_o)
        resid = np.linalg.norm((cand @ R.T + t) - nom_o, axis=1).mean()
        if best is None or resid < best[0]:
            best = (resid, cand, nom_o, R, t)
    return best  # (resid, det_ordered, nom_ordered, R, t)


def knn_spacing(pts2d, k):
    """Median k-th nearest-neighbour distance among 2D points (brute force)."""
    n = len(pts2d)
    if n <= k:
        return float("nan")
    if n > 4000:                        # cap the O(n^2) work
        pts2d = pts2d[np.random.default_rng(0).choice(n, 4000, replace=False)]
        n = 4000
    d = np.linalg.norm(pts2d[:, None, :] - pts2d[None, :, :], axis=2)
    d.sort(axis=1)
    return float(np.median(d[:, k]))    # column 0 is self (0.0)


# --------------------------------------------------------------------------- #
# Phase 2                                                                      #
# --------------------------------------------------------------------------- #
def phase2_analyse(pts, args):
    nominal = _parse_centres(args.hole_centres)
    radius = args.hole_diameter / 2.0
    bw, bh = (float(x) for x in args.board_size.split(","))
    print(f"\n[geometry] hole diameter {args.hole_diameter} m (radius {radius:.3f} m), "
          f"{len(nominal)} nominal centres, board {bw}x{bh} m")

    # optional pass-through crop by range
    rng = np.linalg.norm(pts, axis=1)
    mask = np.ones(len(pts), bool)
    if args.range_min is not None:
        mask &= rng >= args.range_min
    if args.range_max is not None:
        mask &= rng <= args.range_max
    work = pts[mask]
    print(f"[plane] points considered: {len(work):,}"
          + ("" if mask.all() else f" (range-cropped from {len(pts):,})"))

    # RANSAC candidate planes on a subsample, inliers recomputed on all
    rs = np.random.default_rng(args.seed)
    fit_pts = work if len(work) <= args.fit_cap \
        else work[rs.choice(len(work), args.fit_cap, replace=False)]
    planes = extract_planes(
        fit_pts, threshold=args.ransac_threshold, iters=args.ransac_iters,
        seed=args.seed, max_planes=args.max_planes, min_inliers=2000,
    )
    if not planes:
        raise SystemExit("[plane] no plane found -- widen --ransac-threshold or the range crop")

    # pick the plane that actually contains the four holes
    best = None
    for pi, pl in enumerate(planes):
        inl_mask = np.abs((work - pl.point) @ pl.normal) < args.ransac_threshold
        inl = work[inl_mask]
        if len(inl) < 2000:
            continue
        o, eu, ev, n = plane_basis(pl.normal, inl)
        u, v, _ = project_to_plane(inl, o, eu, ev, n)
        holes = detect_holes(u, v, cell=0.01, radius_nom=radius)
        score = len(holes[:len(nominal)])
        print(f"[plane {pi}] n={pl.normal.round(3)} inliers={len(inl):,} "
              f"hole-like voids={len(holes)} -> using top {score}")
        if best is None or score > best[0]:
            best = (score, pl, o, eu, ev, n, holes)
        if score >= len(nominal):
            break
    score, pl, o, eu, ev, n, holes = best
    if score < len(nominal):
        raise SystemExit(
            f"[plane] found only {score} hole-like voids (need {len(nominal)}). "
            f"Try --range-min/--range-max to crop to the board, or adjust --ransac-threshold."
        )
    detected = np.array([[h[0], h[1]] for h in holes[:len(nominal)]])
    print(f"[plane] selected: normal {n.round(3)}, centroid range {np.linalg.norm(o):.2f} m")

    # register detected hole centres onto the nominal board frame
    resid, det_ord, nom_ord, R, t = match_centres(detected, nominal)
    print(f"[register] detected hole centres -> nominal, mean residual {resid*1000:.1f} mm")
    for (du, dv), (nx, ny) in zip(det_ord, nom_ord):
        print(f"[register]   det(u={du:+.3f},v={dv:+.3f}) -> nom({nx:+.2f},{ny:+.2f})")

    # transform ALL working points into the board-local frame (x,y) + keep perp d
    u_all, v_all, d_all = project_to_plane(work, o, eu, ev, n)
    xy = np.column_stack([u_all, v_all]) @ R.T + t     # board-local x,y
    x, y = xy[:, 0], xy[:, 1]

    # keep only what is near the board (in-plane box + a depth window toward
    # background). The box is symmetric and sized to the LARGER board dimension
    # so the full board is kept whichever way the registration oriented it (the
    # 4-hole square is symmetric and can't fix the 100-vs-70 cm axis) -- the
    # holes sit at +/-0.15 m, well inside, so this doesn't change any hole count.
    half = max(bw, bh) / 2 + 0.10
    on_region = (np.abs(x) <= half) & (np.abs(y) <= half) \
        & (np.abs(d_all) <= args.fly_max)
    x, y, d = x[on_region], y[on_region], d_all[on_region]
    surface = np.abs(d) <= args.plane_tol
    print(f"[board] points in board region: {on_region.sum():,} "
          f"(surface |d|<= {args.plane_tol} m: {int(surface.sum()):,})")

    board_xy = np.column_stack([x[surface], y[surface]])   # genuine board-surface pts

    # per-hole analysis ------------------------------------------------------ #
    reports = []
    n_sectors = 12
    for idx, (cx, cy) in enumerate(nom_ord):
        dx, dy = x - cx, y - cy
        rr = np.hypot(dx, dy)
        in_band = (rr >= radius - args.annulus_tol) & (rr <= radius + args.annulus_tol)

        edge = in_band & surface                       # genuine edge points
        fly = in_band & ~surface                       # off-plane ghosts in the band
        inside = (rr <= radius - args.annulus_tol)
        fly_inside = int((inside & ~surface).sum())     # ghosts projected into the hole

        ang = np.arctan2(dy[edge], dx[edge])
        sector = np.floor((ang + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
        occ_sectors = len(np.unique(sector))
        if occ_sectors:
            filled = np.zeros(n_sectors, bool)
            filled[np.unique(sector)] = True
            gaps = np.diff(np.where(np.r_[filled, filled])[0])
            max_gap_deg = (gaps.max() if len(gaps) else n_sectors) * (360 / n_sectors)
        else:
            max_gap_deg = 360.0

        # local spacing on the board surface around this hole (exclude hole interior)
        near = np.hypot(board_xy[:, 0] - cx, board_xy[:, 1] - cy) <= args.local_radius
        ring = near & (np.hypot(board_xy[:, 0] - cx, board_xy[:, 1] - cy) >= radius - args.annulus_tol)
        spacing = knn_spacing(board_xy[ring], args.knn_k)

        ok = (int(edge.sum()) >= args.min_edge_points) and (occ_sectors >= args.min_sectors)
        reports.append(dict(
            idx=idx + 1, cx=cx, cy=cy, edge=int(edge.sum()), fly=int(fly.sum()),
            fly_inside=fly_inside, sectors=occ_sectors, max_gap_deg=max_gap_deg,
            spacing=spacing, ok=ok,
        ))

    _print_report(reports, args, radius)
    if args.plot:
        _make_plot(x, y, d, surface, nom_ord, radius, args, reports)
    return reports


def _print_report(reports, args, radius):
    print("\n" + "=" * 78)
    print("HOLE-EDGE RESOLUTION REPORT")
    print(f"  nominal radius {radius:.3f} m | edge band +/- {args.annulus_tol*100:.1f} cm | "
          f"plane tol {args.plane_tol*100:.1f} cm")
    print(f"  verdict thresholds: >= {args.min_edge_points} edge pts AND "
          f">= {args.min_sectors}/12 angular sectors")
    print("=" * 78)
    hdr = f"{'hole':>4} {'centre(x,y)':>16} {'edge':>5} {'flyBand':>7} {'flyIn':>6} " \
          f"{'sect/12':>7} {'maxGap':>7} {'k-NN mm':>8}  verdict"
    print(hdr)
    for r in reports:
        sp = "  nan" if np.isnan(r["spacing"]) else f"{r['spacing']*1000:6.1f}"
        verdict = "SUFFICIENT" if r["ok"] else "INSUFFICIENT"
        print(f"{r['idx']:>4} ({r['cx']:+.2f},{r['cy']:+.2f})".ljust(22)
              + f"{r['edge']:>5} {r['fly']:>7} {r['fly_inside']:>6} "
              f"{r['sectors']:>5}/12 {r['max_gap_deg']:>5.0f}d {sp:>8}  {verdict}")
    n_ok = sum(r["ok"] for r in reports)
    print("-" * 78)
    print(f"  {n_ok}/{len(reports)} holes SUFFICIENT for a reliable RANSAC circle fit.")
    tot_fly = sum(r["fly"] + r["fly_inside"] for r in reports)
    print(f"  flying/mixed pixels excluded (|d| > {args.plane_tol} m, criterion: perpendicular "
          f"offset\n  from the board plane -> intermediate range toward the background): "
          f"{tot_fly} total.")
    print("=" * 78)


def _make_plot(x, y, d, surface, centres, radius, args, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(x[surface], y[surface], s=1, c="#3b6", label="board surface", rasterized=True)
    ax.scatter(x[~surface], y[~surface], s=2, c="#d33", alpha=0.5,
               label="flying/mixed pixel", rasterized=True)
    th = np.linspace(0, 2 * np.pi, 200)
    for (cx, cy), r in zip(centres, reports):
        ax.plot(cx + radius * np.cos(th), cy + radius * np.sin(th), "k-", lw=1)
        ax.plot(cx + (radius + args.annulus_tol) * np.cos(th),
                cy + (radius + args.annulus_tol) * np.sin(th), "k:", lw=0.6)
        ax.plot(cx + (radius - args.annulus_tol) * np.cos(th),
                cy + (radius - args.annulus_tol) * np.sin(th), "k:", lw=0.6)
        col = "green" if r["ok"] else "red"
        ax.annotate(f"#{r['idx']}\n{r['edge']}pt", (cx, cy), color=col,
                    ha="center", va="center", fontsize=9, fontweight="bold")
    # auto limits from the real point bounding box on the plane (small margin),
    # so the whole board and its physical edges show — no fixed xlim/ylim.
    mx = 0.03
    ax.set_xlim(float(x.min()) - mx, float(x.max()) + mx)
    ax.set_ylim(float(y.min()) - mx, float(y.max()) + mx)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, board frame)")
    ax.set_ylabel("y (m, board frame)")
    ax.set_title("Target plane projection - holes and edge points")
    ax.legend(loc="upper right", markerscale=6)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=140)
    print(f"[plot] wrote {args.plot}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.from_npy:
        pts = np.load(args.from_npy)
        print(f"[load] {len(pts):,} points from {args.from_npy}")
    else:
        pts = phase1_extract(args)
        if args.extract_only:
            return
    phase2_analyse(pts, args)


if __name__ == "__main__":
    main()
