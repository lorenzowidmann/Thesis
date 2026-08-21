"""STAGE 1B -- merge touching same-class segments, then reject the ones whose
SHAPE is not geometrically plausible for a door or a window.

Everything here operates on the pixel MASK, never on the bounding box. The
bbox is derived and secondary: it is used for cheap ratios (height over frame
height) and for reporting, never to decide, draw or export a detection. That
distinction is not cosmetic -- the first QA pass on real session output showed
that bbox-based reasoning invents relationships the masks do not have (see
`door_edge_window_fraction` below).

Pixel-space only, by design: no LiDAR, no depth, no metric threshold. Measured
on session 9, LiDAR returns nothing off glazing -- 11 of 19 opening detections
in the first three frames got ZERO points, including the largest and most
confident window in the session (a 315x735 px glass wall at conf 0.98-0.99,
zero points in all three frames). A metric gate here would fail hardest on
exactly the class it is meant to validate. The metric height/width check
therefore belongs downstream, on the consensus voxels, where multi-view
accumulation and the floor-plane fit make it meaningful.

Every threshold below was measured on session 9's first three frames, not
guessed. Where a default is on thin evidence, the docstring says so.
"""

import numpy as np

try:                                    # imported as part of the package
    from .table import OTHER_CLASS
except ImportError:                     # imported as a bare module by path
    from table import OTHER_CLASS

# --- defaults, all measured on session 9 frames 1-3 ------------------------
DILATE_PX = 2
# Mask-adjacency distance to the floor segment that still counts as standing
# on the floor. NOT the frame bottom: SAM segments the floor as one large
# region reaching y=H, so a door's bottom edge sits at the wall/floor junction
# in mid-frame and zones.py's `y1 >= H-3` test never fires for an opening
# (0 of 19 detections passed it). Measured door gaps: 1, 31, 40, 48, 101, 114
# (standing on the floor) vs 370, 462 (genuinely floating). 120 splits them,
# and absorbs skirting boards and the shadow band under a door.
FLOOR_TOL_PX = 120
# A floor-to-ceiling glazed wall, as a fraction of frame height, on the MERGED
# region. Session 9's real glass wall measures 0.68 and is the largest opening
# in the session; the next candidates down are 0.57-0.60. THIN EVIDENCE: this
# separates by 0.08 on one session -- re-check on the full 107-frame run.
GLASS_WALL_H_RATIO = 0.60
# A door candidate with a window region abutting this fraction of one vertical
# side is a bay-edge frame/reveal strip, not a door. See
# door_edge_window_fraction() for why containment (the obvious test) does not
# work on a mask partition.
DOOR_EDGE_WINDOW_FRAC = 0.40
DOOR_EDGE_BAND_PX = 25
# Sliver guard only. A pixel width is depth-dependent and therefore weak -- the
# real width gate is the metric one at stage 2. Session 9 door candidates span
# 34-136 px at 4.5-7.4 m.
MIN_DOOR_WIDTH_PX = 40
MIN_REGION_AREA_PX = 1500       # matches classify_openings.py's --sam-min-area


def _structure(dilate_px: int) -> np.ndarray:
    k = 2 * max(1, dilate_px) + 1
    return np.ones((k, k), bool)


def merge_regions(mask: np.ndarray, dilate_px: int = DILATE_PX) -> list[np.ndarray]:
    """Connected components of `mask`, bridging hairline gaps.

    The dilation is used ONLY to decide connectivity: each returned region is
    intersected back with the original mask, so a merge never steals pixels
    that belong to another class. This matters because a mullion often leaves
    a 1-2 px unmasked seam between two SAM fragments of the same physical
    window -- the dilation bridges the seam without absorbing it.
    """
    from scipy import ndimage

    if not mask.any():
        return []
    lab, n = ndimage.label(ndimage.binary_dilation(mask, _structure(dilate_px)))
    out = []
    for i in range(1, n + 1):
        region = (lab == i) & mask
        if region.any():
            out.append(region)
    return out


def floor_gap(region: np.ndarray, floor: np.ndarray) -> int:
    """Median vertical gap, in pixels, from the bottom edge of `region` to the
    first floor pixel below it, measured per column and taken over the columns
    where a floor pixel exists below at all.

    Returns -1 when no column has floor beneath the region (nothing to stand
    on -- the region is above the floor line entirely, e.g. a clerestory).
    """
    if not region.any() or not floor.any():
        return -1
    cols = np.where(region.any(axis=0))[0]
    gaps = []
    for c in cols:
        rows = np.where(region[:, c])[0]
        below = np.where(floor[:, c])[0]
        below = below[below > rows.max()]
        if below.size:
            gaps.append(below.min() - rows.max())
    return int(np.median(gaps)) if gaps else -1


def touches_floor(region: np.ndarray, floor: np.ndarray,
                  tol_px: int = FLOOR_TOL_PX) -> bool:
    g = floor_gap(region, floor)
    return 0 <= g <= tol_px


def door_edge_window_fraction(region: np.ndarray, window_mask: np.ndarray,
                              band_px: int = DOOR_EDGE_BAND_PX) -> tuple[float, float]:
    """(left_fraction, right_fraction) of window pixels in a vertical band
    immediately beside `region`, row by row.

    This replaces the containment/overlap veto, which CANNOT work here:
    labels.npy is a PARTITION, so a frame fragment is never *inside* a window
    mask, it is adjacent to it. Measured on session 9, door-vs-window mask
    overlap ran 0.1-6.3% -- the >50% containment test never fires. It looked
    plausible only on bounding boxes, where the thin door box does sit inside
    the window box.

    What actually separates them is one-sided adjacency: every false-positive
    door candidate in session 9 frames 1-3 had a window abutting ONE vertical
    side at 68-96% (86.6/0.0, 0.0/87.6, 96.6/0.0, 0.0/80.6, ...), because they
    are strips at the EDGE of a window bay -- the reveal between bay and wall.
    The one candidate that looks like a real door scored 0.0/1.9.

    Note this is a one-sided test on purpose. A true mullion INSIDE a bay would
    have window on both sides, but none of session 9's false positives are
    that; requiring both sides vetoes none of them.

    KNOWN RISK: a real door standing immediately beside a window in the same
    wall scores high on one side and is falsely vetoed. Stage 2's multi-view
    consensus is the backstop, and --no-geometry-filter disables this.
    """
    h, w = region.shape
    rows = np.where(region.any(axis=1))[0]
    if rows.size == 0:
        return 0.0, 0.0
    left_total = right_total = 0
    left_win = right_win = 0
    for y in rows:
        xs = np.where(region[y])[0]
        x0, x1 = xs.min(), xs.max()
        lo = max(0, x0 - band_px)
        hi = min(w, x1 + 1 + band_px)
        left_total += x0 - lo
        right_total += hi - (x1 + 1)
        left_win += int(window_mask[y, lo:x0].sum())
        right_win += int(window_mask[y, x1 + 1:hi].sum())
    return (left_win / max(1, left_total), right_win / max(1, right_total))


def _region_record(region: np.ndarray, new_id: int, cls: str, members: list[dict],
                   image_shape, zone_fn) -> dict:
    """Build a segments.json record for a merged region.

    Confidence is the area-weighted mean of the absorbed segments' confidences
    -- a merged window's certainty is what its constituent masks said, weighted
    by how much of the region each contributed.
    """
    ys, xs = np.nonzero(region)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    area = int(region.sum())
    total_area = sum(m["area_px"] for m in members) or 1
    conf = sum(m["confidence"] * m["area_px"] for m in members) / total_area
    rec = {
        "id": new_id,
        "bbox": [x0, y0, x1, y1],
        "centroid_px": [float(xs.mean()), float(ys.mean())],
        "area_px": area,
        "top_class": cls,
        "confidence": float(conf),
        "top_k": [(cls, float(conf))],
        "merged_from": sorted(int(m["id"]) for m in members),
    }
    ih, iw = image_shape
    rec["zone"] = zone_fn(rec, ih, iw)
    return rec


def apply_geometry_filter(labels: np.ndarray, segments: list[dict], zone_fn,
                          dilate_px: int = DILATE_PX,
                          floor_tol_px: int = FLOOR_TOL_PX,
                          glass_wall_h_ratio: float = GLASS_WALL_H_RATIO,
                          door_edge_window_frac: float = DOOR_EDGE_WINDOW_FRAC,
                          door_edge_band_px: int = DOOR_EDGE_BAND_PX,
                          min_door_width_px: int = MIN_DOOR_WIDTH_PX,
                          min_region_area_px: int = MIN_REGION_AREA_PX):
    """Merge and filter one frame's opening detections.

    Returns (new_labels, new_segments, report). `new_labels` is a rewritten
    raster: merged regions carry new ids covering all the pixels they absorbed,
    so the mask shape survives into stage 2 and into the overlay. Rejected
    detections keep their pixels but have their class flipped to OTHER_CLASS,
    with a "rejected" block recording which rule fired -- never silent.

    Windows are merged and filtered BEFORE doors, because the door edge-veto
    needs the surviving window regions to test against. A door can therefore
    never veto a window, only the reverse.
    """
    h, w = labels.shape
    by_id = {int(s["id"]): s for s in segments}
    floor = np.isin(labels, [int(s["id"]) for s in segments if s.get("zone") == "floor"])

    report = {"merged": {}, "rejected": {}, "kept": {}}
    new_labels = labels.copy()
    out_segments = []
    consumed: set[int] = set()      # a record for this id has been emitted or replaced
    absorbed: set[int] = set()      # folded into a KEPT merged region (pixels rewritten)
    next_id = max([int(s["id"]) for s in segments], default=-1) + 1

    def members_of(region):
        ids = [int(i) for i in np.unique(labels[region]) if int(i) in by_id]
        return [by_id[i] for i in ids]

    def reject(region, members, rule, detail):
        """Flip a whole merged region back to 'other', keeping its pixels."""
        for m in members:
            rec = dict(m)
            rec["top_class"] = OTHER_CLASS
            rec["rejected"] = {"was": m["top_class"], "rule": rule, **detail}
            out_segments.append(rec)
            consumed.add(int(m["id"]))
        report["rejected"][rule] = report["rejected"].get(rule, 0) + 1

    # ---- 1. windows: merge, then rule 2 (floor contact => glass-wall height)
    window_mask = np.isin(labels, [i for i, s in by_id.items() if s["top_class"] == "window"])
    kept_window_mask = np.zeros_like(window_mask)
    # Union of every merged window region, INCLUDING those rule 2 goes on to
    # reject. This -- not kept_window_mask -- is what the door edge-veto tests
    # against, because the question it asks is "is this strip at the edge of a
    # glazed bay", and the bay is physically there whether or not the region
    # cleared the glass-wall height test. Measured on session 9 frame 1: using
    # only the survivors let 2 of 3 bay-edge strips through, because the window
    # beside them had just been rejected by rule 2.
    all_window_mask = np.zeros_like(window_mask)
    window_regions = [r for r in merge_regions(window_mask, dilate_px)
                      if r.sum() >= min_region_area_px]
    report["merged"]["window"] = len(window_regions)
    for region in window_regions:
        members = members_of(region)
        if not members:
            continue
        all_window_mask |= region
        ys = np.nonzero(region.any(axis=1))[0]
        h_ratio = (ys.max() - ys.min() + 1) / float(h)
        gap = floor_gap(region, floor)
        if 0 <= gap <= floor_tol_px and h_ratio < glass_wall_h_ratio:
            # Standing on the floor but far too short to be a glazed wall --
            # a sliver of frame, a reflection, or a low panel.
            reject(region, members, "window_floor_contact_not_glass_wall",
                   {"h_ratio": round(h_ratio, 3), "floor_gap_px": gap,
                    "needed_h_ratio": glass_wall_h_ratio})
            continue
        kept_window_mask |= region
        rec = _region_record(region, next_id, "window", members, (h, w), zone_fn)
        rec["geometry"] = {"h_ratio": round(h_ratio, 3), "floor_gap_px": gap,
                           "touches_floor": bool(0 <= gap <= floor_tol_px)}
        out_segments.append(rec)
        new_labels[region] = next_id
        next_id += 1
        consumed.update(int(m["id"]) for m in members)
        absorbed.update(int(m["id"]) for m in members)
        report["kept"]["window"] = report["kept"].get("window", 0) + 1

    # ---- 2. doors: merge, then rules 3a-3d against the SURVIVING windows ----
    door_mask = np.isin(labels, [i for i, s in by_id.items() if s["top_class"] == "door"])
    survivors = []
    for region in merge_regions(door_mask, dilate_px):
        members = members_of(region)
        if not members:
            continue
        if region.sum() < min_region_area_px:
            reject(region, members, "door_below_min_area", {"area_px": int(region.sum())})
            continue
        ys = np.nonzero(region.any(axis=1))[0]
        xs = np.nonzero(region.any(axis=0))[0]
        h_ratio = (ys.max() - ys.min() + 1) / float(h)
        width_px = int(xs.max() - xs.min() + 1)
        gap = floor_gap(region, floor)
        lf, rf = door_edge_window_fraction(region, all_window_mask, door_edge_band_px)
        detail = {"h_ratio": round(h_ratio, 3), "width_px": width_px,
                  "floor_gap_px": gap, "window_left": round(lf, 3),
                  "window_right": round(rf, 3)}

        # 3a -- bay-edge frame fragment
        if max(lf, rf) > door_edge_window_frac:
            reject(region, members, "door_window_edge_adjacent", detail)
            continue
        # 3d -- sliver
        if width_px < min_door_width_px:
            reject(region, members, "door_below_min_width", detail)
            continue
        # 3b -- a door has to start at floor level
        if not (0 <= gap <= floor_tol_px):
            reject(region, members, "door_not_floor_standing", detail)
            continue
        # 3c -- taller than a glazed wall means it is not a door
        if h_ratio >= glass_wall_h_ratio:
            reject(region, members, "door_taller_than_glass_wall", detail)
            continue
        survivors.append((region, members, detail))

    # 3f -- re-merge whatever survived, in case a real door was split and the
    # fragments only became adjacent after their neighbours were removed.
    if survivors:
        surv_mask = np.zeros_like(door_mask)
        for region, _m, _d in survivors:
            surv_mask |= region
        detail_by_pixel = {}
        for region, _m, d in survivors:
            detail_by_pixel[id(region)] = d
        remerged = [r for r in merge_regions(surv_mask, dilate_px) if r.any()]
        report["merged"]["door"] = len(remerged)
        for region in remerged:
            members = members_of(region)
            if not members:
                continue
            ys = np.nonzero(region.any(axis=1))[0]
            xs = np.nonzero(region.any(axis=0))[0]
            gap = floor_gap(region, floor)
            rec = _region_record(region, next_id, "door", members, (h, w), zone_fn)
            rec["geometry"] = {
                "h_ratio": round((ys.max() - ys.min() + 1) / float(h), 3),
                "width_px": int(xs.max() - xs.min() + 1),
                "floor_gap_px": gap, "touches_floor": bool(0 <= gap <= floor_tol_px)}
            out_segments.append(rec)
            new_labels[region] = next_id
            next_id += 1
            consumed.update(int(m["id"]) for m in members)
            absorbed.update(int(m["id"]) for m in members)
            report["kept"]["door"] = report["kept"].get("door", 0) + 1
    else:
        report["merged"]["door"] = 0

    # ---- 3. everything untouched keeps its original record and pixels ------
    for s in segments:
        if int(s["id"]) not in consumed:
            out_segments.append(dict(s))

    # ---- 4. normalise against the raster -----------------------------------
    # The raster is the source of truth, and this pass makes segments.json
    # agree with it by construction rather than by bookkeeping.
    #
    # Why it is needed: a SAM segment is absorbed by PIXEL, not by id. The
    # dilation can split one segment across two merged regions, so the same id
    # can be folded into a kept window AND rejected inside another region --
    # which produced both duplicate ids and ids left in the raster with no
    # record at all (an id in labels.npy but absent from segments.json reads as
    # unlabelled to every downstream consumer).
    #
    # So: keep the first record emitted for each id, invent an 'other' record
    # for any raster id that never got one, drop records whose pixels were
    # entirely absorbed, and recompute bbox/area/centroid for everyone from the
    # pixels they actually still own.
    records: dict[int, dict] = {}
    for rec in out_segments:
        records.setdefault(int(rec["id"]), rec)

    for sid in np.unique(new_labels):
        sid = int(sid)
        if sid < 0 or sid in records:
            continue
        src = by_id.get(sid, {})
        records[sid] = {
            "id": sid,
            "top_class": OTHER_CLASS,
            "confidence": float(src.get("confidence", 0.0)),
            "top_k": [(OTHER_CLASS, float(src.get("confidence", 0.0)))],
            "zone": src.get("zone", "any"),
            "residual_of": {"was": src.get("top_class", OTHER_CLASS),
                            "reason": "pixels not absorbed by any merged region"},
        }
        report["rejected"]["residual_fragment"] = \
            report["rejected"].get("residual_fragment", 0) + 1

    final = []
    for sid, rec in records.items():
        mask = new_labels == sid
        if not mask.any():
            continue                    # fully absorbed by a merge
        ys, xs = np.nonzero(mask)
        rec["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        rec["centroid_px"] = [float(xs.mean()), float(ys.mean())]
        rec["area_px"] = int(mask.sum())
        final.append(rec)

    final.sort(key=lambda r: -r["area_px"])
    return new_labels, final, report
