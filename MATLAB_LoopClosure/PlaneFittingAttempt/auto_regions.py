"""Auto-detect corridor/branch regions from a point cloud's top-down
footprint, instead of eyeballing a topdown_view.py plot by hand: builds a
2D occupancy grid, skeletonizes it to a 1-cell-wide network, finds
junction/endpoint nodes, and cuts the skeleton into segments between them
-- one segment per corridor arm. Then asks YOU which segments belong in
the same fit_planes.py box (a real corridor is often split into several
skeleton segments by a side junction, a doorway gap, or plain noise --
only a human looking at the printed list/plot can tell "same corridor" from
"different corridor").

Pipeline:
    occupancy grid (--cell-size cells, largest connected component only,
    small gaps closed) -> skimage.morphology.skeletonize -> networkx graph
    of skeleton cells (8-connectivity) -> nodes with degree != 2 are
    junctions (>=3) or endpoints (1) -> walk the degree-2 chains between
    them to get one path per segment -> segments shorter than
    --min-segment-length are dropped (skeletonization artifacts right at a
    junction, not real corridor arms) -> each kept segment's pixel path
    becomes an axis-aligned XY bounding box, padded by --pad.

Usage:
    python auto_regions.py (--bag <rosbag2_folder> | --pcd <file.pcd>)
        [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--roi xmin,xmax,ymin,ymax,zmin,zmax]
        [--sor] [--declutter]
        [--cell-size 0.3] [--close-iter 2] [--open-iter 2] [--prune-length 1.2]
        [--min-segment-length 1.5] [--pad 1.0]
        [--cache cloud.npy]
        [--no-interactive-merge]
        [--pyvista]
        [--no-show]
        [--out auto_regions.png]

By default a plot window opens (non-blocking) right before the merge
prompt, so you can look at the numbered segments while answering in the
terminal -- --no-show only saves --out instead, for headless/scripted runs.

Interactive merge prompt: type groups of segment indices separated by '|',
e.g. '0,1|2|3,4' merges segments 0+1 into one box, keeps 2 as its own box,
merges 3+4 into another. Empty input keeps every segment as its own box.
Prints ready-to-paste fit_planes.py --roi strings for each resulting box.

Venv: C:\\venvs\\planefit (+ scikit-image, `pip install scikit-image`).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # no Agg backend here on purpose: --show opens a real window
import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from fit_planes import crop_roi, declutter, load_merged_cloud, load_pcd_cloud, statistical_outlier_removal
from topdown_view import show_pyvista

NEIGHBORS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def load_xyz(args):
    if args.cache and Path(args.cache).exists():
        xyz = np.load(args.cache)
        print(f"loaded {len(xyz)} points from cache {args.cache}")
        return xyz

    xyz = load_pcd_cloud(args.pcd) if args.pcd else load_merged_cloud(args.bag, args.topic, args.store)
    if args.roi:
        xyz = crop_roi(xyz, tuple(float(x) for x in args.roi.split(",")))
    if args.sor:
        xyz = statistical_outlier_removal(xyz)
    if args.declutter:
        xyz = declutter(xyz, gap=0.30)

    if args.cache:
        np.save(args.cache, xyz)
        print(f"cached to {args.cache}")
    return xyz


def build_occupancy(xyz, cell_size, close_iter, open_iter):
    """2D XY occupancy grid, largest connected component only (drops
    speckle noise the same way fit_planes.py's declutter does), with small
    gaps closed so real coverage holes (a chair leg's shadow, a sparse
    patch) don't fragment the skeleton into extra spurious segments, and
    small boundary protrusions opened away (a chair, a doorframe reveal, a
    ragged wall edge) so skeletonize doesn't grow a spurious extra branch
    toward every one of them -- skeletonize is very sensitive to boundary
    noise, this is the standard cleanup before it."""
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    ix = np.floor((xyz[:, 0] - x0) / cell_size).astype(np.int64)
    iy = np.floor((xyz[:, 1] - y0) / cell_size).astype(np.int64)
    grid = np.zeros((ix.max() + 1, iy.max() + 1), dtype=bool)
    grid[ix, iy] = True

    struct8 = ndimage.generate_binary_structure(2, 2)
    if close_iter > 0:
        grid = ndimage.binary_closing(grid, structure=struct8, iterations=close_iter)
    if open_iter > 0:
        grid = ndimage.binary_opening(grid, structure=struct8, iterations=open_iter)
    labels, n = ndimage.label(grid, structure=struct8)
    if n > 1:
        sizes = ndimage.sum(grid, labels, range(1, n + 1))
        grid = labels == (int(np.argmax(sizes)) + 1)
    print(f"occupancy grid: {grid.shape[0]}x{grid.shape[1]} cells @ {cell_size}m, "
          f"{int(grid.sum())} occupied ({n} raw component(s), kept largest)")
    return grid, x0, y0


def skeleton_graph(grid):
    skel = skeletonize(grid)
    coords = set(map(tuple, np.argwhere(skel)))
    G = nx.Graph()
    G.add_nodes_from(coords)
    for (i, j) in coords:
        for di, dj in NEIGHBORS8:
            nb = (i + di, j + dj)
            if nb in coords:
                G.add_edge((i, j), nb)
    print(f"skeleton: {G.number_of_nodes()} cell(s), "
          f"{sum(1 for n in G if G.degree(n) == 1)} endpoint(s), "
          f"{sum(1 for n in G if G.degree(n) >= 3)} junction(s)")
    return skel, G


def prune_skeleton(G, prune_cells):
    """Strip short endpoint-terminated spurs from the skeleton graph before
    tracing segments. skeletonize() grows extra short "hairs" off any
    locally wide or irregular patch of the occupancy grid (most visibly at
    junctions, where the blob where corridors meet is wider than the
    corridors themselves) -- these aren't real corridor arms, and left in
    they (a) clutter the segment list with junk entries and (b) turn what
    should be one straight-through corridor into several segments split at
    each fake junction. Removing them lets a real corridor's degree-3
    "junction" collapse back to degree-2 pass-through, so segment tracing
    reconnects it into one piece automatically -- --min-segment-length
    alone can only hide short segments after the fact, it can't undo that
    split. Repeats since removing one spur can drop its parent junction to
    degree 2 (no longer a junction) or degree 1 (now itself a short spur
    one level up), each of which can expose more pruning."""
    changed = True
    n_pruned = 0
    while changed:
        changed = False
        for ep in [n for n in G if G.degree(n) == 1]:
            if ep not in G:
                continue
            path = [ep]
            prev, cur = None, ep
            while G.degree(cur) == 2:
                nxt = next(n for n in G.neighbors(cur) if n != prev)
                path.append(nxt)
                prev, cur = cur, nxt
            # cur is now a junction (>=3), or another endpoint if this whole
            # piece is isolated (degree 1 at both ends) -- only prune spurs
            # that dead-end INTO a junction; an isolated short piece might be
            # a real (if small) disconnected corridor, leave it for
            # --min-segment-length to judge instead.
            if len(path) - 1 < prune_cells and G.degree(cur) >= 3:
                G.remove_nodes_from(path[:-1])
                n_pruned += len(path) - 1
                changed = True
    if n_pruned:
        print(f"pruned {n_pruned} spur cell(s) off junctions "
              f"({G.number_of_nodes()} cell(s) left)")
    return G


def trace_segments(G):
    """Cut the skeleton graph into paths between nodes of interest (degree
    != 2): each path is one candidate corridor arm. A skeleton with no
    junctions/endpoints at all (a closed loop) falls back to the whole
    thing as a single segment."""
    interest = {n for n in G if G.degree(n) != 2}
    if not interest:
        return [list(next(iter(nx.connected_components(G))))] if G.number_of_nodes() else []

    visited_edges = set()
    segments = []
    for start in interest:
        for nbr in G.neighbors(start):
            edge = frozenset((start, nbr))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path = [start, nbr]
            prev, cur = start, nbr
            while cur not in interest:
                nxts = [n for n in G.neighbors(cur) if n != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                visited_edges.add(frozenset((cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
            segments.append(path)
    return segments


def segment_bbox(path, x0, y0, cell_size, pad):
    pts = np.array(path, dtype=float)
    xs = x0 + pts[:, 0] * cell_size
    ys = y0 + pts[:, 1] * cell_size
    length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    return (float(xs.min() - pad), float(xs.max() + pad),
            float(ys.min() - pad), float(ys.max() + pad), length)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, help="rosbag2 folder (raw /cloud_registered)")
    ap.add_argument("--pcd", type=Path, help="load xyz from a .pcd instead of a bag")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE")
    ap.add_argument("--roi", help="xmin,xmax,ymin,ymax,zmin,zmax crop (metres)")
    ap.add_argument("--sor", action="store_true")
    ap.add_argument("--declutter", action="store_true")
    ap.add_argument("--cell-size", type=float, default=0.3, help="occupancy grid cell (m)")
    ap.add_argument("--close-iter", type=int, default=2,
                    help="binary_closing iterations to bridge small coverage gaps (0 = off)")
    ap.add_argument("--open-iter", type=int, default=2,
                    help="binary_opening iterations to shave off small boundary "
                         "protrusions (furniture, ragged wall edges) before "
                         "skeletonizing -- the main lever against a wobbly/branchy "
                         "skeleton and junction over-segmentation (0 = off)")
    ap.add_argument("--prune-length", type=float, default=1.2,
                    help="strip skeleton spurs shorter than this (m) that dead-end into a "
                         "junction, BEFORE tracing segments -- the real fix for junction "
                         "clutter: a spur removed can turn a fake junction back into a "
                         "straight pass-through, reconnecting a corridor that was "
                         "artificially split there (0 = off)")
    ap.add_argument("--min-segment-length", type=float, default=1.5,
                    help="safety net AFTER pruning: drop any segment still shorter than "
                         "this (m) -- e.g. a short isolated piece not touching a junction, "
                         "which pruning deliberately leaves alone")
    ap.add_argument("--pad", type=float, default=1.0,
                    help="margin (m) added around each segment's bounding box, so the "
                         "box reaches the real walls instead of stopping at the skeleton")
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--no-interactive-merge", action="store_true",
                    help="skip the terminal prompt, print/plot every segment as its own box")
    ap.add_argument("--pyvista", action="store_true",
                    help="also open a 3D window with the final (merged) boxes as wireframes")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--no-show", action="store_true",
                    help="don't open the plot in a window, only save --out -- useful for "
                         "headless/automated runs; without it a window opens and stays "
                         "up (non-blocking) while the merge prompt waits in the terminal")
    ap.add_argument("--out", type=Path, default=Path("auto_regions.png"))
    args = ap.parse_args()

    if bool(args.bag) == bool(args.pcd):
        raise SystemExit("pass exactly one of --bag or --pcd")

    xyz = load_xyz(args)
    grid, x0, y0 = build_occupancy(xyz, args.cell_size, args.close_iter, args.open_iter)
    skel, G = skeleton_graph(grid)
    if args.prune_length > 0:
        G = prune_skeleton(G, args.prune_length / args.cell_size)
        print(f"after pruning: {sum(1 for n in G if G.degree(n) == 1)} endpoint(s), "
              f"{sum(1 for n in G if G.degree(n) >= 3)} junction(s)")
    raw_segments = trace_segments(G)

    segments = []  # list of (bbox, length, path)
    for path in raw_segments:
        xmin, xmax, ymin, ymax, length = segment_bbox(path, x0, y0, args.cell_size, args.pad)
        if length < args.min_segment_length:
            continue
        segments.append({"bbox": (xmin, xmax, ymin, ymax), "length": length, "path": path})
    segments.sort(key=lambda s: -s["length"])

    zmin, zmax = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    print(f"\n{len(raw_segments)} raw segment(s), {len(segments)} kept "
          f"(>= --min-segment-length {args.min_segment_length}m):")
    for i, s in enumerate(segments):
        xmin, xmax, ymin, ymax = s["bbox"]
        m = ((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
             (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax))
        print(f"  [{i}] length~{s['length']:.1f}m, {int(m.sum())} points, "
              f"bbox X {xmin:.1f}..{xmax:.1f}  Y {ymin:.1f}..{ymax:.1f}")

    # plot: occupancy + skeleton + numbered segment boxes
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(grid.T, origin="lower",
              extent=[x0, x0 + grid.shape[0] * args.cell_size,
                      y0, y0 + grid.shape[1] * args.cell_size],
              cmap="Greys", alpha=0.4)
    skel_pruned = np.zeros_like(skel)
    if G.number_of_nodes():
        pts = np.array(list(G.nodes()))
        skel_pruned[pts[:, 0], pts[:, 1]] = True
    ys_sk, xs_sk = np.nonzero(skel_pruned.T)
    ax.scatter(x0 + xs_sk * args.cell_size, y0 + ys_sk * args.cell_size,
               s=1, c="blue", label="skeleton")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(segments), 1)))
    for i, s in enumerate(segments):
        xmin, xmax, ymin, ymax = s["bbox"]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              fill=False, edgecolor=colors[i], linewidth=2)
        ax.add_patch(rect)
        ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, str(i),
                color=colors[i], fontsize=14, weight="bold", ha="center")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"auto-detected segments ({len(segments)})")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\nsaved {args.out}")
    if not args.no_show:
        try:
            plt.show(block=False)
            plt.pause(0.5)  # force the window to actually draw before the input() prompt below
        except Exception as e:
            print(f"couldn't open a plot window ({e}) -- open {args.out} by hand instead")

    groups = [[i] for i in range(len(segments))]
    if segments and not args.no_interactive_merge:
        print("\nQuali segmenti appartengono allo stesso box? Gruppi separati da '|', "
              "indici separati da ',' -- es. '0,1|2|3,4'. Invio vuoto = ogni segmento "
              "resta un box separato.")
        answer = input("> ").strip()
        if answer:
            groups = [[int(x) for x in g.split(",")] for g in answer.split("|")]

    print("\n--roi pronti per fit_planes.py:")
    for gi, group in enumerate(groups):
        xmin = min(segments[i]["bbox"][0] for i in group)
        xmax = max(segments[i]["bbox"][1] for i in group)
        ymin = min(segments[i]["bbox"][2] for i in group)
        ymax = max(segments[i]["bbox"][3] for i in group)
        print(f'  box{gi} (segmenti {group}): --roi="{xmin:.2f},{xmax:.2f},'
              f'{ymin:.2f},{ymax:.2f},{zmin:.2f},{zmax:.2f}"')

    if args.pyvista and segments:
        regions = []
        for gi, group in enumerate(groups):
            xmin = min(segments[i]["bbox"][0] for i in group)
            xmax = max(segments[i]["bbox"][1] for i in group)
            ymin = min(segments[i]["bbox"][2] for i in group)
            ymax = max(segments[i]["bbox"][3] for i in group)
            regions.append((f"box{gi}", xmin, xmax, ymin, ymax))
        fake_args = argparse.Namespace(vmin=None, vmax=None, point_size=args.point_size,
                                        top_view=False)
        show_pyvista(xyz, regions, fake_args, title="auto_regions")


if __name__ == "__main__":
    main()
