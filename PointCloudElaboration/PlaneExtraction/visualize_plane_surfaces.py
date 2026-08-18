"""Show the planes themselves as flat, colored rectangles -- not the point
cloud. Works on either extract_planes.py's raw *_planes.ply or
dedupe_planes.py's *_dedup.ply (both just need a "v:primitive_index"
property; use whichever you want a surface-level look at).

For each plane id: PCA-fit its own (normal, centroid) from its member
points, project them into the plane's 2D basis, and take the axis-aligned
bounding rectangle in that basis (same "align_to_structure" approach as
OpenStudioModel/fit_planes.py's fit_oriented_rect -- floor/ceiling rectangles
align to world X/Y, wall rectangles align to world-up + the wall's own
horizontal direction, so a sparse/noisy fragment can't come out
diamond-rotated the way an unconstrained minimum-area rectangle would).

Usage:
    python visualize_plane_surfaces.py <bag>_planes.ply
    python visualize_plane_surfaces.py <bag>_planes_dedup.ply

Venv: C:\\venvs\\planeextraction (easy3d + pyvista + numpy).
"""
import argparse
import colorsys
from pathlib import Path

import easy3d
import numpy as np
import pyvista as pv


def id_to_color(i):
    """Same golden-angle hue stepping as visualize_planes.py, kept
    consistent so the same plane looks the same color in both scripts."""
    hue = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def fit_rect(points):
    """PCA normal/centroid, then an axis-aligned rectangle in the plane's
    own 2D basis (world X/Y for floor/ceiling, world-up + horizontal for a
    wall). Returns 4 corner points (3D), CCW in the local basis."""
    centroid = points.mean(axis=0)
    cov = np.cov((points - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n = eigvecs[:, 0]
    n = n / np.linalg.norm(n)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(n[2]) > 0.5:  # floor/ceiling
        u = np.array([1.0, 0.0, 0.0]) - n[0] * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
    else:  # wall
        u = np.cross(n, world_up)
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-6:
            u = np.array([1.0, 0.0, 0.0]) - n[0] * n
            norm_u = np.linalg.norm(u)
        u /= norm_u
        v = world_up.copy()

    rel = points - centroid
    pts2d = np.stack([rel @ u, rel @ v], axis=1)
    xmin, ymin = pts2d.min(axis=0)
    xmax, ymax = pts2d.max(axis=0)
    box2d = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return centroid + box2d[:, 0:1] * u + box2d[:, 1:2] * v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("planes_file", type=Path, help="*_planes.ply or *_planes_dedup.ply")
    ap.add_argument("--min-points", type=int, default=0,
                    help="skip drawing rectangles for planes with fewer points than this "
                         "(0 = draw all; small/noisy/off-axis fragments can produce "
                         "misleadingly large axis-aligned rectangles -- raise this on raw, "
                         "un-deduped extract_planes.py output)")
    args = ap.parse_args()

    cloud = easy3d.PointCloudIO.load(str(args.planes_file))
    if cloud is None or cloud.n_vertices() == 0:
        raise SystemExit(f"Easy3D failed to load {args.planes_file}")

    prop = cloud.get_vertex_property("v:primitive_index", int)
    if prop is None:
        raise SystemExit(f"{args.planes_file} has no 'v:primitive_index' property")

    pts = cloud.to_numpy()
    idx = np.array(prop.vector(), dtype=np.int64)
    all_ids = sorted(set(idx.tolist()) - {-1})
    if not all_ids:
        raise SystemExit("No segmented planes in this file")

    counts = {i: int((idx == i).sum()) for i in all_ids}
    ids = [i for i in all_ids if counts[i] >= args.min_points]
    if len(ids) < len(all_ids):
        print(f"--min-points {args.min_points}: skipping {len(all_ids) - len(ids)} of "
              f"{len(all_ids)} plane(s)")
    if not ids:
        raise SystemExit("No planes left after --min-points filter")

    verts, faces, scalars, colors = [], [], [], []
    for n, i in enumerate(ids):
        corners = fit_rect(pts[idx == i])
        base = n * 4
        verts.append(corners)
        faces.append([4, base, base + 1, base + 2, base + 3])
        r, g, b = id_to_color(n)
        colors.extend([[r, g, b]] * 4)

    mesh = pv.PolyData(np.vstack(verts), np.hstack(faces))
    mesh["colors"] = np.array(colors, dtype=np.uint8)

    print(f"{len(ids)} plane(s)")

    p = pv.Plotter()
    p.add_mesh(mesh, scalars="colors", rgb=True, show_edges=True,
               edge_color="black", line_width=2, opacity=0.9)
    p.set_background("white")
    p.show_grid(color="black", xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
                font_size=10, fmt="%.1f")
    p.add_axes()
    p.add_text(f"{len(ids)} planes (rectangles) -- close window when done",
               color="black", font_size=10)
    p.show()


if __name__ == "__main__":
    main()
