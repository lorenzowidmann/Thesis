"""Quick look at extract_planes.py's output: one distinct colour per detected
plane, grey for unsegmented points.

Loads the *_planes.ply through Easy3D (not a generic PLY reader -- it wrote
the "v:primitive_index" custom vertex property, so it's the one guaranteed to
read it back correctly) and renders with PyVista, same GPU-accelerated
renderer as PointCloudFilterGUI's preview window.

Usage:
    python visualize_planes.py <bag_name>_planes.ply

Venv: C:\\venvs\\planeextraction (already has easy3d + pyvista + numpy).
"""
import argparse
import colorsys
from pathlib import Path

import easy3d
import numpy as np
import pyvista as pv

UNSEGMENTED_COLOR = (140, 140, 140)  # grey, for primitive_index == -1


def id_to_color(i):
    """Deterministic, maximally-spread-out colour per index via golden-angle
    hue stepping -- no palette to run out of, distinguishable even for
    dozens of planes."""
    hue = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("planes_ply", type=Path, help="the *_planes.ply written by extract_planes.py")
    args = ap.parse_args()

    cloud = easy3d.PointCloudIO.load(str(args.planes_ply))
    if cloud is None or cloud.n_vertices() == 0:
        raise SystemExit(f"Easy3D failed to load {args.planes_ply}")

    prop = cloud.get_vertex_property("v:primitive_index", int)
    if prop is None:
        raise SystemExit(
            f"{args.planes_ply} has no 'v:primitive_index' property -- "
            "is this a *_planes.ply from extract_planes.py?")

    pts = cloud.to_numpy()
    idx = np.array(prop.vector(), dtype=np.int64)

    ids = sorted(set(idx.tolist()) - {-1})
    palette = {i: id_to_color(n) for n, i in enumerate(ids)}

    colors = np.empty((len(idx), 3), dtype=np.uint8)
    colors[:] = UNSEGMENTED_COLOR
    for i in ids:
        colors[idx == i] = palette[i]

    n_unseg = int((idx == -1).sum())
    print(f"{len(ids)} plane(s), {len(idx)} points ({n_unseg} unsegmented, grey)")

    poly = pv.PolyData(pts)
    poly["colors"] = colors
    p = pv.Plotter()
    p.add_mesh(poly, scalars="colors", rgb=True, point_size=2, render_points_as_spheres=False)
    p.set_background("white")
    p.show_grid(color="black", xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
                font_size=10, fmt="%.1f")
    p.add_axes()
    p.add_text(f"{len(ids)} planes, {len(idx)} points ({n_unseg} unsegmented, grey)",
               color="black", font_size=10)
    p.show()


if __name__ == "__main__":
    main()
