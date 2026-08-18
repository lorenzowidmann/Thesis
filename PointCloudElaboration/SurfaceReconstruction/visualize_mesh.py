"""Quick look at reconstruct_mesh.py's output .obj, with edges shown so you
can see individual faces (useful for spotting PolyFit's fitted piecewise
quads, e.g. the crossing diagonals visible in the reference-run screenshot).

Usage:
    python visualize_mesh.py <model.obj>

Venv: C:\\venvs\\planeextraction has pyvista already (surfacereconstruction's
venv doesn't -- pyvista isn't needed there, only for this viewer).
"""
import argparse
from pathlib import Path

import pyvista as pv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj", type=Path, help="mesh file from reconstruct_mesh.py (.obj)")
    args = ap.parse_args()

    mesh = pv.read(str(args.obj))
    print(f"{mesh.n_points} vertices, {mesh.n_cells} face(s)")

    p = pv.Plotter()
    p.add_mesh(mesh, color="#c9a86a", show_edges=True, edge_color="black", line_width=1)
    p.set_background("white")
    p.show_grid(color="black", xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
                font_size=10, fmt="%.1f")
    p.add_axes()
    p.add_text(f"{mesh.n_cells} faces -- close window when done", color="black", font_size=10)
    p.show()


if __name__ == "__main__":
    main()
