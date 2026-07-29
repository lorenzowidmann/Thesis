"""
Visualizzatore rapido per file .pcd (formato ASCII), senza dipendenze PCL.
Parsa l'header e i punti a mano, mostra con pyvista.

Uso:
    py view_pcd.py <file.pcd>
    py view_pcd.py <file.pcd> --point-size 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


def read_pcd_ascii(path: Path):
    with path.open("r", encoding="ascii", errors="replace") as fh:
        lines = fh.readlines()

    fields = None
    n_points_declared = None
    data_line_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("FIELDS"):
            fields = stripped.split()[1:]
        elif stripped.startswith("POINTS"):
            n_points_declared = int(stripped.split()[1])
        elif stripped.startswith("DATA"):
            data_format = stripped.split()[1]
            data_line_idx = i + 1
            if data_format != "ascii":
                raise SystemExit(
                    f"Formato dati '{data_format}' non supportato da questo "
                    f"script (solo 'ascii'). File: {path}"
                )
            break

    if fields is None or data_line_idx is None:
        raise SystemExit(f"Header PCD non valido o incompleto: {path}")

    print(f"Campi dichiarati (FIELDS): {fields}")
    print(f"Punti dichiarati (POINTS): {n_points_declared}")

    try:
        xi = fields.index("x")
        yi = fields.index("y")
        zi = fields.index("z")
    except ValueError:
        raise SystemExit(f"Il file non ha campi x/y/z riconoscibili: {fields}")

    pts = []
    n_fields_expected = len(fields)
    bad_lines = 0
    for line in lines[data_line_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != n_fields_expected:
            bad_lines += 1
            continue
        try:
            x, y, z = float(parts[xi]), float(parts[yi]), float(parts[zi])
        except ValueError:
            bad_lines += 1
            continue
        pts.append((x, y, z))

    if bad_lines:
        print(f"ATTENZIONE: {bad_lines} righe scartate (numero di campi non "
              f"corrispondente a FIELDS, o valori non numerici)")

    pts = np.array(pts, dtype=np.float64)
    print(f"Punti effettivamente letti: {len(pts)}")

    if len(pts) == 0:
        raise SystemExit("Nessun punto valido letto dal file.")

    if n_points_declared is not None and len(pts) != n_points_declared:
        print(f"ATTENZIONE: POINTS dichiarava {n_points_declared} ma ne sono "
              f"stati letti {len(pts)} — possibile mismatch di schema/campi "
              f"(controllare FIELDS/SIZE/COUNT nell'header).")

    return pts


def main():
    parser = argparse.ArgumentParser(description="Visualizza un file .pcd ASCII con pyvista.")
    parser.add_argument("pcd_file", type=Path)
    parser.add_argument("--point-size", type=float, default=6.0)
    args = parser.parse_args()

    if not args.pcd_file.exists():
        sys.exit(f"File non trovato: {args.pcd_file}")

    pts = read_pcd_ascii(args.pcd_file)

    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    print()
    print("Bounding box:")
    print(f"  x: [{xmin:+.4f}, {xmax:+.4f}]  larghezza = {xmax - xmin:.4f} m")
    print(f"  y: [{ymin:+.4f}, {ymax:+.4f}]  altezza   = {ymax - ymin:.4f} m")
    print(f"  z: [{zmin:+.4f}, {zmax:+.4f}]  spessore  = {zmax - zmin:.4f} m")
    print()
    print(f"Atteso per la board: larghezza ~1.00 m, altezza ~0.70 m, spessore ~0 (planare)")

    cloud = pv.PolyData(pts)
    plotter = pv.Plotter()
    plotter.add_points(cloud, color="red", point_size=args.point_size,
                       render_points_as_spheres=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.add_title(str(args.pcd_file.name))
    plotter.show()


if __name__ == "__main__":
    main()
