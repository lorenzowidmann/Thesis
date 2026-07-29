"""
Genera i template PCD per LVT2Calib con la geometria della board reale.

Board: 100 x 70 cm, 4 fori circolari di diametro 13 cm (raggio 0.065 m),
centri nei quattro punti (+-0.15, +-0.15) m rispetto al centro della board.

Produce due file, gli stessi nomi usati da LVT2Calib:
  - four_circle_boundary.pcd : solo i contorni (rettangolo esterno + 4 cerchi).
        E' quello puntato da `model_path` in livox_pattern.launch e usato
        per la registrazione PCA + ICP in isCalibBoard().
  - four_circle_dense.pcd    : superficie piena della board con i 4 fori
        ritagliati (usato per visualizzazione / altri confronti).

Convenzione: nuvola planare sul piano XY (z = 0), centrata nell'origine.
La registrazione e' PCA-based, quindi conta la geometria relativa, non
l'orientamento assoluto.

IMPORTANTE (fix 29/07/2026): il PCD originale del repo
(four_circle_boundary_ORIGINAL.pcd) usa lo schema a 5 campi
"x y z intensity range". La prima versione di questo script scriveva solo
4 campi ("x y z intensity"), mismatch di schema che puo' causare un
caricamento silenzioso errato in pcl::io::loadPCDFile (nessun errore
esplicito, ma dati letti/allineati male). Questa versione scrive lo stesso
schema a 5 campi dell'originale.

Uso:
    py generate_board_template.py --outdir <cartella_di_output>

Poi copiare i due .pcd in  lvt2calib/data/template_pcl/  (sovrascrivendo
gli originali, dopo averne fatto una copia di backup).
"""

import argparse
import math
from pathlib import Path


# ----------------------------- geometria board -----------------------------
BOARD_W = 1.00          # larghezza board [m]
BOARD_H = 0.70          # altezza board [m]
HOLE_DIAMETER = 0.13    # diametro foro [m]
HOLE_OFFSET = 0.15      # offset dei centri dei fori dal centro board [m]


def hole_centers(offset: float):
    """I 4 centri, ai vertici di un quadrato di lato 2*offset."""
    return [
        (-offset, +offset),
        (+offset, +offset),
        (+offset, -offset),
        (-offset, -offset),
    ]


def sample_rectangle_outline(w: float, h: float, spacing: float):
    """Punti lungo il perimetro del rettangolo, passo `spacing`."""
    pts = []
    hw, hh = w / 2.0, h / 2.0

    n_horiz = max(2, int(round(w / spacing)) + 1)
    for i in range(n_horiz):
        x = -hw + w * i / (n_horiz - 1)
        pts.append((x, +hh))
        pts.append((x, -hh))

    n_vert = max(2, int(round(h / spacing)) + 1)
    for i in range(1, n_vert - 1):     # angoli gia' inseriti sopra
        y = -hh + h * i / (n_vert - 1)
        pts.append((+hw, y))
        pts.append((-hw, y))

    return pts


def sample_circle_outline(cx: float, cy: float, radius: float, spacing: float):
    """Punti lungo la circonferenza, passo `spacing` misurato sull'arco."""
    circumference = 2.0 * math.pi * radius
    n = max(8, int(round(circumference / spacing)))
    return [
        (
            cx + radius * math.cos(2.0 * math.pi * i / n),
            cy + radius * math.sin(2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def sample_board_surface(w: float, h: float, radius: float,
                         centers, spacing: float):
    """Griglia sulla superficie della board, escludendo l'interno dei fori."""
    pts = []
    hw, hh = w / 2.0, h / 2.0
    nx = int(round(w / spacing)) + 1
    ny = int(round(h / spacing)) + 1

    r2 = radius * radius
    for ix in range(nx):
        x = -hw + w * ix / (nx - 1)
        for iy in range(ny):
            y = -hh + h * iy / (ny - 1)
            inside_hole = any(
                (x - cx) ** 2 + (y - cy) ** 2 <= r2 for cx, cy in centers
            )
            if not inside_hole:
                pts.append((x, y))
    return pts


def write_pcd(path: Path, points, intensity: float = 0.0) -> None:
    """Scrive un PCD ASCII con campi x y z intensity range (5 campi),
    stesso schema dell'originale four_circle_boundary_ORIGINAL.pcd.
    """
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity range\n"
        "SIZE 4 4 4 4 4\n"
        "TYPE F F F F F\n"
        "COUNT 1 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with path.open("w", encoding="ascii") as fh:
        fh.write(header)
        for x, y in points:
            fh.write(f"{x:.6f} {y:.6f} 0.000000 {intensity:.6f} 0.000000\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera i template PCD della board di calibrazione per LVT2Calib."
    )
    parser.add_argument("--outdir", required=True,
                        help="Cartella di output per i due .pcd")
    parser.add_argument("--board-width", type=float, default=BOARD_W,
                        help=f"Larghezza board in metri (default {BOARD_W})")
    parser.add_argument("--board-height", type=float, default=BOARD_H,
                        help=f"Altezza board in metri (default {BOARD_H})")
    parser.add_argument("--hole-diameter", type=float, default=HOLE_DIAMETER,
                        help=f"Diametro foro in metri (default {HOLE_DIAMETER}). "
                             "MISURARE COL CALIBRO dopo il taglio e aggiornare.")
    parser.add_argument("--hole-offset", type=float, default=HOLE_OFFSET,
                        help=f"Offset dei centri dal centro board (default {HOLE_OFFSET})")
    parser.add_argument("--boundary-spacing", type=float, default=0.004,
                        help="Passo di campionamento dei contorni [m] (default 0.004)")
    parser.add_argument("--dense-spacing", type=float, default=0.004,
                        help="Passo di campionamento della superficie [m] (default 0.004)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    radius = args.hole_diameter / 2.0
    centers = hole_centers(args.hole_offset)

    # --- controlli di sanita' geometrica ---
    hw, hh = args.board_width / 2.0, args.board_height / 2.0
    for cx, cy in centers:
        if abs(cx) + radius > hw or abs(cy) + radius > hh:
            raise SystemExit(
                f"Foro in ({cx:+.3f}, {cy:+.3f}) con raggio {radius:.3f} m "
                f"esce dai bordi della board {args.board_width}x{args.board_height} m."
            )
    gap = 2.0 * args.hole_offset - args.hole_diameter
    if gap <= 0:
        raise SystemExit("I fori si sovrappongono: aumentare --hole-offset "
                         "o ridurre --hole-diameter.")

    # --- boundary: rettangolo esterno + 4 circonferenze ---
    boundary = sample_rectangle_outline(args.board_width, args.board_height,
                                        args.boundary_spacing)
    for cx, cy in centers:
        boundary.extend(sample_circle_outline(cx, cy, radius,
                                              args.boundary_spacing))

    # --- dense: superficie con i fori ritagliati ---
    dense = sample_board_surface(args.board_width, args.board_height,
                                 radius, centers, args.dense_spacing)

    boundary_path = outdir / "four_circle_boundary.pcd"
    dense_path = outdir / "four_circle_dense.pcd"
    write_pcd(boundary_path, boundary)
    write_pcd(dense_path, dense)

    dist_centro = math.hypot(args.hole_offset, args.hole_offset)

    print("Geometria usata:")
    print(f"  board                : {args.board_width:.3f} x {args.board_height:.3f} m")
    print(f"  foro                 : diametro {args.hole_diameter:.3f} m "
          f"(raggio {radius:.4f} m)")
    print(f"  centri fori          : (+-{args.hole_offset:.3f}, +-{args.hole_offset:.3f}) m")
    print(f"  distanza centro-board: {dist_centro:.4f} m")
    print(f"  spazio tra fori      : {gap:.3f} m")
    print()
    print("Parametri coerenti da mettere in config/lidar_pattern_param.yaml:")
    print(f"  circle_radius: {radius:.4f}")
    print(f"  centroid_dis_min: {dist_centro - 0.06:.2f}   # {dist_centro:.4f} con margine")
    print(f"  centroid_dis_max: {dist_centro + 0.06:.2f}")
    print()
    print(f"Scritto {boundary_path}  ({len(boundary)} punti)")
    print(f"Scritto {dense_path}  ({len(dense)} punti)")


if __name__ == "__main__":
    main()
