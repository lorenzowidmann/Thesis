#!/usr/bin/env python3
"""
find_board_limits.py

Stima i limiti passthrough (px/py/pz) per una posa, analizzando la nuvola
grezza accumulata invece della board gia' rilevata.

Va lanciato NEL CONTAINER, mentre il rosbag della posa e' in riproduzione,
con T3 attivo (anche senza filtri: serve solo /livox/points/acc_cloud).

Metodo:
  1. cattura N nuvole accumulate e le unisce
  2. scarta pavimento e soffitto con una fascia Z larga
  3. istogramma in X, trova i blocchi separati da zone vuote
  4. per ogni blocco misura l'estensione in Y e Z
  5. segnala quale blocco ha dimensioni compatibili con la board (100x70 cm)

Uso:
    python3 find_board_limits.py
    python3 find_board_limits.py --frames 5 --xmin 1.5 --xmax 9.0
"""

import argparse
import collections

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

# Dimensioni reali della board (m): lato lungo e lato corto
BOARD_LONG = 1.00
BOARD_SHORT = 0.70
TOL = 0.25          # tolleranza sulle dimensioni attese
BIN = 0.25          # larghezza bin istogramma in X (m)
MIN_PTS_BIN = 150   # punti minimi perche' un bin sia "pieno"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3,
                    help="quante nuvole accumulate unire (default 3)")
    ap.add_argument("--xmin", type=float, default=1.5)
    ap.add_argument("--xmax", type=float, default=9.0)
    ap.add_argument("--ymax", type=float, default=3.0,
                    help="|y| massimo considerato")
    ap.add_argument("--zmin", type=float, default=-0.8)
    ap.add_argument("--zmax", type=float, default=2.5)
    ap.add_argument("--topic", default="/livox/points/acc_cloud")
    args = ap.parse_args()

    rospy.init_node("board_limits", anonymous=True)

    pts = []
    for i in range(args.frames):
        print(f"  cattura nuvola {i+1}/{args.frames} ...")
        m = rospy.wait_for_message(args.topic, PointCloud2, timeout=60)
        pts.extend(pc2.read_points(m, field_names=("x", "y", "z"),
                                   skip_nans=True))

    sel = [p for p in pts
           if args.xmin < p[0] < args.xmax
           and abs(p[1]) < args.ymax
           and args.zmin < p[2] < args.zmax]

    print(f"\npunti totali: {len(pts)}   nel volume di ricerca: {len(sel)}\n")
    if not sel:
        raise SystemExit("Nessun punto nel volume: allargare --xmin/--xmax.")

    # istogramma in X
    hist = collections.Counter(int(p[0] / BIN) for p in sel)
    kmin, kmax = min(hist), max(hist)

    print("Istogramma densita' in X:")
    for k in range(kmin, kmax + 1):
        n = hist.get(k, 0)
        flag = "" if n >= MIN_PTS_BIN else "   (vuoto)"
        print(f"  x {k*BIN:5.2f}-{(k+1)*BIN:5.2f} m : "
              f"{'#' * min(50, n // 300):<50} {n:>7}{flag}")

    # raggruppa bin contigui non vuoti in blocchi
    blocks, cur = [], []
    for k in range(kmin, kmax + 1):
        if hist.get(k, 0) >= MIN_PTS_BIN:
            cur.append(k)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    print(f"\nTrovati {len(blocks)} blocchi separati da zone vuote:\n")

    candidates = []
    for bi, blk in enumerate(blocks, 1):
        x0, x1 = blk[0] * BIN, (blk[-1] + 1) * BIN
        pb = [p for p in sel if x0 <= p[0] < x1]
        ys = [p[1] for p in pb]
        zs = [p[2] for p in pb]
        dy, dz = max(ys) - min(ys), max(zs) - min(zs)

        # una board sta in piedi: una dimensione ~1.0 e l'altra ~0.7
        dims = sorted([dy, dz], reverse=True)
        is_board = (abs(dims[0] - BOARD_LONG) < TOL
                    and abs(dims[1] - BOARD_SHORT) < TOL)
        tag = "  <<< COMPATIBILE CON LA BOARD" if is_board else ""

        print(f"blocco {bi}: x {x0:.2f}-{x1:.2f} m   punti {len(pb):>7}")
        print(f"           y {min(ys):+.2f} .. {max(ys):+.2f}  (estensione {dy:.2f} m)")
        print(f"           z {min(zs):+.2f} .. {max(zs):+.2f}  (estensione {dz:.2f} m){tag}")

        if is_board:
            candidates.append((x0, x1, min(ys), max(ys), min(zs), max(zs)))
        print()

    if not candidates:
        print("Nessun blocco con dimensioni da board.")
        print("La board potrebbe essere fusa col muro: guardare in RViz e")
        print("usare Measure, oppure abbassare MIN_PTS_BIN.")
        return

    print("=" * 62)
    print("COMANDO T3 SUGGERITO")
    print("=" * 62)
    for x0, x1, y0, y1, z0, z1 in candidates:
        # margine di sicurezza
        print(f"\nroslaunch lvt2calib livox_hap_pattern.launch "
              f"cloud_tp:=/livox/points ns_:=livox_hap "
              f"use_passthrough_preprocess:=true "
              f"px_min:={x0-0.3:.1f} px_max:={x1+0.3:.1f} "
              f"py_min:={y0-0.3:.1f} py_max:={y1+0.3:.1f} "
              f"pz_min:={max(z0-0.3, -0.5):.1f} pz_max:={z1+0.3:.1f}")


if __name__ == "__main__":
    main()
