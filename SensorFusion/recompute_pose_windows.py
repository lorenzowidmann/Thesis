#!/usr/bin/env python3
"""
recompute_pose_windows.py

Legge le cartelle FLIR gia' splittate (e successivamente rifinite a mano,
es. rimuovendo i frame in cui l'operatore e' davanti alla board) e ricalcola
le finestre temporali LiDAR corrispondenti, da usare con rosbag play -s/-u.

Relazioni di sincronizzazione (da pose_alignment_Exttr_tryN.md):
    ZED_offset  = FLIR_clock - 20:21:22
    LiDAR_offset = ZED_offset + 4 s

Stampa direttamente i comandi rosbag play pronti da copiare.

Uso:
    py recompute_pose_windows.py
"""

import re
from pathlib import Path

POSES_ROOT = Path(r"C:\Users\loren\Desktop\SLAM\ExtCalibration\Flir\Poses")
BAG = "/data/bags/ExtCalibration/Lidar/rosbag2_2026_07_30-15_02_17_ros1.bag"
FLIR_DIR_IN_CONTAINER = "/data/bags/ExtCalibration/Flir/Poses"

FLIR_START = 20 * 3600 + 21 * 60 + 22   # 20:21:22
LIDAR_MINUS_ZED = 4                      # LiDAR ~ ZED + 4 s

NAME_RE = re.compile(r"^(\d{8})_(\d{6})_R\.jpg$", re.IGNORECASE)


def file_seconds(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    t = m.group(2)
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def hhmmss(sec):
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def main():
    if not POSES_ROOT.is_dir():
        raise SystemExit(f"Cartella non trovata: {POSES_ROOT}")

    dirs = sorted(d for d in POSES_ROOT.iterdir() if d.is_dir())
    if not dirs:
        raise SystemExit(f"Nessuna sottocartella pose_* in {POSES_ROOT}")

    rows = []
    print(f"{'posa':<9} {'frame':>6}  {'FLIR range':<21} {'ZED off':<13} {'LiDAR off':<13} {'durata':>7}")
    print("-" * 82)

    for d in dirs:
        secs = sorted(s for f in d.iterdir()
                      if f.is_file() and (s := file_seconds(f.name)) is not None)
        if not secs:
            print(f"{d.name:<9} {'0':>6}  ATTENZIONE: cartella vuota o nomi non riconosciuti")
            continue

        f_min, f_max = secs[0], secs[-1]
        z_min, z_max = f_min - FLIR_START, f_max - FLIR_START
        l_min, l_max = z_min + LIDAR_MINUS_ZED, z_max + LIDAR_MINUS_ZED
        dur = l_max - l_min

        print(f"{d.name:<9} {len(secs):>6}  "
              f"{hhmmss(f_min)}-{hhmmss(f_max):<10} "
              f"{z_min:>5}-{z_max:<6} "
              f"{l_min:>5}-{l_max:<6} "
              f"{dur:>6}s")

        rows.append((d.name, l_min, l_max, dur, len(secs)))

    print()
    print("=" * 82)
    print("COMANDI PRONTI  (T4a = publisher FLIR, T6 = rosbag play)")
    print("=" * 82)
    for name, l_min, l_max, dur, n in rows:
        print()
        print(f"# --- {name}  ({n} frame FLIR, finestra LiDAR {dur}s) ---")
        print(f"rosrun flir_frame_publisher flir_frame_publisher.py "
              f"--image-dir {FLIR_DIR_IN_CONTAINER}/{name} --image-mode embedded --loop")
        print(f"rosbag play {BAG} -s {l_min} -u {l_max}")

    print()
    print("-" * 82)
    short = [r for r in rows if r[3] < 90]
    if short:
        print("NOTA - finestre piu' corte di 90s (il LiDAR accumula ~0.28 centri/s,")
        print("       quindi con max_frame=25 servono ~90s: queste pose potrebbero")
        print("       non arrivare a 25/25 prima della fine della finestra):")
        for name, l_min, l_max, dur, n in short:
            stimati = int(dur * 0.28)
            print(f"       {name}: {dur}s -> ~{stimati} centri stimati")
        print("       Se una posa non satura, abbassare max_frame per quella posa")
        print("       oppure allargare leggermente -u.")


if __name__ == "__main__":
    main()
