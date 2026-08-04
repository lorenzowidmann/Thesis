#!/usr/bin/env python3
"""
split_flir_poses.py

Divide la cartella FLIR della sessione Exttr_tryN in 12 sottocartelle,
una per posa, cosi' che flir_frame_publisher possa essere puntato alla
singola posa senza che il --loop scorra su inquadrature di pose diverse.

Relazione di sincronizzazione (da pose_alignment_Exttr_tryN.md):
    FLIR_clock = 20:21:22 + ZED_offset
    (il clock FLIR e' sbagliato in assoluto, ma gli offset relativi sono validi)

Uso:
    py split_flir_poses.py
    py split_flir_poses.py --move          # sposta invece di copiare
    py split_flir_poses.py --dry-run       # mostra soltanto cosa farebbe
"""

import argparse
import re
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\loren\Desktop\SLAM\ExtCalibration\Flir\Calibration")
DST_ROOT = Path(r"C:\Users\loren\Desktop\SLAM\ExtCalibration\Flir\Poses")

# Inizio del run reale sul clock FLIR (pose 5 del folder grezzo = posa 1 utile)
FLIR_START = 20 * 3600 + 21 * 60 + 22  # 20:21:22 in secondi

# Offset ZED (secondi da inizio sessione) delle 12 pose utili,
# da pose_alignment_Exttr_tryN.md, colonna "ZED (off s)"
ZED_WINDOWS = [
    (0, 108),
    (116, 223),
    (249, 341),
    (405, 463),
    (481, 573),
    (584, 651),
    (685, 753),
    (810, 901),
    (946, 1029),
    (1052, 1137),
    (1161, 1243),
    (1290, 1377),
]

NAME_RE = re.compile(r"^(\d{8})_(\d{6})_R\.jpg$", re.IGNORECASE)


def file_seconds(name: str):
    """Estrae HHMMSS dal nome file e lo converte in secondi. None se non combacia."""
    m = NAME_RE.match(name)
    if not m:
        return None
    t = m.group(2)
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def hhmmss(sec: int) -> str:
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", action="store_true",
                    help="sposta i file invece di copiarli")
    ap.add_argument("--dry-run", action="store_true",
                    help="non tocca nulla, stampa soltanto il piano")
    args = ap.parse_args()

    if not SRC.is_dir():
        raise SystemExit(f"Cartella sorgente non trovata: {SRC}")

    files = sorted(f for f in SRC.iterdir() if f.is_file() and NAME_RE.match(f.name))
    if not files:
        raise SystemExit(f"Nessun file *_R.jpg trovato in {SRC}")

    print(f"Sorgente : {SRC}")
    print(f"File     : {len(files)}")
    print(f"Range    : {files[0].name}  ->  {files[-1].name}")
    print(f"Modo     : {'DRY-RUN' if args.dry_run else ('MOVE' if args.move else 'COPY')}")
    print()

    assigned = 0
    for i, (z_start, z_end) in enumerate(ZED_WINDOWS, start=1):
        f_start = FLIR_START + z_start
        f_end = FLIR_START + z_end

        sel = [f for f in files
               if (s := file_seconds(f.name)) is not None and f_start <= s <= f_end]

        dst = DST_ROOT / f"pose_{i:02d}"
        print(f"pose_{i:02d}  ZED {z_start:>4}-{z_end:<4}s  "
              f"FLIR {hhmmss(f_start)}-{hhmmss(f_end)}  ->  {len(sel):>3} frame")

        if not sel:
            print("          ATTENZIONE: nessun frame in questa finestra!")
            continue

        if not args.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            for f in sel:
                if args.move:
                    shutil.move(str(f), str(dst / f.name))
                else:
                    shutil.copy2(str(f), str(dst / f.name))

        assigned += len(sel)

    print()
    print(f"Frame assegnati a una posa : {assigned}")
    print(f"Frame non assegnati (gap)  : {len(files) - assigned}")
    if not args.dry_run:
        print(f"Destinazione               : {DST_ROOT}")


if __name__ == "__main__":
    main()
