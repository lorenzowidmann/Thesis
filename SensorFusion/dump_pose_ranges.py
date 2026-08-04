#!/usr/bin/env python3
"""
dump_pose_ranges.py

Stampa in forma compatta (una riga per posa) il primo e l'ultimo timestamp
di ogni cartella pose_*, cosi' da poter ricalcolare le finestre LiDAR.

Uso:
    py dump_pose_ranges.py
"""

import re
from pathlib import Path

POSES_ROOT = Path(r"C:\Users\loren\Desktop\SLAM\ExtCalibration\Flir\Poses")
NAME_RE = re.compile(r"^(\d{8})_(\d{6})_R\.jpg$", re.IGNORECASE)


def main():
    for d in sorted(p for p in POSES_ROOT.iterdir() if p.is_dir()):
        names = sorted(f.name for f in d.iterdir()
                       if f.is_file() and NAME_RE.match(f.name))
        if not names:
            print(f"{d.name} VUOTA")
            continue
        print(f"{d.name} {len(names)} {names[0][9:15]} {names[-1][9:15]}")


if __name__ == "__main__":
    main()
