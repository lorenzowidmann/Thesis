#!/usr/bin/env python3
r"""
split_zed_poses.py

Splits the flat ZED frames/ folder into per-pose subfolders, using the
recorded t_offset_s in metadata.json (frame rate is fixed at 0.4s interval,
~2.5 fps, per zed_record.py). Mirrors split_flir_poses.py for the FLIR side.

Uses the ZED-offset windows from pose_alignment_Exttr_tryN.md ("Final
alignment" table, column "ZED (off s)").

Usage (run locally on Windows, or in container if paths mounted):
    python split_zed_poses.py --metadata C:\Users\loren\Desktop\SLAM\ExtCalibration\ZED\metadata.json ^
        --frames-dir C:\Users\loren\Desktop\SLAM\ExtCalibration\ZED\frames ^
        --out-dir C:\Users\loren\Desktop\SLAM\ExtCalibration\ZED\Poses
"""
import argparse
import copy
import json
import os
import shutil

# ZED session-offset windows (start_s, end_s), from pose_alignment_Exttr_tryN.md
# "Final alignment" table, column "ZED (off s)"
POSES = {
    "pose_01": (0, 108),
    "pose_02": (116, 223),
    "pose_03": (249, 341),
    "pose_04": (405, 463),
    "pose_05": (481, 573),
    "pose_06": (584, 651),
    "pose_07": (685, 753),
    "pose_08": (810, 901),
    "pose_09": (946, 1029),
    "pose_10": (1052, 1137),
    "pose_11": (1161, 1243),
    "pose_12": (1290, 1377),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--poses", nargs="*", default=None,
                    help="subset of pose names to process, e.g. pose_01 pose_02")
    args = ap.parse_args()

    with open(args.metadata) as f:
        meta = json.load(f)
    frames = meta["frames"]

    poses = {k: v for k, v in POSES.items()
             if args.poses is None or k in args.poses}

    for name, (t0, t1) in poses.items():
        dst = os.path.join(args.out_dir, name)
        os.makedirs(dst, exist_ok=True)
        copied = []
        for fr in frames:
            t = fr["t_offset_s"]
            if t0 <= t <= t1:
                src_path = os.path.join(args.frames_dir, fr["file"])
                dst_path = os.path.join(dst, fr["file"])
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                    copied.append(fr)

        # metadata.json per pose: same schema, frames filtered to this window.
        # t_offset_s values are NOT rebased -- zed_frame_publisher.py
        # --stamp-mode absolute uses session.started_utc + t_offset_s.
        pose_meta = copy.deepcopy(meta)
        pose_meta["recording"]["frames_dir"] = "."  # PNGs sit next to metadata.json
        pose_meta["recording"]["n_frames"] = len(copied)
        pose_meta["frames"] = copied
        with open(os.path.join(dst, "metadata.json"), "w") as f:
            json.dump(pose_meta, f, indent=2)

        print(f"{name}: window {t0}-{t1}s -> {len(copied)} frames copied to {dst}")

if __name__ == "__main__":
    main()
