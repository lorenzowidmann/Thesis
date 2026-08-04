#!/usr/bin/env python3
r"""
resync_pose_metadata.py

Prunes a pose folder's metadata.json to the frames actually present on disk.
Use after manually deleting PNGs from a Poses/pose_NN/ folder produced by
split_zed_poses.py -- keeps the zed_record/v1 schema and the original
t_offset_s values, only drops missing entries and fixes recording.n_frames.

Usage:
    python resync_pose_metadata.py C:\path\to\Poses\pose_01 C:\path\to\Poses\pose_02
"""
import json
import os
import sys


def main():
    dirs = sys.argv[1:]
    if not dirs:
        raise SystemExit("usage: resync_pose_metadata.py <pose_dir> [<pose_dir> ...]")

    for d in dirs:
        path = os.path.join(d, "metadata.json")
        with open(path) as f:
            meta = json.load(f)

        kept = [fr for fr in meta["frames"]
                if os.path.isfile(os.path.join(d, fr["file"]))]
        dropped = len(meta["frames"]) - len(kept)

        meta["frames"] = kept
        meta["recording"]["n_frames"] = len(kept)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"{d}: {len(kept)} frames kept, {dropped} dropped -> {path}")


if __name__ == "__main__":
    main()
