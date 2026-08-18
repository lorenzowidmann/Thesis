"""Broadcast a coarse voxel grid's per-voxel material label down onto a
finer voxel grid, one-to-one by containment -- no re-reprojection, no new
FLIR/segment resampling.

Material counterpart of TemperatureToVoxel/broadcast_coarse_temperature.py,
same reasoning: the LiDAR<->FLIR extrinsic calibration RMSE (~5.8cm,
rig_calibration.yaml) and z-buffer tolerance (8cm, material_to_voxel.py's
ZBUFFER_TOL_M) are both coarser than a 5cm voxel, so re-running
material_to_voxel.py binned directly into a 5cm grid would assign spatial
precision the calibration can't back up, and would fragment the already
sparse per-point votes across ~27x more bins -- worse for a majority vote
than for a mean, since a voxel with only 1-2 material-labeled points has no
real "majority" to speak of.

Instead: run material_to_voxel.py ONCE at a voxel size the calibration
actually supports (e.g. the AlignedOctree default 0.15 m) to get
voxels_material.npz, then broadcast each coarse voxel's material_id /
material_confidence / n_material_votes onto every finer voxel (from a
separately regenerated, e.g. 0.05 m, voxels.npz) that falls inside it --
same label repeated across every fine child of one coarse cell, while the
finer grid still renders at its own resolution.

Fine and coarse grids don't need to share an origin or an integer
voxel-size ratio -- containment for each fine voxel center is computed
against the COARSE grid's own origin/voxel_size (floor((center - origin) /
voxel_size)), same convention as material_to_voxel.py's
voxel_index()/voxel_centers_to_index() (copied here, this folder is
self-contained, no cross-import -- same trick as
broadcast_coarse_temperature.py).

Usage:
    python broadcast_coarse_material.py
        [--fine-voxels ../AlignedOctree/voxels.npz]
        [--coarse-material voxels_material.npz]
        [--out voxels_material_broadcast.npz]

Output schema matches voxels_material.npz exactly (centers, counts,
material_id, material_confidence, n_material_votes, materials, voxel_size,
origin, depth -- all at the FINE grid's resolution/geometry, `materials`
copied straight through unchanged since `material_id` indexes into it and
both grids must agree on that vocabulary). n_material_votes here is the
coarse parent voxel's count, repeated across every fine child that maps to
it -- NOT independent per fine voxel; it says how much evidence backs the
(shared) label, not how much is specific to that smaller cell -- same
caveat as broadcast_coarse_temperature.py's n_temp_samples.

Venv: C:\\venvs\\planefit (numpy only).
"""
import argparse
from pathlib import Path

import numpy as np


def voxel_centers_to_index(centers, origin, voxel_size):
    """Invert voxelizer.py's `center = (idx + 0.5) * voxel_size + origin` --
    copied from material_to_voxel.py (same self-contained convention)."""
    return np.round((centers - origin) / voxel_size - 0.5).astype(np.int64)


def voxel_index(points, origin, voxel_size):
    """idx = floor((points - origin) / voxel_size) -- copied from
    material_to_voxel.py's voxel_index() (depth-clipping branch dropped:
    only used here to find which COARSE cell a fine voxel center falls in,
    never to reproduce an octree lattice)."""
    return np.floor((points - origin) / voxel_size).astype(np.int64)


def match_to_grid(query_idx, grid_idx):
    """Row into `grid_idx` that each row of `query_idx` matches, or -1 if
    none. Packs (i,j,k) into one int64 key, sort + searchsorted -- same
    trick as material_to_voxel.py's bin_material_into_voxels."""
    if len(query_idx) == 0 or len(grid_idx) == 0:
        return np.full(len(query_idx), -1, dtype=np.int64)

    combined_lo = np.minimum(query_idx.min(axis=0), grid_idx.min(axis=0))
    combined_hi = np.maximum(query_idx.max(axis=0), grid_idx.max(axis=0))
    ranges = combined_hi - combined_lo + 1

    def pack(idx):
        s = idx - combined_lo
        return (s[:, 0].astype(np.int64)
                + s[:, 1].astype(np.int64) * ranges[0]
                + s[:, 2].astype(np.int64) * ranges[0] * ranges[1])

    grid_key = pack(grid_idx)
    order = np.argsort(grid_key)
    sorted_key = grid_key[order]

    query_key = pack(query_idx)
    pos = np.searchsorted(sorted_key, query_key)
    pos = np.clip(pos, 0, len(sorted_key) - 1)
    found = sorted_key[pos] == query_key

    result = np.full(len(query_idx), -1, dtype=np.int64)
    result[found] = order[pos[found]]
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fine-voxels", type=Path,
                     default=Path(__file__).resolve().parent.parent / "AlignedOctree" / "voxels.npz",
                     help="finer voxels.npz to color (e.g. a 0.05 m grid)")
    ap.add_argument("--coarse-material", type=Path,
                     default=Path("voxels_material.npz"),
                     help="material_to_voxel.py output built at a "
                          "calibration-safe voxel size (e.g. 0.15 m)")
    ap.add_argument("--out", type=Path, default=Path("voxels_material_broadcast.npz"))
    args = ap.parse_args()

    fine = np.load(args.fine_voxels)
    fine_centers = fine["centers"]
    fine_counts = fine["counts"]
    fine_voxel_size = float(fine["voxel_size"])
    fine_origin = fine["origin"]
    fine_depth = int(fine["depth"])
    print(f"loaded {args.fine_voxels}: {len(fine_centers)} voxels, "
          f"voxel_size={fine_voxel_size:.4f} m")

    coarse = np.load(args.coarse_material, allow_pickle=False)
    coarse_centers = coarse["centers"]
    coarse_material_id = coarse["material_id"]
    coarse_material_confidence = coarse["material_confidence"]
    coarse_n_material_votes = coarse["n_material_votes"]
    coarse_materials = coarse["materials"]
    coarse_voxel_size = float(coarse["voxel_size"])
    coarse_origin = coarse["origin"]
    print(f"loaded {args.coarse_material}: {len(coarse_centers)} voxels, "
          f"voxel_size={coarse_voxel_size:.4f} m, {len(coarse_materials)} material(s) "
          f"in vocabulary: {list(coarse_materials)}")

    if coarse_voxel_size < fine_voxel_size:
        print(f"WARNING: --coarse-material voxel_size ({coarse_voxel_size:.4f} m) is "
              f"SMALLER than --fine-voxels voxel_size ({fine_voxel_size:.4f} m) -- this "
              f"script assumes coarse is the bigger grid; check you didn't swap the two inputs")

    coarse_idx = voxel_centers_to_index(coarse_centers, coarse_origin, coarse_voxel_size)
    fine_query_idx = voxel_index(fine_centers, coarse_origin, coarse_voxel_size)

    match = match_to_grid(fine_query_idx, coarse_idx)
    matched = match >= 0

    material_id = np.full(len(fine_centers), -1, dtype=np.int32)
    material_confidence = np.full(len(fine_centers), np.nan, dtype=np.float64)
    n_material_votes = np.zeros(len(fine_centers), dtype=np.int32)
    material_id[matched] = coarse_material_id[match[matched]]
    material_confidence[matched] = coarse_material_confidence[match[matched]]
    n_material_votes[matched] = coarse_n_material_votes[match[matched]]

    has_material = material_id >= 0
    print(f"fine voxels matched to a coarse cell: {int(matched.sum())} / {len(fine_centers)} "
          f"({100 * matched.mean():.1f}%)")
    print(f"fine voxels with a coarse material label: {int(has_material.sum())} / "
          f"{len(fine_centers)} ({100 * has_material.mean():.1f}%)")

    np.savez(
        args.out,
        centers=fine_centers,
        counts=fine_counts,
        material_id=material_id,
        material_confidence=material_confidence,
        n_material_votes=n_material_votes,
        materials=coarse_materials,
        voxel_size=fine_voxel_size,
        origin=fine_origin,
        depth=fine_depth,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
