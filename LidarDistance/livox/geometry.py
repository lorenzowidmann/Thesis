"""Central-square angular filter + distance stats.

Mirrors the EmissivityCalculation "central square" idea (a centred box that is a
`fraction` of the frame), but in the LiDAR's *angular* field of view instead of
image pixels. Forward is +x (sensor frame); azimuth is the horizontal angle off
that axis, elevation the vertical one. The central square keeps points with both
|azimuth| and |elevation| below half the square's angular width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DistanceStats:
    n: int          # points inside the square (after range gating)
    n_total: int    # points received this window
    min: float
    max: float
    mean: float
    median: float   # robust distance estimate used for calibration
    std: float
    # Diagnostics for the n==0 case only (see diagnose_empty below).
    n_valid: int = 0        # points with x > 0 (real returns, sensor invalid-point convention)
    n_in_square: int = 0    # of those, points inside the angular square (any range)
    nearest_in_square: float = 0.0  # closest range among n_in_square, 0.0 if none
    min_abs_azimuth_deg: float = 0.0    # smallest |azimuth| among valid points (n_valid>0 only)
    min_abs_elevation_deg: float = 0.0  # smallest |elevation| among valid points

    def format(self) -> str:
        if self.n == 0:
            if self.n_valid == 0:
                return (f"no returns in central square (received {self.n_total} pts, "
                        f"all invalid/zero -- likely sensor blind zone at this range)")
            if self.n_in_square == 0:
                return (f"no returns in central square (received {self.n_total} pts, "
                        f"{self.n_valid} valid, none inside the angular square -- "
                        f"nearest valid return is {self.min_abs_azimuth_deg:.1f} deg azimuth / "
                        f"{self.min_abs_elevation_deg:.1f} deg elevation off-axis -- widen "
                        f"--square-deg to include it, or this may be a boresight blind cone "
                        f"at this range)")
            return (f"no returns in central square (received {self.n_total} pts, "
                    f"{self.n_in_square} inside the square but outside the "
                    f"[min,max] range gate -- nearest in-square point at "
                    f"{self.nearest_in_square:.3f} m)")
        return (f"distance[m]  median={self.median:.3f}  mean={self.mean:.3f}  "
                f"min={self.min:.3f}  max={self.max:.3f}  std={self.std:.3f}  "
                f"(n={self.n}/{self.n_total})")


def central_square_mask(xyz: np.ndarray, half_angle_rad: float) -> np.ndarray:
    """Boolean mask of points inside the centred angular square around +x."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    horiz = np.hypot(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        azimuth = np.arctan2(y, x)
        elevation = np.arctan2(z, horiz)
    return (x > 0) & (np.abs(azimuth) <= half_angle_rad) & (np.abs(elevation) <= half_angle_rad)


def compute_stats(xyz: np.ndarray, half_angle_rad: float,
                  min_range: float, max_range: float) -> DistanceStats:
    """Range = euclidean distance from the sensor to each point (metres)."""
    n_total = xyz.shape[0]
    rng = np.linalg.norm(xyz, axis=1)
    mask = central_square_mask(xyz, half_angle_rad)
    mask &= (rng >= min_range) & (rng <= max_range)
    sel = rng[mask]
    if sel.size == 0:
        square_mask = central_square_mask(xyz, half_angle_rad)
        valid_mask = xyz[:, 0] > 0
        n_valid = int(valid_mask.sum())
        n_in_square = int(square_mask.sum())
        nearest = float(rng[square_mask].min()) if n_in_square else 0.0
        min_az_deg = min_el_deg = 0.0
        if n_valid and not n_in_square:
            vx, vy, vz = xyz[valid_mask, 0], xyz[valid_mask, 1], xyz[valid_mask, 2]
            horiz = np.hypot(vx, vy)
            azimuth = np.arctan2(vy, vx)
            elevation = np.arctan2(vz, horiz)
            min_az_deg = float(np.degrees(np.abs(azimuth).min()))
            min_el_deg = float(np.degrees(np.abs(elevation).min()))
        return DistanceStats(0, n_total, 0.0, 0.0, 0.0, 0.0, 0.0,
                             n_valid=n_valid, n_in_square=n_in_square,
                             nearest_in_square=nearest,
                             min_abs_azimuth_deg=min_az_deg,
                             min_abs_elevation_deg=min_el_deg)
    return DistanceStats(
        n=int(sel.size), n_total=n_total,
        min=float(sel.min()), max=float(sel.max()),
        mean=float(sel.mean()), median=float(np.median(sel)),
        std=float(sel.std()),
    )
