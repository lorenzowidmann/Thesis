#!/usr/bin/env python3
"""Loop-closure check of the three estimated extrinsics.

The three transforms are estimated independently, so composing two of them
must reproduce the third:

    T_lidar->thermal  ~=  T_rgb->thermal @ T_lidar->rgb

The residual T_err = inv(T_lidar->thermal) @ (T_rgb->thermal @ T_lidar->rgb)
should be near identity; its rotation angle (deg) and translation norm (m)
quantify the internal consistency of the whole calibration. It does NOT prove
absolute accuracy (a shared bias cancels out), but a large residual always
means at least one pairwise calibration is off.

Usage (library; the CLI lives in main.py):
    report = check_consistency(T_lidar_rgb, T_rgb_thermal, T_lidar_thermal)
"""

from __future__ import annotations

import numpy as np

from lidar_camera_extrinsic import invert_transform


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in degrees (0 = identity)."""
    cos = (np.trace(np.asarray(R, np.float64)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def check_consistency(
    T_lidar_rgb: np.ndarray,
    T_rgb_thermal: np.ndarray,
    T_lidar_thermal: np.ndarray,
) -> dict:
    """Compare the composed LiDAR->thermal transform against the direct estimate.

    All inputs are 4x4 rigid transforms with the source->target direction in
    their name (p_target = T @ p_source). Returns a JSON-ready dict with the
    composed transform, the residual rotation angle (deg) and translation
    norm (m), and the residual transform itself.
    """
    T_composed = np.asarray(T_rgb_thermal, np.float64) @ np.asarray(T_lidar_rgb, np.float64)
    T_err = invert_transform(np.asarray(T_lidar_thermal, np.float64)) @ T_composed

    return {
        "composed_lidar_thermal": T_composed.tolist(),
        "residual_transform": T_err.tolist(),
        "residual_rotation_deg": rotation_angle_deg(T_err[:3, :3]),
        "residual_translation_m": float(np.linalg.norm(T_err[:3, 3])),
    }


def print_report(report: dict) -> None:
    """Human-readable summary of a check_consistency() result."""
    print("Consistency check (LiDAR->RGB composed with RGB->thermal vs direct LiDAR->thermal):")
    print(f"  residual rotation:    {report['residual_rotation_deg']:.3f} deg")
    print(f"  residual translation: {report['residual_translation_m']*100:.2f} cm")
