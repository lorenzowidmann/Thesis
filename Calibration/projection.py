"""Project LiDAR points into a camera's pixel space.

Generic: works for either FLIR or ZED, just pass the matching intrinsics
(K, dist) and extrinsic (T_lidar_to_cam) from rig_calibration.py.

Points from /cloud_registered are already in world/map frame (the SLAM node
applies the current LiDAR pose before publishing). To reach the camera's own
frame at that instant: world -> LiDAR-local (undo that same pose) ->
camera-local (T_lidar_to_cam, a fixed rig extrinsic) -> pixels (K + dist).
"""

import cv2
import numpy as np


def quat_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """xyzw quaternion (ROS convention, as stored in sync_manifest.json's
    triplet[i].lidar.orientation) -> 3x3 rotation matrix."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    return np.array([
        [1.0 - (yY + zZ), xY - wZ, xZ + wY],
        [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
        [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
    ])


def world_to_lidar_local(points_world: np.ndarray, position: np.ndarray, orientation_xyzw: np.ndarray) -> np.ndarray:
    """Undo the SLAM pose (lidar-local -> world) to bring world points back
    into the LiDAR's own frame at that instant."""
    R = quat_to_rotation_matrix(*orientation_xyzw)
    return (points_world - position) @ R  # R.T @ (p - t), vectorized as (p-t) @ R


def project_lidar_to_camera(
    points_world: np.ndarray,
    lidar_position: np.ndarray,
    lidar_orientation_xyzw: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-frame LiDAR points into one camera's pixel space.

    Returns:
        uv:    (N, 2) float pixel coordinates (undefined/garbage where invalid)
        depth: (N,) float, camera-frame forward distance (z)
        valid: (N,) bool, True where depth > 0 and the pixel falls inside
               [0, width) x [0, height)
    """
    points_lidar = world_to_lidar_local(points_world, lidar_position, lidar_orientation_xyzw)

    R_lc = T_lidar_to_cam[:3, :3]
    t_lc = T_lidar_to_cam[:3, 3]
    points_cam = points_lidar @ R_lc.T + t_lc

    depth = points_cam[:, 2]
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    uv, _ = cv2.projectPoints(points_cam.astype(np.float64), rvec, tvec, K, dist)
    uv = uv.reshape(-1, 2)

    valid = (
        (depth > 0)
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv, depth, valid
