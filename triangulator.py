"""Triangulate sparse 3D landmarks from two calibrated views.

Triangulation takes two rays, one from each camera center through each matched
pixel, and finds the 3D point whose projection best agrees with both rays. With
near-zero baseline, the rays are almost parallel and depth becomes extremely
uncertain, which is why initialization should wait until the camera has moved.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from config import Config


class Triangulator:
    """Create 3D points and reject geometrically poor landmarks."""

    def __init__(self, config: Config) -> None:
        """Store reprojection thresholds from config."""
        self.config = config

    def triangulate(
        self, pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Triangulate matches and return accepted points plus a validity mask."""
        if len(pts1) == 0 or len(pts2) == 0:
            return np.empty((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)

        points_h = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
        w = points_h[3]
        finite = np.abs(w) > 1e-12
        points_3d = (points_h[:3, finite] / w[finite]).T

        valid = np.zeros(len(pts1), dtype=bool)
        if len(points_3d) == 0:
            return points_3d, valid

        idx = np.flatnonzero(finite)
        raw_depth1 = P1[2, :3] @ points_3d.T + P1[2, 3]
        raw_depth2 = P2[2, :3] @ points_3d.T + P2[2, 3]
        depth1 = raw_depth1 > 0
        depth2 = raw_depth2 > 0
        mean1, per_point1 = self.compute_reprojection_error(points_3d, pts1[idx], P1)
        mean2, per_point2 = self.compute_reprojection_error(points_3d, pts2[idx], P2)
        reproj_ok = (per_point1 < self.config.max_reprojection_error) & (per_point2 < self.config.max_reprojection_error)
        finite_points = np.isfinite(points_3d).all(axis=1)

        provisional = depth1 & depth2 & reproj_ok & finite_points
        depth_ok = np.ones(len(points_3d), dtype=bool)
        if provisional.any():
            # Nearly-parallel rays can still pass the reprojection-error
            # gate while their intersection sits absurdly far past every
            # other point just triangulated -- median depth of this same
            # batch is the only "normal scale" reference available without
            # assuming real-world units monocular geometry doesn't have.
            median_depth = float(np.median(raw_depth1[provisional]))
            if median_depth > 1e-9:
                depth_ok = raw_depth1 < median_depth * self.config.max_depth_ratio

        accepted = provisional & depth_ok
        valid[idx[accepted]] = True
        return points_3d[accepted], valid

    def compute_reprojection_error(
        self, points_3d: np.ndarray, pts_2d: np.ndarray, P: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Project 3D points into pixels and measure Euclidean pixel error."""
        if len(points_3d) == 0:
            return float("inf"), np.empty(0, dtype=np.float64)

        points_h = np.hstack([points_3d, np.ones((len(points_3d), 1), dtype=np.float64)])
        projected = (P @ points_h.T).T
        z = projected[:, 2:3]
        valid_z = np.abs(z[:, 0]) > 1e-12
        pixels = np.full((len(points_3d), 2), np.nan, dtype=np.float64)
        pixels[valid_z] = projected[valid_z, :2] / z[valid_z]
        errors = np.linalg.norm(pixels - pts_2d, axis=1)
        errors[~np.isfinite(errors)] = np.inf
        return float(np.mean(errors[np.isfinite(errors)])) if np.isfinite(errors).any() else float("inf"), errors
