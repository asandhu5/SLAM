"""Essential-matrix estimation and relative-pose recovery.

The essential matrix E encodes the epipolar geometry between two calibrated
camera views. For corresponding normalized image points x1 and x2, the
epipolar constraint is x2.T @ E @ x1 = 0. In plain terms, once a 3D point is
seen in camera 1, its matching point in camera 2 must lie on a specific line.

Real feature matches contain outliers from repeated texture, moving objects,
and descriptor mistakes. RANSAC repeatedly fits E from small random samples and
keeps the model supported by the largest set of inliers. Inliers are matches
consistent with one rigid camera motion; outliers are geometrically impossible
under that motion. The intrinsic matrix K maps between pixel coordinates and
camera-normalized rays using focal lengths and principal point.

Monocular scale ambiguity: from one moving camera, geometry recovers translation
direction but not metric translation length. A one-meter scene and a ten-meter
scaled copy project identically if camera motion is scaled too, so monocular
SLAM needs external scale, known baselines, IMU, wheel odometry, or trajectory
alignment to report metric scale.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from config import Config
from map import Pose


class PoseEstimator:
    """Estimate relative camera motion from 2D-2D feature correspondences."""

    def __init__(self, config: Config) -> None:
        """Store RANSAC thresholds from the central config."""
        self.config = config

    def estimate_essential_matrix(
        self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Fit E with RANSAC and return E, inlier mask, and inlier count."""
        if len(pts1) < 8 or len(pts2) < 8:
            raise ValueError("At least 8 correspondences are required to estimate an essential matrix")

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=self.config.ransac_confidence,
            threshold=self.config.ransac_inlier_threshold_px,
            maxIters=self.config.ransac_max_iterations,
        )

        if E is None or mask is None:
            raise RuntimeError("Essential matrix estimation failed")

        if E.shape[0] > 3:
            E = E[:3, :3]

        mask = mask.astype(bool).reshape(-1)
        return E, mask, int(mask.sum())

    def recover_pose(
        self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Recover relative rotation and translation direction from E.

        E decomposes into four mathematical solutions: two possible rotations
        and two opposite translation directions. OpenCV resolves this with a
        cheirality test, choosing the solution that places the most triangulated
        points in front of both cameras. R is the rotation from camera 1 to
        camera 2. t is the unit-length translation direction, not metric scale.
        """
        inliers, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
        return R.astype(np.float64), t.astype(np.float64).reshape(3, 1), int(inliers), pose_mask.astype(bool).reshape(-1)

    def estimate_relative_pose(self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> Tuple[Pose, np.ndarray]:
        """Estimate relative pose and combined inlier mask in one call."""
        E, e_mask, _ = self.estimate_essential_matrix(pts1, pts2, K)
        R, t, _, p_mask = self.recover_pose(E, pts1[e_mask], pts2[e_mask], K)
        combined = np.zeros(len(pts1), dtype=bool)
        combined[np.flatnonzero(e_mask)[p_mask]] = True
        return Pose(R, t), combined
