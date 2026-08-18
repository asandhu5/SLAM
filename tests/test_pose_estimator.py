"""Synthetic essential-matrix pose tests."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from pose_estimator import PoseEstimator


def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project world points through a calibrated pinhole camera."""
    cam = (R @ points.T + t.reshape(3, 1)).T
    pix = (K @ cam.T).T
    return pix[:, :2] / pix[:, 2:3]


def angle_between_rotations(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Return rotation difference in degrees."""
    R_delta = R_est @ R_gt.T
    value = (np.trace(R_delta) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def direction_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Return translation-direction error in degrees.

    Only direction is tested because monocular geometry cannot recover the
    absolute length of translation; scale is fundamentally ambiguous.
    """
    a = t_est.reshape(3) / np.linalg.norm(t_est)
    b = t_gt.reshape(3) / np.linalg.norm(t_gt)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def synthetic_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate calibrated correspondences from a known relative pose."""
    rng = np.random.default_rng(8)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    points = rng.uniform([-2.0, -1.2, 4.0], [2.0, 1.2, 8.0], size=(220, 3))
    rvec = np.array([0.0, np.deg2rad(4.0), 0.0])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[0.35], [0.0], [0.04]])
    pts1 = project(points, np.eye(3), np.zeros((3, 1)), K)
    pts2 = project(points, R, t, K)
    pts1 += rng.normal(0, 0.15, pts1.shape)
    pts2 += rng.normal(0, 0.15, pts2.shape)
    return pts1.astype(np.float32), pts2.astype(np.float32), K, R, t


def test_recover_pose_close_to_ground_truth() -> None:
    """Essential matrix recovery should match known synthetic motion."""
    pts1, pts2, K, R_gt, t_gt = synthetic_scene()
    estimator = PoseEstimator(Config(ransac_inlier_threshold_px=1.0))
    pose, mask = estimator.estimate_relative_pose(pts1, pts2, K)
    assert mask.sum() > 150
    assert angle_between_rotations(pose.R, R_gt) < 5.0
    assert direction_error(pose.t, t_gt) < 5.0


if __name__ == "__main__":
    pts1, pts2, K, R_gt, t_gt = synthetic_scene()
    estimator = PoseEstimator(Config(ransac_inlier_threshold_px=1.0))
    pose, mask = estimator.estimate_relative_pose(pts1, pts2, K)
    test_recover_pose_close_to_ground_truth()
    # This file only exercises essential-matrix recovery on a synthetic two-
    # view scene, so it reports the errors it actually measures (rotation
    # and translation-direction error). It has never computed trajectory
    # ATE -- an unrelated, unearned "ATE: 0.003m" used to be printed here
    # regardless of what the test actually checked; real ATE numbers now
    # come from a real trajectory run, e.g. benchmark.py against tests/
    # or the SLAM dashboard's results page.
    print(
        f"All tests passed. rotation_error={angle_between_rotations(pose.R, R_gt):.3f} deg, "
        f"translation_direction_error={direction_error(pose.t, t_gt):.3f} deg, inliers={int(mask.sum())}"
    )
