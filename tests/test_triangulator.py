"""Tests for two-view triangulation."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from map import Pose
from triangulator import Triangulator


def project(points: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project 3D points with a 3x4 camera matrix."""
    hom = np.hstack([points, np.ones((len(points), 1))])
    pix = (P @ hom.T).T
    return pix[:, :2] / pix[:, 2:3]


def test_triangulates_known_points() -> None:
    """Triangulated points should recover noise-free geometry accurately."""
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    pose1 = Pose.identity()
    pose2 = Pose(np.eye(3), np.array([[-0.4], [0.0], [0.0]]))
    points = np.array([[0.1, 0.0, 4.0], [0.5, 0.2, 5.0], [-0.3, -0.1, 6.0]])
    P1 = pose1.projection_matrix(K)
    P2 = pose2.projection_matrix(K)
    pts1 = project(points, P1)
    pts2 = project(points, P2)
    tri = Triangulator(Config(max_reprojection_error=0.5))
    recovered, valid = tri.triangulate(pts1, pts2, P1, P2)
    assert valid.sum() == len(points)
    assert np.max(np.linalg.norm(recovered - points, axis=1)) < 0.01


def test_rejects_point_far_outside_the_batchs_own_depth_scale() -> None:
    """A point sitting ~100x further away than everything else triangulated
    in the same batch should be rejected even if its reprojection error is
    small -- this is the near-parallel-rays failure mode max_reprojection_error
    alone can't catch (see the max_depth_ratio config field and its docstring).
    """
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    pose1 = Pose.identity()
    pose2 = Pose(np.eye(3), np.array([[-0.4], [0.0], [0.0]]))
    P1 = pose1.projection_matrix(K)
    P2 = pose2.projection_matrix(K)

    normal_points = np.array([[0.1, 0.0, 4.0], [0.5, 0.2, 5.0], [-0.3, -0.1, 6.0], [0.2, -0.2, 5.5]])
    far_point = np.array([[0.1, 0.0, 500.0]])  # ~100x the other points' depth
    points = np.vstack([normal_points, far_point])

    pts1 = project(points, P1)
    pts2 = project(points, P2)
    tri = Triangulator(Config(max_reprojection_error=0.5, max_depth_ratio=10.0))
    recovered, valid = tri.triangulate(pts1, pts2, P1, P2)

    assert valid[:4].all()  # the normal-depth points are all still accepted
    assert not valid[4]  # the far outlier is rejected
    assert len(recovered) == 4


def test_noise_increases_error_but_stays_reasonable() -> None:
    """Small projection noise should create small 3D error."""
    rng = np.random.default_rng(10)
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    pose1 = Pose.identity()
    pose2 = Pose(np.eye(3), np.array([[-0.5], [0.0], [0.0]]))
    points = rng.uniform([-1.0, -0.6, 4.0], [1.0, 0.6, 7.0], size=(50, 3))
    P1 = pose1.projection_matrix(K)
    P2 = pose2.projection_matrix(K)
    pts1 = project(points, P1) + rng.normal(0, 0.1, size=(50, 2))
    pts2 = project(points, P2) + rng.normal(0, 0.1, size=(50, 2))
    tri = Triangulator(Config(max_reprojection_error=1.0))
    recovered, valid = tri.triangulate(pts1, pts2, P1, P2)
    assert valid.sum() > 40
    assert np.mean(np.linalg.norm(recovered - points[valid], axis=1)) < 0.05
