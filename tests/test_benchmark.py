"""Tests for trajectory alignment and metrics."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import compute_ate, evaluate_trajectory


def test_umeyama_ate_recovers_known_similarity() -> None:
    """ATE should be near zero after aligning a scaled/rotated trajectory."""
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [2.0, 0.0, 0.5], [3.0, 0.0, 0.8]])
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), 0.0, -np.sin(theta)], [0.0, 1.0, 0.0], [np.sin(theta), 0.0, np.cos(theta)]])
    est = 0.5 * (R @ gt.T).T + np.array([5.0, 1.0, -2.0])
    ate = compute_ate(est, gt)
    assert ate["ATE_rmse"] < 0.001


def test_evaluate_known_small_error() -> None:
    """A 1 mm perturbation should produce roughly 1 mm ATE."""
    gt = np.stack([np.linspace(0, 1, 20), np.zeros(20), np.linspace(0, 0.5, 20)], axis=1)
    est = gt.copy()
    est[:, 1] += 0.001
    metrics = evaluate_trajectory(est, gt)
    assert metrics["ATE_rmse"] < 0.0011
