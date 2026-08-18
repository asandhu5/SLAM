"""Trajectory evaluation for monocular SLAM.

ATE measures absolute position agreement after aligning estimated trajectory to
ground truth. RPE measures frame-to-frame relative motion error, so it exposes
local drift even when the global path looks aligned. Monocular SLAM needs
Umeyama similarity alignment because estimated scale is arbitrary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def umeyama_alignment(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Align source points to target with similarity transform."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be N x 3 arrays")

    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)
    src_centered = source - mu_src
    tgt_centered = target - mu_tgt
    cov = (tgt_centered.T @ src_centered) / len(source)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = 1.0
    if with_scale:
        var_src = np.mean(np.sum(src_centered * src_centered, axis=1))
        scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1e-12))
    t = mu_tgt - scale * R @ mu_src
    return scale, R, t


def apply_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply similarity transform to N x 3 points."""
    return scale * (R @ points.T).T + t.reshape(1, 3)


def compute_ate(estimated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute Absolute Trajectory Error after similarity alignment."""
    scale, R, t = umeyama_alignment(estimated, ground_truth, with_scale=True)
    aligned = apply_similarity(estimated, scale, R, t)
    errors = np.linalg.norm(aligned - ground_truth, axis=1)
    return {
        "ATE_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ATE_mean": float(np.mean(errors)),
        "scale": float(scale),
    }


def compute_rpe(estimated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute translation-only Relative Pose Error between consecutive poses."""
    if len(estimated) < 2:
        return {"RPE_rmse": float("nan"), "RPE_mean": float("nan")}
    est_delta = np.diff(estimated, axis=0)
    gt_delta = np.diff(ground_truth, axis=0)
    if np.allclose(est_delta, gt_delta, atol=1e-12):
        return {"RPE_rmse": 0.0, "RPE_mean": 0.0}
    if np.mean(np.sum((est_delta - est_delta.mean(axis=0)) ** 2, axis=1)) < 1e-12:
        errors = np.linalg.norm(est_delta - gt_delta, axis=1)
        return {"RPE_rmse": float(np.sqrt(np.mean(errors ** 2))), "RPE_mean": float(np.mean(errors))}
    scale, R, _ = umeyama_alignment(est_delta, gt_delta, with_scale=True)
    aligned_delta = scale * (R @ est_delta.T).T
    errors = np.linalg.norm(aligned_delta - gt_delta, axis=1)
    return {"RPE_rmse": float(np.sqrt(np.mean(errors ** 2))), "RPE_mean": float(np.mean(errors))}


def evaluate_trajectory(estimated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Return ATE and RPE metrics in one table-friendly dictionary."""
    n = min(len(estimated), len(ground_truth))
    if n < 2:
        raise ValueError("Need at least two poses to evaluate a trajectory")
    estimated = estimated[:n]
    ground_truth = ground_truth[:n]
    return {**compute_ate(estimated, ground_truth), **compute_rpe(estimated, ground_truth)}


def load_kitti_poses(path: str | Path) -> np.ndarray:
    """Load KITTI 3x4 pose rows and return camera centers."""
    poses = []
    for line in Path(path).read_text().strip().splitlines():
        values = np.fromstring(line, sep=" ")
        T = values.reshape(3, 4)
        R = T[:3, :3]
        t = T[:3, 3:4]
        poses.append((-R.T @ t).reshape(3))
    return np.vstack(poses)


def plot_trajectory_comparison(estimated: np.ndarray, ground_truth: np.ndarray, output: str | Path) -> None:
    """Save top-down estimated-vs-ground-truth trajectory plot."""
    scale, R, t = umeyama_alignment(estimated, ground_truth, with_scale=True)
    aligned = apply_similarity(estimated, scale, R, t)
    plt.figure(figsize=(8, 5))
    plt.plot(ground_truth[:, 0], ground_truth[:, 2], label="ground truth", linewidth=2)
    plt.plot(aligned[:, 0], aligned[:, 2], label="estimated aligned", linewidth=2)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    """CLI entry point for evaluating saved trajectory CSV files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimated", required=True, help="CSV with x,y,z estimated camera centers")
    parser.add_argument("--ground-truth", required=True, help="CSV with x,y,z ground-truth camera centers")
    args = parser.parse_args()
    estimated = np.loadtxt(args.estimated, delimiter=",")
    ground_truth = np.loadtxt(args.ground_truth, delimiter=",")
    print(evaluate_trajectory(estimated, ground_truth))


if __name__ == "__main__":
    main()
