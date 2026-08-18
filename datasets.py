"""Dataset loaders: per-dataset calibration, image lists, and (when available)
associated ground-truth trajectories.

Each dataset publishes calibration and ground truth in its own format and
convention; the previous version of this project used one `default_K`
(KITTI's) for every dataset regardless of which camera actually captured the
images, and had no script anywhere that turned TUM's timestamped
groundtruth.txt into something benchmark.py's XYZ-CSV evaluator could
consume. Both are fixed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class DatasetSample:
    """One loadable SLAM dataset: images, intrinsics, and optional ground truth."""

    name: str
    image_paths: List[Path]
    K: np.ndarray
    distortion: Optional[np.ndarray]  # None if the images are already rectified
    ground_truth: Optional[np.ndarray]  # N x 3 camera centers, aligned 1:1 with image_paths


# Freiburg1 (fr1) calibration, published at
# cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats -- distinct from
# both the generic "ROS default" Kinect calibration and the fr2/fr3 cameras.
TUM_FR1_K = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])
TUM_FR1_DISTORTION = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])


def _read_timestamped_file(path: Path) -> List[tuple[float, List[str]]]:
    """Parse a TUM-format '# comment' + 'timestamp value ...' text file."""
    rows: List[tuple[float, List[str]]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), parts[1:]))
    return rows


def load_tum(root: str | Path, max_frames: int = 0) -> DatasetSample:
    """Load a TUM RGB-D sequence (e.g. rgbd_dataset_freiburg1_desk).

    Ground truth is associated to each RGB frame by nearest timestamp
    (standard TUM association, threshold 20ms) when groundtruth.txt exists;
    frames with no association within that window are dropped so image and
    ground-truth arrays stay index-aligned.
    """
    root = Path(root)
    rgb_txt = root / "rgb.txt"
    if not rgb_txt.exists():
        raise FileNotFoundError(f"Expected {rgb_txt} -- not a TUM RGB-D sequence folder")

    rgb_rows = _read_timestamped_file(rgb_txt)
    gt_path = root / "groundtruth.txt"

    images: List[Path] = []
    gt_positions: List[List[float]] = []

    if gt_path.exists():
        gt_rows = _read_timestamped_file(gt_path)
        gt_times = np.array([t for t, _ in gt_rows])
        for t, parts in rgb_rows:
            idx = int(np.argmin(np.abs(gt_times - t)))
            if abs(gt_times[idx] - t) > 0.02:
                continue
            tx, ty, tz = (float(v) for v in gt_rows[idx][1][:3])
            images.append(root / parts[0])
            gt_positions.append([tx, ty, tz])
        ground_truth = np.array(gt_positions, dtype=np.float64) if gt_positions else None
    else:
        images = [root / parts[0] for _, parts in rgb_rows]
        ground_truth = None

    if max_frames:
        images = images[:max_frames]
        if ground_truth is not None:
            ground_truth = ground_truth[:max_frames]

    return DatasetSample(
        name=f"TUM {root.name}",
        image_paths=images,
        K=TUM_FR1_K,
        distortion=TUM_FR1_DISTORTION,
        ground_truth=ground_truth,
    )


def _parse_kitti_calib(calib_path: Path) -> np.ndarray:
    """Read the P0 (rectified left grayscale camera) projection row."""
    for line in calib_path.read_text().splitlines():
        if line.startswith("P0:"):
            values = np.fromstring(line.split(":", 1)[1], sep=" ")
            return values.reshape(3, 4)[:, :3]
    raise ValueError(f"No P0 calibration row found in {calib_path}")


def load_kitti(sequence_dir: str | Path, max_frames: int = 0) -> DatasetSample:
    """Load a KITTI odometry sequence folder (e.g. .../sequences/00).

    Ground truth, when the poses/<seq>.txt file has been downloaded
    alongside the sequence images, is already frame-index-aligned -- KITTI
    publishes exactly one pose per image, in order, with no association
    step needed.
    """
    sequence_dir = Path(sequence_dir)
    images = sorted((sequence_dir / "image_0").glob("*.png"))

    calib_path = sequence_dir / "calib.txt"
    K = _parse_kitti_calib(calib_path) if calib_path.exists() else None
    if K is None:
        raise FileNotFoundError(f"Expected {calib_path} -- not a KITTI odometry sequence folder")

    ground_truth = None
    poses_path = sequence_dir.parent.parent / "poses" / f"{sequence_dir.name}.txt"
    if poses_path.exists():
        centers = []
        for line in poses_path.read_text().strip().splitlines():
            T = np.fromstring(line, sep=" ").reshape(3, 4)
            R, t = T[:, :3], T[:, 3:4]
            centers.append((-R.T @ t).reshape(3))
        ground_truth = np.vstack(centers)

    if max_frames:
        images = images[:max_frames]
        if ground_truth is not None:
            ground_truth = ground_truth[:max_frames]

    return DatasetSample(
        name=f"KITTI seq{sequence_dir.name}",
        image_paths=images,
        K=K,
        distortion=None,  # KITTI's published images are already rectified
        ground_truth=ground_truth,
    )


def load_folder(folder: str | Path, K: np.ndarray, max_frames: int = 0) -> DatasetSample:
    """Load a plain folder of images with an explicitly supplied K and no
    ground truth -- the fallback for anything that isn't TUM or KITTI.
    """
    folder = Path(folder)
    patterns = ("*.png", "*.jpg", "*.jpeg")
    images: List[Path] = sorted({p for pattern in patterns for p in folder.rglob(pattern)})
    if max_frames:
        images = images[:max_frames]
    return DatasetSample(name=folder.name, image_paths=images, K=K, distortion=None, ground_truth=None)
