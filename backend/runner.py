"""Runs the real SLAM pipeline in a background thread and reports live
telemetry + final results back through the RunStore.

This is the only place that imports the actual slam_project modules
(config, datasets, slam_system, benchmark), keeping the Flask layer itself
free of SLAM-specific logic.
"""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets  # noqa: E402
from benchmark import apply_similarity, evaluate_trajectory, umeyama_alignment  # noqa: E402
from config import Config  # noqa: E402
from slam_system import SLAMSystem  # noqa: E402

from .store import RunStore  # noqa: E402

logger = logging.getLogger(__name__)

LOG_TAIL_LENGTH = 60
HISTORY_SAMPLE_EVERY = 1
MAP_PREVIEW_MAX_POINTS = 4000
HISTORY_MAX_POINTS = 400


class _LiveLogStream:
    """A stdout replacement that forwards each printed line into the run
    store's rolling log tail, so the browser can show a live console instead
    of the pipeline's print()s only ever reaching a server-side terminal.

    print() with multiple arguments (e.g. `print("Local BA:", result)`)
    calls the underlying stream's write() once per argument/separator, not
    once for the whole formatted line -- naively treating every write() as
    one line would split those across several log entries. Text is
    line-buffered here instead: only text up to (and including) a newline
    is ever flushed into the log tail, and any trailing partial line is
    held until the rest of it arrives.
    """

    def __init__(self, store: RunStore, run_id: str):
        self._store = store
        self._run_id = run_id
        self._buffer = deque(maxlen=LOG_TAIL_LENGTH)
        self._real_stdout = sys.stdout
        self._pending = ""

    def write(self, text: str) -> None:
        self._real_stdout.write(text)
        self._pending += text
        if "\n" not in self._pending:
            return
        *complete_lines, self._pending = self._pending.split("\n")
        changed = False
        for line in complete_lines:
            if line:
                self._buffer.append(line)
                changed = True
        if changed:
            self._store.update(self._run_id, log_tail=list(self._buffer))

    def flush(self) -> None:
        self._real_stdout.flush()


def _downsample(values: List, max_points: int) -> List:
    if len(values) <= max_points:
        return values
    step = len(values) / max_points
    return [values[int(i * step)] for i in range(max_points)]


def _prepare_map_preview(points: np.ndarray, trajectory: np.ndarray, max_points: int) -> np.ndarray:
    """Drop the most extreme outliers, then subsample for the browser chart.

    The triangulator already rejects points whose depth badly exceeds their
    own triangulation batch's median (see triangulator.py), but that check
    can't catch a batch where most points triangulated were themselves only
    moderately degenerate -- and this map's depth distribution turned out to
    have a long enough tail that a percentile cutoff alone (98th, tried
    first) still left a visually-dominant handful of outliers behind, since
    percentile cutoffs don't help when a meaningful fraction of the tail is
    bad, not just its very extreme.

    The trajectory itself, after Umeyama-aligning to ground truth (or just
    its own internal scale when there's no ground truth to align to), is a
    reference this map has no reason to distrust -- landmarks living many
    times further from it than the camera ever traveled are exactly the
    near-degenerate triangulations this whole check exists to catch, so
    they're clipped here using that as the yardstick instead.
    """
    if len(points) == 0:
        return points
    if len(trajectory) > 0:
        center = np.median(trajectory, axis=0)
        radius = float(np.linalg.norm(trajectory - center, axis=1).max()) if len(trajectory) > 1 else 1.0
        radius = max(radius, 0.5)
        cutoff = radius * 20.0
        distances = np.linalg.norm(points - center, axis=1)
        points = points[distances <= cutoff]
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def run_slam(store: RunStore, run_id: str, dataset_kind: str, dataset_path: str, max_frames: int) -> None:
    started = time.monotonic()
    log_stream = _LiveLogStream(store, run_id)
    real_stdout = sys.stdout
    sys.stdout = log_stream
    try:
        _run_slam_inner(store, run_id, dataset_kind, dataset_path, max_frames, started)
    except Exception as exc:  # noqa: BLE001 - background worker must never crash silently
        logger.exception("Unhandled failure running SLAM (run %s)", run_id)
        store.update(run_id, status="error", error=str(exc))
    finally:
        sys.stdout = real_stdout


def _run_slam_inner(
    store: RunStore, run_id: str, dataset_kind: str, dataset_path: str, max_frames: int, started: float
) -> None:
    if dataset_kind == "tum":
        sample = datasets.load_tum(dataset_path, max_frames=max_frames)
    elif dataset_kind == "kitti":
        sample = datasets.load_kitti(dataset_path, max_frames=max_frames)
    else:
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")

    if not sample.image_paths:
        raise RuntimeError("No images found for this dataset/frame range.")

    total = len(sample.image_paths)
    store.update(run_id, total_frames=total, dataset_label=sample.name)

    cfg = Config()
    slam = SLAMSystem(config=cfg, K=sample.K)

    landmark_history: List[int] = []
    keyframe_history: List[int] = []
    state_timeline: List[str] = []
    processed = 0

    for i, image_path in enumerate(sample.image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if sample.distortion is not None:
            image = cv2.undistort(image, sample.K, sample.distortion)

        state = slam.process_frame(image, float(i))
        processed += 1
        landmark_history.append(len(slam.map.landmarks))
        keyframe_history.append(len(slam.map.keyframes))
        state_timeline.append(state.value)

        if i % HISTORY_SAMPLE_EVERY == 0 or i == total - 1:
            store.update(
                run_id,
                current_frame=i + 1,
                state=state.value,
                landmarks=len(slam.map.landmarks),
                keyframes=len(slam.map.keyframes),
            )

    trajectory = (
        np.vstack([pose.camera_center() for pose in slam.get_trajectory()])
        if slam.get_trajectory()
        else np.empty((0, 3))
    )
    map_points = slam.get_map()

    metrics: dict = {}
    plotted_trajectory = trajectory
    ground_truth_for_plot: Optional[np.ndarray] = None

    if sample.ground_truth is not None and len(trajectory) >= 2:
        n = min(len(trajectory), len(sample.ground_truth))
        try:
            metrics = evaluate_trajectory(trajectory[:n], sample.ground_truth[:n])
            scale, R, t = umeyama_alignment(trajectory[:n], sample.ground_truth[:n], with_scale=True)
            plotted_trajectory = apply_similarity(trajectory[:n], scale, R, t)
            ground_truth_for_plot = sample.ground_truth[:n]
        except Exception:
            logger.exception("Trajectory evaluation failed for run %s", run_id)

    results = {
        # Aligned to ground truth (when available) -- meaningful to compare
        # directly against the ground_truth series below.
        "trajectory": plotted_trajectory.tolist(),
        "ground_truth": ground_truth_for_plot.tolist() if ground_truth_for_plot is not None else None,
        # Raw SLAM-frame trajectory -- NOT the same coordinate frame as
        # "trajectory" once alignment has rotated/scaled/translated it, but
        # it's the frame map_points is actually in, so the map-points chart
        # (trajectory overlaid on the point cloud) needs this one instead,
        # or the overlay would silently be wrong.
        "trajectory_raw": trajectory.tolist(),
        "map_points": _prepare_map_preview(map_points, trajectory, MAP_PREVIEW_MAX_POINTS).tolist(),
        "landmark_history": _downsample(landmark_history, HISTORY_MAX_POINTS),
        "keyframe_history": _downsample(keyframe_history, HISTORY_MAX_POINTS),
        "state_timeline": _downsample(state_timeline, HISTORY_MAX_POINTS),
    }
    store.save_results(run_id, results)
    store.update(
        run_id,
        status="ready",
        processed_frames=processed,
        final_landmarks=len(slam.map.landmarks),
        final_keyframes=len(slam.map.keyframes),
        duration_seconds=round(time.monotonic() - started, 1),
        has_ground_truth=ground_truth_for_plot is not None,
        ATE_rmse=metrics.get("ATE_rmse"),
        ATE_mean=metrics.get("ATE_mean"),
        RPE_rmse=metrics.get("RPE_rmse"),
        RPE_mean=metrics.get("RPE_mean"),
        scale=metrics.get("scale"),
    )
