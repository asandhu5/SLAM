"""Simple ORB bag-of-words style loop detection.

Loop closure detects when the camera returns to a previously seen place. Without
loops, small pose errors accumulate into drift. A classic bag-of-words system
treats descriptors like visual words and frames like documents; TF-IDF weights
words that are distinctive across the sequence. DBoW3 is a strong C++ library,
but Python bindings are not reliable, so this module implements a lightweight
appearance fallback using ORB descriptor matching and temporal consistency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from matcher import Matcher
from pose_estimator import PoseEstimator


@dataclass
class LoopCandidate:
    """A potential loop closure between current frame and an old keyframe."""

    frame_id: int
    score: float
    relative_R: Optional[np.ndarray] = None
    relative_t: Optional[np.ndarray] = None


class LoopDetector:
    """Detect visually similar non-neighbor keyframes."""

    def __init__(self, config: Config, matcher: Matcher, pose_estimator: PoseEstimator, K: np.ndarray) -> None:
        """Create an in-memory descriptor database."""
        self.config = config
        self.matcher = matcher
        self.pose_estimator = pose_estimator
        self.K = K
        self.database: List[Tuple[int, List[cv2.KeyPoint], Optional[np.ndarray]]] = []
        self.recent_candidates: Deque[int] = deque(maxlen=config.loop_min_consistent_frames)

    def add_keyframe(self, frame_id: int, keypoints: List[cv2.KeyPoint], descriptors: Optional[np.ndarray]) -> None:
        """Store descriptors from a keyframe for future place recognition."""
        self.database.append((frame_id, keypoints, descriptors))

    def detect_loop(
        self, frame_id: int, keypoints: List[cv2.KeyPoint], descriptors: Optional[np.ndarray]
    ) -> Optional[LoopCandidate]:
        """Return a temporally consistent loop candidate or None."""
        if descriptors is None or len(descriptors) < 20 or len(self.database) < 5:
            return None

        best: Optional[LoopCandidate] = None
        best_matches = []
        best_kps = []
        for old_id, old_kps, old_desc in self.database:
            if old_desc is None or abs(frame_id - old_id) < 30:
                continue
            matches = self.matcher.match_descriptors(old_desc, descriptors)
            score = len(matches) / max(1, min(len(old_desc), len(descriptors)))
            if score > self.config.loop_min_score and (best is None or score > best.score):
                best = LoopCandidate(old_id, score)
                best_matches = matches
                best_kps = old_kps

        if best is None:
            self.recent_candidates.clear()
            return None

        self.recent_candidates.append(best.frame_id)
        if len(self.recent_candidates) < self.config.loop_min_consistent_frames:
            return None
        if len(set(self.recent_candidates)) != 1:
            return None

        pts_old = np.float32([best_kps[m.queryIdx].pt for m in best_matches])
        pts_new = np.float32([keypoints[m.trainIdx].pt for m in best_matches])
        if len(pts_old) >= 8:
            try:
                pose, _ = self.pose_estimator.estimate_relative_pose(pts_old, pts_new, self.K)
                best.relative_R = pose.R
                best.relative_t = pose.t
            except Exception:
                pass
        return best
