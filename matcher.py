"""ORB descriptor matching.

A descriptor is a compact signature of the image patch around a keypoint.
ORB descriptors are binary, so Hamming distance is the natural metric: it counts
how many bits differ between two 256-bit fingerprints. Lowe's ratio test rejects
ambiguous matches when the best and second-best candidates look too similar.
Cross-checking adds symmetry: a feature in image A must choose B, and the same
feature in B must choose A.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from config import Config


class Matcher:
    """Match ORB binary descriptors with ratio-test and mutual validation."""

    def __init__(self, config: Config) -> None:
        """Create a brute-force Hamming matcher."""
        self.config = config
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _ratio_matches(self, desc_a: np.ndarray, desc_b: np.ndarray) -> List[cv2.DMatch]:
        """Run KNN matching and keep only Lowe-ratio-valid matches."""
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return []

        knn = self.matcher.knnMatch(desc_a.astype(np.uint8), desc_b.astype(np.uint8), k=2)
        good: List[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < self.config.ratio_test_threshold * second.distance:
                good.append(best)
        return good

    def match_descriptors(self, desc_a: np.ndarray, desc_b: np.ndarray) -> List[cv2.DMatch]:
        """Return matches that pass ratio-test in both directions."""
        forward = self._ratio_matches(desc_a, desc_b)
        backward = self._ratio_matches(desc_b, desc_a)
        reverse_pairs = {(m.trainIdx, m.queryIdx) for m in backward}
        mutual = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse_pairs]
        mutual.sort(key=lambda m: m.distance)
        return mutual

    def match_keypoints(
        self,
        keypoints_a: Sequence[cv2.KeyPoint],
        desc_a: np.ndarray,
        keypoints_b: Sequence[cv2.KeyPoint],
        desc_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """Return matched pixel coordinates from two keypoint sets."""
        matches = self.match_descriptors(desc_a, desc_b)
        pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return pts_a, pts_b, matches

    def matched_pairs_array(
        self,
        keypoints_a: Sequence[cv2.KeyPoint],
        desc_a: np.ndarray,
        keypoints_b: Sequence[cv2.KeyPoint],
        desc_b: np.ndarray,
    ) -> np.ndarray:
        """Return matched keypoint pairs as an M x 2 x 2 array."""
        pts_a, pts_b, _ = self.match_keypoints(keypoints_a, desc_a, keypoints_b, desc_b)
        if len(pts_a) == 0:
            return np.empty((0, 2, 2), dtype=np.float32)
        return np.stack([pts_a, pts_b], axis=1)
