"""PnP tracking against an existing sparse 3D map.

After initialization, SLAM usually prefers PnP over a fresh essential matrix
because PnP uses 3D-2D correspondences: known map points matched to current image
features. Essential-matrix tracking only uses 2D-2D matches and again suffers
from unknown scale. EPnP is fast and non-iterative, DLS solves polynomial camera
pose equations, and iterative PnP refines a pose by minimizing reprojection
error. OpenCV's RANSAC wrapper can use EPnP to survive outlier matches.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from config import Config
from feature_extractor import FeatureExtractor, FeatureSet
from map import Map, Pose
from matcher import Matcher


class Tracker:
    """Estimate current camera pose by matching frame features to map points."""

    def __init__(self, config: Config, extractor: FeatureExtractor, matcher: Matcher, K: np.ndarray) -> None:
        """Share extractor/matcher instances with the main SLAM system."""
        self.config = config
        self.extractor = extractor
        self.matcher = matcher
        self.K = K

    def track(
        self, image: np.ndarray, slam_map: Map, last_pose: Pose
    ) -> Tuple[Pose, int, List[int], FeatureSet, List[cv2.DMatch]]:
        """Track one frame using solvePnPRansac and return pose plus inliers."""
        features = self.extractor.detect_and_compute(image)

        recent_keyframe_ids = {kf.frame_id for kf in slam_map.keyframes[-self.config.local_map_keyframes :]}
        map_descriptors, landmark_ids = slam_map.get_local_descriptors_and_ids(recent_keyframe_ids)
        if map_descriptors is None or len(landmark_ids) < 6:
            # Fall back to the full map early on, before enough keyframes
            # exist for a "local" window to contain enough points on its own.
            map_descriptors, landmark_ids = slam_map.get_descriptors_and_ids()

        if features.descriptors is None or map_descriptors is None or len(landmark_ids) < 6:
            raise RuntimeError("Not enough descriptors or map points for PnP tracking")

        matches = self.matcher.match_descriptors(map_descriptors, features.descriptors)
        if len(matches) < 6:
            raise RuntimeError(f"PnP needs at least 6 matches, got {len(matches)}")

        object_points = np.float32([slam_map.landmarks[landmark_ids[m.queryIdx]].position for m in matches])
        image_points = np.float32([features.keypoints[m.trainIdx].pt for m in matches])

        ok, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.K,
            None,
            iterationsCount=self.config.ransac_max_iterations,
            reprojectionError=max(2.0, self.config.ransac_inlier_threshold_px * 2.0),
            confidence=self.config.ransac_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not ok or inlier_idx is None or len(inlier_idx) < 6:
            # EPnP+RANSAC found no consensus pose from a cold start. Before
            # giving up, try one more solve seeded from the last known pose:
            # if the camera moved only a little since the last frame (the
            # common case), an iterative solve starting there can converge
            # where a fresh RANSAC search found nothing to agree on. This is
            # a plain constant-position motion model, not true motion
            # prediction (no velocity is tracked), but it's a real fallback
            # rather than `last_pose` being accepted as a parameter and never
            # used.
            recovered = self._track_from_last_pose(object_points, image_points, last_pose)
            if recovered is None:
                raise RuntimeError("PnP RANSAC failed to find a stable camera pose")
            rvec, tvec, inlier_idx = recovered

        inlier_idx = inlier_idx.reshape(-1)
        inlier_object = object_points[inlier_idx]
        inlier_image = image_points[inlier_idx]
        ok_refine, rvec, tvec = cv2.solvePnP(
            inlier_object,
            inlier_image,
            self.K,
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok_refine:
            raise RuntimeError("Iterative PnP refinement failed")

        R, _ = cv2.Rodrigues(rvec)
        tracked_ids = [landmark_ids[matches[i].queryIdx] for i in inlier_idx]
        inlier_matches = [matches[i] for i in inlier_idx]
        return Pose(R.astype(np.float64), tvec.astype(np.float64).reshape(3, 1)), len(inlier_idx), tracked_ids, features, inlier_matches

    def _track_from_last_pose(
        self, object_points: np.ndarray, image_points: np.ndarray, last_pose: Pose
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Constant-position fallback: solve PnP seeded from the previous
        frame's pose against every candidate correspondence, then keep only
        the ones that actually reproject well under that solution.

        Returns (rvec, tvec, inlier_idx) in the same shape solvePnPRansac
        would, or None if this doesn't recover a plausible pose either.
        """
        rvec0, _ = cv2.Rodrigues(last_pose.R)
        tvec0 = last_pose.t.reshape(3, 1).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, self.K, None, rvec0, tvec0,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.K, None)
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        threshold = max(2.0, self.config.ransac_inlier_threshold_px * 2.0)
        inlier_idx = np.flatnonzero(errors < threshold)
        if len(inlier_idx) < 6:
            return None
        return rvec, tvec, inlier_idx.reshape(-1, 1)
