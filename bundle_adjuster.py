"""Local bundle adjustment (SciPy, sparse).

Bundle adjustment jointly optimizes camera poses and 3D landmarks so their
projected pixels match the observed keypoints. Reprojection error is the right
loss because cameras observe pixels, not abstract pose parameters. Levenberg-
Marquardt blends gradient descent and Gauss-Newton updates, which makes it
useful for nonlinear camera geometry. One pose is fixed because SLAM has gauge
freedom: without an anchor, the whole reconstruction can translate, rotate, or
scale without changing reprojection error. Local BA reduces drift by repeatedly
pulling recent poses and points back into geometric agreement.

On a g2o backend: this project's g2o-python bindings do import successfully,
and g2o's own SE3-only pose graph (see pose_graph.py, used for loop-closure
correction) checks out correctly against a synthetic test. Its joint
pose+landmark bundle-adjustment path (EdgeSE3ProjectXYZ + VertexPointXYZ) is a
different story: multiple vertex/parameter wiring attempts against this
build either produced divergent, physically-nonsensical optimized poses or
segfaulted outright, with no documentation available for this specific
binding's exact conventions to debug further against. Shipping that would
risk silently corrupting the map, which is worse than not having it, so this
stays on the portable, fully-tested SciPy path -- now with a real sparse
Jacobian instead of the dense finite-difference one that made local BA the
slowest part of the whole pipeline once the map started growing continuously.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from config import Config
from map import Keyframe, Map, Pose


class BundleAdjuster:
    """Optimize a small recent window of poses and landmarks."""

    def __init__(self, config: Config, K: np.ndarray) -> None:
        """Store camera intrinsics."""
        self.config = config
        self.K = K.astype(np.float64)
        self.backend = "scipy"

    def optimize(self, slam_map: Map) -> str:
        """Run local BA on the newest keyframes and return a status string."""
        if len(slam_map.keyframes) < 2 or len(slam_map.landmarks) < 6:
            return "skipped: not enough keyframes or landmarks"
        return self._optimize_scipy(slam_map)

    def _window(self, slam_map: Map) -> Tuple[List[Keyframe], List[int]]:
        """Select recent keyframes and landmarks observed inside the window."""
        keyframes = slam_map.keyframes[-self.config.ba_window_size :]
        keyframe_ids = {kf.frame_id for kf in keyframes}
        landmark_ids = [
            lid for lid, lm in slam_map.landmarks.items()
            if sum(1 for fid in lm.observations if fid in keyframe_ids) >= 2
        ]
        return keyframes, landmark_ids

    def _pack(self, keyframes: List[Keyframe], landmark_ids: List[int], slam_map: Map) -> np.ndarray:
        """Pack variable poses except the first anchor plus 3D points."""
        params: List[np.ndarray] = []
        for keyframe in keyframes[1:]:
            rvec, _ = cv2.Rodrigues(keyframe.pose.R)
            params.extend([rvec.reshape(3), keyframe.pose.t.reshape(3)])
        for landmark_id in landmark_ids:
            params.append(slam_map.landmarks[landmark_id].position.reshape(3))
        return np.concatenate(params).astype(np.float64)

    def _unpack(
        self, x: np.ndarray, keyframes: List[Keyframe], landmark_ids: List[int], slam_map: Map
    ) -> Tuple[Dict[int, Pose], Dict[int, np.ndarray]]:
        """Convert a parameter vector back into pose and point dictionaries."""
        cursor = 0
        poses: Dict[int, Pose] = {keyframes[0].frame_id: keyframes[0].pose}
        for keyframe in keyframes[1:]:
            rvec = x[cursor : cursor + 3]
            tvec = x[cursor + 3 : cursor + 6].reshape(3, 1)
            cursor += 6
            R, _ = cv2.Rodrigues(rvec)
            poses[keyframe.frame_id] = Pose(R, tvec)

        points: Dict[int, np.ndarray] = {}
        for landmark_id in landmark_ids:
            points[landmark_id] = x[cursor : cursor + 3]
            cursor += 3
        return poses, points

    def _observation_list(
        self, keyframes: List[Keyframe], landmark_ids: List[int], slam_map: Map
    ) -> List[Tuple[int, int, int]]:
        """Return (landmark_id, frame_id, keypoint_idx) in the exact order
        _residuals() emits residuals for, so the sparsity pattern built from
        this list is guaranteed to line up with the real Jacobian structure
        instead of duplicating (and risking drifting out of sync with) the
        iteration logic in _residuals() itself.
        """
        keyframe_ids = {kf.frame_id for kf in keyframes}
        observations: List[Tuple[int, int, int]] = []
        for landmark_id in landmark_ids:
            landmark = slam_map.landmarks[landmark_id]
            for frame_id, keypoint_idx in landmark.observations.items():
                if frame_id in keyframe_ids:
                    observations.append((landmark_id, frame_id, keypoint_idx))
        return observations

    def _residuals(self, x: np.ndarray, keyframes: List[Keyframe], landmark_ids: List[int], slam_map: Map) -> np.ndarray:
        """Compute stacked robust reprojection residuals for least_squares."""
        poses, points = self._unpack(x, keyframes, landmark_ids, slam_map)
        keyframe_by_id = {kf.frame_id: kf for kf in keyframes}
        residuals: List[float] = []

        for landmark_id, frame_id, keypoint_idx in self._observation_list(keyframes, landmark_ids, slam_map):
            point = points[landmark_id]
            pose = poses[frame_id]
            keypoint = keyframe_by_id[frame_id].keypoints[keypoint_idx]
            point_cam = pose.R @ point.reshape(3, 1) + pose.t
            z = float(point_cam[2])
            if z <= 1e-8:
                residuals.extend([100.0, 100.0])
                continue
            pixel = (self.K @ point_cam).reshape(3)
            pixel = pixel[:2] / pixel[2]
            residuals.extend((pixel - np.array(keypoint.pt, dtype=np.float64)).tolist())

        return np.asarray(residuals, dtype=np.float64)

    def _jacobian_sparsity(
        self, keyframes: List[Keyframe], landmark_ids: List[int], slam_map: Map
    ) -> lil_matrix:
        """Build the block-sparse structure of the Jacobian: each observation's
        2 residuals depend only on its landmark's 3 point parameters and (if
        not the fixed anchor keyframe) its keyframe's 6 pose parameters.

        Passing this to `least_squares` turns finite-difference Jacobian
        estimation from one function evaluation per parameter into one per
        *color group* of non-overlapping parameters, which is what actually
        made local BA usable once maps grew past a couple hundred points
        instead of being reconstructed once and left static.
        """
        n_pose_params = 6 * (len(keyframes) - 1)
        pose_param_start = {kf.frame_id: 6 * i for i, kf in enumerate(keyframes[1:])}
        point_param_start = {lid: n_pose_params + 3 * i for i, lid in enumerate(landmark_ids)}

        observations = self._observation_list(keyframes, landmark_ids, slam_map)
        n_residuals = 2 * len(observations)
        n_params = n_pose_params + 3 * len(landmark_ids)
        sparsity = lil_matrix((n_residuals, n_params), dtype=bool)

        for obs_idx, (landmark_id, frame_id, _keypoint_idx) in enumerate(observations):
            rows = (2 * obs_idx, 2 * obs_idx + 1)
            point_col = point_param_start[landmark_id]
            for row in rows:
                sparsity[row, point_col : point_col + 3] = True
            if frame_id in pose_param_start:  # the anchor keyframe has no free pose params
                pose_col = pose_param_start[frame_id]
                for row in rows:
                    sparsity[row, pose_col : pose_col + 6] = True

        return sparsity

    def _optimize_scipy(self, slam_map: Map) -> str:
        """Run local BA with a sparse Jacobian structure."""
        keyframes, landmark_ids = self._window(slam_map)
        if len(keyframes) < 2 or len(landmark_ids) < 3:
            return "skipped: local BA window too small"

        x0 = self._pack(keyframes, landmark_ids, slam_map)
        if x0.size == 0:
            return "skipped: no BA variables"

        sparsity = self._jacobian_sparsity(keyframes, landmark_ids, slam_map)

        # max_nfev previously reused ba_max_iterations (a small number meant
        # to bound *solver iterations*) directly as a cap on total *function
        # evaluations*. With a dense finite-difference Jacobian, estimating
        # just one Jacobian can cost more evaluations than that on any
        # nontrivial problem, so BA was silently stopping after (partway
        # through) its first iteration regardless of the configured value.
        # A sparse Jacobian needs far fewer evaluations per estimate, so the
        # cap is now sized to the actual problem (with slack for multiple
        # iterations) instead of reusing an unrelated config value.
        max_nfev = max(50, 10 * x0.size)

        result = least_squares(
            self._residuals,
            x0,
            args=(keyframes, landmark_ids, slam_map),
            method="trf",
            loss="huber",
            f_scale=2.0,
            jac_sparsity=sparsity,
            max_nfev=max_nfev,
            verbose=0,
        )

        poses, points = self._unpack(result.x, keyframes, landmark_ids, slam_map)
        for keyframe in keyframes[1:]:
            keyframe.pose = poses[keyframe.frame_id]
        for landmark_id, point in points.items():
            slam_map.landmarks[landmark_id].position = point.reshape(3)
        return f"scipy(sparse): cost={result.cost:.3f}, nfev={result.nfev}, params={x0.size}"
