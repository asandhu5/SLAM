"""Sparse map data structures for monocular visual SLAM.

The map stores keyframes, landmarks, observations, and camera poses. It is small
and explicit on purpose: the target reader should be able to inspect every SLAM
object without needing a database or a graph-optimization framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Pose:
    """Camera pose represented as a world-to-camera transform.

    R rotates world coordinates into the camera frame. t translates world points
    into the camera frame after rotation. The camera center in world coordinates
    is -R.T @ t.
    """

    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def identity() -> "Pose":
        """Create the first-camera pose at the world origin."""
        return Pose(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))

    def matrix(self) -> np.ndarray:
        """Return a 4x4 homogeneous world-to-camera transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3:4] = self.t
        return T

    def camera_center(self) -> np.ndarray:
        """Return the camera center in world coordinates."""
        return (-self.R.T @ self.t).reshape(3)

    def projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """Return K [R|t], the matrix that projects world points into pixels."""
        return K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def compose(self, relative: "Pose") -> "Pose":
        """Apply a relative motion after this pose and return the new pose."""
        R_new = relative.R @ self.R
        t_new = relative.R @ self.t + relative.t
        return Pose(R_new, t_new.reshape(3, 1))


@dataclass
class Frame:
    """One video frame with extracted features and optional estimated pose."""

    frame_id: int
    timestamp: float
    image: np.ndarray
    keypoints: List
    descriptors: Optional[np.ndarray]
    pose: Optional[Pose] = None


@dataclass
class Keyframe(Frame):
    """A frame promoted to a durable optimization and loop-closure anchor."""

    observed_landmarks: Dict[int, int] = field(default_factory=dict)


@dataclass
class Landmark:
    """A sparse 3D point and the frames that observed it."""

    landmark_id: int
    position: np.ndarray
    descriptor: Optional[np.ndarray]
    observations: Dict[int, int] = field(default_factory=dict)


class Map:
    """Container for keyframes, landmarks, and trajectory history."""

    def __init__(self) -> None:
        """Start with an empty map and monotonic landmark ids."""
        self.keyframes: List[Keyframe] = []
        self.landmarks: Dict[int, Landmark] = {}
        self.trajectory: List[Pose] = []
        self._next_landmark_id = 0

    def add_pose(self, pose: Pose) -> None:
        """Append one camera pose to the estimated trajectory."""
        self.trajectory.append(pose)

    def add_keyframe(self, frame: Frame) -> Keyframe:
        """Convert a regular frame into a keyframe and store it."""
        keyframe = Keyframe(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image=frame.image,
            keypoints=frame.keypoints,
            descriptors=frame.descriptors,
            pose=frame.pose,
        )
        self.keyframes.append(keyframe)
        return keyframe

    def add_landmark(
        self,
        position: np.ndarray,
        descriptor: Optional[np.ndarray],
        observations: Optional[Dict[int, int]] = None,
    ) -> int:
        """Insert a 3D point and return its new landmark id."""
        landmark_id = self._next_landmark_id
        self._next_landmark_id += 1
        self.landmarks[landmark_id] = Landmark(
            landmark_id=landmark_id,
            position=np.asarray(position, dtype=np.float64).reshape(3),
            descriptor=None if descriptor is None else np.asarray(descriptor).copy(),
            observations=observations or {},
        )
        return landmark_id

    def add_observation(self, keyframe_id: int, landmark_id: int, keypoint_idx: int) -> None:
        """Record that a keyframe keypoint observes a landmark."""
        if landmark_id in self.landmarks:
            self.landmarks[landmark_id].observations[keyframe_id] = keypoint_idx
        for keyframe in self.keyframes:
            if keyframe.frame_id == keyframe_id:
                keyframe.observed_landmarks[landmark_id] = keypoint_idx
                break

    def get_points_array(self) -> np.ndarray:
        """Return all landmark positions as an N x 3 array."""
        if not self.landmarks:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack([lm.position for lm in self.landmarks.values()])

    def get_descriptors_and_ids(self) -> Tuple[Optional[np.ndarray], List[int]]:
        """Return stacked landmark descriptors and their ids for matching."""
        rows: List[np.ndarray] = []
        ids: List[int] = []
        for landmark_id, landmark in self.landmarks.items():
            if landmark.descriptor is not None:
                rows.append(landmark.descriptor.reshape(1, -1))
                ids.append(landmark_id)
        if not rows:
            return None, []
        return np.vstack(rows).astype(np.uint8), ids

    def get_local_descriptors_and_ids(self, recent_keyframe_ids: set) -> Tuple[Optional[np.ndarray], List[int]]:
        """Same as get_descriptors_and_ids, restricted to landmarks observed by
        at least one of the given (typically: most recent) keyframes.

        Matching PnP candidates against the whole map, rather than this
        "local map," makes each tracked frame progressively more expensive as
        a sequence runs -- landmarks only ever accumulate, so a map with tens
        of thousands of points from earlier, now long-gone parts of a scene
        still gets matched against on every single new frame. Landmarks a
        recent keyframe actually observed are exactly the ones a PnP solve
        for the current pose can plausibly match anyway.
        """
        rows: List[np.ndarray] = []
        ids: List[int] = []
        for landmark_id, landmark in self.landmarks.items():
            if landmark.descriptor is None:
                continue
            if recent_keyframe_ids.isdisjoint(landmark.observations.keys()):
                continue
            rows.append(landmark.descriptor.reshape(1, -1))
            ids.append(landmark_id)
        if not rows:
            return None, []
        return np.vstack(rows).astype(np.uint8), ids
