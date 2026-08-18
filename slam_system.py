"""Main monocular SLAM pipeline.

The state machine keeps the system honest. INITIALIZING waits for two useful
views before creating depth. TRACKING uses map-based PnP. LOST means tracking
failed and the map should not be blindly extended. RELOCALIZATION tries to match
the current view against old keyframes before returning to tracking.
"""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from bundle_adjuster import BundleAdjuster
from config import Config
from feature_extractor import FeatureExtractor, FeatureSet
from keyframe_manager import KeyframeManager
from loop_detector import LoopDetector
from map import Frame, Keyframe, Map, Pose
from matcher import Matcher
from pose_estimator import PoseEstimator
from pose_graph import PoseGraphOptimizer
from tracker import Tracker
from triangulator import Triangulator
from visualizer import Visualizer


class SLAMState(str, Enum):
    """Lifecycle states used by the SLAM system."""

    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RELOCALIZATION = "RELOCALIZATION"


class SLAMSystem:
    """Chain feature extraction, matching, geometry, mapping, BA, and loops."""

    def __init__(self, config: Optional[Config] = None, K: Optional[np.ndarray] = None, visualize: bool = False) -> None:
        """Create all pipeline modules and start in INITIALIZING."""
        self.config = config or Config()
        self.K = K if K is not None else self.config.default_K
        self.extractor = FeatureExtractor(self.config)
        self.matcher = Matcher(self.config)
        self.pose_estimator = PoseEstimator(self.config)
        self.triangulator = Triangulator(self.config)
        self.map = Map()
        self.keyframes = KeyframeManager(self.config)
        self.tracker = Tracker(self.config, self.extractor, self.matcher, self.K)
        self.bundle_adjuster = BundleAdjuster(self.config, self.K)
        self.loop_detector = LoopDetector(self.config, self.matcher, self.pose_estimator, self.K)
        self.pose_graph = PoseGraphOptimizer()
        self.visualizer = Visualizer(self.config) if visualize else None
        self.state = SLAMState.INITIALIZING
        self.frame_id = 0
        self.previous_frame: Optional[Frame] = None
        self.last_pose = Pose.identity()
        self.last_num_features = 0
        # Baseline inlier count new keyframes are compared against when deciding
        # whether tracking quality has degraded enough to insert another one.
        # Reset to the current keyframe's inlier count every time a keyframe is
        # accepted, so it stays meaningful regardless of how large the map grows
        # (dividing by total map size, as this used to, made the ratio shrink
        # forever and become meaningless well before a real sequence finished).
        self.reference_inlier_count = 1
        self.lost_frame_count = 0

    def _make_frame(self, image: np.ndarray, timestamp: float, features: FeatureSet, pose: Optional[Pose]) -> Frame:
        """Create a Frame object from extracted features."""
        return Frame(
            frame_id=self.frame_id,
            timestamp=timestamp,
            image=features.image,
            keypoints=features.keypoints,
            descriptors=features.descriptors,
            pose=pose,
        )

    def _initialize(self, frame: Frame) -> None:
        """Use the first two useful frames to create the initial map."""
        if self.previous_frame is None:
            frame.pose = Pose.identity()
            self.previous_frame = frame
            self.map.add_pose(frame.pose)
            return

        if self.previous_frame.descriptors is None or frame.descriptors is None:
            self.previous_frame = frame
            return

        pts1, pts2, matches = self.matcher.match_keypoints(
            self.previous_frame.keypoints,
            self.previous_frame.descriptors,
            frame.keypoints,
            frame.descriptors,
        )
        if len(matches) < 30:
            self.previous_frame = frame
            return

        try:
            relative_pose, inlier_mask = self.pose_estimator.estimate_relative_pose(pts1, pts2, self.K)
        except Exception:
            self.previous_frame = frame
            return

        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]
        matches_in = [m for m, keep in zip(matches, inlier_mask) if keep]

        # Reject near-identical views before spending a triangulation on them.
        # relative_pose.t is a *unit-length direction* (monocular essential-matrix
        # recovery has no metric scale -- see pose_estimator.py), so comparing its
        # norm against a "minimum baseline in meters" config value is dimensionally
        # meaningless: the norm is always ~1.0 regardless of how much the camera
        # actually moved, so a `norm(t) < 1e-6` guard only ever catches an exact
        # zero vector. What this check is *supposed* to measure -- did the camera
        # move enough for triangulation to be numerically stable -- is the pixel
        # parallax between matched points, so that's what's measured instead.
        parallax_px = float(np.median(np.linalg.norm(pts2_in - pts1_in, axis=1))) if len(pts1_in) else 0.0
        if len(matches_in) < 20 or parallax_px < self.config.min_parallax_px:
            self.previous_frame = frame
            return

        first_pose = Pose.identity()
        second_pose = relative_pose
        self.previous_frame.pose = first_pose
        frame.pose = second_pose
        kf1 = self.map.add_keyframe(self.previous_frame)
        kf2 = self.map.add_keyframe(frame)
        self.keyframes.mark_inserted(frame.frame_id)
        self.loop_detector.add_keyframe(kf1.frame_id, kf1.keypoints, kf1.descriptors)
        self.loop_detector.add_keyframe(kf2.frame_id, kf2.keypoints, kf2.descriptors)
        self.pose_graph.add_pose(kf1.frame_id, kf1.pose)
        self.pose_graph.add_pose(kf2.frame_id, kf2.pose, edge_from=kf1.frame_id)

        added = self._triangulate_between_keyframes(kf1, kf2)
        if added < 15:
            # Not enough usable geometry came out of this pair -- undo the
            # speculative keyframes/landmarks and keep waiting for a better pair.
            self._undo_keyframe(kf1)
            self._undo_keyframe(kf2)
            self.previous_frame = frame
            return

        self.map.add_pose(frame.pose)
        self.last_pose = frame.pose
        self.reference_inlier_count = added
        self.previous_frame = frame
        self.state = SLAMState.TRACKING
        print(f"Initialized map with {added} landmarks from frame {frame.frame_id} (parallax={parallax_px:.1f}px)")

    def _undo_keyframe(self, keyframe: "Keyframe") -> None:
        """Roll back a speculative keyframe (and any landmarks it introduced)
        when an initialization attempt doesn't produce enough geometry.
        """
        stale_landmarks = [lid for lid, kp_idx in keyframe.observed_landmarks.items()]
        for lid in stale_landmarks:
            self.map.landmarks.pop(lid, None)
        if keyframe in self.map.keyframes:
            self.map.keyframes.remove(keyframe)

    def _triangulate_between_keyframes(self, kf_a: "Keyframe", kf_b: "Keyframe") -> int:
        """Triangulate fresh landmarks from matches between two keyframes.

        Skips any keypoint already tied to an existing landmark in either
        frame, so this can be called repeatedly as new keyframes arrive
        without re-triangulating (and duplicating) points that are already
        in the map -- this is what keeps the map growing during tracking
        instead of freezing after the first pair, unlike the original
        implementation which only ever called triangulation once, here.
        """
        if kf_a.descriptors is None or kf_b.descriptors is None:
            return 0

        pts_a, pts_b, matches = self.matcher.match_keypoints(
            kf_a.keypoints, kf_a.descriptors, kf_b.keypoints, kf_b.descriptors
        )
        if len(matches) == 0:
            return 0

        observed_a = set(kf_a.observed_landmarks.values())
        observed_b = set(kf_b.observed_landmarks.values())
        fresh = [
            (i, m) for i, m in enumerate(matches)
            if m.queryIdx not in observed_a and m.trainIdx not in observed_b
        ]
        if not fresh:
            return 0

        fresh_idx = [i for i, _ in fresh]
        fresh_matches = [m for _, m in fresh]
        P_a = kf_a.pose.projection_matrix(self.K)
        P_b = kf_b.pose.projection_matrix(self.K)
        points_3d, valid = self.triangulator.triangulate(pts_a[fresh_idx], pts_b[fresh_idx], P_a, P_b)
        valid_matches = [m for m, keep in zip(fresh_matches, valid) if keep]

        added = 0
        for point, match in zip(points_3d, valid_matches):
            descriptor = kf_b.descriptors[match.trainIdx]
            lid = self.map.add_landmark(
                point, descriptor, {kf_a.frame_id: match.queryIdx, kf_b.frame_id: match.trainIdx}
            )
            kf_a.observed_landmarks[lid] = match.queryIdx
            kf_b.observed_landmarks[lid] = match.trainIdx
            added += 1
        return added

    def _track(self, image: np.ndarray, timestamp: float) -> Frame:
        """Track a normal frame with map-based PnP."""
        pose, num_inliers, tracked_ids, features, inlier_matches = self.tracker.track(image, self.map, self.last_pose)
        frame = self._make_frame(image, timestamp, features, pose)
        self.map.add_pose(pose)
        self.last_pose = pose
        self.lost_frame_count = 0

        tracked_ratio = num_inliers / max(1, self.reference_inlier_count)
        num_new_features = max(0, len(features.keypoints) - num_inliers)
        decision = self.keyframes.should_insert(frame.frame_id, tracked_ratio, num_new_features)

        if decision.insert:
            keyframe = self.map.add_keyframe(frame)
            self.keyframes.mark_inserted(frame.frame_id)
            for landmark_id, match in zip(tracked_ids, inlier_matches):
                self.map.add_observation(keyframe.frame_id, landmark_id, match.trainIdx)
            self.loop_detector.add_keyframe(keyframe.frame_id, keyframe.keypoints, keyframe.descriptors)

            previous_keyframe = self.map.keyframes[-2] if len(self.map.keyframes) >= 2 else None
            if previous_keyframe is not None:
                self.pose_graph.add_pose(keyframe.frame_id, keyframe.pose, edge_from=previous_keyframe.frame_id)
                added = self._triangulate_between_keyframes(previous_keyframe, keyframe)
                if added:
                    print(f"Mapped {added} new landmarks between keyframes {previous_keyframe.frame_id} and {keyframe.frame_id}")

            self.reference_inlier_count = num_inliers

            if len(self.map.keyframes) % self.config.ba_every_n_keyframes == 0:
                print("Local BA:", self.bundle_adjuster.optimize(self.map))
            if len(self.map.keyframes) % self.config.loop_every_n_keyframes == 0:
                candidate = self.loop_detector.detect_loop(frame.frame_id, frame.keypoints, frame.descriptors)
                if candidate is not None and candidate.relative_R is not None:
                    old_keyframe = next((kf for kf in self.map.keyframes if kf.frame_id == candidate.frame_id), None)
                    if old_keyframe is not None:
                        self._apply_loop_closure(old_keyframe, keyframe, candidate)

        if self.visualizer is not None:
            self.visualizer.update(pose, self.map.get_points_array(), num_inliers, 0.0)
        return frame

    def _apply_loop_closure(self, old_keyframe: "Keyframe", new_keyframe: "Keyframe", candidate) -> None:
        """Correct accumulated drift with a real pose-graph optimization.

        The loop edge's rotation comes straight from essential-matrix
        recovery (rotation has no monocular scale ambiguity). Its
        translation is unit-length, so it's rescaled by the trajectory's own
        current estimate of the distance between the two keyframes before
        use -- an approximation (true joint scale recovery is out of scope
        here), but a real geometric correction rather than a loop candidate
        that only ever got printed, as before.
        """
        old_poses = {kf.frame_id: kf.pose for kf in self.map.keyframes}
        separation = float(np.linalg.norm(new_keyframe.pose.camera_center() - old_keyframe.pose.camera_center()))
        scaled_t = candidate.relative_t * max(separation, 1e-3)

        corrected = self.pose_graph.optimize_with_loop(
            old_keyframe.frame_id, new_keyframe.frame_id, candidate.relative_R, scaled_t
        )
        if not corrected:
            return

        # Re-anchor each landmark to whichever keyframe first observed it, so
        # the map moves along with the pose correction instead of the
        # cameras jumping while the points they're supposed to see stay put.
        anchor_keyframe = {
            landmark_id: min(landmark.observations.keys())
            for landmark_id, landmark in self.map.landmarks.items()
            if landmark.observations
        }

        for keyframe in self.map.keyframes:
            if keyframe.frame_id in corrected:
                keyframe.pose = corrected[keyframe.frame_id]

        for landmark_id, anchor_id in anchor_keyframe.items():
            if anchor_id not in old_poses or anchor_id not in corrected:
                continue
            old_pose = old_poses[anchor_id]
            new_pose = corrected[anchor_id]
            landmark = self.map.landmarks[landmark_id]
            point_cam = old_pose.R @ landmark.position.reshape(3, 1) + old_pose.t
            landmark.position = (new_pose.R.T @ (point_cam - new_pose.t)).reshape(3)

        if new_keyframe.frame_id in corrected:
            self.last_pose = corrected[new_keyframe.frame_id]
        print(
            f"Loop closure applied: keyframe {new_keyframe.frame_id} <-> {old_keyframe.frame_id} "
            f"(score={candidate.score:.3f}, {len(corrected)} poses adjusted)"
        )

    def process_frame(self, image: np.ndarray, timestamp: float) -> SLAMState:
        """Process one image and update trajectory/map state."""
        features = self.extractor.detect_and_compute(image)
        frame = self._make_frame(image, timestamp, features, None)
        try:
            if self.state == SLAMState.INITIALIZING:
                self._initialize(frame)
            elif self.state == SLAMState.TRACKING:
                self.previous_frame = self._track(image, timestamp)
            elif self.state in (SLAMState.LOST, SLAMState.RELOCALIZATION):
                self.state = SLAMState.RELOCALIZATION
                self.lost_frame_count += 1
                # Recovering from just-losing-tracking is a different problem
                # than loop closure: it needs to search keyframes close in
                # time (often exactly where the camera still is), but
                # LoopDetector.detect_loop() deliberately excludes anything
                # within 30 frames to avoid treating nearby neighbors as a
                # "loop." Reusing it here made relocalization permanently
                # unreachable for the first 30 frames after getting lost --
                # which, empirically, was most of a real run. Retrying direct
                # map-based PnP tracking every frame is simpler and actually
                # searches the right thing: the whole current map.
                try:
                    self.previous_frame = self._track(image, timestamp)
                    self.state = SLAMState.TRACKING
                except Exception:
                    pass
        except Exception as exc:
            print(f"Tracking failed at frame {self.frame_id}: {exc}")
            self.state = SLAMState.LOST
        finally:
            self.frame_id += 1
        return self.state

    def get_trajectory(self) -> List[Pose]:
        """Return the estimated camera poses."""
        return self.map.trajectory

    def get_map(self) -> np.ndarray:
        """Return sparse 3D point cloud as N x 3."""
        return self.map.get_points_array()


def load_images_from_folder(folder: str | Path) -> List[Path]:
    """Load sorted image paths from a dataset or sample directory."""
    folder = Path(folder)
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(folder.rglob(pattern))
    return sorted(paths)


def main() -> None:
    """Run SLAM from the command line on a folder of images."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Folder containing image files")
    parser.add_argument("--visualize", action="store_true", help="Show Open3D visualization")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames for quick tests")
    args = parser.parse_args()

    images = load_images_from_folder(args.dataset)
    if args.max_frames:
        images = images[: args.max_frames]
    if not images:
        raise FileNotFoundError(f"No images found under {args.dataset}")

    slam = SLAMSystem(visualize=args.visualize)
    for i, path in enumerate(tqdm(images, desc="SLAM")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        slam.process_frame(image, float(i))

    trajectory = np.vstack([pose.camera_center() for pose in slam.get_trajectory()])
    points = slam.get_map()
    np.savetxt("trajectory.csv", trajectory, delimiter=",")
    np.savetxt("points.csv", points, delimiter=",")
    print(f"Saved trajectory.csv with {len(trajectory)} poses and points.csv with {len(points)} points")


if __name__ == "__main__":
    main()
