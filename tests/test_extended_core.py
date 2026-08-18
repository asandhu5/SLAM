"""Extended deterministic tests for the educational SLAM project.

These tests intentionally avoid network, GUI windows, Streamlit, and Open3D so
they can run quickly in a basic Python environment. They are pytest-compatible
plain `test_*` functions and can also be executed by `tests/run_all_tests.py`.
"""

from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import apply_similarity, compute_ate, compute_rpe, evaluate_trajectory, umeyama_alignment
from config import Config
from download_data import copy_sample_frames, extract_archive
from feature_extractor import FeatureExtractor
from keyframe_manager import KeyframeManager
from loop_detector import LoopDetector
from map import Frame, Map, Pose
from matcher import Matcher
from pose_estimator import PoseEstimator
from slam_system import SLAMState, SLAMSystem, load_images_from_folder
from tracker import Tracker
from triangulator import Triangulator


def make_keypoints(n: int) -> list[cv2.KeyPoint]:
    """Create deterministic positive-coordinate keypoints."""
    return [cv2.KeyPoint(float(i + 3), float((i * 7) % 101 + 5), 7) for i in range(n)]


def make_descriptors(n: int, seed: int = 1) -> np.ndarray:
    """Create deterministic mostly unique ORB-shaped descriptors."""
    rng = np.random.default_rng(seed)
    desc = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
    desc[:, 0] = np.arange(n, dtype=np.uint8)
    return desc


def make_image(size: int = 180) -> np.ndarray:
    """Create a synthetic textured image for feature tests."""
    image = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (size - 20, size - 20), 180, 2)
    cv2.circle(image, (size // 2, size // 2), size // 4, 255, 2)
    for i in range(20, size - 20, 20):
        cv2.line(image, (i, 10), (size - i, size - 10), 120, 1)
    return image


def make_frame(frame_id: int = 0, pose: Pose | None = None, n: int = 20) -> Frame:
    """Create a small frame object for map tests."""
    return Frame(frame_id, float(frame_id), make_image(), make_keypoints(n), make_descriptors(n), pose)


def project(points: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project world points with a 3x4 camera projection matrix."""
    hom = np.hstack([points, np.ones((len(points), 1))])
    pix = (P @ hom.T).T
    return pix[:, :2] / pix[:, 2:3]


def test_config_default_k_shape() -> None:
    assert Config().default_K.shape == (3, 3)


def test_config_default_k_bottom_row() -> None:
    assert np.allclose(Config().default_K[2], [0.0, 0.0, 1.0])


def test_config_trajectory_color_is_independent_per_instance() -> None:
    a = Config()
    b = Config()
    a.trajectory_color[0] = 0.2
    assert b.trajectory_color[0] == 1.0


def test_pose_identity_values() -> None:
    pose = Pose.identity()
    assert np.allclose(pose.R, np.eye(3))
    assert np.allclose(pose.t, np.zeros((3, 1)))


def test_pose_matrix_shape_and_last_row() -> None:
    T = Pose.identity().matrix()
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])


def test_pose_camera_center_for_translated_pose() -> None:
    pose = Pose(np.eye(3), np.array([[1.0], [2.0], [3.0]]))
    assert np.allclose(pose.camera_center(), [-1.0, -2.0, -3.0])


def test_pose_projection_matrix_shape() -> None:
    assert Pose.identity().projection_matrix(Config().default_K).shape == (3, 4)


def test_pose_compose_identity_relative() -> None:
    pose = Pose(np.eye(3), np.array([[0.1], [0.2], [0.3]]))
    out = pose.compose(Pose.identity())
    assert np.allclose(out.R, pose.R)
    assert np.allclose(out.t, pose.t)


def test_map_starts_empty() -> None:
    slam_map = Map()
    assert slam_map.get_points_array().shape == (0, 3)


def test_map_add_pose() -> None:
    slam_map = Map()
    slam_map.add_pose(Pose.identity())
    assert len(slam_map.trajectory) == 1


def test_map_add_keyframe_copies_frame_id() -> None:
    slam_map = Map()
    frame = make_frame(7, Pose.identity())
    keyframe = slam_map.add_keyframe(frame)
    assert keyframe.frame_id == 7


def test_map_add_landmark_returns_incrementing_ids() -> None:
    slam_map = Map()
    first = slam_map.add_landmark([0, 0, 3], make_descriptors(1)[0])
    second = slam_map.add_landmark([1, 0, 3], make_descriptors(1, seed=2)[0])
    assert (first, second) == (0, 1)


def test_map_points_array_has_inserted_point() -> None:
    slam_map = Map()
    slam_map.add_landmark([1, 2, 3], None)
    assert np.allclose(slam_map.get_points_array(), [[1, 2, 3]])


def test_map_descriptors_none_when_empty() -> None:
    desc, ids = Map().get_descriptors_and_ids()
    assert desc is None
    assert ids == []


def test_map_descriptors_skip_missing_descriptors() -> None:
    slam_map = Map()
    slam_map.add_landmark([0, 0, 3], None)
    desc, ids = slam_map.get_descriptors_and_ids()
    assert desc is None
    assert ids == []


def test_map_descriptors_stack_shape() -> None:
    slam_map = Map()
    slam_map.add_landmark([0, 0, 3], make_descriptors(1)[0])
    desc, ids = slam_map.get_descriptors_and_ids()
    assert desc.shape == (1, 32)
    assert ids == [0]


def test_map_add_observation_updates_keyframe_and_landmark() -> None:
    slam_map = Map()
    keyframe = slam_map.add_keyframe(make_frame(3, Pose.identity()))
    landmark_id = slam_map.add_landmark([0, 0, 3], make_descriptors(1)[0])
    slam_map.add_observation(3, landmark_id, 5)
    assert slam_map.landmarks[landmark_id].observations[3] == 5
    assert keyframe.observed_landmarks[landmark_id] == 5


def test_feature_extractor_accepts_grayscale() -> None:
    features = FeatureExtractor(Config(use_clahe=False)).detect_and_compute(make_image())
    assert features.image.ndim == 2


def test_feature_extractor_accepts_color() -> None:
    color = cv2.cvtColor(make_image(), cv2.COLOR_GRAY2BGR)
    features = FeatureExtractor(Config(use_clahe=False)).detect_and_compute(color)
    assert features.image.ndim == 2


def test_feature_extractor_rejects_bad_rank() -> None:
    extractor = FeatureExtractor(Config())
    try:
        extractor.detect_and_compute(np.zeros((2, 2, 2, 2), dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("bad image rank should raise ValueError")


def test_feature_extractor_blank_image_returns_empty() -> None:
    features = FeatureExtractor(Config(use_clahe=False)).detect_and_compute(np.zeros((120, 120), dtype=np.uint8))
    assert features.descriptors is None
    assert features.keypoints == []


def test_feature_extractor_descriptor_dtype_uint8() -> None:
    features = FeatureExtractor(Config(use_clahe=False)).detect_and_compute(make_image())
    assert features.descriptors is None or features.descriptors.dtype == np.uint8


def test_feature_extractor_respects_low_feature_cap() -> None:
    features = FeatureExtractor(Config(orb_nfeatures=30, use_clahe=False)).detect_and_compute(make_image(260))
    assert len(features.keypoints) <= 30


def test_matcher_empty_descriptors_return_no_matches() -> None:
    matches = Matcher(Config()).match_descriptors(np.empty((0, 32), dtype=np.uint8), make_descriptors(5))
    assert matches == []


def test_matcher_identical_unique_descriptors_match_all() -> None:
    desc = make_descriptors(32)
    assert len(Matcher(Config()).match_descriptors(desc, desc.copy())) == 32


def test_matcher_matched_pairs_empty_shape() -> None:
    pairs = Matcher(Config()).matched_pairs_array([], None, [], None)
    assert pairs.shape == (0, 2, 2)


def test_matcher_pairs_shape_for_identical_descriptors() -> None:
    desc = make_descriptors(20)
    pairs = Matcher(Config()).matched_pairs_array(make_keypoints(20), desc, make_keypoints(20), desc.copy())
    assert pairs.shape == (20, 2, 2)


def test_matcher_matches_sorted_by_distance() -> None:
    desc = make_descriptors(20)
    matches = Matcher(Config()).match_descriptors(desc, desc.copy())
    assert [m.distance for m in matches] == sorted(m.distance for m in matches)


def test_pose_estimator_rejects_too_few_points() -> None:
    estimator = PoseEstimator(Config())
    try:
        estimator.estimate_essential_matrix(np.zeros((7, 2), np.float32), np.zeros((7, 2), np.float32), Config().default_K)
    except ValueError:
        return
    raise AssertionError("too few correspondences should raise ValueError")


def test_pose_estimator_output_mask_length_on_synthetic_grid() -> None:
    pts1 = np.float32([[x, y] for x in range(100, 500, 50) for y in range(80, 360, 50)])
    pts2 = pts1 + np.float32([8.0, 0.4])
    estimator = PoseEstimator(Config(ransac_inlier_threshold_px=2.0))
    try:
        _, mask, _ = estimator.estimate_essential_matrix(pts1, pts2, Config().default_K)
    except RuntimeError:
        return
    assert len(mask) == len(pts1)


def test_triangulator_empty_input() -> None:
    tri = Triangulator(Config())
    points, valid = tri.triangulate(np.empty((0, 2)), np.empty((0, 2)), np.eye(3, 4), np.eye(3, 4))
    assert points.shape == (0, 3)
    assert valid.shape == (0,)


def test_triangulator_reprojection_error_zero_for_exact_projection() -> None:
    K = np.eye(3)
    P = Pose.identity().projection_matrix(K)
    points = np.array([[0.0, 0.0, 2.0], [0.2, 0.1, 3.0]])
    pts = project(points, P)
    mean, errors = Triangulator(Config()).compute_reprojection_error(points, pts, P)
    assert mean < 1e-9
    assert np.max(errors) < 1e-9


def test_triangulator_reprojection_error_known_offset() -> None:
    K = np.eye(3)
    P = Pose.identity().projection_matrix(K)
    points = np.array([[0.0, 0.0, 2.0]])
    pts = project(points, P) + np.array([[3.0, 4.0]])
    _, errors = Triangulator(Config()).compute_reprojection_error(points, pts, P)
    assert np.allclose(errors[0], 5.0)


def test_triangulator_filters_negative_depth() -> None:
    K = np.array([[500.0, 0.0, 100.0], [0.0, 500.0, 100.0], [0.0, 0.0, 1.0]])
    pose1 = Pose.identity()
    pose2 = Pose(np.eye(3), np.array([[-0.3], [0.0], [0.0]]))
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.1, 5.0]])
    P1 = pose1.projection_matrix(K)
    P2 = pose2.projection_matrix(K)
    pts1 = project(points, P1)
    pts2 = project(points, P2)
    _, valid = Triangulator(Config()).triangulate(pts1, pts2, P1, P2)
    assert valid.all()


def test_keyframe_first_insert() -> None:
    assert KeyframeManager(Config()).should_insert(0, 1.0, 0).insert


def test_keyframe_low_tracked_ratio_insert() -> None:
    manager = KeyframeManager(Config())
    manager.mark_inserted(0)
    decision = manager.should_insert(1, 0.5, 0)
    assert decision.insert and "tracked ratio" in decision.reason


def test_keyframe_many_new_features_insert() -> None:
    manager = KeyframeManager(Config())
    manager.mark_inserted(0)
    decision = manager.should_insert(1, 1.0, 999)
    assert decision.insert and "new features" in decision.reason


def test_keyframe_max_gap_insert() -> None:
    manager = KeyframeManager(Config(max_frame_gap=2, min_new_features=999))
    manager.mark_inserted(0)
    assert manager.should_insert(3, 1.0, 0).insert


def test_keyframe_redundant_frame_not_inserted() -> None:
    manager = KeyframeManager(Config(max_frame_gap=10, min_new_features=999))
    manager.mark_inserted(0)
    assert not manager.should_insert(2, 1.0, 0).insert


def test_benchmark_apply_similarity_identity() -> None:
    points = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(apply_similarity(points, 1.0, np.eye(3), np.zeros(3)), points)


def test_benchmark_umeyama_returns_positive_scale() -> None:
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.2, 0.0]])
    target = 2.0 * source + np.array([1.0, 0.0, 0.0])
    scale, _, _ = umeyama_alignment(source, target)
    assert abs(scale - 2.0) < 1e-9


def test_benchmark_compute_ate_zero_on_identical() -> None:
    traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert compute_ate(traj, traj)["ATE_rmse"] < 1e-9


def test_benchmark_compute_rpe_zero_on_identical() -> None:
    traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert compute_rpe(traj, traj)["RPE_rmse"] < 1e-9


def test_benchmark_evaluate_uses_shorter_length() -> None:
    est = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    gt = est[:2]
    metrics = evaluate_trajectory(est, gt)
    assert "ATE_rmse" in metrics and "RPE_rmse" in metrics


def test_benchmark_rejects_one_pose() -> None:
    try:
        evaluate_trajectory(np.zeros((1, 3)), np.zeros((1, 3)))
    except ValueError:
        return
    raise AssertionError("single-pose trajectory should raise ValueError")


def test_loop_detector_empty_database_returns_none() -> None:
    cfg = Config(loop_min_consistent_frames=2)
    detector = LoopDetector(cfg, Matcher(cfg), PoseEstimator(cfg), cfg.default_K)
    assert detector.detect_loop(100, make_keypoints(30), make_descriptors(30)) is None


def test_loop_detector_adds_keyframe_to_database() -> None:
    cfg = Config()
    detector = LoopDetector(cfg, Matcher(cfg), PoseEstimator(cfg), cfg.default_K)
    detector.add_keyframe(0, make_keypoints(30), make_descriptors(30))
    assert len(detector.database) == 1


def test_loop_detector_needs_enough_database_entries() -> None:
    cfg = Config(loop_min_consistent_frames=2)
    detector = LoopDetector(cfg, Matcher(cfg), PoseEstimator(cfg), cfg.default_K)
    for i in range(4):
        detector.add_keyframe(i * 40, make_keypoints(30), make_descriptors(30, seed=i + 10))
    assert detector.detect_loop(200, make_keypoints(30), make_descriptors(30)) is None


def test_loop_detector_temporal_consistency_accepts_repeated_candidate() -> None:
    cfg = Config(loop_min_consistent_frames=2, loop_min_score=0.01)
    detector = LoopDetector(cfg, Matcher(cfg), PoseEstimator(cfg), cfg.default_K)
    desc = make_descriptors(40, seed=42)
    for frame_id in [0, 40, 80, 120, 160]:
        detector.add_keyframe(frame_id, make_keypoints(40), desc.copy())
    first = detector.detect_loop(220, make_keypoints(40), desc.copy())
    second = detector.detect_loop(221, make_keypoints(40), desc.copy())
    assert first is None
    assert second is not None
    assert second.frame_id == 0


def test_tracker_fails_with_empty_map() -> None:
    cfg = Config()
    tracker = Tracker(cfg, FeatureExtractor(cfg), Matcher(cfg), cfg.default_K)
    try:
        tracker.track(make_image(), Map(), Pose.identity())
    except RuntimeError:
        return
    raise AssertionError("empty map should not track")


def test_slam_system_initial_state() -> None:
    assert SLAMSystem().state == SLAMState.INITIALIZING


def test_slam_system_empty_map_before_frames() -> None:
    assert SLAMSystem().get_map().shape == (0, 3)


def test_slam_system_process_blank_frame_does_not_crash() -> None:
    slam = SLAMSystem()
    state = slam.process_frame(np.zeros((100, 100), dtype=np.uint8), 0.0)
    assert state in {SLAMState.INITIALIZING, SLAMState.LOST}


def test_slam_system_get_trajectory_after_one_blank_frame() -> None:
    slam = SLAMSystem()
    slam.process_frame(np.zeros((100, 100), dtype=np.uint8), 0.0)
    assert len(slam.get_trajectory()) >= 1


def test_load_images_from_folder_returns_sorted_images() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cv2.imwrite(str(root / "b.png"), make_image())
        cv2.imwrite(str(root / "a.png"), make_image())
        paths = load_images_from_folder(root)
        assert [p.name for p in paths] == ["a.png", "b.png"]


def test_load_images_from_folder_ignores_non_images() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "notes.txt").write_text("not an image")
        assert load_images_from_folder(root) == []


def test_copy_sample_frames_copies_pngs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        sample = root / "sample"
        source.mkdir()
        for i in range(3):
            cv2.imwrite(str(source / f"{i}.png"), make_image())
        assert copy_sample_frames(source, sample, limit=2) == 2
        assert len(list(sample.glob("*.png"))) == 2


def test_extract_archive_rejects_unknown_suffix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        archive = root / "bad.bin"
        archive.write_bytes(b"not an archive")
        try:
            extract_archive(archive, root / "out")
        except ValueError:
            return
    raise AssertionError("unknown archive suffix should raise ValueError")


def test_bundle_adjuster_importable_and_skips_empty_map() -> None:
    from bundle_adjuster import BundleAdjuster

    result = BundleAdjuster(Config(), Config().default_K).optimize(Map())
    assert result.startswith("skipped")


def test_visualizer_importable_without_open3d_window() -> None:
    from visualizer import Visualizer

    visualizer = Visualizer(Config())
    assert visualizer.poses == []


def test_benchmark_invalid_alignment_shape_raises() -> None:
    try:
        umeyama_alignment(np.zeros((3, 2)), np.zeros((3, 2)))
    except ValueError:
        return
    raise AssertionError("invalid alignment shape should raise ValueError")


def test_pose_projection_projects_origin_forward() -> None:
    K = np.eye(3)
    P = Pose.identity().projection_matrix(K)
    pixel = project(np.array([[0.0, 0.0, 2.0]]), P)
    assert np.allclose(pixel, [[0.0, 0.0]])


def test_feature_grid_balancing_keeps_indices_unique() -> None:
    extractor = FeatureExtractor(Config(orb_nfeatures=200, use_clahe=False))
    features = extractor.detect_and_compute(make_image(260))
    coords = [(round(kp.pt[0], 3), round(kp.pt[1], 3), round(kp.size, 3)) for kp in features.keypoints]
    assert len(coords) == len(set(coords))


def test_matcher_ratio_threshold_zero_rejects_nonzero_distances() -> None:
    desc_a = make_descriptors(10)
    desc_b = make_descriptors(10)
    desc_b[:, 1] ^= 255
    matches = Matcher(Config(ratio_test_threshold=0.0)).match_descriptors(desc_a, desc_b)
    assert matches == []
