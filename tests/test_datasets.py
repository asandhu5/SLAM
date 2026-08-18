"""Tests for datasets.py: TUM timestamp association and KITTI calib/pose parsing."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_kitti, load_tum  # noqa: E402


def _write_tum_fixture(root: Path, n_frames: int = 5) -> None:
    (root / "rgb").mkdir(parents=True)
    rgb_lines = ["# color images", "# file: 'x'", "# timestamp filename"]
    gt_lines = ["# ground truth trajectory", "# file: 'x'", "# timestamp tx ty tz qx qy qz qw"]
    for i in range(n_frames):
        t = 100.0 + i * 0.1
        fname = f"rgb/{t:.6f}.png"
        (root / fname).write_bytes(b"\x89PNG\r\n\x1a\n")  # not a real PNG, just needs to exist
        rgb_lines.append(f"{t:.6f} {fname}")
        # ground truth sampled at a slightly different rate/offset than rgb,
        # like the real TUM files, so association actually has to do work
        gt_t = t + 0.005
        gt_lines.append(f"{gt_t:.6f} {i * 1.0} {i * 2.0} {i * 0.5} 0 0 0 1")
    (root / "rgb.txt").write_text("\n".join(rgb_lines))
    (root / "groundtruth.txt").write_text("\n".join(gt_lines))


def test_load_tum_associates_every_frame_within_fixture(tmp_path):
    root = tmp_path / "rgbd_dataset_freiburg1_test"
    _write_tum_fixture(root, n_frames=5)

    sample = load_tum(root)

    assert len(sample.image_paths) == 5
    assert sample.ground_truth is not None
    assert sample.ground_truth.shape == (5, 3)
    np.testing.assert_allclose(sample.ground_truth[2], [2.0, 4.0, 1.0])
    assert sample.K.shape == (3, 3)
    assert sample.distortion is not None


def test_load_tum_respects_max_frames(tmp_path):
    root = tmp_path / "rgbd_dataset_freiburg1_test"
    _write_tum_fixture(root, n_frames=5)

    sample = load_tum(root, max_frames=2)

    assert len(sample.image_paths) == 2
    assert len(sample.ground_truth) == 2


def test_load_tum_drops_frames_with_no_nearby_ground_truth(tmp_path):
    root = tmp_path / "rgbd_dataset_freiburg1_test"
    root.mkdir()
    (root / "rgb").mkdir()
    (root / "rgb" / "1.0.png").write_bytes(b"x")
    (root / "rgb.txt").write_text("# c\n# f\n# t\n1.000000 rgb/1.0.png\n")
    # groundtruth is 5 seconds away -- nowhere near the 20ms association window
    (root / "groundtruth.txt").write_text("# g\n# f\n# t\n6.000000 0 0 0 0 0 0 1\n")

    sample = load_tum(root)

    assert len(sample.image_paths) == 0
    assert sample.ground_truth is None


def test_load_tum_without_groundtruth_file(tmp_path):
    root = tmp_path / "rgbd_dataset_freiburg1_test"
    root.mkdir()
    (root / "rgb").mkdir()
    (root / "rgb" / "1.0.png").write_bytes(b"x")
    (root / "rgb.txt").write_text("# c\n# f\n# t\n1.000000 rgb/1.0.png\n")

    sample = load_tum(root)

    assert len(sample.image_paths) == 1
    assert sample.ground_truth is None


def _write_kitti_fixture(root: Path, n_frames: int = 5) -> Path:
    seq_dir = root / "dataset" / "sequences" / "00"
    (seq_dir / "image_0").mkdir(parents=True)
    for i in range(n_frames):
        (seq_dir / "image_0" / f"{i:06d}.png").write_bytes(b"x")
    (seq_dir / "calib.txt").write_text(
        "P0: 718.856 0.0 607.1928 0.0 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n"
        "P1: 718.856 0.0 607.1928 -386.1448 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n"
    )
    poses_dir = root / "dataset" / "poses"
    poses_dir.mkdir(parents=True)
    lines = []
    for i in range(n_frames):
        # identity rotation, translating along x by i meters
        lines.append(f"1 0 0 {float(i)} 0 1 0 0 0 0 1 0")
    (poses_dir / "00.txt").write_text("\n".join(lines))
    return seq_dir


def test_load_kitti_parses_calibration_and_poses(tmp_path):
    seq_dir = _write_kitti_fixture(tmp_path, n_frames=5)

    sample = load_kitti(seq_dir)

    assert len(sample.image_paths) == 5
    np.testing.assert_allclose(sample.K, [[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    assert sample.distortion is None
    assert sample.ground_truth.shape == (5, 3)
    # fixture poses use identity R with t=[i,0,0]; camera center is -R.T@t = [-i,0,0]
    np.testing.assert_allclose(sample.ground_truth[3], [-3.0, 0.0, 0.0])


def test_load_kitti_without_poses_file(tmp_path):
    seq_dir = _write_kitti_fixture(tmp_path, n_frames=3)
    (seq_dir.parent.parent / "poses" / "00.txt").unlink()

    sample = load_kitti(seq_dir)

    assert len(sample.image_paths) == 3
    assert sample.ground_truth is None


def test_load_kitti_max_frames(tmp_path):
    seq_dir = _write_kitti_fixture(tmp_path, n_frames=5)

    sample = load_kitti(seq_dir, max_frames=2)

    assert len(sample.image_paths) == 2
    assert len(sample.ground_truth) == 2
