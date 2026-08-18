"""Tests for pose_graph.py's g2o-backed loop-closure optimization."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from map import Pose  # noqa: E402
from pose_graph import PoseGraphOptimizer  # noqa: E402

g2o_available = PoseGraphOptimizer().available


@pytest.mark.skipif(not g2o_available, reason="g2o-python not importable in this environment")
def test_loop_closure_pulls_drifted_pose_back_toward_the_loop_constraint():
    """Three poses roughly 1 unit apart in a line, but the third has drifted
    to 2.3 instead of 2.0. A loop edge saying pose 2 should be 2.0 away from
    pose 0 should pull the whole chain back toward consistency, not just
    leave the drift sitting on the last pose untouched.
    """
    graph = PoseGraphOptimizer()
    identity = np.eye(3)

    pose0 = Pose(identity, np.zeros((3, 1)))
    pose1 = Pose(identity, np.array([[1.0], [0.0], [0.0]]))
    pose2 = Pose(identity, np.array([[2.3], [0.0], [0.0]]))  # drifted

    graph.add_pose(0, pose0)
    graph.add_pose(1, pose1, edge_from=0)
    graph.add_pose(2, pose2, edge_from=1)

    corrected = graph.optimize_with_loop(
        loop_from_id=0, loop_to_id=2, relative_R=identity, relative_t=np.array([2.0, 0.0, 0.0])
    )

    assert corrected is not None
    assert set(corrected.keys()) == {0, 1, 2}
    # anchor (id 0, fixed) never moves
    np.testing.assert_allclose(corrected[0].t.reshape(3), [0, 0, 0], atol=1e-6)
    # pose 2 should have moved measurably closer to 2.0 than its drifted 2.3
    corrected_x2 = float(corrected[2].t.reshape(3)[0])
    assert abs(corrected_x2 - 2.0) < abs(2.3 - 2.0)
    # pose 1 (in between) should also have been adjusted by the correction,
    # not just the pose the loop edge directly touches
    corrected_x1 = float(corrected[1].t.reshape(3)[0])
    assert corrected_x1 != pytest.approx(1.0, abs=1e-6)


def test_optimize_with_loop_returns_none_with_too_few_poses():
    graph = PoseGraphOptimizer()
    graph.add_pose(0, Pose.identity())
    graph.add_pose(1, Pose.identity(), edge_from=0)

    result = graph.optimize_with_loop(0, 1, np.eye(3), np.zeros(3))

    assert result is None
