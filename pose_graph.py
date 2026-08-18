"""Global pose-graph optimization for loop-closure correction.

Without this, a detected loop closure is just a print statement: the
trajectory keeps whatever drift had already accumulated. With it, a loop
closure adds one more constraint edge to a graph of keyframe poses (already
connected by "odometry" edges from consecutive tracking) and re-optimizes
the whole graph, so the correction spreads back across the trajectory
instead of sitting entirely on the newest pose.

Uses g2o (the same library ORB-SLAM uses for this exact step) when it's
importable; the rest of the SLAM pipeline works identically without it,
just without loop-closure correction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from map import Pose

try:  # g2o's Python bindings aren't available on every platform.
    import g2o
except Exception:  # pragma: no cover - exercised on machines without g2o.
    g2o = None


def _relative_pose(pose_a: Pose, pose_b: Pose) -> Pose:
    """Return the pose transform from a's camera frame to b's camera frame.

    Both poses are world-to-camera. Solving pose_b = relative.compose(pose_a)
    (this project's own Pose.compose convention) for `relative` gives
    R = R_b @ R_a.T, t = t_b - R @ t_a.
    """
    R = pose_b.R @ pose_a.R.T
    t = pose_b.t - R @ pose_a.t
    return Pose(R, t.reshape(3, 1))


class PoseGraphOptimizer:
    """Incrementally tracks keyframe poses and their odometry edges, and can
    re-optimize the whole graph once a loop-closure edge is detected.
    """

    def __init__(self) -> None:
        self.available = g2o is not None
        self._poses: Dict[int, Pose] = {}
        self._odometry_edges: List[Tuple[int, int]] = []

    def add_pose(self, frame_id: int, pose: Pose, edge_from: Optional[int] = None) -> None:
        """Register a keyframe's current pose estimate.

        `edge_from`, when given, records an odometry edge from that earlier
        keyframe to this one (using their current relative pose as the
        measurement) -- call this every time a new keyframe is accepted,
        chaining from the previous keyframe.
        """
        self._poses[frame_id] = pose
        if edge_from is not None and edge_from in self._poses:
            self._odometry_edges.append((edge_from, frame_id))

    def optimize_with_loop(
        self,
        loop_from_id: int,
        loop_to_id: int,
        relative_R: np.ndarray,
        relative_t: np.ndarray,
    ) -> Optional[Dict[int, Pose]]:
        """Optimize the graph with one extra loop-closure edge and return
        corrected {frame_id: Pose} for every registered keyframe, or None
        if g2o isn't available or there isn't enough graph to optimize.
        """
        if not self.available or len(self._poses) < 3:
            return None

        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(algorithm)

        ordered_ids = sorted(self._poses.keys())
        anchor_id = ordered_ids[0]

        for frame_id in ordered_ids:
            pose = self._poses[frame_id]
            vertex = g2o.VertexSE3Expmap()
            vertex.set_id(frame_id)
            vertex.set_estimate(g2o.SE3Quat(pose.R, pose.t.reshape(3, 1)))
            vertex.set_fixed(frame_id == anchor_id)
            optimizer.add_vertex(vertex)

        def add_edge(id_a: int, id_b: int, rel_pose: Pose, information: float) -> None:
            edge = g2o.EdgeSE3Expmap()
            edge.set_vertex(0, optimizer.vertex(id_a))
            edge.set_vertex(1, optimizer.vertex(id_b))
            edge.set_measurement(g2o.SE3Quat(rel_pose.R, rel_pose.t.reshape(3, 1)))
            edge.set_information(np.eye(6) * information)
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            optimizer.add_edge(edge)

        for id_a, id_b in self._odometry_edges:
            if id_a not in self._poses or id_b not in self._poses:
                continue
            add_edge(id_a, id_b, _relative_pose(self._poses[id_a], self._poses[id_b]), information=1.0)

        # The loop edge gets a higher information (confidence) weight than a
        # single odometry edge, since it's the constraint actually correcting
        # accumulated drift -- but not so high that one noisy detection can
        # freely override the whole rest of the trajectory.
        add_edge(loop_from_id, loop_to_id, Pose(relative_R, relative_t.reshape(3, 1)), information=10.0)

        optimizer.initialize_optimization()
        optimizer.optimize(20)

        corrected: Dict[int, Pose] = {}
        for frame_id in ordered_ids:
            estimate = optimizer.vertex(frame_id).estimate()
            R = estimate.rotation().matrix()
            t = np.asarray(estimate.translation()).reshape(3, 1)
            corrected[frame_id] = Pose(R, t)
        return corrected
