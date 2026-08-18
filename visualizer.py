"""Non-blocking visualization helpers for SLAM outputs."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional GUI dependency.
    o3d = None

from config import Config
from map import Pose


class Visualizer:
    """Show a sparse 3D map, trajectory, and simple diagnostics."""

    def __init__(self, config: Config) -> None:
        """Create plotting state and lazily initialize Open3D."""
        self.config = config
        self.poses: List[Pose] = []
        self.tracked_counts: List[int] = []
        self.reprojection_errors: List[float] = []
        self.o3d_ready = False
        self.vis = None
        self.cloud = None

    def _init_open3d(self) -> None:
        """Open an Open3D window when the optional dependency is available."""
        if o3d is None or self.o3d_ready:
            return
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Monocular SLAM Map", width=1000, height=700)
        self.cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.cloud)
        render = self.vis.get_render_option()
        render.point_size = self.config.point_size
        self.o3d_ready = True

    def update(
        self,
        new_pose: Pose,
        new_points: np.ndarray,
        tracked_count: int = 0,
        reprojection_error: float = 0.0,
    ) -> None:
        """Update Open3D without blocking the main SLAM loop."""
        self.poses.append(new_pose)
        self.tracked_counts.append(tracked_count)
        self.reprojection_errors.append(reprojection_error)
        self._init_open3d()
        if self.o3d_ready and self.cloud is not None and len(new_points) > 0:
            self.cloud.points = o3d.utility.Vector3dVector(new_points)
            colors = np.tile(np.array([[0.0, 1.0, 1.0]]), (len(new_points), 1))
            self.cloud.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.cloud)
            self.vis.poll_events()
            self.vis.update_renderer()

    def plot_diagnostics(self):
        """Return a matplotlib figure with trajectory and tracking diagnostics."""
        centers = np.vstack([pose.camera_center() for pose in self.poses]) if self.poses else np.empty((0, 3))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        if len(centers) > 0:
            axes[0].plot(centers[:, 0], centers[:, 2], color="green")
        axes[0].set_title("Top-down trajectory")
        axes[0].axis("equal")
        axes[1].plot(self.tracked_counts, color="tab:blue")
        axes[1].set_title("Tracked features")
        axes[2].plot(self.reprojection_errors, color="tab:red")
        axes[2].set_title("Reprojection error")
        fig.tight_layout()
        return fig
