"""Central configuration for the educational monocular SLAM system.

Keeping all hyperparameters in one dataclass makes experiments repeatable:
change a value here, rerun the pipeline, and compare the effect in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Hyperparameters shared by the SLAM pipeline.

    Each value is intentionally conservative for a beginner-friendly ORB
    pipeline. Real deployments tune these values per camera, frame rate,
    scene texture, and compute budget.
    """

    # ORB detects up to this many features per frame. Increasing it gives more
    # tracking candidates but costs CPU; decreasing it is faster but less robust.
    orb_nfeatures: int = 2000

    # ORB pyramid scale between levels. Larger values cover scale faster but may
    # miss subtle scale changes; smaller values are denser and slower.
    orb_scaleFactor: float = 1.2

    # Number of pyramid levels. More levels help scale invariance; fewer levels
    # reduce compute and can work when the camera motion is gentle.
    orb_nlevels: int = 8

    # Border where ORB avoids keypoints. Increasing it avoids edge artifacts;
    # decreasing it can find features near image borders but risks bad patches.
    orb_edgeThreshold: int = 31

    # Grid cells distribute ORB features spatially so corners do not all cluster
    # on one textured object. More cells improve coverage; fewer cells are faster.
    grid_rows: int = 4
    grid_cols: int = 6

    # Cap per grid cell. Increasing it admits richer local texture; decreasing it
    # forces stronger global spread across the image.
    max_features_per_cell: int = 90

    # CLAHE improves local contrast in low-texture scenes. Disable if lighting is
    # stable and raw image photometry should be preserved.
    use_clahe: bool = True

    # Lowe ratio threshold. Higher keeps more matches, including ambiguous ones;
    # lower is stricter and can reject useful but repetitive texture matches.
    ratio_test_threshold: float = 0.75

    # RANSAC confidence. Higher seeks a more reliable consensus but may run more
    # iterations; lower is faster with slightly more risk.
    ransac_confidence: float = 0.999

    # RANSAC max iterations. More iterations can survive more outliers; fewer
    # iterations speeds up easy scenes.
    ransac_max_iterations: int = 1000

    # Essential/PnP inlier threshold in pixels. Larger tolerates noise but admits
    # geometric mistakes; smaller is precise but fragile with motion blur.
    ransac_inlier_threshold_px: float = 1.0

    # Minimum median pixel parallax between matched points before accepting an
    # initialization pair. This -- not a "meters" baseline, which monocular
    # essential-matrix recovery has no way to know -- is what actually
    # measures whether the camera moved enough for triangulation to be
    # numerically stable: recoverPose()'s translation is a unit-length
    # direction regardless of how far the camera actually moved, so pixel
    # parallax is the only observable proxy available before real scale
    # exists. Larger avoids unstable near-identical views; smaller
    # initializes sooner but with noisier depth.
    min_parallax_px: float = 8.0

    # Reprojection-error cutoff for accepting new 3D points. Larger accepts more
    # points but noisier map; smaller makes a cleaner but sparser map.
    max_reprojection_error: float = 2.0

    # Reject a triangulated point if its depth exceeds this multiple of the
    # median depth among the other points triangulated in the same batch.
    # Reprojection error alone doesn't catch near-degenerate ray geometry:
    # two rays that are almost parallel can still reproject within a couple
    # pixels of the observed keypoints while their actual intersection point
    # sits enormously (sometimes literally kilometers, in a room-scale
    # scene) further away than everything else just triangulated, because a
    # tiny pixel error translates into a huge depth error when the rays
    # barely converge. Comparing against the batch's own median depth (not
    # an absolute distance, which monocular SLAM has no fixed scale for)
    # catches this without needing to know the scene's real-world scale.
    max_depth_ratio: float = 40.0

    # Insert a keyframe when fewer than this ratio of previous tracks survive.
    # Higher inserts more keyframes; lower keeps the map smaller but risks drift.
    min_tracked_ratio: float = 0.9

    # Insert a keyframe when enough unmatched features appear. Higher waits for
    # strong novelty; lower maps new areas more aggressively.
    min_new_features: int = 200

    # Force a keyframe after this many frames. Higher reduces compute; lower gives
    # BA and loop closure more anchors.
    max_frame_gap: int = 15

    # Number of recent keyframes optimized by local BA. Larger improves local
    # consistency but grows nonlinear solve cost quickly.
    ba_window_size: int = 7

    # Number of recent keyframes whose landmarks are eligible for PnP
    # tracking matches. Restricting to a "local map" (landmarks recent
    # keyframes actually observed) instead of the whole map -- which only
    # ever grows -- is what keeps per-frame tracking cost roughly constant
    # over a long sequence instead of climbing as the map does. Larger
    # widens what tracking can match against (more robust to reappearing
    # points) but costs more per frame; smaller is faster but can lose track
    # of points seen further back.
    local_map_keyframes: int = 15

    # Iterations for bundle adjustment. More can reduce reprojection error; fewer
    # keep interactive demos responsive.
    ba_max_iterations: int = 20

    # Run local BA every N accepted keyframes. Lower is more accurate but slower.
    ba_every_n_keyframes: int = 3

    # Loop candidate score threshold. Higher requires stronger visual similarity;
    # lower detects more candidates but increases false positives.
    loop_min_score: float = 0.013

    # Number of consecutive compatible loop candidates before accepting a loop.
    # Higher is safer; lower closes loops sooner.
    loop_min_consistent_frames: int = 3

    # Run loop detection every N keyframes. Lower catches loops sooner; higher is
    # cheaper and can be enough for short sequences.
    loop_every_n_keyframes: int = 5

    # Open3D map point size. Larger is easier to see; smaller looks cleaner.
    point_size: float = 2.0

    # RGB trajectory color. Red is visually separated from cyan map points.
    trajectory_color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])

    # Default camera intrinsics used by synthetic tests and uploaded videos when
    # calibration is unknown. Real datasets should load their own K matrix.
    default_fx: float = 718.856
    default_fy: float = 718.856
    default_cx: float = 607.1928
    default_cy: float = 185.2157

    @property
    def default_K(self):
        """Return the default pinhole camera matrix as a NumPy array."""
        import numpy as np

        return np.array(
            [[self.default_fx, 0.0, self.default_cx],
             [0.0, self.default_fy, self.default_cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
