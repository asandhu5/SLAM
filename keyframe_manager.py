"""Keyframe insertion policy.

Keyframes are selected frames kept as durable map anchors. Keeping every frame
would create a large, redundant optimization problem; keeping too few frames
starves triangulation and relocalization. The tracked ratio measures how many
map points survived into the current frame. A low ratio usually means the camera
has moved enough that a new anchor is useful.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import Config


@dataclass
class KeyframeDecision:
    """Reasoned result of the keyframe insertion test."""

    insert: bool
    reason: str


class KeyframeManager:
    """Decide when a tracked frame should become a keyframe."""

    def __init__(self, config: Config) -> None:
        """Initialize with no prior keyframe."""
        self.config = config
        self.last_keyframe_id = -1

    def should_insert(
        self,
        frame_id: int,
        tracked_ratio: float,
        num_new_features: int,
    ) -> KeyframeDecision:
        """Insert when tracking drops, new texture appears, or the gap is large."""
        if self.last_keyframe_id < 0:
            return KeyframeDecision(True, "first keyframe")

        if tracked_ratio < self.config.min_tracked_ratio:
            return KeyframeDecision(True, f"tracked ratio {tracked_ratio:.2f} below threshold")

        if num_new_features > self.config.min_new_features:
            return KeyframeDecision(True, f"{num_new_features} new features visible")

        if frame_id - self.last_keyframe_id > self.config.max_frame_gap:
            return KeyframeDecision(True, "maximum frame gap reached")

        return KeyframeDecision(False, "frame is redundant")

    def mark_inserted(self, frame_id: int) -> None:
        """Record the frame id of the newest keyframe."""
        self.last_keyframe_id = frame_id
