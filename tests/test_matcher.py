"""Tests for ORB descriptor matching."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from matcher import Matcher


def keypoints(n: int) -> list[cv2.KeyPoint]:
    """Create dummy keypoints with positive coordinates."""
    return [cv2.KeyPoint(float(i + 1), float(i + 2), 7) for i in range(n)]


def test_identical_descriptors_match() -> None:
    """Identical descriptor sets should mostly match one-to-one."""
    rng = np.random.default_rng(4)
    desc = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
    matcher = Matcher(Config())
    matches = matcher.match_descriptors(desc, desc.copy())
    assert len(matches) == 64


def test_random_descriptors_have_fewer_ratio_matches() -> None:
    """Random unrelated descriptors should produce fewer confident matches."""
    rng = np.random.default_rng(5)
    desc_a = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
    desc_b = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
    matcher = Matcher(Config())
    matches = matcher.match_descriptors(desc_a, desc_b)
    assert len(matches) < 25


def test_output_shape_is_pair_array() -> None:
    """The helper should return an M x 2 x 2 coordinate array."""
    rng = np.random.default_rng(6)
    desc = rng.integers(0, 256, size=(16, 32), dtype=np.uint8)
    matcher = Matcher(Config())
    pairs = matcher.matched_pairs_array(keypoints(16), desc, keypoints(16), desc.copy())
    assert pairs.ndim == 3
    assert pairs.shape[1:] == (2, 2)
