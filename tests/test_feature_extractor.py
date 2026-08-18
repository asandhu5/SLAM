"""Tests for ORB feature extraction on synthetic corner-rich images."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from feature_extractor import FeatureExtractor


def synthetic_checkerboard(size: int = 320, step: int = 40) -> np.ndarray:
    """Create a high-contrast image with many known corners."""
    image = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, step):
        for x in range(0, size, step):
            if (x // step + y // step) % 2 == 0:
                image[y : y + step, x : x + step] = 255
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def test_feature_extractor_detects_orb_descriptors() -> None:
    """ORB should find keypoints and produce N x 32 descriptors."""
    extractor = FeatureExtractor(Config(orb_nfeatures=500, use_clahe=False))
    features = extractor.detect_and_compute(synthetic_checkerboard())
    assert len(features.keypoints) >= 20
    assert features.descriptors is not None
    assert features.descriptors.shape == (len(features.keypoints), 32)
    assert all(kp.pt[0] >= 0 and kp.pt[1] >= 0 for kp in features.keypoints)
