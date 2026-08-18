"""ORB feature extraction with grid-based spatial balancing.

ORB is a good default for educational SLAM because it is fast, free to use,
rotation-aware, and produces compact binary descriptors. SIFT is often more
distinctive but slower; SURF has a more complicated patent history and is not
as universally available. ORB builds an image scale pyramid, detects corners at
multiple resolutions, and describes each keypoint with a 256-bit binary
fingerprint. Each bit is the result of comparing the brightness of two learned
pixel locations in a patch; matching then becomes fast Hamming-distance lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from config import Config


@dataclass
class FeatureSet:
    """Features extracted from one image.

    keypoints are OpenCV keypoint objects, descriptors are ORB's N x 32 uint8
    binary descriptor matrix, and image is the grayscale image used for feature
    detection.
    """

    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    image: np.ndarray


class FeatureExtractor:
    """Detect ORB keypoints and compute descriptors for a video frame."""

    def __init__(self, config: Config) -> None:
        """Create ORB and optional CLAHE contrast enhancer from config values."""
        self.config = config
        self.orb = cv2.ORB_create(
            nfeatures=config.orb_nfeatures,
            scaleFactor=config.orb_scaleFactor,
            nlevels=config.orb_nlevels,
            edgeThreshold=config.orb_edgeThreshold,
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR/RGB images to grayscale while accepting gray input."""
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
        return gray.astype(np.uint8)

    def _balanced_indices(self, keypoints: List[cv2.KeyPoint], image_shape: tuple[int, int]) -> List[int]:
        """Select strong keypoints from each image grid cell.

        Clustering is bad for SLAM because epipolar geometry and PnP need points
        spread across the camera view. If all features sit on one poster, the
        solver can fit that poster while ignoring the rest of the scene.
        """
        if not keypoints:
            return []

        height, width = image_shape
        cell_h = max(1, height // self.config.grid_rows)
        cell_w = max(1, width // self.config.grid_cols)
        cells: dict[tuple[int, int], List[int]] = {}

        for idx, kp in enumerate(keypoints):
            col = min(self.config.grid_cols - 1, int(kp.pt[0] // cell_w))
            row = min(self.config.grid_rows - 1, int(kp.pt[1] // cell_h))
            cells.setdefault((row, col), []).append(idx)

        selected: List[int] = []
        for indices in cells.values():
            indices.sort(key=lambda i: keypoints[i].response, reverse=True)
            selected.extend(indices[: self.config.max_features_per_cell])

        selected.sort(key=lambda i: keypoints[i].response, reverse=True)
        return selected[: self.config.orb_nfeatures]

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        """Return grid-balanced ORB keypoints and descriptors for an image."""
        gray = self._to_gray(image)
        if self.config.use_clahe:
            gray = self.clahe.apply(gray)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        keypoints = list(keypoints or [])

        if descriptors is None or len(keypoints) == 0:
            return FeatureSet([], None, gray)

        keep = self._balanced_indices(keypoints, gray.shape[:2])
        balanced_keypoints = [keypoints[i] for i in keep]
        balanced_descriptors = descriptors[keep].astype(np.uint8) if keep else None
        return FeatureSet(balanced_keypoints, balanced_descriptors, gray)
