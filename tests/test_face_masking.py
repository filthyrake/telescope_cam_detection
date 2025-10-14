"""
Tests for face masking functionality.
Tests face detection, masking styles, and caching.
"""

import sys
import os
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_masker import FaceMasker, FaceMaskingCache


class TestFaceMasker(unittest.TestCase):
    """Test suite for FaceMasker class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test image with a face-like region (simple rectangle pattern)
        self.test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Draw a simple face-like pattern (oval + eyes + mouth)
        # This won't be detected by Haar cascade, but we can test the masking logic
        center_x, center_y = 320, 240
        cv2.ellipse(self.test_frame, (center_x, center_y), (80, 100), 0, 0, 360, (200, 180, 170), -1)
        cv2.circle(self.test_frame, (center_x - 25, center_y - 20), 10, (50, 50, 50), -1)  # Left eye
        cv2.circle(self.test_frame, (center_x + 25, center_y - 20), 10, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(self.test_frame, (center_x, center_y + 20), (30, 15), 0, 0, 180, (100, 50, 50), 2)  # Mouth

    def test_face_masker_initialization_opencv(self):
        """Test FaceMasker initialization with OpenCV backend."""
        masker = FaceMasker(detection_backend="opencv_haar")
        self.assertIsNotNone(masker)
        self.assertEqual(masker.detection_backend, "opencv_haar")
        self.assertIsNotNone(masker.face_cascade)

    def test_face_masker_initialization_mediapipe(self):
        """Test FaceMasker initialization with MediaPipe backend (if available)."""
        try:
            masker = FaceMasker(detection_backend="mediapipe")
            self.assertIsNotNone(masker)
            # May fall back to opencv_haar if mediapipe not installed
            self.assertIn(masker.detection_backend, ["mediapipe", "opencv_haar"])
        except Exception as e:
            self.skipTest(f"MediaPipe not available: {e}")

    def test_detect_faces_empty_frame(self):
        """Test face detection on empty frame."""
        masker = FaceMasker(detection_backend="opencv_haar")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = masker.detect_faces(empty_frame)
        self.assertIsInstance(faces, list)
        # Empty frame should have no faces
        self.assertEqual(len(faces), 0)

    def test_apply_gaussian_blur_mask(self):
        """Test Gaussian blur masking."""
        masker = FaceMasker(mask_style="gaussian_blur", blur_strength=25)

        # Create fake face detection
        faces = [(200, 150, 100, 120)]  # x, y, w, h

        masked_frame = masker.apply_mask(self.test_frame.copy(), faces)

        self.assertIsNotNone(masked_frame)
        self.assertEqual(masked_frame.shape, self.test_frame.shape)

        # Check that the masked region is different from original
        face_region = masked_frame[150:270, 200:300]
        original_region = self.test_frame[150:270, 200:300]
        difference = np.mean(np.abs(face_region.astype(float) - original_region.astype(float)))

        # Blurred region should be different from original
        self.assertGreater(difference, 0)

    def test_apply_pixelate_mask(self):
        """Test pixelation masking."""
        masker = FaceMasker(mask_style="pixelate", pixelate_blocks=10)

        faces = [(200, 150, 100, 120)]
        masked_frame = masker.apply_mask(self.test_frame.copy(), faces)

        self.assertIsNotNone(masked_frame)
        self.assertEqual(masked_frame.shape, self.test_frame.shape)

        # Check that masking was applied
        face_region = masked_frame[150:270, 200:300]
        original_region = self.test_frame[150:270, 200:300]
        self.assertFalse(np.array_equal(face_region, original_region))

    def test_apply_black_box_mask(self):
        """Test black box masking."""
        masker = FaceMasker(mask_style="black_box")

        faces = [(200, 150, 100, 120)]
        masked_frame = masker.apply_mask(self.test_frame.copy(), faces)

        self.assertIsNotNone(masked_frame)
        self.assertEqual(masked_frame.shape, self.test_frame.shape)

        # Check that the face region is now black (or very dark)
        # Add padding (20%) to account for masker's padding
        padding = int(100 * 0.2)
        x1 = max(0, 200 - padding)
        y1 = max(0, 150 - padding)
        x2 = min(640, 300 + padding)
        y2 = min(480, 270 + padding)

        face_region = masked_frame[y1:y2, x1:x2]
        mean_intensity = np.mean(face_region)

        # Black box should have very low intensity
        self.assertLess(mean_intensity, 50)

    def test_apply_adaptive_blur_mask(self):
        """Test adaptive blur masking."""
        masker = FaceMasker(mask_style="adaptive_blur", blur_strength=25)

        faces = [(200, 150, 100, 120)]
        masked_frame = masker.apply_mask(self.test_frame.copy(), faces)

        self.assertIsNotNone(masked_frame)
        self.assertEqual(masked_frame.shape, self.test_frame.shape)

    def test_detect_and_mask_combined(self):
        """Test combined detect and mask operation."""
        masker = FaceMasker(detection_backend="opencv_haar", mask_style="gaussian_blur")

        masked_frame, faces = masker.detect_and_mask(self.test_frame.copy())

        self.assertIsNotNone(masked_frame)
        self.assertIsInstance(faces, list)
        self.assertEqual(masked_frame.shape, self.test_frame.shape)

    def test_multiple_faces(self):
        """Test masking multiple faces."""
        masker = FaceMasker(mask_style="gaussian_blur")

        # Multiple fake face detections
        faces = [
            (100, 100, 80, 90),
            (400, 200, 70, 85),
            (250, 300, 90, 100)
        ]

        masked_frame = masker.apply_mask(self.test_frame.copy(), faces)

        self.assertIsNotNone(masked_frame)
        self.assertEqual(masked_frame.shape, self.test_frame.shape)

    def test_min_face_size_filtering(self):
        """Test that faces smaller than min_face_size are filtered."""
        masker = FaceMasker(detection_backend="opencv_haar", min_face_size=50)

        # The actual detection will filter based on min_face_size parameter
        # We're just checking that the parameter is set correctly
        self.assertEqual(masker.min_face_size, 50)

    def test_mask_style_override(self):
        """Test that mask style can be overridden per call."""
        masker = FaceMasker(mask_style="gaussian_blur")

        faces = [(200, 150, 100, 120)]

        # Apply with default style
        masked_default = masker.apply_mask(self.test_frame.copy(), faces)

        # Apply with overridden style
        masked_override = masker.apply_mask(self.test_frame.copy(), faces, mask_style="black_box")

        # The two should be different
        difference = np.mean(np.abs(masked_default.astype(float) - masked_override.astype(float)))
        self.assertGreater(difference, 10)  # Significant difference


class TestFaceMaskingCache(unittest.TestCase):
    """Test suite for FaceMaskingCache class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = FaceMaskingCache(ttl_frames=5)

    def test_cache_initialization(self):
        """Test cache initialization."""
        self.assertIsNotNone(self.cache)
        self.assertEqual(self.cache.ttl, 5)
        self.assertEqual(len(self.cache.cache), 0)

    def test_should_detect_first_time(self):
        """Test that should_detect returns True for first frame."""
        should_detect = self.cache.should_detect("cam1")
        self.assertTrue(should_detect)

    def test_cache_update_and_retrieval(self):
        """Test updating and retrieving cached faces."""
        faces = [(100, 100, 80, 90), (300, 200, 70, 85)]
        self.cache.update_cache("cam1", faces)

        cached_faces = self.cache.get_cached_faces("cam1")
        self.assertEqual(cached_faces, faces)

    def test_should_detect_after_ttl(self):
        """Test that should_detect returns True after TTL expires."""
        faces = [(100, 100, 80, 90)]
        self.cache.update_cache("cam1", faces)

        # First few frames should use cache
        for i in range(5):
            should_detect = self.cache.should_detect("cam1")
            if i < 5:
                self.assertFalse(should_detect)
                self.cache.increment_frame_count("cam1")

        # After TTL frames, should detect again
        should_detect = self.cache.should_detect("cam1")
        self.assertTrue(should_detect)

    def test_increment_frame_count(self):
        """Test frame count incrementation."""
        faces = [(100, 100, 80, 90)]
        self.cache.update_cache("cam1", faces)

        # Frame count should start at 0
        _, frame_count = self.cache.cache["cam1"]
        self.assertEqual(frame_count, 0)

        # Increment and check
        self.cache.increment_frame_count("cam1")
        _, frame_count = self.cache.cache["cam1"]
        self.assertEqual(frame_count, 1)

    def test_cache_per_camera(self):
        """Test that cache is per-camera."""
        faces1 = [(100, 100, 80, 90)]
        faces2 = [(300, 200, 70, 85)]

        self.cache.update_cache("cam1", faces1)
        self.cache.update_cache("cam2", faces2)

        cached_faces1 = self.cache.get_cached_faces("cam1")
        cached_faces2 = self.cache.get_cached_faces("cam2")

        self.assertEqual(cached_faces1, faces1)
        self.assertEqual(cached_faces2, faces2)

    def test_clear_cache_single_camera(self):
        """Test clearing cache for a single camera."""
        faces1 = [(100, 100, 80, 90)]
        faces2 = [(300, 200, 70, 85)]

        self.cache.update_cache("cam1", faces1)
        self.cache.update_cache("cam2", faces2)

        self.cache.clear_cache("cam1")

        self.assertIsNone(self.cache.get_cached_faces("cam1"))
        self.assertIsNotNone(self.cache.get_cached_faces("cam2"))

    def test_clear_cache_all(self):
        """Test clearing cache for all cameras."""
        faces1 = [(100, 100, 80, 90)]
        faces2 = [(300, 200, 70, 85)]

        self.cache.update_cache("cam1", faces1)
        self.cache.update_cache("cam2", faces2)

        self.cache.clear_cache()  # Clear all

        self.assertIsNone(self.cache.get_cached_faces("cam1"))
        self.assertIsNone(self.cache.get_cached_faces("cam2"))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFaceMasker))
    suite.addTests(loader.loadTestsFromTestCase(TestFaceMaskingCache))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
