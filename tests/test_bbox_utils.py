#!/usr/bin/env python3
"""
Tests for bounding box validation utilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from src.bbox_utils import (
    validate_bbox,
    bbox_area,
    bbox_iou,
    ensure_valid_bbox,
    validate_bbox_coords,
    is_valid_bbox
)


class TestValidateBbox(unittest.TestCase):
    """Tests for validate_bbox function."""

    def test_valid_bbox(self):
        """Test with valid bounding box."""
        bbox = (10, 20, 100, 200)
        result = validate_bbox(bbox)
        self.assertEqual(result, (10, 20, 100, 200))

    def test_inverted_coordinates(self):
        """Test that inverted coordinates are swapped."""
        bbox = (100, 200, 10, 20)
        result = validate_bbox(bbox)
        self.assertEqual(result, (10, 20, 100, 200))

    def test_minimum_size_enforcement(self):
        """Test that minimum size is enforced."""
        # Width too small
        bbox = (10, 20, 10, 100)
        result = validate_bbox(bbox, min_size=1)
        self.assertEqual(result, (10, 20, 11, 100))
        
        # Height too small
        bbox = (10, 20, 100, 20)
        result = validate_bbox(bbox, min_size=1)
        self.assertEqual(result, (10, 20, 100, 21))

    def test_frame_clipping(self):
        """Test clipping to frame bounds."""
        frame_shape = (480, 640)  # height, width
        
        # Bbox extends beyond frame
        bbox = (600, 450, 700, 500)
        result = validate_bbox(bbox, frame_shape=frame_shape)
        self.assertEqual(result, (600, 450, 640, 480))
        
        # Negative coordinates
        bbox = (-10, -20, 100, 200)
        result = validate_bbox(bbox, frame_shape=frame_shape)
        self.assertEqual(result, (0, 0, 100, 200))

    def test_invalid_bbox_after_normalization(self):
        """Test that bbox is properly clipped even when extending beyond frame."""
        # Bbox extends beyond frame but can be clipped to valid region
        frame_shape = (480, 640)
        bbox = (639, 20, 650, 100)
        result = validate_bbox(bbox, frame_shape=frame_shape)
        # Should be clipped to (639, 20, 640, 100)
        self.assertEqual(result, (639, 20, 640, 100))

    def test_invalid_input_types(self):
        """Test with invalid input types."""
        result = validate_bbox(None)
        self.assertIsNone(result)
        
        result = validate_bbox("invalid")
        self.assertIsNone(result)
        
        result = validate_bbox((1, 2))  # Not enough values
        self.assertIsNone(result)

    def test_list_input(self):
        """Test that list input works."""
        bbox = [10, 20, 100, 200]
        result = validate_bbox(bbox)
        self.assertEqual(result, (10, 20, 100, 200))

    def test_float_coordinates(self):
        """Test with float coordinates (should convert to int)."""
        bbox = (10.5, 20.7, 100.2, 200.9)
        result = validate_bbox(bbox)
        self.assertEqual(result, (10, 20, 100, 200))


class TestBboxArea(unittest.TestCase):
    """Tests for bbox_area function."""

    def test_valid_tuple_bbox(self):
        """Test area calculation with tuple bbox."""
        bbox = (10, 20, 100, 200)
        area = bbox_area(bbox)
        self.assertEqual(area, 90 * 180)  # (100-10) * (200-20)

    def test_valid_dict_bbox(self):
        """Test area calculation with dict bbox."""
        bbox = {'x1': 10, 'y1': 20, 'x2': 100, 'y2': 200}
        area = bbox_area(bbox)
        self.assertEqual(area, 90 * 180)

    def test_zero_area(self):
        """Test with zero area bbox."""
        bbox = (10, 20, 10, 20)
        area = bbox_area(bbox)
        self.assertEqual(area, 0)

    def test_negative_area(self):
        """Test with inverted bbox (still calculates absolute area)."""
        bbox = (100, 200, 10, 20)
        area = bbox_area(bbox)
        # Note: bbox_area calculates absolute value, doesn't validate order
        # Use validate_bbox first if you need ordering
        self.assertGreater(area, 0)

    def test_invalid_input(self):
        """Test with invalid input."""
        self.assertEqual(bbox_area(None), 0)
        self.assertEqual(bbox_area("invalid"), 0)
        self.assertEqual(bbox_area((1, 2)), 0)  # Not enough values
        self.assertEqual(bbox_area({}), 0)  # Missing keys


class TestBboxIou(unittest.TestCase):
    """Tests for bbox_iou function."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes."""
        bbox1 = (10, 20, 100, 200)
        bbox2 = (10, 20, 100, 200)
        iou = bbox_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 1.0)

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes."""
        bbox1 = (10, 20, 100, 200)
        bbox2 = (200, 300, 300, 400)
        iou = bbox_iou(bbox1, bbox2)
        self.assertEqual(iou, 0.0)

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 150, 150)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 = 0.142857...
        iou = bbox_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 2500 / 17500, places=5)

    def test_one_inside_another(self):
        """Test IoU when one box is inside another."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (25, 25, 75, 75)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 2500 - 2500 = 10000
        # IoU: 2500 / 10000 = 0.25
        iou = bbox_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 0.25)

    def test_touching_boxes(self):
        """Test IoU of boxes that touch but don't overlap."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (100, 0, 200, 100)
        iou = bbox_iou(bbox1, bbox2)
        self.assertEqual(iou, 0.0)

    def test_invalid_input(self):
        """Test with invalid input."""
        bbox1 = (10, 20, 100, 200)
        self.assertEqual(bbox_iou(bbox1, None), 0.0)
        self.assertEqual(bbox_iou(None, bbox1), 0.0)
        self.assertEqual(bbox_iou((1, 2), bbox1), 0.0)


class TestEnsureValidBbox(unittest.TestCase):
    """Tests for ensure_valid_bbox function (existing functionality)."""

    def test_valid_bbox(self):
        """Test with valid bounding box."""
        bbox = {'x1': 10, 'y1': 20, 'x2': 100, 'y2': 200}
        result = ensure_valid_bbox(bbox)
        self.assertEqual(result['x1'], 10)
        self.assertEqual(result['y1'], 20)
        self.assertEqual(result['x2'], 100)
        self.assertEqual(result['y2'], 200)
        self.assertEqual(result['width'], 90)
        self.assertEqual(result['height'], 180)
        self.assertEqual(result['area'], 90 * 180)

    def test_inverted_coordinates(self):
        """Test that inverted coordinates are swapped."""
        bbox = {'x1': 100, 'y1': 200, 'x2': 10, 'y2': 20}
        result = ensure_valid_bbox(bbox)
        self.assertEqual(result['x1'], 10)
        self.assertEqual(result['y1'], 20)
        self.assertEqual(result['x2'], 100)
        self.assertEqual(result['y2'], 200)


class TestValidateBboxCoords(unittest.TestCase):
    """Tests for validate_bbox_coords function (existing functionality)."""

    def test_valid_coords(self):
        """Test with valid coordinates."""
        result = validate_bbox_coords(10, 20, 100, 200)
        self.assertEqual(result, (10, 20, 100, 200))

    def test_inverted_coords(self):
        """Test with inverted coordinates."""
        result = validate_bbox_coords(100, 200, 10, 20)
        self.assertEqual(result, (10, 20, 100, 200))


class TestIsValidBbox(unittest.TestCase):
    """Tests for is_valid_bbox function (existing functionality)."""

    def test_valid_bbox(self):
        """Test with valid bbox."""
        bbox = {'x1': 10, 'y1': 20, 'x2': 100, 'y2': 200}
        self.assertTrue(is_valid_bbox(bbox))

    def test_inverted_bbox(self):
        """Test with inverted bbox."""
        bbox = {'x1': 100, 'y1': 20, 'x2': 10, 'y2': 200}
        self.assertFalse(is_valid_bbox(bbox))

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        bbox = {'x1': -10, 'y1': 20, 'x2': 100, 'y2': 200}
        self.assertFalse(is_valid_bbox(bbox))


if __name__ == '__main__':
    unittest.main()
