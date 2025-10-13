"""
Motion Detection Filter
Filters out static objects (rocks, shadows) by detecting frame-to-frame motion.
Uses background subtraction to identify regions with actual movement.
"""

import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class MotionFilter:
    """
    Motion detection filter using background subtraction.
    Filters out detections in static regions (no motion).
    """

    def __init__(
        self,
        history: int = 500,  # Number of frames for background model
        var_threshold: int = 16,  # Threshold for background/foreground classification
        detect_shadows: bool = True,  # Detect shadows (slower but more accurate)
        min_motion_area: int = 100,  # Minimum motion area in pixels² to consider
        motion_required: bool = True,  # Require motion for detection to be valid
        motion_blur_size: int = 21,  # Gaussian blur kernel size for noise reduction
        min_motion_ratio: float = 0.05,  # Minimum motion ratio (5% of bbox must have motion)
    ):
        """
        Initialize motion filter.

        Args:
            history: Number of frames for background model (higher = slower adaptation)
            var_threshold: Threshold for background/foreground (higher = less sensitive)
            detect_shadows: Detect and filter shadows
            min_motion_area: Minimum motion area to consider valid
            motion_required: If True, filter detections without motion
            motion_blur_size: Gaussian blur kernel size (must be odd)
            min_motion_ratio: Minimum ratio of motion pixels to bbox area (0.0-1.0)
        """
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.min_motion_area = min_motion_area
        self.motion_required = motion_required
        # Gaussian blur kernel size must be odd (required by OpenCV cv2.GaussianBlur)
        self.motion_blur_size = motion_blur_size if motion_blur_size % 2 == 1 else motion_blur_size + 1
        self.min_motion_ratio = min_motion_ratio

        # Background subtractor (MOG2 - mixture of Gaussians)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

        # Stats
        self.total_frames = 0
        self.total_detections_filtered = 0

        logger.info(f"MotionFilter initialized (history={history}, var_threshold={var_threshold})")

    def has_motion_in_bbox(
        self,
        frame: np.ndarray,
        bbox: Dict[str, float],
        min_motion_pixels: int = 10
    ) -> Tuple[bool, float]:
        """
        Check if there's motion in a bounding box region.

        Args:
            frame: Current frame (BGR)
            bbox: Bounding box dict with x1, y1, x2, y2
            min_motion_pixels: Minimum number of motion pixels required

        Returns:
            Tuple of (has_motion, motion_ratio)
            - has_motion: True if motion detected
            - motion_ratio: Ratio of motion pixels to total bbox area
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (if enabled, shadows are marked as 127, foreground as 255)
        if self.detect_shadows:
            fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Apply Gaussian blur to reduce noise
        fg_mask = cv2.GaussianBlur(fg_mask, (self.motion_blur_size, self.motion_blur_size), 0)

        # Threshold again after blur
        fg_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)[1]

        # Extract ROI from motion mask
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        x2, y2 = int(bbox['x2']), int(bbox['y2'])

        # Ensure bbox is within frame bounds
        h, w = fg_mask.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False, 0.0

        roi = fg_mask[y1:y2, x1:x2]

        # Count motion pixels
        motion_pixels = cv2.countNonZero(roi)
        bbox_area = (x2 - x1) * (y2 - y1)
        motion_ratio = motion_pixels / bbox_area if bbox_area > 0 else 0.0

        # Check if motion exceeds minimum threshold
        has_motion = motion_pixels >= min_motion_pixels and motion_ratio > self.min_motion_ratio

        return has_motion, motion_ratio

    def filter_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter detections based on motion.

        Args:
            frame: Current frame (BGR)
            detections: List of detection dictionaries

        Returns:
            Filtered list of detections with motion
        """
        if not self.motion_required or len(detections) == 0:
            return detections

        self.total_frames += 1
        filtered = []

        for det in detections:
            has_motion, motion_ratio = self.has_motion_in_bbox(frame, det['bbox'])

            if has_motion:
                # Add motion metadata
                det['has_motion'] = True
                det['motion_ratio'] = motion_ratio
                filtered.append(det)
            else:
                self.total_detections_filtered += 1

        if len(detections) > len(filtered):
            logger.debug(f"Motion filter: {len(detections)} → {len(filtered)} detections "
                        f"({len(detections) - len(filtered)} filtered)")

        return filtered

    def reset_background(self):
        """Reset background model (useful when camera moves or lighting changes)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        logger.info("Background model reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get motion filter statistics."""
        return {
            'total_frames': self.total_frames,
            'total_detections_filtered': self.total_detections_filtered,
            'motion_required': self.motion_required
        }


class AdaptiveMotionFilter(MotionFilter):
    """
    Adaptive motion filter that adjusts sensitivity based on time of day.
    More sensitive during daytime (less noise), less sensitive at night (more IR noise).
    """

    def __init__(
        self,
        day_var_threshold: int = 16,
        night_var_threshold: int = 32,
        day_start_hour: int = 6,
        day_end_hour: int = 20,
        **kwargs
    ):
        """
        Initialize adaptive motion filter.

        Args:
            day_var_threshold: Threshold during daytime
            night_var_threshold: Threshold during nighttime (higher = less sensitive)
            day_start_hour: Hour when day starts (0-23)
            day_end_hour: Hour when day ends (0-23)
            **kwargs: Other MotionFilter parameters
        """
        super().__init__(var_threshold=day_var_threshold, **kwargs)
        self.day_var_threshold = day_var_threshold
        self.night_var_threshold = night_var_threshold
        self.day_start_hour = day_start_hour
        self.day_end_hour = day_end_hour

    def _is_daytime(self) -> bool:
        """Check if it's currently daytime."""
        current_hour = datetime.now().hour
        return self.day_start_hour <= current_hour < self.day_end_hour

    def filter_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter detections with adaptive sensitivity."""
        # Adjust threshold based on time of day
        is_day = self._is_daytime()
        current_threshold = self.day_var_threshold if is_day else self.night_var_threshold

        # Update background subtractor if threshold changed
        if current_threshold != self.var_threshold:
            self.var_threshold = current_threshold
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows
            )

        return super().filter_detections(frame, detections)


if __name__ == "__main__":
    # Test motion filter
    logger.info("Testing MotionFilter")

    # Create motion filter
    motion_filter = MotionFilter(
        history=100,
        var_threshold=16,
        min_motion_area=100,
        motion_required=True
    )

    # Create test frames (simulated motion)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    cv2.rectangle(frame2, (100, 100), (200, 200), (255, 255, 255), -1)  # Add white square (motion)

    # Test detections
    test_detections = [
        {
            'class_name': 'bird',
            'confidence': 0.8,
            'bbox': {'x1': 90, 'y1': 90, 'x2': 210, 'y2': 210, 'area': 14400}
        },
        {
            'class_name': 'rock',
            'confidence': 0.6,
            'bbox': {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400, 'area': 10000}
        }
    ]

    # Process frames
    filtered1 = motion_filter.filter_detections(frame1, test_detections)
    logger.info(f"Frame 1: {len(test_detections)} → {len(filtered1)} detections")

    filtered2 = motion_filter.filter_detections(frame2, test_detections)
    logger.info(f"Frame 2: {len(test_detections)} → {len(filtered2)} detections")

    # Stats
    logger.info(f"Stats: {motion_filter.get_stats()}")

    logger.info("MotionFilter test completed")
