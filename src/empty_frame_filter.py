"""
Empty Frame Filter (L-filter method) - Issue #160
Lightweight motion detection to skip inference on frames with no motion.
Expected: 30-50% throughput improvement on wildlife footage (70-90% empty frames).
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EmptyFrameFilter:
    """
    Lightweight motion detector for skipping inference on empty frames.
    Uses frame differencing with Gaussian blur for noise reduction.
    """

    def __init__(
        self,
        min_motion_area: int = 200,
        threshold: int = 25,
        blur_size: int = 21
    ):
        """
        Initialize empty frame filter.

        Args:
            min_motion_area: Minimum motion area (pixels²) to trigger inference
            threshold: Frame difference threshold (0-255, lower = more sensitive)
            blur_size: Gaussian blur kernel size (must be odd)
        """
        self.min_motion_area = min_motion_area
        self.threshold = threshold
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  # Ensure odd

        self.prev_gray: Optional[np.ndarray] = None

        # Statistics
        self.total_frames = 0
        self.skipped_frames = 0
        self.motion_frames = 0

        logger.info(
            f"EmptyFrameFilter initialized: min_motion_area={min_motion_area}px², "
            f"threshold={threshold}, blur_size={self.blur_size}"
        )

    def has_motion(self, frame: np.ndarray) -> bool:
        """
        Check if frame has significant motion.

        Args:
            frame: Input frame (BGR or grayscale)

        Returns:
            True if motion detected (run inference), False if empty (skip inference)
        """
        self.total_frames += 1

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        # First frame - no comparison possible
        if self.prev_gray is None:
            self.prev_gray = blurred
            self.motion_frames += 1
            return True

        # Compute absolute difference between frames
        frame_delta = cv2.absdiff(self.prev_gray, blurred)

        # Threshold the delta image
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Count non-zero pixels (motion area)
        motion_area = cv2.countNonZero(thresh)

        # Update previous frame
        self.prev_gray = blurred

        # Check if motion exceeds threshold
        has_motion = motion_area >= self.min_motion_area

        if has_motion:
            self.motion_frames += 1
        else:
            self.skipped_frames += 1

        return has_motion

    def reset(self):
        """Reset filter state (for camera reconnection, etc.)"""
        self.prev_gray = None

    def get_stats(self) -> dict:
        """Get filter statistics."""
        skip_rate = self.skipped_frames / max(self.total_frames, 1)
        return {
            'total_frames': self.total_frames,
            'skipped_frames': self.skipped_frames,
            'motion_frames': self.motion_frames,
            'skip_rate': skip_rate,
            'skip_rate_percent': skip_rate * 100
        }
