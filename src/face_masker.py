"""
Privacy-preserving face detection and masking module.

This module provides face detection and various masking strategies to protect
individual privacy in both live video feeds and saved clips.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceMasker:
    """
    Face detection and masking for privacy protection.

    Supports multiple detection backends (OpenCV Haar, MediaPipe, YOLOX)
    and various masking styles (blur, pixelate, black box, adaptive blur).
    """

    # Padding percentage around detected faces (20% of face width)
    FACE_PADDING_PERCENT = 0.2

    def __init__(
        self,
        detection_backend: str = "opencv_haar",
        mask_style: str = "gaussian_blur",
        min_face_size: int = 30,
        blur_strength: int = 25,
        pixelate_blocks: int = 10,
        scale_factor: float = 1.1,
        min_neighbors: int = 5
    ):
        """
        Initialize face masker.

        Args:
            detection_backend: Backend to use for face detection
                              ('opencv_haar', 'mediapipe', 'yolox')
            mask_style: Masking style to apply
                       ('gaussian_blur', 'pixelate', 'black_box', 'adaptive_blur')
            min_face_size: Minimum face size in pixels to detect (filters false positives)
            blur_strength: Gaussian blur kernel size (must be odd number)
            pixelate_blocks: Pixel block size for pixelation effect
            scale_factor: Haar cascade scale factor (lower = more detections, slower)
            min_neighbors: Haar cascade min neighbors (higher = fewer false positives)
        """
        self.detection_backend = detection_backend
        self.mask_style = mask_style
        self.min_face_size = min_face_size

        # Validate blur_strength is positive
        if blur_strength <= 0:
            raise ValueError(f"blur_strength must be positive, got {blur_strength}")

        # Gaussian blur requires odd kernel size - auto-adjust if needed
        if blur_strength % 2 == 1:
            self.blur_strength = blur_strength
        else:
            adjusted_blur_strength = blur_strength + 1
            logger.warning(
                f"blur_strength must be odd, got {blur_strength}. "
                f"Automatically adjusted to {adjusted_blur_strength}."
            )
            self.blur_strength = adjusted_blur_strength

        self.pixelate_blocks = pixelate_blocks
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        # Initialize detection backend
        if detection_backend == "opencv_haar":
            self._init_opencv_haar()
        elif detection_backend == "mediapipe":
            self._init_mediapipe()
        elif detection_backend == "yolox":
            # TODO: YOLOX backend planned for Phase 2 (leverage existing YOLOX GPU pipeline)
            # Would require adding face detection class to YOLOX model
            # See issue #87 Phase 2 enhancement roadmap
            logger.warning("YOLOX backend not yet implemented, falling back to OpenCV Haar")
            self.detection_backend = "opencv_haar"
            self._init_opencv_haar()
        else:
            logger.error(f"Unknown detection backend: {detection_backend}, using opencv_haar")
            self.detection_backend = "opencv_haar"
            self._init_opencv_haar()

        logger.info(
            f"FaceMasker initialized: backend={self.detection_backend}, "
            f"style={self.mask_style}, min_size={self.min_face_size}"
        )

    def _init_opencv_haar(self):
        """Initialize OpenCV Haar Cascade face detector."""
        try:
            # Use built-in OpenCV Haar cascade for frontal faces
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                raise ValueError("Failed to load Haar cascade")

            logger.info(f"OpenCV Haar cascade loaded from: {cascade_path}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV Haar cascade: {e}")
            raise

    def _init_mediapipe(self):
        """Initialize MediaPipe face detector."""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range (< 2m), 1 for full-range
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe face detector initialized")
        except ImportError:
            logger.error("MediaPipe not installed. Install with: pip install mediapipe")
            logger.warning("Falling back to OpenCV Haar cascade")
            self.detection_backend = "opencv_haar"
            self._init_opencv_haar()
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            logger.warning("Falling back to OpenCV Haar cascade")
            self.detection_backend = "opencv_haar"
            self._init_opencv_haar()

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        if self.detection_backend == "opencv_haar":
            return self._detect_opencv_haar(frame)
        elif self.detection_backend == "mediapipe":
            return self._detect_mediapipe(frame)
        else:
            logger.error(f"Unknown detection backend: {self.detection_backend}")
            return []

    def _detect_opencv_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar Cascade."""
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size)
        )

        # Convert from numpy array to list of tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def _detect_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe."""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)

        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Filter by minimum size
                if width >= self.min_face_size and height >= self.min_face_size:
                    faces.append((x, y, width, height))

        return faces

    def apply_mask(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        mask_style: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply masking to detected faces.

        Args:
            frame: Input frame (BGR format)
            faces: List of face bounding boxes as (x, y, width, height)
            mask_style: Override default mask style (optional)

        Returns:
            Masked frame (BGR format)
        """
        if not faces:
            return frame

        style = mask_style or self.mask_style
        masked_frame = frame.copy()

        for (x, y, w, h) in faces:
            if style == "gaussian_blur":
                masked_frame = self._apply_gaussian_blur(masked_frame, x, y, w, h)
            elif style == "pixelate":
                masked_frame = self._apply_pixelate(masked_frame, x, y, w, h)
            elif style == "black_box":
                masked_frame = self._apply_black_box(masked_frame, x, y, w, h)
            elif style == "adaptive_blur":
                masked_frame = self._apply_adaptive_blur(masked_frame, x, y, w, h)
            else:
                logger.warning(f"Unknown mask style: {style}, using gaussian_blur")
                masked_frame = self._apply_gaussian_blur(masked_frame, x, y, w, h)

        return masked_frame

    def _apply_gaussian_blur(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> np.ndarray:
        """Apply Gaussian blur to face region."""
        # Add padding around face for better coverage
        padding = int(w * self.FACE_PADDING_PERCENT)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        # Extract face region
        face_region = frame[y1:y2, x1:x2]

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(face_region, (self.blur_strength, self.blur_strength), 0)

        # Replace face region with blurred version
        frame[y1:y2, x1:x2] = blurred

        return frame

    def _apply_pixelate(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> np.ndarray:
        """Apply pixelation effect to face region."""
        # Add padding
        padding = int(w * self.FACE_PADDING_PERCENT)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        # Extract face region
        face_region = frame[y1:y2, x1:x2]

        # Resize down and back up for pixelation effect
        h_region, w_region = face_region.shape[:2]
        small_h = max(1, h_region // self.pixelate_blocks)
        small_w = max(1, w_region // self.pixelate_blocks)

        # Downscale
        small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # Upscale back to original size with nearest neighbor (creates pixelation)
        pixelated = cv2.resize(small, (w_region, h_region), interpolation=cv2.INTER_NEAREST)

        # Replace face region
        frame[y1:y2, x1:x2] = pixelated

        return frame

    def _apply_black_box(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> np.ndarray:
        """Apply solid black rectangle over face."""
        # Add padding
        padding = int(w * self.FACE_PADDING_PERCENT)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        # Draw black rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        return frame

    def _apply_adaptive_blur(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> np.ndarray:
        """Apply adaptive blur - stronger blur for larger/closer faces."""
        # Calculate adaptive blur strength based on face size
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        face_ratio = face_area / frame_area

        # Scale blur strength (larger faces get more blur)
        adaptive_strength = int(self.blur_strength * (1 + face_ratio * 10))
        adaptive_strength = adaptive_strength if adaptive_strength % 2 == 1 else adaptive_strength + 1
        adaptive_strength = min(99, adaptive_strength)  # Cap at 99

        # Add padding
        padding = int(w * self.FACE_PADDING_PERCENT)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        # Extract face region
        face_region = frame[y1:y2, x1:x2]

        # Apply adaptive Gaussian blur
        blurred = cv2.GaussianBlur(face_region, (adaptive_strength, adaptive_strength), 0)

        # Replace face region
        frame[y1:y2, x1:x2] = blurred

        return frame

    def detect_and_mask(
        self,
        frame: np.ndarray,
        mask_style: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detect faces and apply masking in one step.

        Args:
            frame: Input frame (BGR format)
            mask_style: Override default mask style (optional)

        Returns:
            Tuple of (masked_frame, faces) where faces is list of bounding boxes
        """
        faces = self.detect_faces(frame)
        masked_frame = self.apply_mask(frame, faces, mask_style)
        return masked_frame, faces


class FaceMaskingCache:
    """
    Cache face positions and apply to nearby frames for performance optimization.

    Reduces CPU usage by detecting faces only every N frames and reusing
    face positions for intermediate frames.
    """

    def __init__(self, ttl_frames: int = 5):
        """
        Initialize face masking cache.

        Args:
            ttl_frames: Number of frames to cache face positions for
        """
        self.cache: Dict[str, Tuple[List[Tuple[int, int, int, int]], int]] = {}
        self.ttl = ttl_frames

    def should_detect(self, camera_id: str) -> bool:
        """
        Check if face detection should be run for this frame.

        Args:
            camera_id: Camera identifier

        Returns:
            True if detection should run, False if cached positions should be used
        """
        if camera_id not in self.cache:
            return True

        _, frame_count = self.cache[camera_id]
        return frame_count >= self.ttl

    def get_cached_faces(
        self,
        camera_id: str
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """
        Get cached face positions for camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of face bounding boxes, or None if not cached
        """
        if camera_id not in self.cache:
            return None

        faces, _ = self.cache[camera_id]
        return faces

    def update_cache(
        self,
        camera_id: str,
        faces: List[Tuple[int, int, int, int]]
    ):
        """
        Update cached face positions for camera.

        Args:
            camera_id: Camera identifier
            faces: List of face bounding boxes
        """
        self.cache[camera_id] = (faces, 0)

    def increment_frame_count(self, camera_id: str):
        """
        Increment frame count for camera.

        Args:
            camera_id: Camera identifier
        """
        if camera_id in self.cache:
            faces, frame_count = self.cache[camera_id]
            self.cache[camera_id] = (faces, frame_count + 1)

    def clear_cache(self, camera_id: Optional[str] = None):
        """
        Clear cache for specific camera or all cameras.

        Args:
            camera_id: Camera identifier, or None to clear all
        """
        if camera_id is None:
            self.cache.clear()
        elif camera_id in self.cache:
            del self.cache[camera_id]
