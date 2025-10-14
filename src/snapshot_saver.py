"""
Snapshot Saver Module
Saves images/clips when specific detection events occur.
"""

import cv2
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from queue import Queue
from threading import Thread, Event, Lock
from collections import deque
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnapshotSaver:
    """
    Saves snapshots when detection events occur.
    Supports both single images and video clips with pre/post buffers.
    """

    def __init__(
        self,
        output_dir: str = "clips",
        save_mode: str = "image",  # "image" or "clip"
        trigger_classes: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        cooldown_seconds: int = 30,
        clip_duration: int = 10,
        pre_buffer_seconds: int = 5,
        fps: int = 30,
        save_annotated: bool = True
    ):
        """
        Initialize snapshot saver.

        Args:
            output_dir: Directory to save snapshots/clips
            save_mode: "image" for single frame, "clip" for video
            trigger_classes: List of class names that trigger save (None = all)
            min_confidence: Minimum confidence to trigger save
            cooldown_seconds: Seconds between saves for same class
            clip_duration: Duration of video clip in seconds
            pre_buffer_seconds: Seconds of video before detection
            fps: Frames per second for video clips
            save_annotated: Save with bounding boxes drawn
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_mode = save_mode
        self.trigger_classes = set(trigger_classes) if trigger_classes else None
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self.clip_duration = clip_duration
        self.pre_buffer_seconds = pre_buffer_seconds
        self.fps = fps
        self.save_annotated = save_annotated

        # Ring buffer for pre-detection frames
        buffer_size = pre_buffer_seconds * fps
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = Lock()

        # Cooldown tracking
        self.last_save_times: Dict[str, float] = {}
        self.cooldown_lock = Lock()

        # Statistics
        self.total_saved = 0
        self.save_failures = 0
        self.saves_by_class: Dict[str, int] = {}

        logger.info(f"SnapshotSaver initialized: mode={save_mode}, output={output_dir}")

    def add_frame_to_buffer(self, frame: Any, timestamp: float):
        """
        Add frame to ring buffer for pre-detection recording.

        Args:
            frame: Video frame
            timestamp: Frame timestamp
        """
        with self.buffer_lock:
            self.frame_buffer.append({
                'frame': frame.copy() if frame is not None else None,
                'timestamp': timestamp
            })

    def should_save(self, detection_result: Dict[str, Any]) -> bool:
        """
        Check if detection result should trigger a save.

        Args:
            detection_result: Detection result dictionary

        Returns:
            True if should save, False otherwise
        """
        detections = detection_result.get('detections', [])

        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Check confidence threshold
            if confidence < self.min_confidence:
                continue

            # Check if class should trigger save
            if self.trigger_classes and class_name not in self.trigger_classes:
                continue

            # Check cooldown
            with self.cooldown_lock:
                last_save = self.last_save_times.get(class_name, 0)
                current_time = time.time()  # Call once for consistency
                if current_time - last_save < self.cooldown_seconds:
                    continue

                # Update last save time (use same timestamp as check)
                self.last_save_times[class_name] = current_time

            return True

        return False

    def save_snapshot(
        self,
        frame: Any,
        detection_result: Dict[str, Any],
        annotated_frame: Optional[Any] = None
    ) -> Optional[str]:
        """
        Save a single snapshot image.
        Saves both raw (for training) and annotated (for web display) versions.

        Args:
            frame: Raw video frame
            detection_result: Detection result
            annotated_frame: Frame with annotations (if save_annotated=True)

        Returns:
            Path to saved file, or None if save failed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Get primary detection class for filename
        detections = detection_result.get('detections', [])
        if not detections:
            return None

        primary_class = detections[0]['class_name']
        confidence = detections[0]['confidence']

        # Get camera identifier
        camera_id = detection_result.get('camera_id', 'default')

        # Create base filename with camera ID
        base_filename = f"{camera_id}_{primary_class}_{timestamp}_conf{confidence:.2f}"
        raw_filepath = self.output_dir / f"{base_filename}.jpg"
        annotated_filepath = self.output_dir / f"{base_filename}_annotated.jpg"

        # Save images
        try:
            # Always save raw frame (for training data)
            success = cv2.imwrite(str(raw_filepath), frame)
            if not success:
                logger.error(f"Failed to save raw frame: {raw_filepath}")
                logger.error("Possible causes: disk full, permission denied, or invalid path")
                self.save_failures += 1
                return None

            # Save annotated frame if available (for web display)
            if self.save_annotated and annotated_frame is not None:
                success = cv2.imwrite(str(annotated_filepath), annotated_frame)
                if not success:
                    logger.warning(f"Failed to save annotated frame: {annotated_filepath}")
                    logger.warning("Raw frame saved successfully, continuing without annotated version")
                    # Don't fail - raw frame is more important

            # Save metadata JSON
            metadata = {
                'timestamp': detection_result.get('timestamp'),
                'camera_id': detection_result.get('camera_id', 'default'),
                'camera_name': detection_result.get('camera_name', 'Default Camera'),
                'detections': detections,
                'detection_counts': detection_result.get('detection_counts', {}),
                'latency_ms': detection_result.get('total_latency_ms', 0),
                'filename': f"{base_filename}.jpg",
                'annotated_filename': f"{base_filename}_annotated.jpg" if (self.save_annotated and annotated_frame is not None) else None
            }

            metadata_file = raw_filepath.with_suffix('.json')
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save metadata {metadata_file}: {e}")
                # Continue - snapshot is more important than metadata

            # Update statistics
            self.total_saved += 1
            self.saves_by_class[primary_class] = self.saves_by_class.get(primary_class, 0) + 1

            logger.info(f"📸 Saved snapshot: {base_filename}.jpg (raw + annotated)")
            return str(raw_filepath)

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            self.save_failures += 1
            return None

    def save_clip(
        self,
        current_frame: Any,
        detection_result: Dict[str, Any],
        annotated_frame: Optional[Any] = None
    ) -> Optional[str]:
        """
        Save a video clip with pre and post detection frames.

        Args:
            current_frame: Current video frame
            detection_result: Detection result
            annotated_frame: Current frame with annotations

        Returns:
            Path to saved file, or None if save failed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get primary detection class
        detections = detection_result.get('detections', [])
        if not detections:
            return None

        primary_class = detections[0]['class_name']

        # Get camera identifier
        camera_id = detection_result.get('camera_id', 'default')

        # Create filename with camera ID
        filename = f"{camera_id}_{primary_class}_{timestamp}.mp4"
        filepath = self.output_dir / filename

        try:
            # Get buffered frames (pre-detection)
            with self.buffer_lock:
                buffered_frames = list(self.frame_buffer)

            if not buffered_frames:
                logger.warning("No buffered frames available for clip")
                return None

            # Get frame dimensions
            sample_frame = buffered_frames[0]['frame']
            height, width = sample_frame.shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))

            # Write pre-detection frames
            for frame_data in buffered_frames:
                if frame_data['frame'] is not None:
                    out.write(frame_data['frame'])

            # Note: Post-detection frames would need to be captured in real-time
            # This is a simplified implementation that saves pre-buffer only
            # For full pre+post recording, you'd need to continue recording after detection

            out.release()

            # Save metadata
            metadata = {
                'timestamp': detection_result.get('timestamp'),
                'camera_id': detection_result.get('camera_id', 'default'),
                'camera_name': detection_result.get('camera_name', 'Default Camera'),
                'detections': detections,
                'detection_counts': detection_result.get('detection_counts', {}),
                'clip_duration_seconds': len(buffered_frames) / self.fps
            }

            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Update statistics
            self.total_saved += 1
            self.saves_by_class[primary_class] = self.saves_by_class.get(primary_class, 0) + 1

            logger.info(f"🎬 Saved clip: {filename} ({len(buffered_frames)} frames)")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save clip: {e}")
            return None

    def process_detection(
        self,
        frame: Any,
        detection_result: Dict[str, Any],
        annotated_frame: Optional[Any] = None
    ) -> Optional[str]:
        """
        Process detection and save if triggered.

        Args:
            frame: Raw video frame
            detection_result: Detection result
            annotated_frame: Frame with annotations

        Returns:
            Path to saved file, or None if not saved
        """
        if not self.should_save(detection_result):
            return None

        if self.save_mode == "image":
            return self.save_snapshot(frame, detection_result, annotated_frame)
        elif self.save_mode == "clip":
            return self.save_clip(frame, detection_result, annotated_frame)
        else:
            logger.error(f"Unknown save mode: {self.save_mode}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get snapshot saver statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_saved': self.total_saved,
            'save_failures': self.save_failures,
            'saves_by_class': self.saves_by_class,
            'output_dir': str(self.output_dir),
            'save_mode': self.save_mode
        }

    def cleanup_old_files(self, max_age_days: int = 7):
        """
        Remove old snapshot files.

        Args:
            max_age_days: Maximum age of files to keep
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed = 0

        for file in self.output_dir.glob("*"):
            if file.is_file() and file.stat().st_mtime < cutoff_time:
                try:
                    file.unlink()
                    removed += 1
                except Exception as e:
                    logger.error(f"Failed to remove {file}: {e}")

        if removed > 0:
            logger.info(f"🗑️  Cleaned up {removed} old files")


if __name__ == "__main__":
    # Test the snapshot saver
    import numpy as np

    logger.info("Testing SnapshotSaver")

    saver = SnapshotSaver(
        output_dir="clips/test",
        save_mode="image",
        trigger_classes=["person", "cat"],
        min_confidence=0.5,
        cooldown_seconds=5
    )

    # Create test frame
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Create test detection
    test_detection = {
        'timestamp': time.time(),
        'detections': [
            {
                'class_name': 'person',
                'confidence': 0.95,
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 300}
            }
        ],
        'detection_counts': {'person': 1},
        'total_latency_ms': 50.0
    }

    # Test save
    saved_path = saver.process_detection(test_frame, test_detection)

    if saved_path:
        logger.info(f"✅ Test successful! Saved to: {saved_path}")
        logger.info(f"Stats: {saver.get_stats()}")
    else:
        logger.error("❌ Test failed")
