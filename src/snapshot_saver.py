"""
Snapshot Saver Module
Saves images/clips when specific detection events occur.
Supports privacy-preserving face masking with dual storage.
"""

import cv2
import time
import logging
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, TYPE_CHECKING
from queue import Queue
from threading import Thread, Event, Lock
from collections import deque
import json
import numpy as np

if TYPE_CHECKING:
    from .face_masker import FaceMasker

logger = logging.getLogger(__name__)

# Maximum buffer memory in MB (per SnapshotSaver instance)
# With multi-camera systems, this helps prevent OOM errors
MAX_BUFFER_MEMORY_MB = 500

# Frequency to check memory usage (in frames)
# Check every N frames to avoid excessive overhead
MEMORY_CHECK_INTERVAL = 30


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
        save_annotated: bool = True,
        face_masker: Optional['FaceMasker'] = None,
        enable_face_masking: bool = False
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
            face_masker: FaceMasker instance for privacy-preserving face masking
            enable_face_masking: Enable face masking feature
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Face masking
        self.face_masker = face_masker
        self.enable_face_masking = enable_face_masking and face_masker is not None

        self.save_mode = save_mode
        self.trigger_classes = set(trigger_classes) if trigger_classes else None
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self.clip_duration = clip_duration
        self.pre_buffer_seconds = pre_buffer_seconds
        self.fps = fps
        self.save_annotated = save_annotated

        # Ring buffer for pre-detection frames
        # Use JPEG compression to reduce memory usage
        buffer_size = pre_buffer_seconds * fps
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = Lock()
        self.use_compressed_buffer = True  # Compress frames in buffer to save memory

        # Memory tracking
        self.estimated_buffer_memory_mb = 0.0

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
        Uses JPEG compression to reduce memory usage by ~10x.

        Args:
            frame: Video frame (BGR numpy array or GPU tensor)
            timestamp: Frame timestamp
        """
        if frame is None:
            return

        # Convert GPU tensor to NumPy if needed (cv2.imencode needs NumPy)
        # Free GPU memory explicitly to avoid leak (Issue #98)
        if isinstance(frame, torch.Tensor):
            frame_cpu = frame.cpu().numpy()
            del frame  # Explicitly delete GPU tensor to free VRAM
            frame = frame_cpu

        with self.buffer_lock:
            if self.use_compressed_buffer:
                # Compress frame to JPEG to save memory (~10x reduction)
                # Quality 90 provides good balance of quality vs size
                ret, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret:
                    self.frame_buffer.append({
                        'frame_compressed': encoded,
                        'timestamp': timestamp
                    })

                    # Update memory estimate (periodically)
                    if len(self.frame_buffer) % MEMORY_CHECK_INTERVAL == 0:
                        self._update_memory_estimate()
                else:
                    logger.warning("Failed to compress frame for buffer, skipping")
            else:
                # Fallback: store uncompressed (uses more memory)
                self.frame_buffer.append({
                    'frame': frame.copy(),
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

    def _update_memory_estimate(self):
        """
        Estimate current buffer memory usage.
        Called periodically to track memory consumption.
        """
        if not self.frame_buffer:
            self.estimated_buffer_memory_mb = 0.0
            return

        total_bytes = 0
        for frame_data in self.frame_buffer:
            if 'frame_compressed' in frame_data:
                # Compressed frame (JPEG bytes)
                total_bytes += frame_data['frame_compressed'].nbytes
            elif 'frame' in frame_data and frame_data['frame'] is not None:
                # Uncompressed frame (numpy array)
                total_bytes += frame_data['frame'].nbytes

        self.estimated_buffer_memory_mb = total_bytes / (1024 * 1024)

        # Warn if buffer is using excessive memory
        if self.estimated_buffer_memory_mb > MAX_BUFFER_MEMORY_MB:
            logger.warning(f"Frame buffer using {self.estimated_buffer_memory_mb:.1f}MB "
                          f"(max recommended: {MAX_BUFFER_MEMORY_MB}MB). "
                          f"Consider reducing pre_buffer_seconds or fps.")

    def _decode_frame(self, frame_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Decode frame from buffer (handles both compressed and uncompressed).

        Args:
            frame_data: Frame data dictionary

        Returns:
            Decoded frame or None if decoding fails
        """
        if 'frame_compressed' in frame_data:
            # Decompress JPEG frame
            frame = cv2.imdecode(frame_data['frame_compressed'], cv2.IMREAD_COLOR)
            return frame
        elif 'frame' in frame_data:
            # Already uncompressed
            return frame_data['frame']
        else:
            return None

    def save_snapshot(
        self,
        frame: Any,
        detection_result: Dict[str, Any],
        annotated_frame: Optional[Any] = None
    ) -> Optional[str]:
        """
        Save a single snapshot image.
        Saves three versions:
        - raw/ : Unmasked frame (restricted access, for security investigation)
        - masked/ : Masked frame without annotations (for web gallery)
        - annotated/ : Masked frame with detection boxes (for web display)

        Args:
            frame: Raw video frame (numpy array or GPU tensor)
            detection_result: Detection result
            annotated_frame: Frame with annotations (if save_annotated=True)

        Returns:
            Path to masked file (for web serving), or None if save failed
        """
        # Convert GPU tensors to NumPy (cv2.imwrite needs NumPy)
        # Free GPU memory explicitly to avoid leak (Issue #98)
        if isinstance(frame, torch.Tensor):
            frame_cpu = frame.cpu().numpy()
            del frame  # Explicitly delete GPU tensor to free VRAM
            frame = frame_cpu
        if isinstance(annotated_frame, torch.Tensor):
            annotated_cpu = annotated_frame.cpu().numpy()
            del annotated_frame  # Explicitly delete GPU tensor to free VRAM
            annotated_frame = annotated_cpu

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Get primary detection class for filename
        detections = detection_result.get('detections', [])
        if not detections:
            return None

        primary_class = detections[0]['class_name']
        confidence = detections[0]['confidence']

        # Get camera identifier
        camera_id = detection_result.get('camera_id', 'default')

        # Create camera-specific directory structure
        camera_dir = self.output_dir / camera_id
        raw_dir = camera_dir / "raw"
        masked_dir = camera_dir / "masked"
        annotated_dir = camera_dir / "annotated"

        # Create directories if they don't exist
        raw_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_face_masking:
            masked_dir.mkdir(parents=True, exist_ok=True)
            annotated_dir.mkdir(parents=True, exist_ok=True)

        # Create base filename
        base_filename = f"{primary_class}_{timestamp}_conf{confidence:.2f}.jpg"

        # File paths
        raw_filepath = raw_dir / base_filename
        masked_filepath = masked_dir / base_filename if self.enable_face_masking else None

        # Determine annotated file path based on face masking setting
        if self.enable_face_masking:
            annotated_filepath = annotated_dir / base_filename
        else:
            # Legacy path for non-masked mode (backward compatibility)
            annotated_filepath = camera_dir / f"{primary_class}_{timestamp}_conf{confidence:.2f}_annotated.jpg"

        # Save images
        try:
            # 1. Always save unmasked frame in raw/ (for security investigation)
            success = cv2.imwrite(str(raw_filepath), frame)
            if not success:
                logger.error(f"Failed to save raw frame: {raw_filepath}")
                logger.error("Possible causes: disk full, permission denied, or invalid path")
                self.save_failures += 1
                return None

            # 2. Apply face masking if enabled
            masked_frame = frame
            masked_annotated_frame = annotated_frame
            faces_detected = []

            if self.enable_face_masking and self.face_masker:
                try:
                    # Detect faces once
                    faces_detected = self.face_masker.detect_faces(frame)

                    if faces_detected:
                        logger.info(f"Detected {len(faces_detected)} face(s) in snapshot, applying mask")

                        # Mask the raw frame
                        masked_frame = self.face_masker.apply_mask(frame.copy(), faces_detected)

                        # Mask the annotated frame if available
                        if annotated_frame is not None:
                            masked_annotated_frame = self.face_masker.apply_mask(annotated_frame.copy(), faces_detected)
                except Exception as e:
                    logger.error(f"Face masking failed: {e}, saving unmasked versions")
                    masked_frame = frame
                    masked_annotated_frame = annotated_frame

            # 3. Save masked frame (served by web UI)
            if self.enable_face_masking and masked_filepath:
                success = cv2.imwrite(str(masked_filepath), masked_frame)
                if not success:
                    logger.warning(f"Failed to save masked frame: {masked_filepath}")
                    # Continue - raw frame is saved

            # 4. Save annotated frame (masked if face masking enabled)
            if self.save_annotated and masked_annotated_frame is not None:
                success = cv2.imwrite(str(annotated_filepath), masked_annotated_frame)
                if not success:
                    logger.warning(f"Failed to save annotated frame: {annotated_filepath}")
                    # Continue - raw and masked frames are more important
            elif self.save_annotated and annotated_frame is not None and not self.enable_face_masking:
                # Fallback for no face masking - save annotated in old location
                success = cv2.imwrite(str(annotated_filepath), annotated_frame)
                if not success:
                    logger.warning(f"Failed to save annotated frame: {annotated_filepath}")

            # 5. Save metadata JSON (in masked/ or camera root)
            metadata_dir = masked_dir if self.enable_face_masking else camera_dir
            # Build annotated filename path based on directory structure
            if self.save_annotated:
                if self.enable_face_masking:
                    annotated_filename = f"annotated/{primary_class}_{timestamp}_conf{confidence:.2f}.jpg"
                else:
                    annotated_filename = f"{primary_class}_{timestamp}_conf{confidence:.2f}_annotated.jpg"
            else:
                annotated_filename = None

            metadata = {
                'timestamp': detection_result.get('timestamp'),
                'camera_id': detection_result.get('camera_id', 'default'),
                'camera_name': detection_result.get('camera_name', 'Default Camera'),
                'detections': detections,
                'detection_counts': detection_result.get('detection_counts', {}),
                'latency_ms': detection_result.get('total_latency_ms', 0),
                'filename': base_filename,
                'annotated_filename': annotated_filename,
                'face_masking_enabled': self.enable_face_masking,
                'faces_detected': len(faces_detected) if faces_detected else 0
            }

            metadata_file = metadata_dir / f"{primary_class}_{timestamp}_conf{confidence:.2f}.json"
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save metadata {metadata_file}: {e}")
                # Continue - snapshot is more important than metadata

            # Update statistics
            self.total_saved += 1
            self.saves_by_class[primary_class] = self.saves_by_class.get(primary_class, 0) + 1

            mask_info = f" (masked, {len(faces_detected)} face(s))" if self.enable_face_masking and faces_detected else ""
            logger.info(f"📸 Saved snapshot: {camera_id}/{base_filename}{mask_info}")

            # Return masked path for web serving (or raw path if face masking disabled)
            return str(masked_filepath) if self.enable_face_masking and masked_filepath else str(raw_filepath)

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
            current_frame: Current video frame (numpy array or GPU tensor)
            detection_result: Detection result
            annotated_frame: Current frame with annotations

        Returns:
            Path to saved file, or None if save failed
        """
        # Convert GPU tensors to NumPy (cv2.VideoWriter needs NumPy)
        # Free GPU memory explicitly to avoid leak (Issue #98)
        if isinstance(current_frame, torch.Tensor):
            current_cpu = current_frame.cpu().numpy()
            del current_frame  # Explicitly delete GPU tensor to free VRAM
            current_frame = current_cpu
        if isinstance(annotated_frame, torch.Tensor):
            annotated_cpu = annotated_frame.cpu().numpy()
            del annotated_frame  # Explicitly delete GPU tensor to free VRAM
            annotated_frame = annotated_cpu

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

            # Decode first frame to get dimensions
            sample_frame = self._decode_frame(buffered_frames[0])
            if sample_frame is None:
                logger.error("Failed to decode sample frame for clip")
                return None

            height, width = sample_frame.shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))

            # Check if VideoWriter opened successfully
            if not out.isOpened():
                logger.error(f"Failed to open VideoWriter for {filepath}")
                logger.error("Possible causes: missing codec, invalid path, or disk full")
                self.save_failures += 1
                return None

            # Write pre-detection frames (decode as needed)
            frames_written = 0
            for frame_data in buffered_frames:
                decoded_frame = self._decode_frame(frame_data)
                if decoded_frame is not None:
                    out.write(decoded_frame)
                    frames_written += 1
                else:
                    logger.warning("Failed to decode frame in buffer, skipping")

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

            logger.info(f"🎬 Saved clip: {filename} ({frames_written}/{len(buffered_frames)} frames)")
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

    def update_settings(
        self,
        cooldown_seconds: Optional[int] = None,
        trigger_classes: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        save_annotated: Optional[bool] = None
    ):
        """
        Update snapshot settings without restarting (hot-reload).

        Args:
            cooldown_seconds: New cooldown duration
            trigger_classes: New list of trigger classes
            min_confidence: New minimum confidence threshold
            save_annotated: Whether to save annotated frames
        """
        updated_settings = []

        if cooldown_seconds is not None and cooldown_seconds != self.cooldown_seconds:
            self.cooldown_seconds = cooldown_seconds
            updated_settings.append(f"cooldown_seconds: {cooldown_seconds}")

        if trigger_classes is not None:
            new_trigger_set = set(trigger_classes) if trigger_classes else None
            if new_trigger_set != self.trigger_classes:
                self.trigger_classes = new_trigger_set
                updated_settings.append(f"trigger_classes: {trigger_classes}")

        if min_confidence is not None and min_confidence != self.min_confidence:
            self.min_confidence = min_confidence
            updated_settings.append(f"min_confidence: {min_confidence}")

        if save_annotated is not None and save_annotated != self.save_annotated:
            self.save_annotated = save_annotated
            updated_settings.append(f"save_annotated: {save_annotated}")

        if updated_settings:
            logger.info(f"SnapshotSaver settings updated: {', '.join(updated_settings)}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get snapshot saver statistics.

        Returns:
            Dictionary with statistics
        """
        # Update memory estimate before returning stats
        with self.buffer_lock:
            self._update_memory_estimate()

        return {
            'total_saved': self.total_saved,
            'save_failures': self.save_failures,
            'saves_by_class': self.saves_by_class,
            'output_dir': str(self.output_dir),
            'save_mode': self.save_mode,
            'buffer_size': len(self.frame_buffer),
            'buffer_memory_mb': self.estimated_buffer_memory_mb,
            'compressed_buffer': self.use_compressed_buffer
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
