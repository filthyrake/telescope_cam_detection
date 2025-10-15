"""
Detection Post-Processor
Processes raw detections and prepares them for web display.
"""

import time
import logging
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
from threading import Thread, Event
from collections import deque
from visualization_utils import draw_detections
from src.motion_filter import MotionFilter
from src.time_of_day_filter import TimeOfDayFilter
from src.constants import (
    QUEUE_GET_TIMEOUT_SECONDS,
    LOG_DROPPED_EVERY_N,
    ERROR_SLEEP_SECONDS,
    THREAD_JOIN_TIMEOUT_SECONDS,
    MIN_TIME_DELTA
)

logger = logging.getLogger(__name__)


class DetectionProcessor:
    """
    Processes detection results and manages detection state.
    Can be extended for collision detection and zone-based alerts.
    """

    def __init__(
        self,
        input_queue: Optional[Queue] = None,
        output_queue: Optional[Queue] = None,
        detection_history_size: int = 30,
        snapshot_saver: Optional[Any] = None,
        frame_source: Optional[Any] = None,
        enable_motion_filter: bool = False,
        motion_filter_config: Optional[Dict[str, Any]] = None,
        enable_time_of_day_filter: bool = False,
        time_of_day_filter_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize detection processor.

        Args:
            input_queue: Queue to receive detections from
            output_queue: Queue to send processed detections to
            detection_history_size: Number of frames to keep in history
            snapshot_saver: SnapshotSaver instance (optional)
            frame_source: Source to get latest frames (for snapshot saving)
            enable_motion_filter: Enable motion-based filtering
            motion_filter_config: Configuration dict for motion filter
            enable_time_of_day_filter: Enable time-of-day filtering
            time_of_day_filter_config: Configuration dict for time-of-day filter
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detection_history_size = detection_history_size
        self.snapshot_saver = snapshot_saver
        self.frame_source = frame_source

        self.detection_history = deque(maxlen=detection_history_size)
        self.stop_event = Event()
        self.processor_thread: Optional[Thread] = None

        # Motion filter
        self.enable_motion_filter = enable_motion_filter
        self.motion_filter = None
        if enable_motion_filter:
            config = motion_filter_config or {}
            self.motion_filter = MotionFilter(**config)
            logger.info("Motion filter enabled")

        # Time-of-day filter
        self.enable_time_of_day_filter = enable_time_of_day_filter
        self.time_of_day_filter = None
        if enable_time_of_day_filter:
            config = time_of_day_filter_config or {}
            self.time_of_day_filter = TimeOfDayFilter(**config)
            logger.info("Time-of-day filter enabled")

        # Statistics
        self.processed_count = 0
        self.last_detection_time = None
        self.dropped_results = 0  # Track dropped results when queue is full
        self.last_drop_warning_time = 0  # Track last drop warning for rate limiting
        self.drop_count_since_warning = 0  # Track drops since last warning

    def start(self) -> bool:
        """
        Start the processor thread.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.input_queue or not self.output_queue:
            logger.error("Input and output queues must be provided")
            return False

        self.stop_event.clear()
        self.processor_thread = Thread(target=self._processing_loop, daemon=True)
        self.processor_thread.start()
        logger.info("Detection processor thread started")
        return True

    def stop(self):
        """Stop the processor thread and clean up resources."""
        logger.info("Stopping detection processor thread...")
        self.stop_event.set()

        if self.processor_thread:
            self.processor_thread.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)

            # Check if thread actually stopped
            if self.processor_thread.is_alive():
                logger.error(
                    f"CRITICAL: Detection processor thread did not stop after {THREAD_JOIN_TIMEOUT_SECONDS}s timeout (thread may be blocked). "
                    f"Thread is orphaned and will continue running - potential resource leak (Issue #96). "
                    f"This may indicate a deadlock or blocking queue operation."
                )
            else:
                logger.info("Detection processor thread stopped successfully")

        # Clean up motion filter resources
        if self.motion_filter is not None:
            self.motion_filter.cleanup()
            logger.debug("Motion filter cleaned up")

    def _get_frame_copy(self) -> Optional[Any]:
        """
        Get a thread-safe copy of the latest frame from frame source.
        Handles both NumPy arrays and GPU tensors.

        Returns:
            Copy of latest frame, or None if unavailable
        """
        if not self.frame_source or not hasattr(self.frame_source, 'latest_frame'):
            return None

        with self.frame_source.frame_lock:
            if self.frame_source.latest_frame is None:
                return None

            # Handle GPU tensors vs NumPy arrays
            if isinstance(self.frame_source.latest_frame, torch.Tensor):
                return self.frame_source.latest_frame.clone()
            else:
                return self.frame_source.latest_frame.copy()

    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        logger.info("Detection processing loop started")

        while not self.stop_event.is_set():
            try:
                # Get detections from input queue (blocking with timeout)
                try:
                    detection_result = self.input_queue.get(timeout=QUEUE_GET_TIMEOUT_SECONDS)
                except Empty:
                    continue

                # Get current frame for motion filtering (thread-safe)
                current_frame = self._get_frame_copy()

                # Process detections (includes motion filtering)
                processed_result = self._process_detections(detection_result, current_frame)

                # Add to history
                self.detection_history.append(processed_result)

                # Save snapshot if enabled and triggered (thread-safe frame access)
                if self.snapshot_saver and self.frame_source:
                    frame_copy = self._get_frame_copy()
                    if frame_copy is not None:
                        # Add current frame to buffer for clip mode
                        self.snapshot_saver.add_frame_to_buffer(
                            frame_copy,
                            processed_result['timestamp']
                        )

                        # Check if snapshot should be saved
                        if processed_result['detections']:
                            # Create annotated frame with bounding boxes
                            annotated_frame = draw_detections(
                                frame_copy,
                                processed_result['detections']
                            )

                            saved_path = self.snapshot_saver.process_detection(
                                frame_copy,
                                processed_result,
                                annotated_frame=annotated_frame
                            )
                            if saved_path:
                                processed_result['snapshot_saved'] = saved_path

                # Send to output queue
                try:
                    self.output_queue.put_nowait(processed_result)
                except Exception as e:
                    self.dropped_results += 1
                    self.drop_count_since_warning += 1

                    # Log with drop rate when drops are frequent (improved observability)
                    current_time = time.time()
                    time_since_last_warning = current_time - self.last_drop_warning_time

                    # Log every Nth drop OR every 10 seconds (whichever comes first)
                    should_log = (self.dropped_results % LOG_DROPPED_EVERY_N == 0) or (time_since_last_warning >= 10.0)

                    if should_log:
                        drop_rate = self.drop_count_since_warning / max(time_since_last_warning, MIN_TIME_DELTA)
                        total_drop_rate = self.dropped_results / max(self.processed_count, 1)
                        logger.warning(
                            f"Output queue full: dropped {self.dropped_results} total results "
                            f"(drop rate: {drop_rate:.2f}/s, {total_drop_rate*100:.1f}% overall) - system overloaded"
                        )
                        self.last_drop_warning_time = current_time
                        self.drop_count_since_warning = 0

                self.processed_count += 1
                if processed_result['detections']:
                    self.last_detection_time = time.time()

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(ERROR_SLEEP_SECONDS)

    def _process_detections(
        self,
        detection_result: Dict[str, Any],
        frame: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Process raw detection results.

        Args:
            detection_result: Raw detection result from inference engine
            frame: Current frame (for motion filtering)

        Returns:
            Processed detection result
        """
        detections = detection_result['detections']
        frame_id = detection_result['frame_id']
        timestamp = detection_result['timestamp']
        inference_time = detection_result['inference_time']

        # Extract camera metadata if present
        camera_id = detection_result.get('camera_id', 'default')
        camera_name = detection_result.get('camera_name', 'Default Camera')

        # Apply motion filter if enabled
        detections_before_motion = len(detections)
        if self.enable_motion_filter and self.motion_filter and frame is not None:
            detections = self.motion_filter.filter_detections(frame, detections)

        # Apply time-of-day filter if enabled
        detections_before_time_filter = len(detections)
        if self.enable_time_of_day_filter and self.time_of_day_filter:
            detections = self.time_of_day_filter.filter_detections(detections, datetime.fromtimestamp(timestamp))
        detections_after_time_filter = len(detections)

        # Calculate total latency (from frame capture to now)
        current_time = time.time()
        total_latency = current_time - timestamp

        # Group detections by class
        detections_by_class = self._group_by_class(detections)

        # Calculate detection counts
        detection_counts = {
            class_name: len(dets) for class_name, dets in detections_by_class.items()
        }

        # Prepare result
        processed_result = {
            'frame_id': frame_id,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'timestamp': timestamp,
            'processing_timestamp': current_time,
            'inference_time_ms': inference_time * 1000,
            'total_latency_ms': total_latency * 1000,
            'detections': detections,
            'detections_by_class': detections_by_class,
            'detection_counts': detection_counts,
            'total_detections': len(detections),
            'motion_filtered': detections_before_motion - len(detections) if self.enable_motion_filter else 0,
            'time_filtered': detections_before_time_filter - detections_after_time_filter if self.enable_time_of_day_filter else 0
        }

        return processed_result

    def _group_by_class(self, detections: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group detections by class name.

        Args:
            detections: List of detection dictionaries

        Returns:
            Dictionary mapping class names to lists of detections
        """
        grouped = {}
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in grouped:
                grouped[class_name] = []
            grouped[class_name].append(detection)

        return grouped

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary containing processor stats
        """
        stats = {
            'processed_count': self.processed_count,
            'history_size': len(self.detection_history),
            'last_detection_time': self.last_detection_time,
            'motion_filter_enabled': self.enable_motion_filter,
            'time_of_day_filter_enabled': self.enable_time_of_day_filter,
            'dropped_results': self.dropped_results,
        }

        # Add drop rate
        if self.processed_count > 0:
            stats['drop_rate'] = self.dropped_results / self.processed_count
        else:
            stats['drop_rate'] = 0.0

        # Add motion filter stats if enabled
        if self.enable_motion_filter and self.motion_filter:
            stats['motion_filter_stats'] = self.motion_filter.get_stats()

        # Add time-of-day filter stats if enabled
        if self.enable_time_of_day_filter and self.time_of_day_filter:
            stats['time_of_day_filter_stats'] = self.time_of_day_filter.get_stats()

        # Add memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            stats['memory_mb'] = mem_info.rss / 1024 / 1024
            stats['memory_percent'] = process.memory_percent()
        except ImportError:
            pass  # psutil not available

        return stats

    def get_recent_detections(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent detection results.

        Args:
            n: Number of recent results to return

        Returns:
            List of recent detection results
        """
        return list(self.detection_history)[-n:]


class CollisionDetector:
    """
    Collision detection for telescope equipment.
    This is a placeholder for Phase 3 custom detection.
    """

    def __init__(self, danger_threshold: float = 50.0):
        """
        Initialize collision detector.

        Args:
            danger_threshold: Minimum distance in pixels for collision warning
        """
        self.danger_threshold = danger_threshold
        self.danger_zones = []

    def add_danger_zone(self, zone: Dict[str, Any]):
        """
        Add a danger zone for collision detection.

        Args:
            zone: Dictionary defining the danger zone
        """
        self.danger_zones.append(zone)

    def check_collision_risk(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for collision risk between detected objects.

        Args:
            detections: List of detection dictionaries

        Returns:
            Dictionary containing collision risk information
        """
        # Placeholder implementation
        # Will be expanded in Phase 3 with custom telescope detection

        collision_risk = {
            'has_risk': False,
            'risk_level': 0.0,
            'risk_objects': []
        }

        # TODO: Implement actual collision detection logic
        # - Calculate distances between telescope parts
        # - Track velocity vectors
        # - Predict intersection points

        return collision_risk


if __name__ == "__main__":
    # Test the detection processor
    from queue import Queue

    logger.info("Testing Detection Processor")

    # Create queues
    input_queue = Queue(maxsize=10)
    output_queue = Queue(maxsize=10)

    # Initialize processor
    processor = DetectionProcessor(
        input_queue=input_queue,
        output_queue=output_queue
    )

    # Start processor
    if not processor.start():
        logger.error("Failed to start processor")
        exit(1)

    # Test with dummy detection data
    test_detection = {
        'frame_id': 1,
        'timestamp': time.time(),
        'inference_time': 0.015,
        'detections': [
            {
                'class_id': 0,
                'class_name': 'person',
                'confidence': 0.95,
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 300}
            },
            {
                'class_id': 16,
                'class_name': 'cat',
                'confidence': 0.87,
                'bbox': {'x1': 300, 'y1': 200, 'x2': 400, 'y2': 350}
            }
        ],
        'frame_shape': (720, 1280, 3)
    }

    input_queue.put(test_detection)

    # Wait for processing
    time.sleep(0.5)

    # Get result
    if not output_queue.empty():
        result = output_queue.get()
        logger.info(f"Processed result: {result['total_detections']} detections")
        logger.info(f"Latency: {result['total_latency_ms']:.1f}ms")
        logger.info(f"Detection counts: {result['detection_counts']}")
    else:
        logger.error("No result in output queue")

    # Stop processor
    processor.stop()

    logger.info("Detection processor test completed successfully")
