"""
Detection Post-Processor
Processes raw detections and prepares them for web display.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from queue import Queue
from threading import Thread, Event
from collections import deque
from visualization_utils import draw_detections

logging.basicConfig(level=logging.INFO)
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
        frame_source: Optional[Any] = None
    ):
        """
        Initialize detection processor.

        Args:
            input_queue: Queue to receive detections from
            output_queue: Queue to send processed detections to
            detection_history_size: Number of frames to keep in history
            snapshot_saver: SnapshotSaver instance (optional)
            frame_source: Source to get latest frames (for snapshot saving)
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detection_history_size = detection_history_size
        self.snapshot_saver = snapshot_saver
        self.frame_source = frame_source

        self.detection_history = deque(maxlen=detection_history_size)
        self.stop_event = Event()
        self.processor_thread: Optional[Thread] = None

        # Statistics
        self.processed_count = 0
        self.last_detection_time = None

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
        """Stop the processor thread."""
        logger.info("Stopping detection processor thread...")
        self.stop_event.set()

        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)

        logger.info("Detection processor thread stopped")

    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        logger.info("Detection processing loop started")

        while not self.stop_event.is_set():
            try:
                # Get detections from input queue
                if self.input_queue.empty():
                    time.sleep(0.001)
                    continue

                detection_result = self.input_queue.get()

                # Process detections
                processed_result = self._process_detections(detection_result)

                # Add to history
                self.detection_history.append(processed_result)

                # Save snapshot if enabled and triggered
                if self.snapshot_saver and self.frame_source:
                    if hasattr(self.frame_source, 'latest_frame') and self.frame_source.latest_frame is not None:
                        # Add current frame to buffer for clip mode
                        self.snapshot_saver.add_frame_to_buffer(
                            self.frame_source.latest_frame,
                            processed_result['timestamp']
                        )

                        # Check if snapshot should be saved
                        if processed_result['detections']:
                            # Create annotated frame with bounding boxes
                            annotated_frame = draw_detections(
                                self.frame_source.latest_frame,
                                processed_result['detections']
                            )

                            saved_path = self.snapshot_saver.process_detection(
                                self.frame_source.latest_frame,
                                processed_result,
                                annotated_frame=annotated_frame
                            )
                            if saved_path:
                                processed_result['snapshot_saved'] = saved_path

                # Send to output queue
                try:
                    self.output_queue.put_nowait(processed_result)
                except:
                    pass  # Drop if queue is full

                self.processed_count += 1
                if processed_result['detections']:
                    self.last_detection_time = time.time()

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def _process_detections(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw detection results.

        Args:
            detection_result: Raw detection result from inference engine

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
            'total_detections': len(detections)
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
        return {
            'processed_count': self.processed_count,
            'history_size': len(self.detection_history),
            'last_detection_time': self.last_detection_time
        }

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
