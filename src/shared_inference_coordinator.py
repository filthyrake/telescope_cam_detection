"""
SharedInferenceCoordinator - Batched GPU Inference Coordinator

Coordinates batched inference across multiple camera threads for improved GPU utilization.
Collects frames from multiple cameras and processes them in a single GPU forward pass.
"""

import threading
import time
import logging
from typing import List, Callable, Optional, Any, Dict
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PendingInference:
    """Represents a pending inference request"""
    frame: Any  # np.ndarray or torch.Tensor
    callback: Callable[[List[Dict[str, Any]]], None]
    enqueue_time: float
    camera_id: Optional[str] = None


class SharedInferenceCoordinator:
    """
    Coordinates batched inference across multiple camera threads.

    Collects inference requests from multiple cameras and processes them
    in batches for improved GPU utilization (30-40% → 80-90%).

    Performance:
    - Single frame inference: ~11-21ms per camera
    - Batched inference (4 frames): ~15-30ms total (3-4x throughput!)
    """

    def __init__(
        self,
        detector: Any,  # YOLOXDetector instance
        max_batch_size: int = 4,
        max_batch_wait_ms: float = 10.0,
        enable_metrics: bool = True
    ):
        """
        Initialize batched inference coordinator.

        Args:
            detector: YOLOXDetector instance with detect_batch() method
            max_batch_size: Maximum frames to batch together (default: 4)
            max_batch_wait_ms: Maximum time to wait for full batch in milliseconds (default: 10ms)
            enable_metrics: Track and log performance metrics
        """
        self.detector = detector
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = max_batch_wait_ms / 1000.0  # Convert to seconds
        self.enable_metrics = enable_metrics

        # Thread-safe queue for pending inference requests
        self.pending_queue: deque[PendingInference] = deque()
        self.queue_lock = threading.Lock()
        self.queue_condition = threading.Condition(self.queue_lock)

        # Coordinator thread
        self.coordinator_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.running = False

        # Performance metrics
        self.total_batches = 0
        self.total_frames = 0
        self.total_batch_time_ms = 0.0
        self.batch_sizes: List[int] = []
        self.wait_times_ms: List[float] = []

        logger.info(f"SharedInferenceCoordinator initialized (max_batch_size={max_batch_size}, "
                   f"max_wait={max_batch_wait_ms:.1f}ms)")

    def start(self):
        """Start the coordinator thread"""
        if self.running:
            logger.warning("Coordinator already running")
            return

        self.running = True
        self.stop_event.clear()
        self.coordinator_thread = threading.Thread(
            target=self._coordinator_loop,
            name="InferenceCoordinator",
            daemon=True
        )
        self.coordinator_thread.start()
        logger.info("Coordinator thread started")

    def stop(self):
        """Stop the coordinator thread"""
        if not self.running:
            return

        logger.info("Stopping coordinator thread...")
        self.running = False
        self.stop_event.set()

        # Wake up coordinator thread
        with self.queue_condition:
            self.queue_condition.notify()

        # Wait for thread to finish
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=2.0)
            if self.coordinator_thread.is_alive():
                logger.warning("Coordinator thread did not stop cleanly")

        logger.info("Coordinator thread stopped")

    def infer_async(
        self,
        frame: Any,
        callback: Callable[[List[Dict[str, Any]]], None],
        camera_id: Optional[str] = None
    ):
        """
        Queue a frame for batched inference.

        Args:
            frame: Input frame (BGR) - NumPy array or GPU tensor
            callback: Callback function to receive detection results
            camera_id: Optional camera identifier for debugging
        """
        if not self.running:
            raise RuntimeError("Coordinator not running - call start() first")

        # Create pending inference request
        request = PendingInference(
            frame=frame,
            callback=callback,
            enqueue_time=time.time(),
            camera_id=camera_id
        )

        # Add to queue and notify coordinator
        with self.queue_condition:
            self.pending_queue.append(request)
            # Always notify coordinator on new frame (reduces latency for first frame)
            self.queue_condition.notify()

    def _coordinator_loop(self):
        """Main coordinator loop - collects and processes batches"""
        logger.info("Coordinator loop started")

        while not self.stop_event.is_set():
            try:
                # Collect a batch of frames
                batch = self._collect_batch()

                if not batch:
                    continue

                # Process the batch
                self._process_batch(batch)

            except Exception as e:
                logger.error(f"Error in coordinator loop: {e}", exc_info=True)

    def _collect_batch(self) -> List[PendingInference]:
        """
        Collect a batch of pending inference requests.

        Returns:
            List of pending inference requests (up to max_batch_size)
        """
        batch = []

        with self.queue_condition:
            # Wait for frames or timeout
            while len(self.pending_queue) == 0 and not self.stop_event.is_set():
                # Wait with timeout to check stop event periodically
                self.queue_condition.wait(timeout=0.1)

            if self.stop_event.is_set():
                return []

            # Collect frames up to max_batch_size
            batch_start_time = time.time()

            while len(batch) < self.max_batch_size and len(self.pending_queue) > 0:
                batch.append(self.pending_queue.popleft())

                # If we have at least one frame, wait a bit for more (up to max_wait_ms)
                if len(batch) < self.max_batch_size and len(self.pending_queue) == 0:
                    elapsed = (time.time() - batch_start_time)
                    remaining_wait = self.max_batch_wait_ms - elapsed

                    if remaining_wait > 0:
                        # Wait for more frames
                        self.queue_condition.wait(timeout=remaining_wait)
                    else:
                        # Timeout - process what we have
                        break

        return batch

    def _process_batch(self, batch: List[PendingInference]):
        """
        Process a batch of inference requests.

        Args:
            batch: List of pending inference requests
        """
        if not batch:
            return

        batch_start_time = time.time()

        # Extract frames from batch
        frames = [req.frame for req in batch]

        # Calculate wait times (time between enqueue and batch processing)
        if self.enable_metrics:
            for req in batch:
                wait_time_ms = (batch_start_time - req.enqueue_time) * 1000
                self.wait_times_ms.append(wait_time_ms)

        try:
            # Run batched inference
            inference_start = time.time()
            all_detections = self.detector.detect_batch(frames)
            inference_time_ms = (time.time() - inference_start) * 1000

            # Dispatch results to callbacks
            for req, detections in zip(batch, all_detections):
                try:
                    req.callback(detections)
                except Exception as e:
                    logger.error(f"Error in callback for camera {req.camera_id}: {e}")

            # Update metrics
            if self.enable_metrics:
                self.total_batches += 1
                self.total_frames += len(batch)
                self.total_batch_time_ms += inference_time_ms
                self.batch_sizes.append(len(batch))

                # Log metrics periodically
                if self.total_batches % 100 == 0:
                    self._log_metrics()

            logger.debug(f"Processed batch of {len(batch)} frames in {inference_time_ms:.1f}ms")

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)

            # Call callbacks with empty results on error
            for req in batch:
                try:
                    req.callback([])
                except Exception as cb_error:
                    logger.error(f"Error calling callback on error: {cb_error}")

    def _log_metrics(self):
        """Log performance metrics"""
        if not self.enable_metrics or self.total_batches == 0:
            return

        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes)
        avg_batch_time = self.total_batch_time_ms / self.total_batches
        avg_wait_time = sum(self.wait_times_ms) / len(self.wait_times_ms) if self.wait_times_ms else 0

        logger.info(f"Coordinator metrics (after {self.total_batches} batches):")
        logger.info(f"  Avg batch size: {avg_batch_size:.1f} frames")
        logger.info(f"  Avg batch time: {avg_batch_time:.1f}ms")
        logger.info(f"  Avg wait time: {avg_wait_time:.1f}ms")
        logger.info(f"  Total frames processed: {self.total_frames}")
        logger.info(f"  Throughput: {self.total_frames / (self.total_batch_time_ms / 1000):.1f} fps")

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        if not self.enable_metrics or self.total_batches == 0:
            return {
                'enabled': False,
                'total_batches': 0,
                'total_frames': 0
            }

        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes)
        avg_batch_time = self.total_batch_time_ms / self.total_batches
        avg_wait_time = sum(self.wait_times_ms) / len(self.wait_times_ms) if self.wait_times_ms else 0
        throughput = self.total_frames / (self.total_batch_time_ms / 1000) if self.total_batch_time_ms > 0 else 0

        return {
            'enabled': True,
            'total_batches': self.total_batches,
            'total_frames': self.total_frames,
            'avg_batch_size': round(avg_batch_size, 2),
            'avg_batch_time_ms': round(avg_batch_time, 2),
            'avg_wait_time_ms': round(avg_wait_time, 2),
            'throughput_fps': round(throughput, 1),
            'queue_depth': len(self.pending_queue)
        }

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
