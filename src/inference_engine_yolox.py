"""
GPU Inference Engine - YOLOX Version
Fast Stage 1 detection with YOLOX (Apache 2.0 License)
47x faster than GroundingDINO (11-21ms vs 560ms)
"""

import torch
import time
import logging
from typing import Optional, List, Dict, Any
from queue import Queue, Empty
from threading import Thread, Event
import numpy as np

from src.yolox_detector import YOLOXDetector
from src.constants import (
    QUEUE_GET_TIMEOUT_SECONDS,
    LOG_DROPPED_EVERY_N,
    ERROR_SLEEP_SECONDS,
    THREAD_JOIN_TIMEOUT_SECONDS,
    FPS_CALCULATION_INTERVAL_SECONDS
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    GPU-accelerated inference engine using YOLOX.
    Drop-in replacement for GroundingDINO-based inference engine.
    """

    def __init__(
        self,
        model_name: str = "yolox-s",
        model_path: str = "models/yolox/yolox_s.pth",
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: tuple = (640, 640),
        input_queue: Optional[Queue] = None,
        output_queue: Optional[Queue] = None,
        min_box_area: int = 0,
        max_det: int = 300,
        use_two_stage: bool = False,
        two_stage_pipeline: Optional[Any] = None,
        class_confidence_overrides: Optional[Dict[str, float]] = None,
        class_size_constraints: Optional[Dict[str, Dict[str, int]]] = None,
        wildlife_only: bool = True,
        **kwargs  # Ignore extra params from old config
    ):
        """
        Initialize YOLOX inference engine.

        Args:
            model_name: YOLOX model variant (yolox-s, yolox-m, etc.)
            model_path: Path to YOLOX weights
            device: Device to run inference on
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            input_size: Input image size (height, width) - larger = better for small objects
            input_queue: Queue to receive frames from
            output_queue: Queue to send detections to
            min_box_area: Minimum bounding box area (pixels²)
            max_det: Maximum detections per image
            use_two_stage: Enable two-stage detection (YOLOX → iNaturalist)
            two_stage_pipeline: TwoStageDetectionPipeline instance
            class_confidence_overrides: Per-class confidence thresholds
            class_size_constraints: Per-class min/max box area constraints (e.g., {'bird': {'max': 15000}})
            wildlife_only: Filter to wildlife-relevant classes only
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.min_box_area = min_box_area
        self.max_det = max_det
        self.use_two_stage = use_two_stage
        self.two_stage_pipeline = two_stage_pipeline
        self.class_confidence_overrides = class_confidence_overrides or {}
        self.class_size_constraints = class_size_constraints or {}
        self.wildlife_only = wildlife_only

        self.detector: Optional[YOLOXDetector] = None
        self.is_loaded = False
        self.stop_event = Event()
        self.inference_thread: Optional[Thread] = None

        # Performance metrics
        self.inference_count = 0  # Reset every second for FPS calculation
        self.total_inference_count = 0  # Never reset - cumulative total
        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0
        self.fps = 0.0
        self.last_fps_check = time.time()
        self.dropped_results = 0  # Track dropped results when queue is full
        self.last_drop_warning_time = 0  # Track last drop warning for rate limiting
        self.drop_count_since_warning = 0  # Track drops since last warning

    def load_model(self) -> bool:
        """Load YOLOX model"""
        try:
            logger.info(f"Loading YOLOX inference engine...")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Weights: {self.model_path}")
            logger.info(f"Device: {self.device}")

            # Create detector
            self.detector = YOLOXDetector(
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                input_size=self.input_size,
                wildlife_only=self.wildlife_only,
            )

            # Load model
            if not self.detector.load_model():
                logger.error("Failed to load YOLOX model")
                return False

            # Log GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            logger.info(f"Two-stage detection: {'enabled' if self.use_two_stage else 'disabled'}")

            self.is_loaded = True
            logger.info("YOLOX inference engine loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load inference engine: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def start(self) -> bool:
        """Start the inference thread"""
        if not self.is_loaded:
            if not self.load_model():
                return False

        if not self.input_queue or not self.output_queue:
            logger.error("Input and output queues must be provided")
            return False

        self.stop_event.clear()
        self.inference_thread = Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        logger.info("Inference thread started")
        return True

    def stop(self):
        """Stop the inference thread"""
        logger.info("Stopping inference thread...")
        self.stop_event.set()

        if self.inference_thread:
            self.inference_thread.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)

            # Check if thread actually stopped
            if self.inference_thread.is_alive():
                logger.error(
                    f"CRITICAL: Inference thread did not stop after {THREAD_JOIN_TIMEOUT_SECONDS}s timeout (thread may be blocked). "
                    f"Thread is orphaned and will continue running - potential resource leak (Issue #96). "
                    f"This may indicate a deadlock or blocking operation in the inference loop."
                )
            else:
                logger.info("Inference thread stopped successfully")

    def _inference_loop(self):
        """Main inference loop running in separate thread"""
        logger.info("Inference loop started")

        while not self.stop_event.is_set():
            try:
                # Get frame from input queue (blocking with timeout)
                try:
                    frame_data = self.input_queue.get(timeout=QUEUE_GET_TIMEOUT_SECONDS)
                except Empty:
                    continue

                frame = frame_data['frame']
                frame_timestamp = frame_data['timestamp']
                frame_id = frame_data['frame_id']
                camera_id = frame_data.get('camera_id', 'default')
                camera_name = frame_data.get('camera_name', 'Default Camera')

                # Run inference
                inference_start = time.time()
                detections = self._run_inference(frame)
                inference_time = time.time() - inference_start

                # Update metrics
                self.inference_count += 1
                self.total_inference_count += 1  # Cumulative total (never reset)
                self.total_inference_time += inference_time
                self.avg_inference_time = self.total_inference_time / self.total_inference_count

                # Calculate FPS
                if time.time() - self.last_fps_check >= FPS_CALCULATION_INTERVAL_SECONDS:
                    self.fps = self.inference_count / (time.time() - self.last_fps_check)
                    self.last_fps_check = time.time()
                    self.inference_count = 0
                    logger.debug(f"Inference FPS: {self.fps:.1f}, Avg time: {self.avg_inference_time*1000:.1f}ms")

                # Prepare result
                result = {
                    'frame_id': frame_id,
                    'timestamp': frame_timestamp,
                    'inference_time': inference_time,
                    'detections': detections,
                    'frame_shape': frame.shape,
                    'camera_id': camera_id,
                    'camera_name': camera_name
                }

                # Send to output queue
                try:
                    self.output_queue.put_nowait(result)
                except Exception as e:
                    self.dropped_results += 1
                    self.drop_count_since_warning += 1

                    # Log with drop rate when drops are frequent (improved observability)
                    current_time = time.time()
                    time_since_last_warning = current_time - self.last_drop_warning_time

                    # Log every Nth drop OR every 10 seconds (whichever comes first)
                    should_log = (self.dropped_results % LOG_DROPPED_EVERY_N == 0) or (time_since_last_warning >= 10.0)

                    if should_log:
                        drop_rate = self.drop_count_since_warning / max(time_since_last_warning, 0.001)
                        total_drop_rate = self.dropped_results / max(self.total_inference_count, 1)
                        logger.warning(
                            f"Output queue full: dropped {self.dropped_results} total results "
                            f"(drop rate: {drop_rate:.2f}/s, {total_drop_rate*100:.1f}% overall) - system overloaded"
                        )
                        self.last_drop_warning_time = current_time
                        self.drop_count_since_warning = 0

            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(ERROR_SLEEP_SECONDS)

    def _run_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detection dictionaries
        """
        # Stage 1: Run YOLOX detection
        detections = self.detector.detect(frame)

        # Filter by confidence overrides and size constraints
        filtered_detections = []
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            box_area = det['bbox']['area']

            # Apply per-class confidence threshold
            class_threshold = self.class_confidence_overrides.get(class_name, self.conf_threshold)
            if confidence < class_threshold:
                continue

            # Filter by minimum box area (global)
            if self.min_box_area > 0 and box_area < self.min_box_area:
                continue

            # Filter by per-class size constraints
            if class_name in self.class_size_constraints:
                constraints = self.class_size_constraints[class_name]
                if 'min' in constraints and box_area < constraints['min']:
                    continue
                if 'max' in constraints and box_area > constraints['max']:
                    continue

            filtered_detections.append(det)

            # Stop if we hit max detections
            if len(filtered_detections) >= self.max_det:
                break

        # Stage 2: Run species classification if enabled
        if self.use_two_stage and self.two_stage_pipeline:
            filtered_detections = self.two_stage_pipeline.process_detections(frame, filtered_detections)

        return filtered_detections

    def infer_single(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single frame (synchronous).

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detection dictionaries
        """
        if not self.is_loaded:
            logger.error("Model not loaded")
            return []

        return self._run_inference(frame)

    def update_settings(
        self,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        min_box_area: Optional[int] = None,
        max_det: Optional[int] = None,
        class_confidence_overrides: Optional[Dict[str, float]] = None,
        class_size_constraints: Optional[Dict[str, Dict[str, int]]] = None
    ):
        """
        Update inference settings without restarting (hot-reload).

        Args:
            conf_threshold: New confidence threshold
            nms_threshold: New NMS threshold
            min_box_area: New minimum box area
            max_det: New maximum detections
            class_confidence_overrides: New per-class confidence overrides
            class_size_constraints: New per-class size constraints
        """
        updated_settings = []

        if conf_threshold is not None and conf_threshold != self.conf_threshold:
            self.conf_threshold = conf_threshold
            if self.detector:
                self.detector.conf_threshold = conf_threshold
            updated_settings.append(f"conf_threshold: {conf_threshold}")

        if nms_threshold is not None and nms_threshold != self.nms_threshold:
            self.nms_threshold = nms_threshold
            if self.detector:
                self.detector.nms_threshold = nms_threshold
            updated_settings.append(f"nms_threshold: {nms_threshold}")

        if min_box_area is not None and min_box_area != self.min_box_area:
            self.min_box_area = min_box_area
            updated_settings.append(f"min_box_area: {min_box_area}")

        if max_det is not None and max_det != self.max_det:
            self.max_det = max_det
            updated_settings.append(f"max_det: {max_det}")

        if class_confidence_overrides is not None:
            self.class_confidence_overrides = class_confidence_overrides
            updated_settings.append(f"class_confidence_overrides: {class_confidence_overrides}")

        if class_size_constraints is not None:
            self.class_size_constraints = class_size_constraints
            updated_settings.append(f"class_size_constraints: {class_size_constraints}")

        if updated_settings:
            logger.info(f"InferenceEngine settings updated: {', '.join(updated_settings)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        stats = {
            'device': self.device,
            'fps': self.fps,
            'avg_inference_time': self.avg_inference_time,
            'avg_inference_time_ms': self.avg_inference_time * 1000,
            'total_inferences': self.total_inference_count,  # Use cumulative count
            'dropped_results': self.dropped_results,
        }

        # Add drop rate
        if self.total_inference_count > 0:
            stats['drop_rate'] = self.dropped_results / self.total_inference_count
        else:
            stats['drop_rate'] = 0.0

        # Add GPU memory usage if CUDA is available
        if torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved(self.device) / 1024 / 1024

        # Add system memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            stats['memory_mb'] = mem_info.rss / 1024 / 1024
            stats['memory_percent'] = process.memory_percent()
        except ImportError:
            pass  # psutil not available

        return stats
