"""
GPU Inference Engine - YOLOX Version
Fast Stage 1 detection with YOLOX (Apache 2.0 License)
47x faster than GroundingDINO (11-21ms vs 560ms)
"""

import torch
import time
import logging
from typing import Optional, List, Dict, Any
from queue import Queue
from threading import Thread, Event
import numpy as np

from src.yolox_detector import YOLOXDetector

logging.basicConfig(level=logging.INFO)
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
        self.wildlife_only = wildlife_only

        self.detector: Optional[YOLOXDetector] = None
        self.is_loaded = False
        self.stop_event = Event()
        self.inference_thread: Optional[Thread] = None

        # Performance metrics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0
        self.fps = 0.0
        self.last_fps_check = time.time()

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
            self.inference_thread.join(timeout=5.0)

        logger.info("Inference thread stopped")

    def _inference_loop(self):
        """Main inference loop running in separate thread"""
        logger.info("Inference loop started")

        while not self.stop_event.is_set():
            try:
                # Get frame from input queue (non-blocking)
                if self.input_queue.empty():
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                    continue

                frame_data = self.input_queue.get()
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
                self.total_inference_time += inference_time
                self.avg_inference_time = self.total_inference_time / self.inference_count

                # Calculate FPS
                if time.time() - self.last_fps_check >= 1.0:
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
                except:
                    pass  # Drop if queue is full

            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(0.1)

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

        # Filter by confidence overrides
        filtered_detections = []
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            box_area = det['bbox']['area']

            # Apply per-class confidence threshold
            class_threshold = self.class_confidence_overrides.get(class_name, self.conf_threshold)
            if confidence < class_threshold:
                continue

            # Filter by minimum box area
            if self.min_box_area > 0 and box_area < self.min_box_area:
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

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            'fps': self.fps,
            'avg_inference_time': self.avg_inference_time,
            'total_inferences': self.inference_count,
        }
