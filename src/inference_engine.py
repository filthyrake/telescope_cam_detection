"""
GPU Inference Engine
Handles GroundingDINO object detection with NVIDIA A30 GPU optimization.
Supports open-vocabulary detection with text prompts.
"""

import torch
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from queue import Queue
from threading import Thread, Event
import numpy as np
import cv2
from pathlib import Path
from groundingdino.util.inference import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    GPU-accelerated inference engine for object detection.
    Optimized for low latency on NVIDIA A30.
    Uses GroundingDINO for open-vocabulary detection.
    """

    def __init__(
        self,
        model_config: str,
        model_weights: str,
        device: str = "cuda:0",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        input_queue: Optional[Queue] = None,
        output_queue: Optional[Queue] = None,
        text_prompts: Optional[List[str]] = None,
        min_box_area: int = 0,  # Minimum bounding box area (pixels²)
        max_det: int = 300,  # Maximum detections per image
        use_two_stage: bool = False,  # Enable two-stage detection
        two_stage_pipeline: Optional[Any] = None,  # TwoStageDetectionPipeline instance
        class_confidence_overrides: Optional[Dict[str, float]] = None  # Per-class confidence thresholds
    ):
        """
        Initialize inference engine.

        Args:
            model_config: Path to GroundingDINO config file
            model_weights: Path to GroundingDINO model weights
            device: Device to run inference on (cuda:0, cpu, etc.)
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text-image matching
            input_queue: Queue to receive frames from
            output_queue: Queue to send detections to
            text_prompts: List of text prompts for open-vocabulary detection
            min_box_area: Minimum bounding box area (pixels²)
            max_det: Maximum detections per image
            use_two_stage: Enable two-stage detection pipeline
            two_stage_pipeline: TwoStageDetectionPipeline instance
            class_confidence_overrides: Per-class confidence thresholds
        """
        self.model_config = model_config
        self.model_weights = model_weights
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.text_prompts = text_prompts or []
        self.min_box_area = min_box_area
        self.max_det = max_det
        self.use_two_stage = use_two_stage
        self.two_stage_pipeline = two_stage_pipeline
        self.class_confidence_overrides = class_confidence_overrides or {}

        # Create text prompt caption (period-separated)
        self.caption = " . ".join(self.text_prompts) + " ."

        self.model: Optional[Model] = None
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
        """
        Load GroundingDINO model and move to GPU.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Use two-stage pipeline if enabled
            if self.use_two_stage and self.two_stage_pipeline:
                logger.info("Loading two-stage detection pipeline...")
                if self.two_stage_pipeline.load_detector():
                    self.is_loaded = True
                    logger.info("Two-stage pipeline loaded successfully")
                    return True
                else:
                    logger.error("Failed to load two-stage pipeline")
                    return False

            # Standard GroundingDINO loading
            logger.info(f"Loading GroundingDINO model from {self.model_weights}")
            logger.info(f"Using config: {self.model_config}")
            logger.info(f"Using device: {self.device}")

            # Check CUDA availability
            if "cuda" in self.device and not torch.cuda.is_available():
                logger.error("CUDA requested but not available")
                return False

            # Load GroundingDINO model
            self.model = Model(
                model_config_path=self.model_config,
                model_checkpoint_path=self.model_weights,
                device=self.device
            )

            # Log GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            # Log text prompts
            logger.info(f"Open-vocabulary prompts: {self.text_prompts}")
            logger.info(f"Caption: {self.caption}")

            # Warm up the model with a dummy inference
            logger.info("Warming up model...")
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict_with_caption(
                image=dummy_input,
                caption=self.caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )

            self.is_loaded = True
            logger.info("GroundingDINO model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def start(self) -> bool:
        """
        Start the inference thread.

        Returns:
            True if started successfully, False otherwise
        """
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
        """Stop the inference thread."""
        logger.info("Stopping inference thread...")
        self.stop_event.set()

        if self.inference_thread:
            self.inference_thread.join(timeout=5.0)

        logger.info("Inference thread stopped")

    def _inference_loop(self):
        """Main inference loop running in separate thread."""
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
                    'frame_shape': frame.shape
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
        # Use two-stage pipeline if enabled
        if self.use_two_stage and self.two_stage_pipeline:
            result = self.two_stage_pipeline.detect(frame)
            return result.get('detections', [])

        # Run GroundingDINO inference
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict with GroundingDINO
        # Returns: (Detections object, List[str] labels)
        detections_obj, labels = self.model.predict_with_caption(
            image=image_rgb,
            caption=self.caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        detections = []

        # Extract data from Detections object
        # detections_obj.xyxy: bounding boxes in [x1, y1, x2, y2] format (already in pixels)
        # detections_obj.confidence: confidence scores
        # labels: list of class names

        if len(detections_obj.xyxy) > 0:
            for box, conf, label in zip(detections_obj.xyxy, detections_obj.confidence, labels):
                x1, y1, x2, y2 = box

                # Clean up label (remove periods and extra spaces)
                class_name = label.strip().strip('.')

                # Apply per-class confidence threshold if specified
                class_conf_threshold = self.class_confidence_overrides.get(class_name, self.box_threshold)
                if conf < class_conf_threshold:
                    continue

                # Calculate bounding box area
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height

                # Filter by minimum box area (skip tiny detections)
                if self.min_box_area > 0 and box_area < self.min_box_area:
                    continue

                detection = {
                    'class_id': -1,  # GroundingDINO doesn't have class IDs
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'area': int(box_area)
                    }
                }

                detections.append(detection)

                # Stop if we hit max detections
                if len(detections) >= self.max_det:
                    break

        return detections

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
        """
        Get inference statistics.

        Returns:
            Dictionary containing inference stats
        """
        return {
            'is_loaded': self.is_loaded,
            'device': self.device,
            'fps': self.fps,
            'avg_inference_time_ms': self.avg_inference_time * 1000,
            'total_inferences': self.inference_count,
            'box_threshold': self.box_threshold,
            'text_threshold': self.text_threshold
        }


if __name__ == "__main__":
    # Test the inference engine
    import cv2
    from queue import Queue

    logger.info("Testing Inference Engine with GroundingDINO")

    # Check CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Create queues
    input_queue = Queue(maxsize=2)
    output_queue = Queue(maxsize=10)

    # Test prompts
    text_prompts = ["person", "dog", "cat", "bird"]

    # Initialize engine
    engine = InferenceEngine(
        model_config="models/GroundingDINO_SwinT_OGC.py",
        model_weights="models/groundingdino_swint_ogc.pth",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        box_threshold=0.25,
        text_threshold=0.25,
        input_queue=input_queue,
        output_queue=output_queue,
        text_prompts=text_prompts
    )

    # Load model
    if not engine.load_model():
        logger.error("Failed to load model")
        exit(1)

    # Test with a dummy frame
    test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    detections = engine.infer_single(test_frame)

    logger.info(f"Test inference completed. Detections: {len(detections)}")
    logger.info(f"Stats: {engine.get_stats()}")

    logger.info("Inference engine test completed successfully")
