"""
GPU Inference Engine
Handles YOLOv8/YOLO-World object detection with NVIDIA A30 GPU optimization.
Supports both single-stage and two-stage detection pipelines.
"""

import torch
import time
import logging
from typing import Optional, List, Dict, Any
from queue import Queue
from threading import Thread, Event
import numpy as np
from ultralytics import YOLO
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    GPU-accelerated inference engine for object detection.
    Optimized for low latency on NVIDIA A30.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        input_queue: Optional[Queue] = None,
        output_queue: Optional[Queue] = None,
        target_classes: Optional[List[str]] = None,
        min_box_area: int = 0,  # Minimum bounding box area (pixels²)
        max_det: int = 300,  # Maximum detections per image
        use_two_stage: bool = False,  # Enable two-stage detection
        two_stage_pipeline: Optional[Any] = None,  # TwoStageDetectionPipeline instance
        class_confidence_overrides: Optional[Dict[str, float]] = None  # Per-class confidence thresholds
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to YOLOv8/YOLO-World model weights
            device: Device to run inference on (cuda:0, cpu, etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            input_queue: Queue to receive frames from
            output_queue: Queue to send detections to
            target_classes: List of class names to detect (None = all classes)
            use_two_stage: Enable two-stage detection pipeline
            two_stage_pipeline: TwoStageDetectionPipeline instance
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.target_classes = target_classes
        self.min_box_area = min_box_area
        self.max_det = max_det
        self.use_two_stage = use_two_stage
        self.two_stage_pipeline = two_stage_pipeline
        self.class_confidence_overrides = class_confidence_overrides or {}

        self.model: Optional[YOLO] = None
        self.is_loaded = False
        self.is_yolo_world = False
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
        Load YOLOv8/YOLO-World model and move to GPU.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Use two-stage pipeline if enabled
            if self.use_two_stage and self.two_stage_pipeline:
                logger.info("Loading two-stage detection pipeline...")
                if self.two_stage_pipeline.load_detector():
                    self.is_loaded = True
                    self.is_yolo_world = True
                    logger.info("Two-stage pipeline loaded successfully")
                    return True
                else:
                    logger.error("Failed to load two-stage pipeline")
                    return False

            # Standard YOLO loading
            logger.info(f"Loading model from {self.model_path}")
            logger.info(f"Using device: {self.device}")

            # Check CUDA availability
            if "cuda" in self.device and not torch.cuda.is_available():
                logger.error("CUDA requested but not available")
                return False

            # Detect if this is a YOLO-World model
            model_name = Path(self.model_path).stem.lower()
            self.is_yolo_world = 'world' in model_name

            # Load YOLO model
            if self.is_yolo_world:
                from ultralytics import YOLOWorld
                self.model = YOLOWorld(self.model_path)
                # Set custom classes if provided
                if self.target_classes:
                    self.model.set_classes(self.target_classes)
                    logger.info(f"YOLO-World classes set: {self.target_classes}")
            else:
                self.model = YOLO(self.model_path)

            # Move model to device
            self.model.to(self.device)

            # Log GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            # Warm up the model with a dummy inference
            logger.info("Warming up model...")
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_input, verbose=False)

            self.is_loaded = True
            model_type = "YOLO-World" if self.is_yolo_world else "YOLO"
            logger.info(f"{model_type} model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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

        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False
        )[0]

        detections = []

        # Extract detection results
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                class_name = results.names[class_id]

                # Filter by target classes if specified
                if self.target_classes and class_name not in self.target_classes:
                    continue

                # Apply per-class confidence threshold if specified
                class_conf_threshold = self.class_confidence_overrides.get(class_name, self.conf_threshold)
                if conf < class_conf_threshold:
                    continue

                # Calculate bounding box area
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                box_area = box_width * box_height

                # Filter by minimum box area (skip tiny detections)
                if self.min_box_area > 0 and box_area < self.min_box_area:
                    continue

                detection = {
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3]),
                        'area': int(box_area)
                    }
                }

                detections.append(detection)

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
            'conf_threshold': self.conf_threshold
        }


# COCO class names for reference
COCO_PERSON_ANIMAL_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]


if __name__ == "__main__":
    # Test the inference engine
    import cv2
    from queue import Queue

    logger.info("Testing Inference Engine")

    # Check CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Create queues
    input_queue = Queue(maxsize=2)
    output_queue = Queue(maxsize=10)

    # Initialize engine
    engine = InferenceEngine(
        model_path="yolov8n.pt",  # Will download automatically
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        conf_threshold=0.5,
        input_queue=input_queue,
        output_queue=output_queue,
        target_classes=COCO_PERSON_ANIMAL_CLASSES
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
