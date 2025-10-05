"""
YOLOX Detector - Fast Stage 1 detection (Apache 2.0 License)
Replaces GroundingDINO for 47x faster inference (11.8ms vs 560ms)
"""
import sys
sys.path.insert(0, 'YOLOX')

import torch
import cv2
import numpy as np
import logging
from typing import List, Dict, Any
from yolox.exp import get_exp
from yolox.utils import postprocess

logger = logging.getLogger(__name__)

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Wildlife-relevant COCO classes (indices)
WILDLIFE_CLASSES = {
    0: "person",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


class YOLOXDetector:
    """Fast YOLOX detector for Stage 1 detection"""

    def __init__(
        self,
        model_name: str = "yolox-s",
        model_path: str = "models/yolox/yolox_s.pth",
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: tuple = (640, 640),
        wildlife_only: bool = True,
    ):
        """
        Initialize YOLOX detector

        Args:
            model_name: YOLOX model variant (yolox-s, yolox-m, yolox-l, etc.)
            model_path: Path to model weights
            device: Device to run on
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            input_size: Input image size (height, width)
            wildlife_only: If True, only return wildlife-relevant classes (no umbrellas/backpacks/etc.)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.wildlife_only = wildlife_only

        self.model = None
        self.exp = None

    def load_model(self) -> bool:
        """Load YOLOX model"""
        try:
            logger.info(f"Loading YOLOX model: {self.model_name}")
            logger.info(f"Weights: {self.model_path}")

            # Get experiment config
            self.exp = get_exp(None, self.model_name)
            self.exp.test_size = self.input_size

            # Load model
            self.model = self.exp.get_model()
            ckpt = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ YOLOX loaded successfully")
            logger.info(f"  Input size: {self.input_size}")
            logger.info(f"  Num classes: {self.exp.num_classes} (COCO)")
            logger.info(f"  Confidence threshold: {self.conf_threshold}")

            return True

        except Exception as e:
            logger.error(f"Failed to load YOLOX model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for YOLOX

        Args:
            img: Input image (BGR, HxWxC)

        Returns:
            Preprocessed tensor (1xCxHxW)
        """
        # Resize
        img_resized = cv2.resize(img, self.input_size)

        # Convert to tensor (HxWxC -> 1xCxHxW)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()

        # Move to device
        img_tensor = img_tensor.to(self.device)

        return img_tensor

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []

        # Get original image size
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        img_tensor = self.preprocess(frame)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)

            # Post-process (NMS, etc.)
            outputs = postprocess(
                outputs,
                self.exp.num_classes,
                self.conf_threshold,
                self.nms_threshold,
            )

        # Convert to detection format
        detections = []

        if outputs[0] is not None:
            output = outputs[0].cpu().numpy()

            # Scale boxes to original image size
            ratio_h = orig_h / self.input_size[0]
            ratio_w = orig_w / self.input_size[1]

            for detection in output:
                x1, y1, x2, y2, obj_conf, class_conf, class_id = detection

                # Scale coordinates
                x1 = float(x1 * ratio_w)
                y1 = float(y1 * ratio_h)
                x2 = float(x2 * ratio_w)
                y2 = float(y2 * ratio_h)

                class_id = int(class_id)
                confidence = float(obj_conf * class_conf)

                # Filter out non-wildlife classes if wildlife_only is enabled
                if self.wildlife_only and not self.is_wildlife_relevant(class_id):
                    continue

                # Get class name
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

                # Calculate box area
                box_area = int((x2 - x1) * (y2 - y1))

                detection_dict = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'area': box_area
                    }
                }

                detections.append(detection_dict)

        return detections

    def is_wildlife_relevant(self, class_id: int) -> bool:
        """Check if detection is wildlife-relevant"""
        return class_id in WILDLIFE_CLASSES

    def get_class_category(self, class_id: int) -> str:
        """
        Get general category for class

        Returns:
            "person", "bird", "mammal", or "other"
        """
        if class_id == 0:
            return "person"
        elif class_id == 14:
            return "bird"
        elif class_id in [15, 16, 17, 18, 19, 20, 21, 22, 23]:  # cat, dog, horse, etc.
            return "mammal"
        else:
            return "other"
