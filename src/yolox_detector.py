"""
YOLOX Detector - Fast Stage 1 detection (Apache 2.0 License)
Replaces GroundingDINO for 47x faster inference (11.8ms vs 560ms)
"""
import sys
import torch
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Union
from pathlib import Path

# Add YOLOX directory to sys.path dynamically (relative to this file)
_yolox_path = Path(__file__).parent.parent / "YOLOX"
_yolox_exp_file = _yolox_path / "yolox" / "exp" / "__init__.py"

# Validate YOLOX installation before modifying sys.path
if not _yolox_path.exists():
    raise ImportError(
        f"YOLOX not found at {_yolox_path}. "
        "Please install YOLOX according to the official project documentation."
    )

if not _yolox_exp_file.exists():
    raise ImportError(
        f"YOLOX installation incomplete: {_yolox_exp_file} not found. "
        "Please ensure YOLOX is properly installed with all required modules."
    )

if str(_yolox_path) not in sys.path:
    sys.path.insert(0, str(_yolox_path))

from yolox.exp import get_exp
from yolox.utils import postprocess
from src.coco_constants import COCO_CLASSES, WILDLIFE_CLASSES, MAMMAL_CLASS_IDS

logger = logging.getLogger(__name__)


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

    def preprocess(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image for YOLOX with GPU-accelerated resize

        Args:
            img: Input image (BGR, HxWxC) - can be NumPy array or GPU tensor

        Returns:
            Preprocessed tensor (1xCxHxW)
        """
        # Convert to tensor if needed (HxWxC -> 1xCxHxW)
        if isinstance(img, torch.Tensor):
            # Already a tensor - just add batch dimension and ensure float
            img_tensor = img.permute(2, 0, 1).unsqueeze(0).float()
            # Ensure it's on the correct device (proper device comparison to avoid unnecessary transfers)
            target_device = torch.device(self.device)
            if img_tensor.device != target_device:
                img_tensor = img_tensor.to(target_device)
        else:
            # NumPy array - convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            # Move to device BEFORE resize (so resize happens on GPU!)
            img_tensor = img_tensor.to(self.device)

        # GPU-accelerated resize using torch.nn.functional
        # Note: Both img_tensor.shape[2:] and self.input_size are (H, W) format for PyTorch
        if img_tensor.shape[2:] != self.input_size:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor,
                size=self.input_size,  # Expects (H, W) - matches our convention
                mode='bilinear',
                align_corners=False
            )

        return img_tensor

    def _format_model_output_to_detections(
        self,
        output: np.ndarray,
        orig_h: int,
        orig_w: int
    ) -> List[Dict[str, Any]]:
        """
        Convert model output to detection format with scaling and filtering.

        Args:
            output: Model output array (N x 7: x1, y1, x2, y2, obj_conf, class_conf, class_id)
            orig_h: Original image height
            orig_w: Original image width

        Returns:
            List of detection dictionaries
        """
        detections = []

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

    def detect(self, frame: Union[np.ndarray, torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Run detection on frame

        Args:
            frame: Input frame (BGR format) - can be NumPy array or GPU tensor

        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []

        # Get original image size (NumPy or torch.Tensor both support shape[:2])
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

        # Convert to detection format using shared helper
        if outputs[0] is not None:
            output = outputs[0].cpu().numpy()
            return self._format_model_output_to_detections(output, orig_h, orig_w)

        return []

    def detect_batch(self, frames: List[Union[np.ndarray, torch.Tensor]]) -> List[List[Dict[str, Any]]]:
        """
        Run batched detection on multiple frames (GPU-optimized)

        This method processes multiple frames in a single forward pass for improved
        GPU utilization. Recommended for multi-camera setups with 2-4+ cameras.

        Args:
            frames: List of input frames (BGR format) - can be NumPy arrays or GPU tensors

        Returns:
            List of detection lists (one list per frame)

        Performance Notes:
            - Single frame (batch=1): ~11-21ms
            - Batch of 4 frames: ~15-30ms (3-4x throughput!)
            - GPU utilization: 30-40% → 80-90%
        """
        if self.model is None:
            logger.error("Model not loaded")
            return [[] for _ in frames]

        if not frames:
            return []

        # Get original image sizes (extract H and W for both formats)
        # NumPy array (HWC): shape[0]=H, shape[1]=W
        # Torch tensor (CHW): shape[-2]=H, shape[-1]=W
        orig_sizes = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                # Torch tensor (CHW or NCHW) - H and W are last two dimensions
                h, w = frame.shape[-2], frame.shape[-1]
            else:
                # NumPy array (HWC) - H and W are first two dimensions
                h, w = frame.shape[0], frame.shape[1]
            orig_sizes.append((h, w))

        # Preprocess all frames and stack into batch
        preprocessed_frames = [self.preprocess(frame) for frame in frames]
        batch_tensor = torch.cat(preprocessed_frames, dim=0)  # (B, C, H, W)

        # Batched inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)

            # Post-process (NMS, etc.) - returns list of tensors (one per frame)
            outputs = postprocess(
                outputs,
                self.exp.num_classes,
                self.conf_threshold,
                self.nms_threshold,
            )

        # Convert to detection format for each frame using shared helper
        all_detections = []

        for frame_idx, output in enumerate(outputs):
            if output is not None:
                output_np = output.cpu().numpy()
                orig_h, orig_w = orig_sizes[frame_idx]
                detections = self._format_model_output_to_detections(output_np, orig_h, orig_w)
            else:
                detections = []

            all_detections.append(detections)

        return all_detections

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
        elif class_id in MAMMAL_CLASS_IDS:
            return "mammal"
        else:
            return "other"
