"""
RT-DETRv2 Detector - Transformer-based detection (Apache 2.0 License)
Better small object detection and more robust to occlusion than YOLOX

Repository: https://github.com/lyuwenyu/RT-DETR (Apache 2.0)
"""
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Union
from pathlib import Path

# Import from our src
from src.coco_constants import COCO_CLASSES, WILDLIFE_CLASSES

logger = logging.getLogger(__name__)

# Global reference for RT-DETR path (will be set in load_model)
_RTDETR_PATH = Path(__file__).parent.parent / "RT-DETR" / "rtdetrv2_pytorch"


class RTDETRDetector:
    """RT-DETRv2 detector for Stage 1 detection (transformer-based)"""

    def __init__(
        self,
        config_path: str = "RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
        model_path: str = "models/rtdetr/rtdetrv2_r18vd.pth",
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        input_size: tuple = (640, 640),
        wildlife_only: bool = True,
    ):
        """
        Initialize RT-DETRv2 detector

        Args:
            config_path: Path to RT-DETR config file
            model_path: Path to model weights (.pth)
            device: Device to run on
            conf_threshold: Confidence threshold
            input_size: Input image size (height, width)
            wildlife_only: If True, only return wildlife-relevant classes
        """
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.wildlife_only = wildlife_only

        self.model = None
        self.postprocessor = None
        self.transforms = None

    def load_model(self, max_retries: int = 3) -> bool:
        """
        Load RT-DETR model with retry logic

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            True if loaded successfully, False otherwise
        """
        import time

        # Lazy import of RT-DETR components to avoid sys.path conflicts at module load time
        if not _RTDETR_PATH.exists():
            logger.error(f"RT-DETR not found at {_RTDETR_PATH}")
            logger.error("Please clone https://github.com/lyuwenyu/RT-DETR to RT-DETR/")
            return False

        # Import RT-DETR YAMLConfig by temporarily manipulating sys.modules and sys.path
        # NOTE: We use sys.modules manipulation instead of importlib.util.spec_from_file_location
        # because RT-DETR's YAMLConfig has dependencies on other RT-DETR modules (src.core.yaml_utils, etc.)
        # that also need to be imported from RT-DETR's src. Using importlib for individual files would
        # break these inter-module dependencies. This approach swaps the entire 'src' namespace temporarily.
        # Future improvement: Consider creating a wrapper module or vendoring RT-DETR components to avoid
        # namespace conflicts, but this works reliably for now and is only executed once during model loading.
        _rtdetr_str = str(_RTDETR_PATH)

        # Save original state
        _original_path = sys.path.copy()
        _original_src = sys.modules.get('src')  # Save our src module

        try:
            # Temporarily remove our 'src' module from sys.modules
            if 'src' in sys.modules:
                del sys.modules['src']

            # Add RT-DETR to beginning of sys.path
            sys.path.insert(0, _rtdetr_str)

            # Import YAMLConfig from RT-DETR's src.core
            # NOTE: This import is intentionally from RT-DETR's src directory, not the project's src.
            # The sys.path and sys.modules manipulation above ensures we import from RT-DETR's src.
            from src.core import YAMLConfig as RTDETRYAMLConfig

            # Clean up RT-DETR's src from sys.modules (we only needed YAMLConfig)
            if 'src' in sys.modules:
                _rtdetr_src = sys.modules['src']  # Save RT-DETR's src
                del sys.modules['src']
            else:
                _rtdetr_src = None

        except ImportError as e:
            logger.error(f"Failed to import RT-DETR YAMLConfig: {e}")
            # Restore our src module
            if _original_src is not None:
                sys.modules['src'] = _original_src
            sys.path = _original_path
            return False
        finally:
            # Restore original state
            sys.path = _original_path
            # Restore our src module
            if _original_src is not None:
                sys.modules['src'] = _original_src

        for attempt in range(max_retries):
            try:
                logger.info(f"Loading RT-DETRv2 model")
                logger.info(f"Config: {self.config_path}")
                logger.info(f"Weights: {self.model_path}")

                # Load config and checkpoint
                cfg = RTDETRYAMLConfig(self.config_path, resume=self.model_path)

                checkpoint = torch.load(self.model_path, map_location='cpu')
                if 'ema' in checkpoint:
                    state = checkpoint['ema']['module']
                else:
                    state = checkpoint['model']

                # Load state dict
                cfg.model.load_state_dict(state)

                # CRITICAL FIX for variable input sizes:
                # Disable cached positional embeddings (eval_spatial_size) to enable dynamic generation
                # This allows RT-DETR to work at any resolution, not just the 640x640 it was trained on
                # Must disable for BOTH encoder (positional embeddings) and decoder (anchor generation)
                #
                # Performance trade-off: Dynamic generation adds ~1-2ms overhead vs cached embeddings,
                # but enables high-resolution detection (1920x1920) for small/distant wildlife.
                # At 640x640 native resolution, the overhead is negligible (~20ms → ~21ms).
                # At 1920x1920, total inference time is ~150-250ms (mostly from increased feature map size).
                if hasattr(cfg.model, 'encoder') and hasattr(cfg.model.encoder, 'eval_spatial_size'):
                    logger.info(f"Disabling encoder eval_spatial_size for dynamic positional embeddings")
                    cfg.model.encoder.eval_spatial_size = None

                if hasattr(cfg.model, 'decoder') and hasattr(cfg.model.decoder, 'eval_spatial_size'):
                    logger.info(f"Disabling decoder eval_spatial_size for dynamic anchor generation")
                    cfg.model.decoder.eval_spatial_size = None

                # Create deployment model
                class Model(nn.Module):
                    def __init__(self, model, postprocessor):
                        super().__init__()
                        self.model = model.deploy()
                        self.postprocessor = postprocessor.deploy()

                    def forward(self, images, orig_target_sizes):
                        outputs = self.model(images)
                        outputs = self.postprocessor(outputs, orig_target_sizes)
                        return outputs

                self.model = Model(cfg.model, cfg.postprocessor).to(self.device)
                self.model.eval()

                # Setup transforms
                self.transforms = T.Compose([
                    T.ToPILImage(),
                    T.Resize(self.input_size),
                    T.ToTensor(),
                ])

                logger.info(f"✓ RT-DETRv2 loaded successfully")
                logger.info(f"  Input size: {self.input_size}")
                logger.info(f"  Num classes: 80 (COCO)")
                logger.info(f"  Confidence threshold: {self.conf_threshold}")
                logger.info(f"  Architecture: ResNet-18 + Transformer")

                return True

            except (RuntimeError, OSError, IOError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Model load failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to load RT-DETR model after {max_retries} attempts: {e}", exc_info=True)
                    return False

            except Exception as e:
                logger.error(f"Failed to load RT-DETR model: {e}", exc_info=True)
                return False

        return False

    def preprocess(self, img: Union[np.ndarray, torch.Tensor]) -> tuple:
        """
        Preprocess image for RT-DETR

        Args:
            img: Input image (BGR, HxWxC)

        Returns:
            Tuple of (preprocessed tensor, original size tensor)
        """
        # Get original size
        if isinstance(img, torch.Tensor):
            orig_h, orig_w = img.shape[:2]
            img_np = img.cpu().numpy() if img.is_cuda else img.numpy()
        else:
            orig_h, orig_w = img.shape[:2]
            img_np = img

        # Convert BGR to RGB
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_np

        # Apply transforms
        img_tensor = self.transforms(img_rgb)[None].to(self.device)

        # Original size tensor (width, height format for RT-DETR)
        orig_size = torch.tensor([orig_w, orig_h])[None].to(self.device)

        return img_tensor, orig_size

    def detect(self, frame: Union[np.ndarray, torch.Tensor]) -> List[Dict[str, Any]]:
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

        # Preprocess
        img_tensor, orig_size = self.preprocess(frame)

        # Inference
        with torch.no_grad():
            labels, boxes, scores = self.model(img_tensor, orig_size)

        # Convert to detection format
        detections = []

        # labels, boxes, scores are lists with one element (batch size 1)
        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

        for i in range(len(labels)):
            score = float(scores[i])

            # Apply confidence threshold
            if score < self.conf_threshold:
                continue

            class_id = int(labels[i])

            # Filter wildlife classes if enabled
            if self.wildlife_only and class_id not in WILDLIFE_CLASSES:
                continue

            # Get box coordinates (already scaled to original size by postprocessor)
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

            # Get class name
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

            # Calculate box area
            box_area = int((x2 - x1) * (y2 - y1))

            detection_dict = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': score,
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

    def detect_batch(self, frames: List[Union[np.ndarray, torch.Tensor]]) -> List[List[Dict[str, Any]]]:
        """
        Run batched detection on multiple frames

        Args:
            frames: List of input frames (BGR format)

        Returns:
            List of detection lists (one list per frame)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return [[] for _ in frames]

        if not frames:
            return []

        # Preprocess all frames
        img_tensors = []
        orig_sizes = []

        for frame in frames:
            img_tensor, orig_size = self.preprocess(frame)
            img_tensors.append(img_tensor)
            orig_sizes.append(orig_size)

        # Stack into batch
        batch_tensor = torch.cat(img_tensors, dim=0)
        orig_sizes_tensor = torch.cat(orig_sizes, dim=0)

        try:
            # Batched inference
            with torch.no_grad():
                labels, boxes, scores = self.model(batch_tensor, orig_sizes_tensor)

            # Convert to detection format for each frame
            all_detections = []

            for frame_idx in range(len(frames)):
                detections = []

                frame_labels = labels[frame_idx].cpu().numpy()
                frame_boxes = boxes[frame_idx].cpu().numpy()
                frame_scores = scores[frame_idx].cpu().numpy()

                for i in range(len(frame_labels)):
                    score = float(frame_scores[i])

                    if score < self.conf_threshold:
                        continue

                    class_id = int(frame_labels[i])

                    if self.wildlife_only and class_id not in WILDLIFE_CLASSES:
                        continue

                    x1, y1, x2, y2 = frame_boxes[i]
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                    box_area = int((x2 - x1) * (y2 - y1))

                    detection_dict = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': score,
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'area': box_area
                        }
                    }

                    detections.append(detection_dict)

                all_detections.append(detections)

            return all_detections

        finally:
            # Explicit GPU memory cleanup for batched inference
            # While Python's GC will eventually free these, explicit deletion provides deterministic
            # memory release which is critical for GPU tensors in high-throughput scenarios.
            # PyTorch keeps GPU tensors alive until GC runs, which can be unpredictable.
            # See: https://pytorch.org/docs/stable/notes/faq.html#my-gpu-memory-isn-t-freed-properly
            # Using try-finally ensures cleanup occurs even if an exception is raised during processing.
            del batch_tensor
            del orig_sizes_tensor
            del img_tensors
            if 'labels' in locals():
                del labels
            if 'boxes' in locals():
                del boxes
            if 'scores' in locals():
                del scores

    def is_wildlife_relevant(self, class_id: int) -> bool:
        """Check if detection is wildlife-relevant"""
        return class_id in WILDLIFE_CLASSES

    def get_class_category(self, class_id: int) -> str:
        """
        Get general category for class

        Returns:
            "person", "bird", "mammal", or "other"
        """
        from src.coco_constants import MAMMAL_CLASS_IDS

        if class_id == 0:
            return "person"
        elif class_id == 14:
            return "bird"
        elif class_id in MAMMAL_CLASS_IDS:
            return "mammal"
        else:
            return "other"
