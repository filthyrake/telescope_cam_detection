"""
Two-Stage Detection Pipeline
Stage 1: YOLO-World for fast animal detection and localization
Stage 2: Species classifier for fine-grained identification
"""

import cv2
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ultralytics import YOLOWorld
from species_classifier import SpeciesClassifier, TaxonomySpecificClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageDetectionPipeline:
    """
    Two-stage pipeline for wildlife detection and species identification.

    Stage 1: YOLO-World detects animals and creates bounding boxes
    Stage 2: Species classifier identifies exact species from crops
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8x-worldv2.pt",
        device: str = "cuda",
        enable_species_classification: bool = True,
        stage2_confidence_threshold: float = 0.3,
        stage1_confidence_threshold: float = 0.25
    ):
        """
        Initialize two-stage pipeline.

        Args:
            yolo_model_path: Path to YOLO-World model
            device: Device to run on
            enable_species_classification: Whether to run Stage 2
            stage2_confidence_threshold: Min confidence for species classification
            stage1_confidence_threshold: Min confidence for detection
        """
        self.device = device
        self.enable_species_classification = enable_species_classification
        self.stage2_confidence_threshold = stage2_confidence_threshold
        self.stage1_confidence_threshold = stage1_confidence_threshold

        # Stage 1: YOLO-World detector
        self.detector = None
        self.yolo_model_path = yolo_model_path

        # Stage 2: Species classifiers
        self.species_classifier = None
        if enable_species_classification:
            self.species_classifier = TaxonomySpecificClassifier(device=device)

        # Detection classes for Stage 1
        self.detection_classes = [
            'person',
            'bird',
            'lizard',
            'iguana',
            'tortoise',
            'rabbit',
            'coyote',
            'fox',
            'squirrel',
            'snake',
            'deer',
            'javelina'
        ]

        # Mapping from detection class to taxonomy group
        self.class_to_taxonomy = {
            'bird': 'bird',
            'lizard': 'reptile',
            'iguana': 'reptile',
            'tortoise': 'reptile',
            'snake': 'reptile',
            'rabbit': 'mammal',
            'coyote': 'mammal',
            'fox': 'mammal',
            'squirrel': 'mammal',
            'deer': 'mammal',
            'javelina': 'mammal'
        }

        logger.info("TwoStageDetectionPipeline initialized")

    def load_detector(self) -> bool:
        """
        Load YOLO-World detector (Stage 1).

        Returns:
            True if loaded successfully
        """
        try:
            self.detector = YOLOWorld(self.yolo_model_path)
            self.detector.set_classes(self.detection_classes)
            logger.info(f"YOLO-World loaded with classes: {self.detection_classes}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO-World: {e}")
            return False

    def add_species_classifier(
        self,
        taxonomy_group: str,
        classifier: SpeciesClassifier
    ):
        """
        Add a species classifier for a taxonomic group.

        Args:
            taxonomy_group: Group name ('bird', 'mammal', 'reptile')
            classifier: SpeciesClassifier instance
        """
        if self.species_classifier:
            self.species_classifier.add_classifier(taxonomy_group, classifier)
            logger.info(f"Added species classifier for {taxonomy_group}")

    def detect(
        self,
        frame: np.ndarray,
        run_stage2: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Run two-stage detection pipeline.

        Args:
            frame: Input image
            run_stage2: Override to enable/disable Stage 2

        Returns:
            Detection result with species identification
        """
        if self.detector is None:
            logger.error("Detector not loaded")
            return {'detections': [], 'error': 'Detector not loaded'}

        # Stage 1: YOLO-World detection
        try:
            results = self.detector.predict(
                frame,
                conf=self.stage1_confidence_threshold,
                verbose=False
            )

            # Parse Stage 1 results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    # Extract detection info
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.detection_classes[cls_id] if cls_id < len(self.detection_classes) else f"class_{cls_id}"

                    detection = {
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': {
                            'x1': float(xyxy[0]),
                            'y1': float(xyxy[1]),
                            'x2': float(xyxy[2]),
                            'y2': float(xyxy[3])
                        },
                        'stage1_class': class_name,  # Keep original detection
                        'stage2_enabled': False
                    }

                    # Stage 2: Species classification (if enabled)
                    if (run_stage2 or (run_stage2 is None and self.enable_species_classification)) and \
                       self.species_classifier and class_name != 'person':

                        species_result = self._classify_species(frame, detection, class_name)
                        if species_result:
                            detection.update(species_result)
                            detection['stage2_enabled'] = True

                    detections.append(detection)

            return {
                'detections': detections,
                'stage1_count': len(detections),
                'stage2_count': sum(1 for d in detections if d.get('stage2_enabled', False))
            }

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {'detections': [], 'error': str(e)}

    def _classify_species(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        class_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run Stage 2 species classification on detected animal.

        Args:
            frame: Full frame
            detection: Detection dict with bbox
            class_name: Detected class from Stage 1

        Returns:
            Species classification results or None
        """
        try:
            # Get taxonomy group for this class
            taxonomy_group = self.class_to_taxonomy.get(class_name)
            if not taxonomy_group:
                return None

            # Check if we have a classifier for this group
            if taxonomy_group not in self.species_classifier.get_available_groups():
                return None

            # Crop animal from frame
            bbox = detection['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])

            # Add some padding
            h, w = frame.shape[:2]
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = frame[y1:y2, x1:x2]

            # Classify species
            species_results = self.species_classifier.classify(
                crop,
                taxonomy_group,
                top_k=3
            )

            if species_results:
                return {
                    'species': species_results[0]['species'],
                    'species_confidence': species_results[0]['confidence'],
                    'species_alternatives': species_results[1:],
                    'taxonomy_group': taxonomy_group
                }

            return None

        except Exception as e:
            logger.error(f"Species classification failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            'stage1_enabled': self.detector is not None,
            'stage2_enabled': self.enable_species_classification,
            'detection_classes': self.detection_classes,
        }

        if self.species_classifier:
            stats['available_classifiers'] = self.species_classifier.get_available_groups()

        return stats


class PipelineConfig:
    """Configuration for two-stage pipeline."""

    # Desert wildlife detection classes
    DESERT_DETECTION_CLASSES = [
        'person',
        # Birds
        'bird', 'quail', 'roadrunner', 'hawk', 'raven', 'dove',
        # Mammals
        'coyote', 'rabbit', 'fox', 'squirrel', 'javelina', 'deer', 'bobcat',
        # Reptiles
        'lizard', 'iguana', 'tortoise', 'snake'
    ]

    # Broader classes for better detection
    COARSE_DETECTION_CLASSES = [
        'person',
        'bird',
        'mammal',
        'reptile',
        'lizard'
    ]

    @staticmethod
    def get_class_mapping() -> Dict[str, str]:
        """Get mapping from detection class to taxonomy group."""
        return {
            'bird': 'bird', 'quail': 'bird', 'roadrunner': 'bird',
            'hawk': 'bird', 'raven': 'bird', 'dove': 'bird',
            'coyote': 'mammal', 'rabbit': 'mammal', 'fox': 'mammal',
            'squirrel': 'mammal', 'javelina': 'mammal', 'deer': 'mammal',
            'bobcat': 'mammal', 'mammal': 'mammal',
            'lizard': 'reptile', 'iguana': 'reptile', 'tortoise': 'reptile',
            'snake': 'reptile', 'reptile': 'reptile'
        }


if __name__ == "__main__":
    # Test the two-stage pipeline
    logger.info("Testing TwoStageDetectionPipeline")

    # Initialize pipeline
    pipeline = TwoStageDetectionPipeline(
        yolo_model_path="yolov8x-worldv2.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_species_classification=False  # Disable Stage 2 for basic test
    )

    # Load detector
    if pipeline.load_detector():
        logger.info("✅ Pipeline initialized successfully")

        # Create test frame
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Run detection
        results = pipeline.detect(test_frame)
        logger.info(f"Detection results: {results}")

        # Get stats
        stats = pipeline.get_stats()
        logger.info(f"Pipeline stats: {stats}")
    else:
        logger.error("❌ Failed to initialize pipeline")
