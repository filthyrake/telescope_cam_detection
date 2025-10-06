"""
Two-Stage Detection Pipeline - YOLOX Version
Stage 1: YOLOX for fast detection (11-21ms)
Stage 2: iNaturalist for species classification (~20-30ms)
Total: 30-50ms end-to-end
"""

import cv2
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from species_classifier import SpeciesClassifier
from src.coco_constants import CLASS_ID_TO_CATEGORY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageDetectionPipeline:
    """
    Two-stage pipeline for wildlife detection and species identification.

    Stage 1: YOLOX detects animals and creates bounding boxes (handled externally)
    Stage 2: iNaturalist classifier identifies exact species from crops
    """

    def __init__(
        self,
        enable_species_classification: bool = True,
        stage2_confidence_threshold: float = 0.3,
        device: str = "cuda:0",
    ):
        """
        Initialize two-stage pipeline.

        Args:
            enable_species_classification: Whether to run Stage 2
            stage2_confidence_threshold: Min confidence for species classification
            device: Device to run on
        """
        self.enable_species_classification = enable_species_classification
        self.stage2_confidence_threshold = stage2_confidence_threshold
        self.device = device

        # Species classifiers (will be added via add_species_classifier)
        self.species_classifiers: Dict[str, SpeciesClassifier] = {}

        # Use shared COCO class mapping for Stage 2 routing
        self.class_id_to_category = CLASS_ID_TO_CATEGORY

        logger.info("Two-stage pipeline initialized (YOLOX + iNaturalist)")

    def add_species_classifier(self, category: str, classifier: SpeciesClassifier):
        """
        Add a species classifier for a category.

        Args:
            category: Category name (e.g., 'bird', 'mammal', 'reptile')
            classifier: SpeciesClassifier instance
        """
        self.species_classifiers[category] = classifier
        logger.info(f"Added species classifier for category: {category}")

    def classify_detection(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Stage 2 classification on a detection.

        Args:
            frame: Full frame image (BGR)
            detection: Detection dict with bbox and class info

        Returns:
            Updated detection dict with species info
        """
        if not self.enable_species_classification:
            return detection

        # Get detection info
        class_id = detection.get('class_id')
        class_name = detection.get('class_name', '')
        bbox = detection.get('bbox', {})

        # Determine classifier category
        category = self.class_id_to_category.get(class_id)

        # Check if we have a classifier for this category
        if category not in self.species_classifiers:
            # No classifier for this category, return as-is
            detection['species'] = None
            detection['species_confidence'] = 0.0
            return detection

        # Extract crop
        x1 = int(bbox['x1'])
        y1 = int(bbox['y1'])
        x2 = int(bbox['x2'])
        y2 = int(bbox['y2'])

        # Ensure valid crop
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            # Invalid crop
            detection['species'] = None
            detection['species_confidence'] = 0.0
            return detection

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            detection['species'] = None
            detection['species_confidence'] = 0.0
            return detection

        # Run species classification
        classifier = self.species_classifiers[category]

        try:
            # classifier.classify() returns List[Dict[str, Any]]
            results = classifier.classify(crop, top_k=1)

            if results and len(results) > 0:
                # Get top prediction
                top_result = results[0]
                species_name = top_result['species']
                confidence = top_result['confidence']
                taxonomic_level = top_result.get('taxonomic_level', 'species')

                # Note: Hierarchical mode accepts confidence >= 0.1 (class level)
                # This is intentional - lower confidence returns coarser taxonomic levels
                # (e.g., "Mammalia (class)" at 0.15 instead of null)
                # Original stage2_confidence_threshold is enforced at classifier level
                detection['species'] = species_name
                detection['species_confidence'] = float(confidence)
                detection['stage2_category'] = category
                detection['taxonomic_level'] = taxonomic_level

                logger.debug(f"Classified as {species_name} ({taxonomic_level}, conf: {confidence:.2f})")
            else:
                # No results above threshold
                detection['species'] = None
                detection['species_confidence'] = 0.0
                detection['stage2_category'] = category
                detection['taxonomic_level'] = None

        except Exception as e:
            logger.error(f"Species classification failed: {e}")
            detection['species'] = None
            detection['species_confidence'] = 0.0
            detection['taxonomic_level'] = None

        return detection

    def process_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process all detections through Stage 2.

        Args:
            frame: Full frame image (BGR)
            detections: List of detection dicts from Stage 1

        Returns:
            List of detection dicts with species info added
        """
        if not self.enable_species_classification:
            # Add empty species fields
            for det in detections:
                det['species'] = None
                det['species_confidence'] = 0.0
            return detections

        # Classify each detection
        processed_detections = []
        for detection in detections:
            processed_det = self.classify_detection(frame, detection)
            processed_detections.append(processed_det)

        return processed_detections

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'stage2_enabled': self.enable_species_classification,
            'classifiers_loaded': list(self.species_classifiers.keys()),
            'num_classifiers': len(self.species_classifiers),
        }
