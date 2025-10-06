"""
Two-Stage Detection Pipeline - YOLOX Version
Stage 1: YOLOX for fast detection (11-21ms)
Stage 2: iNaturalist for species classification (~20-30ms)
Stage 2 Enhancement (optional): Real-ESRGAN + CLAHE + bilateral (~1s)
Total: 30-50ms (no enhancement) or ~1s (with Real-ESRGAN)
"""

import cv2
import torch
import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple

from species_classifier import SpeciesClassifier
from src.coco_constants import CLASS_ID_TO_CATEGORY
from src.image_enhancement import ImageEnhancer

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
        enhancement_config: Optional[Dict[str, Any]] = None,
        rejected_taxonomic_levels: Optional[List[str]] = None,
    ):
        """
        Initialize two-stage pipeline.

        Args:
            enable_species_classification: Whether to run Stage 2
            stage2_confidence_threshold: Min confidence for species classification
            device: Device to run on
            enhancement_config: Image enhancement config (e.g., {"method": "realesrgan", ...})
            rejected_taxonomic_levels: Taxonomic levels to reject (e.g., ['order', 'class'])
        """
        self.enable_species_classification = enable_species_classification
        self.stage2_confidence_threshold = stage2_confidence_threshold
        self.device = device

        # Configurable rejected taxonomic levels (default: order and class are too vague)
        self.rejected_taxonomic_levels = rejected_taxonomic_levels or ['order', 'class']

        # Species classifiers (will be added via add_species_classifier)
        self.species_classifiers: Dict[str, SpeciesClassifier] = {}

        # Use shared COCO class mapping for Stage 2 routing
        self.class_id_to_category = CLASS_ID_TO_CATEGORY

        # Image enhancement (optional)
        self.enhancer = None
        if enhancement_config and enhancement_config.get('enabled', False):
            try:
                logger.info(f"Initializing image enhancer: method={enhancement_config.get('method', 'none')}")

                # Flatten nested config structure for ImageEnhancer
                enhancer_params = {
                    'method': enhancement_config.get('method', 'none'),
                    'device': enhancement_config.get('device', self.device)
                }

                # Add Real-ESRGAN params with realesrgan_ prefix
                if 'realesrgan' in enhancement_config:
                    for key, value in enhancement_config['realesrgan'].items():
                        enhancer_params[f'realesrgan_{key}'] = value

                # Add CLAHE params with clahe_ prefix
                if 'clahe' in enhancement_config:
                    for key, value in enhancement_config['clahe'].items():
                        enhancer_params[f'clahe_{key}'] = value

                # Add bilateral params with bilateral_ prefix
                if 'bilateral' in enhancement_config:
                    for key, value in enhancement_config['bilateral'].items():
                        enhancer_params[f'bilateral_{key}'] = value

                self.enhancer = ImageEnhancer(**enhancer_params)
                logger.info("✓ Image enhancer loaded")
            except Exception as e:
                logger.error(f"Failed to load image enhancer: {e}")
                logger.warning("Continuing without image enhancement")

        # Performance tracking
        self.enhancement_times = []
        self.classification_times = []
        self.last_perf_log_time = time.time()
        self.perf_log_interval = 30.0  # Log performance stats every 30 seconds

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

    def _set_detection_species_fields(
        self,
        detection: Dict[str, Any],
        species: Optional[str],
        confidence: float,
        category: str,
        taxonomic_level: Optional[str]
    ):
        """
        Helper method to set species classification fields on detection dict.

        Args:
            detection: Detection dict to update
            species: Species name (or None)
            confidence: Classification confidence
            category: Stage 2 category (bird, mammal, reptile)
            taxonomic_level: Taxonomic level (species, genus, family, order, class)
        """
        detection['species'] = species
        detection['species_confidence'] = float(confidence)
        detection['stage2_category'] = category
        detection['taxonomic_level'] = taxonomic_level

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

        # Apply image enhancement if configured
        if self.enhancer is not None:
            try:
                enhancement_start = time.time()
                crop = self.enhancer.enhance(crop)
                enhancement_time = (time.time() - enhancement_start) * 1000
                self.enhancement_times.append(enhancement_time)

                # Log enhancement performance periodically (time-based)
                current_time = time.time()
                if current_time - self.last_perf_log_time >= self.perf_log_interval:
                    if self.enhancement_times:
                        recent_count = min(len(self.enhancement_times), 10)
                        avg_enhancement = np.mean(self.enhancement_times[-recent_count:])
                        logger.info(f"Enhancement performance (last {recent_count}): {avg_enhancement:.1f}ms avg")
                    self.last_perf_log_time = current_time
            except Exception as e:
                logger.error(f"Enhancement failed, using original crop: {e}")

        # Run species classification
        classifier = self.species_classifiers[category]

        try:
            classification_start = time.time()
            # classifier.classify() returns List[Dict[str, Any]]
            results = classifier.classify(crop, top_k=1)
            classification_time = (time.time() - classification_start) * 1000
            self.classification_times.append(classification_time)

            if results and len(results) > 0:
                # Get top prediction
                top_result = results[0]
                species_name = top_result['species']
                confidence = top_result['confidence']
                taxonomic_level = top_result.get('taxonomic_level', 'species')

                # Filter out vague taxonomic level classifications (Option 2)
                # Only accept specific identifications (species, genus, family)
                if taxonomic_level in self.rejected_taxonomic_levels:
                    logger.info(f"Stage 2: Rejected vague classification: {species_name} ({taxonomic_level}, conf: {confidence:.2f})")
                    self._set_detection_species_fields(detection, None, 0.0, category, None)
                else:
                    # Accept specific identifications
                    self._set_detection_species_fields(detection, species_name, confidence, category, taxonomic_level)
                    logger.info(f"Stage 2: Classified as {species_name} ({taxonomic_level}, conf: {confidence:.2f})")
            else:
                # No results above threshold
                logger.info(f"Stage 2: No results above threshold for {class_name} (category: {category})")
                self._set_detection_species_fields(detection, None, 0.0, category, None)

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
        stats = {
            'stage2_enabled': self.enable_species_classification,
            'classifiers_loaded': list(self.species_classifiers.keys()),
            'num_classifiers': len(self.species_classifiers),
            'enhancement_enabled': self.enhancer is not None,
        }

        # Add enhancement statistics if available
        if self.enhancer is not None:
            stats['enhancement_method'] = self.enhancer.method
            if self.enhancement_times:
                stats['avg_enhancement_ms'] = float(np.mean(self.enhancement_times))
                stats['enhancement_count'] = len(self.enhancement_times)

        # Add classification statistics if available
        if self.classification_times:
            stats['avg_classification_ms'] = float(np.mean(self.classification_times))
            stats['classification_count'] = len(self.classification_times)

        return stats
