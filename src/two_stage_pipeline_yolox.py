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
import hashlib
import itertools
from collections import OrderedDict, deque
from typing import Dict, Any, List, Optional, Tuple, Union

from species_classifier import SpeciesClassifier
from src.coco_constants import CLASS_ID_TO_CATEGORY
from src.image_enhancement import ImageEnhancer
from src.species_activity_patterns import is_species_likely_active, get_species_activity
from src.bbox_utils import ensure_valid_bbox

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
        crop_padding_percent: int = 20,
        min_crop_size: int = 64,
        time_of_day_top_k: int = 5,
        time_of_day_penalty: float = 0.3,
        enhancement_cache_size: int = 100,
    ):
        """
        Initialize two-stage pipeline.

        Args:
            enable_species_classification: Whether to run Stage 2
            stage2_confidence_threshold: Min confidence for species classification
            device: Device to run on
            enhancement_config: Image enhancement config (e.g., {"method": "realesrgan", ...})
            rejected_taxonomic_levels: Taxonomic levels to reject (e.g., ['order', 'class'])
            crop_padding_percent: Percent to expand bbox for better context (e.g., 20 = 20% padding)
            min_crop_size: Skip Stage 2 if crop smaller than this (e.g., 64 = 64x64 pixels minimum)
            time_of_day_top_k: Number of species to consider for time-of-day re-ranking (default: 5)
            time_of_day_penalty: Confidence penalty for unlikely species (default: 0.3 = 70% reduction)
            enhancement_cache_size: Max number of enhanced crops to cache (default: 100)
        """
        self.enable_species_classification = enable_species_classification
        self.stage2_confidence_threshold = stage2_confidence_threshold
        self.device = device

        # Configurable rejected taxonomic levels (default: order and class are too vague)
        self.rejected_taxonomic_levels = rejected_taxonomic_levels or ['order', 'class']

        # Preprocessing configuration
        self.crop_padding_percent = crop_padding_percent
        self.min_crop_size = min_crop_size

        # Time-of-day filtering configuration
        self.time_of_day_top_k = time_of_day_top_k
        self.time_of_day_penalty = time_of_day_penalty

        # Species classifiers (will be added via add_species_classifier)
        self.species_classifiers: Dict[str, SpeciesClassifier] = {}

        # Use shared COCO class mapping for Stage 2 routing
        self.class_id_to_category = CLASS_ID_TO_CATEGORY

        # Enhancement LRU cache (for Real-ESRGAN caching)
        self.enhancement_cache_size = enhancement_cache_size
        self.enhancement_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

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
                logger.info(f"✓ Image enhancer loaded (cache size: {self.enhancement_cache_size})")
            except Exception as e:
                logger.error(f"Failed to load image enhancer (method={enhancement_config.get('method', 'none')}): {e}", exc_info=True)
                logger.warning("Continuing without image enhancement")

        # Performance tracking (bounded deque to prevent memory growth)
        self.enhancement_times = deque(maxlen=1000)
        self.classification_times = deque(maxlen=1000)
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

    def _compute_crop_hash(self, crop: Union[np.ndarray, torch.Tensor]) -> str:
        """
        Compute hash of crop for enhancement cache key.
        Uses exact hash of 8x8 downsampled grayscale thumbnail for cache lookups.

        Note: This is an exact hash (MD5) of a downsampled image, not a true
        perceptual hash. Only crops that downsample to identical 8x8 grayscale
        values will share the same hash; this is not a true perceptual hash.

        Args:
            crop: Crop image (BGR format) - can be NumPy array or GPU tensor

        Returns:
            Hash string for cache lookup
        """
        # Convert to NumPy if GPU tensor
        if isinstance(crop, torch.Tensor):
            # Detach from computation graph and move to CPU
            crop_np = crop.detach().cpu().numpy()

            # Convert CHW to HWC for OpenCV (if needed)
            if crop_np.ndim == 3 and crop_np.shape[0] in [1, 3]:  # CHW format
                crop_np = crop_np.transpose(1, 2, 0)  # CHW → HWC
        else:
            crop_np = crop

        # Downsample to 8x8 for similarity matching
        small = cv2.resize(crop_np, (8, 8), interpolation=cv2.INTER_AREA)

        # Convert to grayscale for hashing
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small

        # Compute MD5 hash of downsampled thumbnail
        return hashlib.md5(gray.tobytes()).hexdigest()

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
        frame: Union[np.ndarray, torch.Tensor],
        detection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Stage 2 classification on a detection (GPU-accelerated).

        Args:
            frame: Full frame image (BGR) - can be NumPy array or GPU tensor
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

        # Validate and normalize bbox (Issue #117)
        validated_bbox = ensure_valid_bbox(bbox)
        if validated_bbox != bbox:
            detection['bbox'] = validated_bbox
            bbox = validated_bbox
        else:
            bbox = validated_bbox

        # Determine classifier category
        category = self.class_id_to_category.get(class_id)

        # Check if we have a classifier for this category
        if category not in self.species_classifiers:
            # No classifier for this category, return as-is
            detection['species'] = None
            detection['species_confidence'] = 0.0
            return detection

        # Extract crop with padding
        x1 = int(bbox['x1'])
        y1 = int(bbox['y1'])
        x2 = int(bbox['x2'])
        y2 = int(bbox['y2'])

        # Calculate crop dimensions before padding
        crop_w = x2 - x1
        crop_h = y2 - y1

        # Check minimum size BEFORE padding (skip Stage 2 for tiny detections)
        # Note: ensure_valid_bbox() already enforces min_size=1, so crop dimensions are always >= 1
        if crop_w < self.min_crop_size or crop_h < self.min_crop_size:
            logger.debug(f"Skipping Stage 2: crop too small ({crop_w}x{crop_h} < {self.min_crop_size}x{self.min_crop_size})")
            self._set_detection_species_fields(detection, None, 0.0, category, None)
            return detection

        # Add padding for better context
        padding_x = int(crop_w * self.crop_padding_percent / 100)
        padding_y = int(crop_h * self.crop_padding_percent / 100)

        x1_padded = x1 - padding_x
        y1_padded = y1 - padding_y
        x2_padded = x2 + padding_x
        y2_padded = y2 + padding_y

        # Ensure valid crop within frame bounds
        if isinstance(frame, torch.Tensor):
            h, w = frame.shape[0], frame.shape[1]
        else:
            h, w = frame.shape[:2]

        x1_padded = max(0, min(x1_padded, w - 1))
        y1_padded = max(0, min(y1_padded, h - 1))
        x2_padded = max(0, min(x2_padded, w))
        y2_padded = max(0, min(y2_padded, h))

        if x2_padded <= x1_padded or y2_padded <= y1_padded:
            # Invalid crop
            logger.debug(f"Invalid crop coordinates: x[{x1_padded}:{x2_padded}] y[{y1_padded}:{y2_padded}]")
            self._set_detection_species_fields(detection, None, 0.0, category, None)
            return detection

        # Crop frame (GPU tensor slicing or NumPy slicing)
        try:
            crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        except (IndexError, RuntimeError) as e:
            logger.warning(f"Failed to crop frame: {e}")
            self._set_detection_species_fields(detection, None, 0.0, category, None)
            return detection

        # Check crop validity (empty tensor/array check)
        if isinstance(crop, torch.Tensor):
            if crop.numel() == 0:
                logger.debug(f"Empty crop tensor for detection at bbox: {bbox}")
                self._set_detection_species_fields(detection, None, 0.0, category, None)
                return detection
        elif isinstance(crop, np.ndarray):
            if crop.size == 0:
                logger.debug(f"Empty crop array for detection at bbox: {bbox}")
                self._set_detection_species_fields(detection, None, 0.0, category, None)
                return detection
        else:
            # Unknown type
            logger.warning(f"Unknown crop type: {type(crop)}, skipping Stage 2")
            self._set_detection_species_fields(detection, None, 0.0, category, None)
            return detection

        # Apply image enhancement if configured (with LRU caching)
        if self.enhancer is not None:
            try:
                # Skip caching if cache size is 0
                if self.enhancement_cache_size <= 0:
                    # No caching - always enhance
                    enhancement_start = time.time()
                    crop = self.enhancer.enhance(crop)
                    enhancement_time = (time.time() - enhancement_start) * 1000
                    self.enhancement_times.append(enhancement_time)
                    logger.debug(f"Enhancement (no cache): {enhancement_time:.1f}ms")
                else:
                    # Compute perceptual hash for cache lookup
                    crop_hash = self._compute_crop_hash(crop)

                    # Check cache for existing enhanced version
                    if crop_hash in self.enhancement_cache:
                        # Cache hit! Use cached enhanced crop
                        crop = self.enhancement_cache[crop_hash]
                        self.cache_hits += 1

                        # Move to end (mark as recently used)
                        self.enhancement_cache.move_to_end(crop_hash)

                        logger.debug(f"Enhancement cache hit (total hits: {self.cache_hits}, misses: {self.cache_misses})")
                    else:
                        # Cache miss - enhance and store
                        enhancement_start = time.time()
                        enhanced_crop = self.enhancer.enhance(crop)
                        enhancement_time = (time.time() - enhancement_start) * 1000
                        self.enhancement_times.append(enhancement_time)
                        self.cache_misses += 1

                        # Store in cache with LRU eviction
                        if len(self.enhancement_cache) >= self.enhancement_cache_size:
                            # Remove least-recently-used entry (LRU eviction)
                            self.enhancement_cache.popitem(last=False)

                        self.enhancement_cache[crop_hash] = enhanced_crop
                        crop = enhanced_crop

                        logger.debug(f"Enhancement cache miss: {enhancement_time:.1f}ms (cache: {len(self.enhancement_cache)}/{self.enhancement_cache_size})")

                # Log enhancement performance periodically (time-based)
                current_time = time.time()
                if current_time - self.last_perf_log_time >= self.perf_log_interval:
                    if self.enhancement_times or self.cache_hits > 0:
                        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
                        logger.info(f"Enhancement cache: {cache_hit_rate:.1f}% hit rate ({self.cache_hits} hits, {self.cache_misses} misses)")

                        if self.enhancement_times:
                            recent_count = min(len(self.enhancement_times), 10)
                            # Use islice for efficient access to last N elements without converting entire deque
                            start_idx = max(0, len(self.enhancement_times) - recent_count)
                            recent_times = list(itertools.islice(self.enhancement_times, start_idx, len(self.enhancement_times)))
                            avg_enhancement = np.mean(recent_times)
                            logger.info(f"Enhancement performance (last {recent_count} enhancements): {avg_enhancement:.1f}ms avg")
                    self.last_perf_log_time = current_time
            except Exception as e:
                logger.error(f"Enhancement failed for {class_name} detection (bbox={bbox}), using original crop: {e}", exc_info=True)

        # Run species classification
        classifier = self.species_classifiers[category]

        # Get time-of-day context if available (from time-of-day filter)
        time_of_day = detection.get('time_of_day')
        time_alternatives = detection.get('time_of_day_alternatives', [])

        try:
            classification_start = time.time()
            # classifier.classify() returns List[Dict[str, Any]]
            # Request more results if we need to filter by time of day
            top_k = self.time_of_day_top_k if time_of_day else 1
            results = classifier.classify(crop, top_k=top_k)
            classification_time = (time.time() - classification_start) * 1000
            self.classification_times.append(classification_time)

            if results and len(results) > 0:
                # Filter results by time-of-day activity if we have the context
                if time_of_day:
                    # Re-rank results based on activity patterns
                    filtered_results = []
                    for result in results:
                        species_name = result['species']
                        confidence = result['confidence']
                        is_active = is_species_likely_active(species_name, time_of_day)

                        # Boost confidence for species likely active at this time
                        # Reduce confidence for species unlikely to be active
                        if is_active:
                            result['activity_boosted'] = True
                            result['confidence_original'] = confidence
                            # Keep confidence as-is or slightly boost it
                        else:
                            result['activity_boosted'] = False
                            result['confidence_original'] = confidence
                            # Penalize unlikely species using configured penalty
                            result['confidence'] = confidence * self.time_of_day_penalty

                        filtered_results.append(result)

                    # Re-sort by adjusted confidence
                    filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
                    results = filtered_results

                    if time_alternatives:
                        logger.debug(f"Stage 2 filtering by time-of-day: {time_of_day} (alternatives: {time_alternatives})")

                # Get top prediction after filtering
                top_result = results[0]
                species_name = top_result['species']
                confidence = top_result['confidence']
                taxonomic_level = top_result.get('taxonomic_level', 'species')

                # Log if confidence was adjusted
                if top_result.get('confidence_original'):
                    original_conf = top_result['confidence_original']
                    if abs(original_conf - confidence) > 0.01:
                        activity = get_species_activity(species_name)
                        logger.debug(f"Time-of-day adjusted: {species_name} @ {time_of_day} ({activity.value}) {original_conf:.2f} → {confidence:.2f}")

                # Filter out vague taxonomic level classifications (Option 2)
                # Only accept specific identifications (species, genus, family)
                if taxonomic_level in self.rejected_taxonomic_levels:
                    logger.debug(f"Rejected vague classification: {species_name} ({taxonomic_level}, conf: {confidence:.2f})")
                    self._set_detection_species_fields(detection, None, 0.0, category, None)
                else:
                    # Accept specific identifications
                    self._set_detection_species_fields(detection, species_name, confidence, category, taxonomic_level)
                    logger.debug(f"Classified as {species_name} ({taxonomic_level}, conf: {confidence:.2f})")
            else:
                # No results above threshold
                self._set_detection_species_fields(detection, None, 0.0, category, None)

        except Exception as e:
            logger.error(f"Species classification failed for {class_name} (category={category}, bbox={bbox}): {e}", exc_info=True)
            detection['species'] = None
            detection['species_confidence'] = 0.0
            detection['taxonomic_level'] = None

        return detection

    def process_detections(
        self,
        frame: Union[np.ndarray, torch.Tensor],
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process all detections through Stage 2 (GPU-accelerated).

        Args:
            frame: Full frame image (BGR) - can be NumPy array or GPU tensor
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

            # Add cache statistics
            total_requests = self.cache_hits + self.cache_misses
            if total_requests > 0:
                stats['enhancement_cache_hit_rate'] = float(self.cache_hits / total_requests)
                stats['enhancement_cache_hits'] = self.cache_hits
                stats['enhancement_cache_misses'] = self.cache_misses
                stats['enhancement_cache_size'] = len(self.enhancement_cache)
                stats['enhancement_cache_max_size'] = self.enhancement_cache_size

        # Add classification statistics if available
        if self.classification_times:
            stats['avg_classification_ms'] = float(np.mean(self.classification_times))
            stats['classification_count'] = len(self.classification_times)

        return stats
