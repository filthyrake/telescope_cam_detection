"""
Species Classifier Module
Fine-grained species identification for detected animals.
"""

import cv2
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import timm

logger = logging.getLogger(__name__)


class SpeciesClassifier:
    """
    Fine-grained species classification for wildlife.
    Uses pre-trained models (iNaturalist, etc.) to identify specific species.
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnet_b4_ns",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        taxonomy_file: Optional[str] = None,
        input_size: int = 224,
        use_hierarchical: bool = True,
        allowed_species: Optional[List[str]] = None,
        enable_geographic_filter: bool = False
    ):
        """
        Initialize species classifier.

        Args:
            model_name: Model architecture (timm model name)
            checkpoint_path: Path to pre-trained weights
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for classification
            taxonomy_file: Path to taxonomy mapping file
            input_size: Input image size (224, 336, etc.)
            use_hierarchical: Use hierarchical taxonomy fallback
            allowed_species: Whitelist of allowed species names (geographic filter)
            enable_geographic_filter: Whether to apply geographic filtering
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.taxonomy = {}
        self.use_hierarchical = use_hierarchical

        # Geographic filtering
        self.enable_geographic_filter = enable_geographic_filter
        self.allowed_species = set(allowed_species) if allowed_species else set()
        if self.enable_geographic_filter:
            logger.info(f"Geographic filter enabled with {len(self.allowed_species)} allowed species")

        # Image preprocessing settings
        self.input_size = input_size  # Configurable input size
        self.mean = [0.485, 0.456, 0.406]  # ImageNet normalization
        self.std = [0.229, 0.224, 0.225]

        # Cached GPU tensors for normalization (created after model is loaded)
        self._mean_tensor: Optional[torch.Tensor] = None
        self._std_tensor: Optional[torch.Tensor] = None

        # Hierarchical taxonomy thresholds
        self.hierarchy_thresholds = {
            'species': 0.5,    # Full species name (e.g., "Desert Cottontail")
            'genus': 0.4,      # Genus (e.g., "Sylvilagus")
            'family': 0.3,     # Family (e.g., "Leporidae")
            'order': 0.3,      # Order (e.g., "Lagomorpha") - raised from 0.2
            'class': 0.2,      # Class (e.g., "Mammalia") - raised from 0.1
        }

        # Load taxonomy if provided
        if taxonomy_file:
            self._load_taxonomy(taxonomy_file)

        logger.info(f"SpeciesClassifier initialized: {model_name}")
        if use_hierarchical:
            logger.info("  Hierarchical taxonomy fallback enabled")

    def _load_taxonomy(self, taxonomy_file: str):
        """
        Load taxonomy mapping (class_id -> species_name).

        Args:
            taxonomy_file: Path to taxonomy JSON or text file
        """
        import json

        try:
            taxonomy_path = Path(taxonomy_file)
            if taxonomy_path.exists():
                with open(taxonomy_path, 'r') as f:
                    if taxonomy_path.suffix == '.json':
                        self.taxonomy = json.load(f)
                    else:
                        # Text file: one species per line
                        self.taxonomy = {i: line.strip() for i, line in enumerate(f.readlines())}
                logger.info(f"Loaded taxonomy with {len(self.taxonomy)} species")
            else:
                logger.warning(f"Taxonomy file not found: {taxonomy_file}")
        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")

    def get_hierarchical_label(self, class_id: int, confidence: float) -> Tuple[str, str]:
        """
        Get appropriate taxonomic label based on confidence level.

        Args:
            class_id: Predicted class ID
            confidence: Prediction confidence

        Returns:
            Tuple of (label, taxonomic_level) e.g., ("Mammalia", "class")
        """
        if not self.use_hierarchical:
            # Just return species name
            tax_entry = self.taxonomy.get(str(class_id), {})
            if isinstance(tax_entry, str):
                return (tax_entry, "species")
            elif isinstance(tax_entry, dict):
                # Try common_name, then name, then fallback
                label = tax_entry.get('common_name') or tax_entry.get('name')
                if label:
                    return (label, "species")
                else:
                    return (f"species_{class_id}", "species")
            else:
                # Unexpected type, fallback
                return (f"species_{class_id}", "species")

        # Get taxonomy entry
        tax_entry = self.taxonomy.get(str(class_id), {})

        # Handle simple taxonomy (just strings)
        if isinstance(tax_entry, str):
            return (tax_entry, "species")

        # Hierarchical fallback based on confidence
        if confidence >= self.hierarchy_thresholds['species']:
            # High confidence: return common name or scientific name
            label = tax_entry.get('common_name') or tax_entry.get('name', f"species_{class_id}")
            return (label, "species")

        elif confidence >= self.hierarchy_thresholds['genus']:
            # Medium-high: return genus
            genus = tax_entry.get('genus', '')
            if genus:
                return (genus, "genus")

        elif confidence >= self.hierarchy_thresholds['family']:
            # Medium: return family
            family = tax_entry.get('family', '')
            if family:
                return (family, "family")

        elif confidence >= self.hierarchy_thresholds['order']:
            # Medium-low: return order
            order = tax_entry.get('order', '')
            if order:
                return (order, "order")

        elif confidence >= self.hierarchy_thresholds['class']:
            # Low: return class (e.g., "Mammalia", "Aves", "Reptilia")
            cls = tax_entry.get('class', '')
            if cls:
                return (cls, "class")

        # Too low confidence - return null
        return (None, None)

    def load_model(self, num_classes: int = 1000) -> bool:
        """
        Load the classification model.

        Args:
            num_classes: Number of species classes

        Returns:
            True if loaded successfully
        """
        try:
            # Handle HuggingFace hub models (format: hf-hub:org/model_name or hf_hub:org/model_name)
            model_name = self.model_name
            if self.model_name.startswith('timm/') or '/' in self.model_name:
                # For HuggingFace models, use hf-hub: prefix
                model_name = f"hf-hub:{self.model_name}"
                logger.info(f"Loading from HuggingFace Hub: {model_name}")

            # Create model using timm
            self.model = timm.create_model(
                model_name,
                pretrained=(self.checkpoint_path is None),
                num_classes=num_classes
            )

            # Load custom checkpoint if provided
            if self.checkpoint_path:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")

            self.model = self.model.to(self.device)
            self.model.eval()

            # Cache mean/std tensors on GPU for efficient preprocessing
            self._mean_tensor = torch.tensor(self.mean, device=self.device).view(3, 1, 1)
            self._std_tensor = torch.tensor(self.std, device=self.device).view(3, 1, 1)

            logger.info(f"Species classifier loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def preprocess(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image for classification (GPU-accelerated).

        Args:
            image: Input image (BGR format) - can be NumPy array or GPU tensor

        Returns:
            Preprocessed tensor on GPU
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            # NumPy array: convert to tensor (HxWxC)
            image = torch.from_numpy(image).to(self.device, non_blocking=True)
        elif not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(image)}")

        # Ensure tensor is on correct device
        if image.device != self.device:
            image = image.to(self.device, non_blocking=True)

        # Convert BGR to RGB (just swap channels: [H, W, 3] with channels 2,1,0 → 0,1,2)
        image = image[:, :, [2, 1, 0]]

        # Resize using GPU (HxWxC → CxHxW for interpolate, then back)
        if image.shape[0] != self.input_size or image.shape[1] != self.input_size:
            # torch.nn.functional.interpolate expects (N, C, H, W)
            image = image.permute(2, 0, 1).unsqueeze(0).float()  # HxWxC → 1xCxHxW
            image = torch.nn.functional.interpolate(
                image,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
            image = image.squeeze(0)  # 1xCxHxW → CxHxW
        else:
            # Already correct size, just permute
            image = image.permute(2, 0, 1).float()  # HxWxC → CxHxW

        # Normalize (GPU operation using cached tensors)
        image = image / 255.0
        # Use cached mean/std tensors (created once in load_model)
        if self._mean_tensor is None or self._std_tensor is None:
            # Fallback: create on-the-fly if not cached (shouldn't happen normally)
            mean = torch.tensor(self.mean, device=self.device).view(3, 1, 1)
            std = torch.tensor(self.std, device=self.device).view(3, 1, 1)
        else:
            mean = self._mean_tensor
            std = self._std_tensor
        image = (image - mean) / std

        # Add batch dimension
        image = image.unsqueeze(0)  # CxHxW → 1xCxHxW

        return image

    def classify(
        self,
        image: Union[np.ndarray, torch.Tensor],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify species from image crop (GPU-accelerated).

        Args:
            image: Cropped image containing the animal (NumPy array or GPU tensor)
            top_k: Return top K predictions

        Returns:
            List of predictions with species name and confidence
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        try:
            # Preprocess
            input_tensor = self.preprocess(image)

            # Inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)

            # Get top K predictions
            top_probs, top_indices = torch.topk(probs[0], top_k)

            # Format results with hierarchical taxonomy
            results = []
            for prob, idx in zip(top_probs, top_indices):
                prob = prob.item()
                idx = idx.item()

                # For hierarchical mode, allow lower threshold (class level = 0.1)
                # For non-hierarchical mode, enforce standard threshold
                min_threshold = 0.1 if self.use_hierarchical else self.confidence_threshold
                if prob < min_threshold:
                    continue

                # Get hierarchical label based on confidence
                label, tax_level = self.get_hierarchical_label(idx, prob)

                if label is not None:
                    # Apply geographic filter if enabled
                    if self.enable_geographic_filter and self.allowed_species:
                        # Check if species is in whitelist
                        if label not in self.allowed_species:
                            logger.debug(f"Filtered out non-local species: {label} (confidence: {prob:.3f})")
                            continue

                    results.append({
                        'species': label,
                        'confidence': prob,
                        'class_id': idx,
                        'taxonomic_level': tax_level  # e.g., "species", "genus", "family", "order", "class"
                    })

            return results

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return []

    def classify_batch(
        self,
        images: List[np.ndarray],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Classify multiple images in batch.

        Args:
            images: List of cropped images
            top_k: Return top K predictions per image

        Returns:
            List of predictions for each image
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        try:
            # Preprocess batch
            batch_tensors = [self.preprocess(img) for img in images]
            batch = torch.cat(batch_tensors, dim=0)

            # Inference
            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1)

            # Get top K for each image
            results = []
            for i in range(len(images)):
                top_probs, top_indices = torch.topk(probs[i], top_k)

                image_results = []
                for prob, idx in zip(top_probs, top_indices):
                    prob = prob.item()
                    idx = idx.item()

                    if prob >= self.confidence_threshold:
                        species_name = self.taxonomy.get(idx, f"species_{idx}")
                        image_results.append({
                            'species': species_name,
                            'confidence': prob,
                            'class_id': idx
                        })

                results.append(image_results)

            return results

        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            return []


class TaxonomySpecificClassifier:
    """
    Manages multiple specialized classifiers for different taxonomic groups.
    Routes detections to appropriate classifier (bird, mammal, reptile, etc.)
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize taxonomy-specific classifier manager.

        Args:
            device: Device to run on
        """
        self.device = device
        self.classifiers: Dict[str, SpeciesClassifier] = {}
        logger.info("TaxonomySpecificClassifier initialized")

    def add_classifier(
        self,
        taxonomy_group: str,
        classifier: SpeciesClassifier
    ):
        """
        Add a classifier for a specific taxonomic group.

        Args:
            taxonomy_group: Group name (e.g., 'bird', 'mammal', 'reptile')
            classifier: SpeciesClassifier instance
        """
        self.classifiers[taxonomy_group] = classifier
        logger.info(f"Added classifier for {taxonomy_group}")

    def classify(
        self,
        image: np.ndarray,
        taxonomy_group: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Classify using appropriate taxonomic classifier.

        Args:
            image: Cropped animal image
            taxonomy_group: Which classifier to use
            top_k: Number of top predictions

        Returns:
            List of species predictions
        """
        if taxonomy_group not in self.classifiers:
            logger.warning(f"No classifier for {taxonomy_group}")
            return []

        return self.classifiers[taxonomy_group].classify(image, top_k)

    def get_available_groups(self) -> List[str]:
        """Get list of available taxonomic groups."""
        return list(self.classifiers.keys())


# Desert wildlife species lists for your location
DESERT_BIRD_SPECIES = [
    "Gambel's Quail",
    "Greater Roadrunner",
    "Curve-billed Thrasher",
    "Cactus Wren",
    "Common Raven",
    "Red-tailed Hawk",
    "Harris's Hawk",
    "Cooper's Hawk",
    "White-winged Dove",
    "Mourning Dove",
    "Gila Woodpecker",
    "Gilded Flicker",
    "Verdin",
    "Black-throated Sparrow"
]

DESERT_MAMMAL_SPECIES = [
    "Coyote",
    "Desert Cottontail",
    "Black-tailed Jackrabbit",
    "Round-tailed Ground Squirrel",
    "Harris's Antelope Squirrel",
    "Bobcat",
    "Gray Fox",
    "Javelina (Collared Peccary)",
    "Mule Deer",
    "Desert Bighorn Sheep"
]

DESERT_REPTILE_SPECIES = [
    "Desert Iguana",
    "Desert Spiny Lizard",
    "Zebra-tailed Lizard",
    "Desert Horned Lizard",
    "Chuckwalla",
    "Western Diamondback Rattlesnake",
    "Gopher Snake",
    "Desert Tortoise",
    "Sonoran Desert Toad",
    "Couch's Spadefoot"
]


if __name__ == "__main__":
    # Test the species classifier
    logger.info("Testing SpeciesClassifier")

    # Create classifier
    classifier = SpeciesClassifier(
        model_name="tf_efficientnet_b4_ns",
        device="cuda" if torch.cuda.is_available() else "cpu",
        confidence_threshold=0.3
    )

    # Load model (will download pretrained ImageNet weights)
    if classifier.load_model(num_classes=1000):
        logger.info("✅ Model loaded successfully")

        # Create test image
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        # Test classification
        results = classifier.classify(test_image, top_k=5)
        logger.info(f"Classification results: {results}")
    else:
        logger.error("❌ Failed to load model")
