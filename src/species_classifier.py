"""
Species Classifier Module
Fine-grained species identification for detected animals.
"""

import cv2
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import timm

logging.basicConfig(level=logging.INFO)
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
        taxonomy_file: Optional[str] = None
    ):
        """
        Initialize species classifier.

        Args:
            model_name: Model architecture (timm model name)
            checkpoint_path: Path to pre-trained weights
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for classification
            taxonomy_file: Path to taxonomy mapping file
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.taxonomy = {}

        # Image preprocessing settings
        self.input_size = 224  # Standard for most models
        self.mean = [0.485, 0.456, 0.406]  # ImageNet normalization
        self.std = [0.229, 0.224, 0.225]

        # Load taxonomy if provided
        if taxonomy_file:
            self._load_taxonomy(taxonomy_file)

        logger.info(f"SpeciesClassifier initialized: {model_name}")

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

    def load_model(self, num_classes: int = 1000) -> bool:
        """
        Load the classification model.

        Args:
            num_classes: Number of species classes

        Returns:
            True if loaded successfully
        """
        try:
            # Create model using timm
            self.model = timm.create_model(
                self.model_name,
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

            logger.info(f"Species classifier loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for classification.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, (self.input_size, self.input_size))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.unsqueeze(0)

        return image.to(self.device)

    def classify(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify species from image crop.

        Args:
            image: Cropped image containing the animal
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

            # Format results
            results = []
            for prob, idx in zip(top_probs, top_indices):
                prob = prob.item()
                idx = idx.item()

                if prob >= self.confidence_threshold:
                    species_name = self.taxonomy.get(idx, f"species_{idx}")
                    results.append({
                        'species': species_name,
                        'confidence': prob,
                        'class_id': idx
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
