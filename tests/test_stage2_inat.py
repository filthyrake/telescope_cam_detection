#!/usr/bin/env python3
"""
Test Stage 2 iNaturalist species classification.
"""

import sys
import logging
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from two_stage_pipeline import TwoStageDetectionPipeline
from species_classifier import SpeciesClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_species_classifier():
    """Test iNaturalist classifier loading and inference."""
    logger.info("=" * 80)
    logger.info("TEST 1: iNaturalist Species Classifier")
    logger.info("=" * 80)

    try:
        # Initialize classifier
        classifier = SpeciesClassifier(
            model_name="timm/eva02_large_patch14_clip_336.merged2b_ft_inat21",
            checkpoint_path=None,
            device="cuda:0",
            confidence_threshold=0.3,
            taxonomy_file="models/inat2021_taxonomy_simple.json",
            input_size=336
        )

        # Load model
        logger.info("Loading iNaturalist model (this may take a minute)...")
        if not classifier.load_model(num_classes=10000):
            logger.error("❌ Failed to load classifier")
            return False

        logger.info("✅ Classifier loaded successfully")

        # Test with random image
        test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

        logger.info("Running test classification...")
        results = classifier.classify(test_image, top_k=5)

        if results:
            logger.info(f"✅ Classification successful, got {len(results)} predictions")
            logger.info("\nTop predictions:")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. {result['species']} (confidence: {result['confidence']:.3f})")
        else:
            logger.warning("⚠️  No predictions (expected for random noise)")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_two_stage_pipeline():
    """Test full two-stage pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Two-Stage Detection Pipeline")
    logger.info("=" * 80)

    try:
        # Initialize pipeline
        pipeline = TwoStageDetectionPipeline(
            yolo_model_path="yolov8x-worldv2.pt",
            device="cuda:0",
            enable_species_classification=True,
            stage2_confidence_threshold=0.3,
            stage1_confidence_threshold=0.25
        )

        # Set YOLO-World classes
        pipeline.detection_classes = [
            'person', 'coyote', 'rabbit', 'ground squirrel',
            'roadrunner', 'quail', 'hawk', 'raven', 'dove',
            'lizard', 'iguana', 'tortoise', 'snake',
            'deer', 'javelina', 'bobcat', 'fox'
        ]

        # Load Stage 1 (YOLO-World)
        logger.info("Loading YOLO-World detector...")
        if not pipeline.load_detector():
            logger.error("❌ Failed to load YOLO-World")
            return False
        logger.info("✅ YOLO-World loaded")

        # Add iNaturalist classifier (Stage 2)
        logger.info("Loading iNaturalist classifier...")
        inat_classifier = SpeciesClassifier(
            model_name="timm/eva02_large_patch14_clip_336.merged2b_ft_inat21",
            checkpoint_path=None,
            device="cuda:0",
            confidence_threshold=0.3,
            taxonomy_file="models/inat2021_taxonomy_simple.json",
            input_size=336
        )

        if not inat_classifier.load_model(num_classes=10000):
            logger.error("❌ Failed to load iNaturalist classifier")
            return False

        # Add to pipeline for all animal groups
        pipeline.add_species_classifier('bird', inat_classifier)
        pipeline.add_species_classifier('mammal', inat_classifier)
        pipeline.add_species_classifier('reptile', inat_classifier)
        logger.info("✅ iNaturalist classifier added to pipeline")

        # Test detection on random image
        logger.info("\nRunning test detection on random image...")
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        result = pipeline.detect(test_frame, run_stage2=True)

        logger.info(f"✅ Detection complete")
        logger.info(f"   Stage 1 detections: {result.get('stage1_count', 0)}")
        logger.info(f"   Stage 2 classifications: {result.get('stage2_count', 0)}")

        if result.get('detections'):
            logger.info("\nDetections:")
            for i, det in enumerate(result['detections'][:5], 1):
                stage1 = det['class_name']
                stage2 = det.get('species', 'N/A')
                conf2 = det.get('species_confidence', 0)
                logger.info(f"  {i}. Stage 1: {stage1} -> Stage 2: {stage2} ({conf2:.3f})")

        # Get pipeline stats
        stats = pipeline.get_stats()
        logger.info("\nPipeline Stats:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("Stage 2 iNaturalist Classification Tests")
    logger.info("=" * 80)

    # Check GPU
    import torch
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available, tests require GPU")
        return 1

    logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    logger.info("")

    # Run tests
    test_results = []

    test_results.append(("Species Classifier", test_species_classifier()))
    test_results.append(("Two-Stage Pipeline", test_two_stage_pipeline()))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}  {test_name}")

    all_passed = all(result for _, result in test_results)

    if all_passed:
        logger.info("\n🎉 All tests passed! Stage 2 is ready to use.")
        logger.info("\nNext step: Run 'python main.py' to test with live camera")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
