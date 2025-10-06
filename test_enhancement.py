#!/usr/bin/env python3
"""
Test script for image enhancement module.
Tests all three methods: none, clahe, realesrgan
"""

import sys
import time
import cv2
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from image_enhancement import ImageEnhancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image():
    """Create a simulated low-light, low-resolution wildlife image"""
    # Create a 128x128 dark image with low contrast
    img = np.random.randint(20, 60, (128, 128, 3), dtype=np.uint8)

    # Add a simulated animal shape (lighter blob)
    cv2.ellipse(img, (64, 64), (30, 20), 0, 0, 360, (80, 70, 60), -1)

    # Add some noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def test_enhancement_methods():
    """Test all enhancement methods"""
    logger.info("="*80)
    logger.info("Testing Image Enhancement Methods")
    logger.info("="*80)

    # Create test image
    logger.info("\n1. Creating test image (simulated low-light wildlife)...")
    test_img = create_test_image()
    logger.info(f"   Test image size: {test_img.shape}")

    # Save original
    cv2.imwrite("test_output_original.jpg", test_img)
    logger.info("   ✓ Saved: test_output_original.jpg")

    # Test Method 1: None (passthrough)
    logger.info("\n2. Testing method='none' (no enhancement)...")
    enhancer_none = ImageEnhancer(method="none")

    start = time.time()
    enhanced_none = enhancer_none.enhance(test_img)
    elapsed_none = (time.time() - start) * 1000

    cv2.imwrite("test_output_none.jpg", enhanced_none)
    logger.info(f"   Time: {elapsed_none:.1f}ms")
    logger.info(f"   Output size: {enhanced_none.shape}")
    logger.info("   ✓ Saved: test_output_none.jpg")

    # Test Method 2: CLAHE + bilateral
    logger.info("\n3. Testing method='clahe' (CLAHE + bilateral denoising)...")
    enhancer_clahe = ImageEnhancer(method="clahe")

    start = time.time()
    enhanced_clahe = enhancer_clahe.enhance(test_img)
    elapsed_clahe = (time.time() - start) * 1000

    cv2.imwrite("test_output_clahe.jpg", enhanced_clahe)
    logger.info(f"   Time: {elapsed_clahe:.1f}ms")
    logger.info(f"   Output size: {enhanced_clahe.shape}")
    logger.info("   ✓ Saved: test_output_clahe.jpg")

    # Test Method 3: Real-ESRGAN (if model available)
    model_path = "models/enhancement/RealESRGAN_x4plus.pth"
    if Path(model_path).exists():
        logger.info("\n4. Testing method='realesrgan' (Real-ESRGAN 4x + CLAHE + bilateral)...")

        try:
            enhancer_esrgan = ImageEnhancer(
                method="realesrgan",
                realesrgan_model_path=model_path,
                device="cuda:0"
            )

            start = time.time()
            enhanced_esrgan = enhancer_esrgan.enhance(test_img)
            elapsed_esrgan = (time.time() - start) * 1000

            cv2.imwrite("test_output_realesrgan.jpg", enhanced_esrgan)
            logger.info(f"   Time: {elapsed_esrgan:.1f}ms")
            logger.info(f"   Input size: {test_img.shape}")
            logger.info(f"   Output size: {enhanced_esrgan.shape}")
            logger.info(f"   Upscale: {test_img.shape[0]}x{test_img.shape[1]} → {enhanced_esrgan.shape[0]}x{enhanced_esrgan.shape[1]}")
            logger.info("   ✓ Saved: test_output_realesrgan.jpg")

        except Exception as e:
            logger.error(f"   ✗ Real-ESRGAN test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"\n4. Real-ESRGAN model not found at: {model_path}")
        logger.warning("   Download with:")
        logger.warning("   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P models/enhancement/")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Test Summary:")
    logger.info(f"  Method 'none':       {elapsed_none:.1f}ms")
    logger.info(f"  Method 'clahe':      {elapsed_clahe:.1f}ms")
    if Path(model_path).exists():
        logger.info(f"  Method 'realesrgan': {elapsed_esrgan:.1f}ms")
    logger.info("\nOutput images saved to current directory:")
    logger.info("  - test_output_original.jpg (original)")
    logger.info("  - test_output_none.jpg (no enhancement)")
    logger.info("  - test_output_clahe.jpg (CLAHE + bilateral)")
    if Path(model_path).exists():
        logger.info("  - test_output_realesrgan.jpg (Real-ESRGAN 4x upscaled)")
    logger.info("="*80)


if __name__ == "__main__":
    test_enhancement_methods()
