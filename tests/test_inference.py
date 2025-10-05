#!/usr/bin/env python3
"""
Test script to benchmark GPU inference performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import time
import numpy as np
import torch
from inference_engine import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_inference_performance(
    model_path: str = "yolov8n.pt",
    device: str = "cuda:0",
    num_iterations: int = 100
):
    """
    Benchmark inference performance.

    Args:
        model_path: Path to YOLOv8 model
        device: Device to run inference on
        num_iterations: Number of iterations for benchmark
    """
    logger.info("=" * 80)
    logger.info("GPU Inference Performance Test")
    logger.info("=" * 80)

    # Check CUDA availability
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Initialize inference engine
    logger.info(f"\nLoading model: {model_path}")
    engine = InferenceEngine(
        model_path=model_path,
        device=device,
        conf_threshold=0.5
    )

    if not engine.load_model():
        logger.error("Failed to load model")
        return False

    logger.info("Model loaded successfully")

    # Test different resolutions
    resolutions = [
        (640, 640),
        (1280, 720),
        (1920, 1080)
    ]

    for width, height in resolutions:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing resolution: {width}x{height}")
        logger.info(f"{'=' * 80}")

        # Create test frame
        test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Warmup
        logger.info("Warming up...")
        for _ in range(10):
            _ = engine.infer_single(test_frame)

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Benchmark
        logger.info(f"Running {num_iterations} iterations...")
        inference_times = []
        start_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0

        for i in range(num_iterations):
            start = time.time()
            detections = engine.infer_single(test_frame)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.time() - start
            inference_times.append(elapsed)

            if i % 20 == 0:
                logger.info(f"  Progress: {i}/{num_iterations}")

        end_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0

        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        p50_time = np.percentile(inference_times, 50)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)

        # Report results
        logger.info(f"\nResults for {width}x{height}:")
        logger.info(f"  - Average: {avg_time*1000:.2f}ms ({1/avg_time:.1f} FPS)")
        logger.info(f"  - Std dev: {std_time*1000:.2f}ms")
        logger.info(f"  - Min: {min_time*1000:.2f}ms")
        logger.info(f"  - Max: {max_time*1000:.2f}ms")
        logger.info(f"  - P50: {p50_time*1000:.2f}ms")
        logger.info(f"  - P95: {p95_time*1000:.2f}ms")
        logger.info(f"  - P99: {p99_time*1000:.2f}ms")

        if torch.cuda.is_available():
            memory_used = (end_memory - start_memory) / 1e6
            logger.info(f"  - GPU memory used: {memory_used:.1f} MB")

        # Performance assessment
        if avg_time < 0.015:  # < 15ms = excellent
            logger.info("  ✓ Performance: EXCELLENT (suitable for real-time)")
        elif avg_time < 0.033:  # < 33ms = good
            logger.info("  ✓ Performance: GOOD (real-time capable)")
        elif avg_time < 0.100:  # < 100ms = acceptable
            logger.info("  ⚠ Performance: ACCEPTABLE (may need optimization)")
        else:
            logger.warning("  ✗ Performance: POOR (optimization required)")

    logger.info("\n" + "=" * 80)
    logger.info("Inference performance test completed")
    logger.info("=" * 80)

    return True


def test_model_comparison():
    """Compare different YOLOv8 model sizes."""
    logger.info("\n" + "=" * 80)
    logger.info("YOLOv8 Model Comparison (640x640)")
    logger.info("=" * 80)

    models = [
        ("yolov8n.pt", "Nano"),
        ("yolov8s.pt", "Small"),
        ("yolov8m.pt", "Medium"),
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    results = []

    for model_path, name in models:
        logger.info(f"\nTesting {name} model...")

        engine = InferenceEngine(model_path=model_path, device=device)

        if not engine.load_model():
            logger.error(f"Failed to load {name} model")
            continue

        # Warmup
        for _ in range(10):
            _ = engine.infer_single(test_frame)

        # Benchmark
        times = []
        for _ in range(50):
            start = time.time()
            _ = engine.infer_single(test_frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time = np.mean(times)
        results.append((name, avg_time))

        logger.info(f"  - Avg inference time: {avg_time*1000:.2f}ms ({1/avg_time:.1f} FPS)")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Model Comparison Summary:")
    logger.info("=" * 80)
    for name, avg_time in results:
        logger.info(f"  {name:10s}: {avg_time*1000:6.2f}ms ({1/avg_time:5.1f} FPS)")


if __name__ == "__main__":
    # Run inference performance test
    test_inference_performance(
        model_path="yolov8n.pt",  # Start with nano model
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        num_iterations=100
    )

    # Compare models
    if torch.cuda.is_available():
        test_model_comparison()
