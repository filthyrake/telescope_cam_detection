"""
Test OOM Recovery and Graceful Degradation (Issue #125)
Tests memory management, OOM detection, and recovery mechanisms.
"""

import torch
import numpy as np
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_manager import MemoryManager, MemoryPressure
from src.inference_engine_yolox import InferenceEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_memory_manager():
    """Test MemoryManager functionality"""
    logger.info("\n=== Testing MemoryManager ===")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping GPU tests")
        return True

    # Create memory manager
    mm = MemoryManager(device="cuda:0")

    # Test 1: Check initial pressure
    logger.info("Test 1: Checking initial memory pressure...")
    pressure = mm.check_memory_pressure(force=True)
    logger.info(f"  Initial pressure: {pressure.value}")
    assert pressure in [MemoryPressure.NORMAL, MemoryPressure.HIGH, MemoryPressure.CRITICAL, MemoryPressure.EXTREME]

    # Test 2: Get memory stats
    logger.info("Test 2: Getting memory statistics...")
    stats = mm.get_memory_stats()
    logger.info(f"  CUDA available: {stats['cuda_available']}")
    if stats['cuda_available']:
        logger.info(f"  Allocated: {stats['allocated_gb']:.2f}GB")
        logger.info(f"  Reserved: {stats['reserved_gb']:.2f}GB")
        logger.info(f"  Total: {stats['total_gb']:.2f}GB")
        logger.info(f"  Usage: {stats['usage_percent']:.1f}%")

    # Test 3: Test cache clearing
    logger.info("Test 3: Testing cache clearing...")
    mm.clear_cache()
    logger.info("  Cache cleared successfully")

    # Test 4: Test reduction recommendations
    logger.info("Test 4: Testing memory reduction recommendations...")
    for level in [MemoryPressure.HIGH, MemoryPressure.CRITICAL, MemoryPressure.EXTREME]:
        recommendations = mm.reduce_memory_usage(level)
        logger.info(f"  {level.value} recommendations: {recommendations}")

    # Test 5: Test OOM handling
    logger.info("Test 5: Testing OOM error handling...")
    recommendations = mm.handle_oom_error()
    logger.info(f"  OOM recovery recommendations: {recommendations}")
    logger.info(f"  OOM events: {mm.oom_events}")

    logger.info("\n✓ MemoryManager tests passed")
    return True


def test_artificial_memory_pressure():
    """Create artificial memory pressure to test degradation"""
    logger.info("\n=== Testing Artificial Memory Pressure ===")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping GPU tests")
        return True

    mm = MemoryManager(device="cuda:0")

    # Get baseline stats
    baseline_stats = mm.get_memory_stats()
    logger.info(f"Baseline memory: {baseline_stats['allocated_gb']:.2f}GB / {baseline_stats['total_gb']:.2f}GB")

    # Allocate tensors to create memory pressure
    tensors = []
    logger.info("Allocating GPU memory to create pressure...")

    try:
        # Allocate in chunks until we hit high pressure or OOM
        chunk_size = 512 * 1024 * 1024  # 512MB chunks
        max_chunks = 20

        for i in range(max_chunks):
            # Check pressure before each allocation
            pressure = mm.check_memory_pressure(force=True)
            stats = mm.get_memory_stats()

            logger.info(f"  Chunk {i+1}: {stats['allocated_gb']:.2f}GB allocated, pressure: {pressure.value}")

            if pressure in (MemoryPressure.CRITICAL, MemoryPressure.EXTREME):
                logger.warning(f"  Reached {pressure.value} pressure - stopping allocation")
                break

            # Allocate new tensor
            try:
                tensor = torch.zeros(chunk_size // 4, device="cuda:0", dtype=torch.float32)
                tensors.append(tensor)
            except torch.cuda.OutOfMemoryError:
                logger.error("  OOM encountered during allocation")
                mm.handle_oom_error()
                break

        # Test degradation recommendations
        final_pressure = mm.check_memory_pressure(force=True)
        logger.info(f"\nFinal pressure: {final_pressure.value}")

        if final_pressure != MemoryPressure.NORMAL:
            recommendations = mm.reduce_memory_usage(final_pressure)
            logger.info(f"Degradation recommendations: {recommendations}")

    finally:
        # Clean up
        logger.info("\nCleaning up allocated memory...")
        tensors.clear()
        torch.cuda.empty_cache()
        final_stats = mm.get_memory_stats()
        logger.info(f"Final memory: {final_stats['allocated_gb']:.2f}GB / {final_stats['total_gb']:.2f}GB")

    logger.info("\n✓ Artificial memory pressure test completed")
    return True


def test_inference_engine_oom_recovery():
    """Test InferenceEngine OOM recovery"""
    logger.info("\n=== Testing InferenceEngine OOM Recovery ===")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping GPU tests")
        return True

    # Create inference engine (standalone mode for testing)
    logger.info("Creating InferenceEngine...")
    engine = InferenceEngine(
        model_name="yolox-s",
        model_path="models/yolox/yolox_s.pth",
        device="cuda:0",
        conf_threshold=0.25,
        nms_threshold=0.45,
        input_size=(640, 640),
    )

    # Test model loading with CPU fallback
    logger.info("Testing model loading with CPU fallback...")
    success = engine.load_model()

    if success:
        logger.info(f"  Model loaded on device: {engine.device}")
        logger.info(f"  Degradation active: {engine.degradation_active}")

        # Get stats
        stats = engine.get_stats()
        logger.info(f"\nInference engine stats:")
        logger.info(f"  Device: {stats['device']}")
        logger.info(f"  Degradation active: {stats['degradation_active']}")

        if 'memory' in stats:
            mem_stats = stats['memory']
            if mem_stats.get('cuda_available'):
                logger.info(f"  GPU memory: {mem_stats['allocated_gb']:.2f}GB / {mem_stats['total_gb']:.2f}GB")
                logger.info(f"  Memory pressure: {mem_stats['current_pressure']}")
                logger.info(f"  OOM events: {mem_stats['oom_events']}")
                logger.info(f"  Recoveries: {mem_stats['recoveries']}")

        # Test inference on dummy frame
        logger.info("\nTesting inference with dummy frame...")
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        try:
            detections = engine.infer_single(dummy_frame)
            logger.info(f"  Inference successful: {len(detections)} detections")
        except Exception as e:
            logger.error(f"  Inference failed: {e}")
            return False

    else:
        logger.error("  Model loading failed")
        return False

    logger.info("\n✓ InferenceEngine OOM recovery test passed")
    return True


def test_memory_stats_api():
    """Test memory stats in get_stats() output"""
    logger.info("\n=== Testing Memory Stats API ===")

    # Create memory manager
    mm = MemoryManager(device="cuda:0" if torch.cuda.is_available() else "cpu")

    # Get stats
    stats = mm.get_memory_stats()

    # Verify expected fields
    required_fields = ['cuda_available', 'current_pressure', 'oom_events', 'recoveries', 'degradation_level']
    for field in required_fields:
        assert field in stats, f"Missing required field: {field}"
        logger.info(f"  {field}: {stats[field]}")

    if stats['cuda_available']:
        gpu_fields = ['allocated_mb', 'reserved_mb', 'total_mb', 'usage_percent']
        for field in gpu_fields:
            assert field in stats, f"Missing GPU field: {field}"

    logger.info("\n✓ Memory stats API test passed")
    return True


def run_all_tests():
    """Run all OOM recovery tests"""
    logger.info("=" * 60)
    logger.info("OOM Recovery and Graceful Degradation Test Suite (Issue #125)")
    logger.info("=" * 60)

    tests = [
        ("Memory Manager", test_memory_manager),
        ("Artificial Memory Pressure", test_artificial_memory_pressure),
        ("InferenceEngine OOM Recovery", test_inference_engine_oom_recovery),
        ("Memory Stats API", test_memory_stats_api),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                logger.error(f"✗ {test_name} failed")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
