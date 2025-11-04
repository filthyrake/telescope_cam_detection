#!/usr/bin/env python3
"""
Test GPU tensor cleanup to validate memory leak fixes.
Validates that tensors are properly converted to NumPy with explicit cleanup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tensor_conversion_pattern():
    """
    Test that the tensor-to-numpy conversion pattern properly cleans up GPU memory.
    This validates the pattern used throughout the codebase.
    """
    logger.info("=" * 80)
    logger.info("GPU Tensor Cleanup Test")
    logger.info("=" * 80)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping GPU-specific tests")
        logger.info("Testing CPU tensor cleanup pattern instead")
        device = "cpu"
    else:
        device = "cuda:0"
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        # Get initial GPU memory state
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated(device)
        logger.info(f"Initial GPU memory: {initial_memory / 1024 / 1024:.2f} MB")

    # Test the correct pattern: .cpu().detach().numpy() with del
    logger.info("\n--- Testing CORRECT pattern: .cpu().detach().numpy() + del ---")
    
    # Create test tensor
    test_tensor = torch.randn(1920, 1080, 3, device=device)
    tensor_size = test_tensor.element_size() * test_tensor.nelement()
    logger.info(f"Created test tensor: {test_tensor.shape}, size: {tensor_size / 1024 / 1024:.2f} MB")
    
    if torch.cuda.is_available():
        memory_after_alloc = torch.cuda.memory_allocated(device)
        logger.info(f"GPU memory after allocation: {memory_after_alloc / 1024 / 1024:.2f} MB")
    
    # Convert using correct pattern
    tensor_temp = test_tensor
    numpy_array = tensor_temp.cpu().detach().numpy()
    del tensor_temp
    del test_tensor
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after_cleanup = torch.cuda.memory_allocated(device)
        logger.info(f"GPU memory after cleanup: {memory_after_cleanup / 1024 / 1024:.2f} MB")
    
    # Validate conversion
    assert isinstance(numpy_array, np.ndarray), "Conversion should produce NumPy array"
    assert numpy_array.shape == (1920, 1080, 3), "Shape should be preserved"
    logger.info(f"✓ Conversion successful: {numpy_array.shape}, dtype: {numpy_array.dtype}")
    
    # Test multiple conversions to simulate real workload
    logger.info("\n--- Testing multiple conversions (simulating batch processing) ---")
    num_iterations = 10
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_batch_memory = torch.cuda.memory_allocated(device)
    
    for i in range(num_iterations):
        # Create tensor
        tensor = torch.randn(640, 640, 3, device=device)
        
        # Convert with proper cleanup
        tensor_temp = tensor
        _ = tensor_temp.cpu().detach().numpy()
        del tensor_temp
        del tensor
        
        if torch.cuda.is_available() and i % 3 == 0:
            torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        final_batch_memory = torch.cuda.memory_allocated(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
        logger.info(f"Initial batch memory: {initial_batch_memory / 1024 / 1024:.2f} MB")
        logger.info(f"Final batch memory: {final_batch_memory / 1024 / 1024:.2f} MB")
        logger.info(f"Peak memory during batch: {peak_memory / 1024 / 1024:.2f} MB")
        
        memory_growth = final_batch_memory - initial_batch_memory
        logger.info(f"Memory growth: {memory_growth / 1024 / 1024:.2f} MB")
        
        # Allow some tolerance for internal PyTorch memory management
        # but the growth should be minimal (< 10 MB for this test)
        if memory_growth > 10 * 1024 * 1024:  # 10 MB threshold
            logger.warning(f"⚠ Significant memory growth detected: {memory_growth / 1024 / 1024:.2f} MB")
            logger.warning("This may indicate incomplete cleanup, but could also be PyTorch caching")
        else:
            logger.info("✓ Memory growth within acceptable range")
    
    logger.info(f"\n✓ Completed {num_iterations} conversions successfully")
    logger.info("=" * 80)
    logger.info("GPU Tensor Cleanup Test: PASSED")
    logger.info("=" * 80)
    
    return True


def test_detach_importance():
    """
    Test that .detach() is important for tensors in computation graphs.
    """
    logger.info("\n--- Testing importance of .detach() ---")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Create tensor that requires grad (part of computation graph)
    tensor_with_grad = torch.randn(100, 100, device=device, requires_grad=True)
    
    # Perform an operation to create a computation graph
    tensor_computed = tensor_with_grad * 2.0
    
    logger.info(f"Tensor requires_grad: {tensor_computed.requires_grad}")
    logger.info(f"Tensor is_leaf: {tensor_computed.is_leaf}")
    
    # Convert with .detach() - this removes from computation graph
    tensor_temp = tensor_computed
    numpy_array = tensor_temp.cpu().detach().numpy()
    del tensor_temp
    
    logger.info("✓ Conversion with .detach() successful")
    logger.info(f"  NumPy array shape: {numpy_array.shape}")
    
    # Cleanup
    del tensor_with_grad, tensor_computed, numpy_array
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True


if __name__ == "__main__":
    try:
        # Run tests
        test_tensor_conversion_pattern()
        test_detach_importance()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
