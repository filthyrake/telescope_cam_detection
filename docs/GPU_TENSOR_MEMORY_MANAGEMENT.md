# GPU Tensor Memory Management

This document describes the proper pattern for GPU tensor cleanup to prevent VRAM leaks in the telescope detection system.

## Problem

GPU tensors in PyTorch are not immediately freed when they go out of scope. Python's garbage collector (GC) eventually reclaims the memory, but this can be unpredictable and lead to VRAM exhaustion in long-running applications.

**Impact:**
- VRAM exhaustion over time
- Out-of-Memory (OOM) errors in long-running deployments
- Performance degradation
- System instability

## Solution

Explicitly free GPU tensors after converting them to NumPy arrays using a consistent pattern.

### Correct Pattern

```python
# Before (BAD - causes memory leaks)
numpy_array = tensor.cpu().numpy()

# After (GOOD - explicit cleanup)
tensor_temp = tensor
numpy_array = tensor_temp.cpu().detach().numpy()
del tensor_temp  # Explicitly free GPU tensor
torch.cuda.empty_cache()  # Optional: Aggressive cleanup for batch operations
```

### Why .detach()?

The `.detach()` method is crucial for tensors that are part of a computation graph:
- Removes the tensor from the autograd computation graph
- Prevents memory leaks from gradient tracking
- Safe to use on all tensors (no-op for tensors without gradients)

### Why del?

The `del` statement explicitly marks the tensor for garbage collection:
- Provides deterministic memory release
- Critical for GPU tensors in high-throughput scenarios
- PyTorch keeps GPU tensors alive until GC runs, which can be unpredictable

Reference: [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

### When to use torch.cuda.empty_cache()?

Use aggressive cache clearing in batch operations or after processing multiple tensors:
- Batch inference (`detect_batch()` methods)
- Frame processing loops
- After large tensor operations

**Note:** `torch.cuda.empty_cache()` only releases cached memory, not allocated memory. It should be used in conjunction with `del`, not as a replacement.

## Implementation

All tensor-to-numpy conversions in the codebase now follow this pattern:

### Files Modified

1. **yolox_detector.py** - Stage 1 detection
   - `detect()` - Single frame detection
   - `detect_batch()` - Batch detection with aggressive cleanup

2. **rtdetr_detector.py** - Alternative Stage 1 detector
   - `preprocess()` - Image preprocessing
   - `detect()` - Single frame detection
   - `detect_batch()` - Batch detection with try-finally cleanup

3. **two_stage_pipeline_yolox.py** - Stage 2 pipeline
   - `_compute_crop_hash()` - Crop hashing for caching

4. **visualization_utils.py** - Visualization functions
   - `draw_detections()` - Drawing bounding boxes
   - `draw_info_overlay()` - Drawing info text

5. **stream_capture_gpu_ffmpeg.py** - GPU-accelerated capture
   - `get_latest_frame_as_numpy()` - Frame retrieval

6. **two_stage_pipeline.py** - Legacy Stage 2 pipeline
   - `process_detections()` - Box coordinate extraction

7. **web_server.py** - Web interface
   - WebSocket frame serving

8. **snapshot_saver.py** - Reference implementation
   - `_convert_tensor_to_numpy()` - Helper method (pattern reference)

## Testing

A dedicated test validates the cleanup pattern:

```bash
python3 tests/test_gpu_tensor_cleanup.py
```

The test:
1. Validates tensor-to-numpy conversion with cleanup
2. Simulates batch processing (10 iterations)
3. Monitors GPU memory growth
4. Tests importance of `.detach()` for computation graph tensors

**Expected Results:**
- Memory growth < 10 MB for batch processing
- No sustained memory leaks
- Proper cleanup of computation graph tensors

## Best Practices

### DO:
✓ Always use `.cpu().detach().numpy()` for tensor conversions  
✓ Always `del` temporary tensor references immediately  
✓ Use `torch.cuda.empty_cache()` after batch operations  
✓ Be careful with stored tensor references (don't delete `self.latest_frame`, etc.)  
✓ Use try-finally blocks for cleanup in batch operations

### DON'T:
✗ Use `.cpu().numpy()` without `.detach()`  
✗ Rely on Python GC for GPU memory management  
✗ Delete stored tensor attributes (only delete temporary references)  
✗ Skip `del` statements thinking they're unnecessary  
✗ Use `torch.cuda.empty_cache()` as a replacement for `del`

## Special Cases

### Stored Tensors

When working with stored tensor attributes, only delete temporary references:

```python
# WRONG - deletes the stored reference
if isinstance(self.latest_frame, torch.Tensor):
    frame_np = self.latest_frame.cpu().detach().numpy()
    del self.latest_frame  # ❌ BAD - deletes stored attribute!
    return frame_np

# CORRECT - only delete temporary reference
if isinstance(self.latest_frame, torch.Tensor):
    frame_tensor = self.latest_frame  # Create temporary reference
    frame_np = frame_tensor.cpu().detach().numpy()
    del frame_tensor  # ✓ Good - only deletes temporary reference
    return frame_np
```

### Batch Operations with try-finally

For batch operations, use try-finally to ensure cleanup even on exceptions:

```python
try:
    # Batched inference
    with torch.no_grad():
        outputs = self.model(batch_tensor)
    
    # Process outputs...
    
finally:
    # Guaranteed cleanup
    del batch_tensor
    if 'outputs' in locals():
        del outputs
    torch.cuda.empty_cache()
```

## Monitoring

Monitor GPU memory usage to validate effectiveness:

```python
# Check current GPU memory
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
```

Add to health checks:
```python
stats = engine.get_stats()
gpu_memory_mb = stats.get('gpu_memory_allocated_mb', 0)
```

## Related Issues

- Issue #98: GPU memory leak in snapshot frame buffer (CLOSED)
- Issue #125: Graceful degradation for GPU OOM (CLOSED)
- This fix: Insufficient GPU tensor cleanup (HIGH SEVERITY)

## References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [PyTorch Best Practices - GPU Memory](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-vs-gpu)
- [Tensor.detach() Documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)
