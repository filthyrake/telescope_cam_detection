# GPU OOM Graceful Degradation

**Issue:** [#125](https://github.com/filthyrake/telescope_cam_detection/issues/125)
**Status:** ✓ Implemented
**Related Issues:** #98 (GPU memory leak), #99 (Device handling), #110 (Model loading), #122 (Resource tracking)

## Overview

The GPU OOM (Out of Memory) graceful degradation system prevents system crashes when GPU memory is exhausted. Instead of crashing, the system automatically degrades performance to maintain stability.

## Features

### 1. Real-Time Memory Monitoring

The `MemoryManager` class continuously monitors GPU memory usage and detects pressure levels:

- **Normal** (< 75%): No action needed
- **High** (75-85%): Warning logged, cache cleared
- **Critical** (85-95%): Reduce batch size, clear cache
- **Extreme** (> 95%): Full degradation mode

### 2. Progressive Degradation Strategies

When memory pressure is detected, the system applies progressive degradation:

#### Level 1: High Pressure (75-85%)
- Clear GPU cache
- Log warning
- Monitor more frequently

#### Level 2: Critical Pressure (85-95%)
- Clear GPU cache
- Reduce batch size to 1
- Log warning with recovery suggestions

#### Level 3: Extreme Pressure (> 95%)
- Clear GPU cache
- Reduce input size to 640x640
- Set batch size to 1
- Increase degradation level counter
- **Note**: Stage 2 classification is NEVER disabled, even under extreme pressure

### 3. OOM Detection and Recovery

When OOM errors occur during inference:

1. **Catch Exception**: `torch.cuda.OutOfMemoryError` is caught
2. **Clear Memory**: GPU cache is cleared and synchronized
3. **Apply Degradation**: Progressive degradation strategies applied
4. **Retry Inference**: Inference retried with reduced settings
5. **Log Recovery**: Success/failure logged with metrics

After 3+ OOM events, CPU fallback is recommended.

### 4. CPU Fallback

If GPU is unavailable or OOM persists:

1. Model loading attempts GPU first
2. On OOM/failure, automatically falls back to CPU
3. System continues running (reduced performance)
4. CPU mode clearly indicated in logs and UI

### 5. Web UI Monitoring

Real-time memory monitoring in the web dashboard:

#### Stats Panel
- **GPU Memory**: Shows used/total memory and percentage
- **Memory Status**: Color-coded pressure indicator (Normal/High/Critical/Extreme)
- **Degradation**: Shows active degradation level and OOM event count

#### Alert Banner
- **High Memory (>80%)**: Warning banner with yellow styling
- **Critical Memory (>85%)**: Critical banner with red styling and pulse animation
- Shows current usage percentage and degradation status

## Architecture

### MemoryManager Class

**File:** `src/memory_manager.py`

**Responsibilities:**
- Monitor GPU memory usage
- Detect memory pressure levels
- Provide degradation recommendations
- Track OOM events and recoveries
- Implement hysteresis to prevent thrashing

**Key Methods:**
- `check_memory_pressure()`: Check current pressure level with rate limiting
- `reduce_memory_usage(level)`: Get degradation recommendations for pressure level
- `handle_oom_error()`: Handle OOM with recovery recommendations
- `get_memory_stats()`: Get comprehensive memory statistics

### InferenceEngine Integration

**File:** `src/inference_engine_yolox.py`

**Changes:**
- Added `MemoryManager` instance to each engine
- OOM detection in `_run_inference()` method
- CPU fallback in `load_model()` method
- `_apply_degradation()` method to apply recommendations
- Memory stats included in `get_stats()` output

### Web UI Integration

**Files:** `web/index.html`, `web/app.js`

**Changes:**
- Memory stats panel in single view
- Memory alert banner at top of page
- Periodic stats polling (every 5 seconds)
- Color-coded pressure indicators
- Pulse animation for critical alerts

## Configuration

No configuration changes required - graceful degradation is automatic.

### Thresholds (Advanced)

You can customize memory thresholds when creating a `MemoryManager` instance:

```python
from src.memory_manager import MemoryManager

mm = MemoryManager(
    device="cuda:0",
    warning_threshold=0.75,   # 75% - start warning
    high_threshold=0.85,      # 85% - high pressure
    critical_threshold=0.95,  # 95% - critical pressure
    check_interval=1.0,       # Check every 1 second
    hysteresis=0.05           # 5% hysteresis to prevent thrashing
)
```

## Degradation Behavior

### What Happens During Degradation?

1. **Input Size Reduction**
   - Original: e.g., 1920x1920 or 1280x1280
   - Degraded: 640x640
   - Impact: Faster inference, reduced GPU memory, but lower accuracy for small objects

2. **Batch Size Reduction**
   - Coordinator batch size reduced to 1
   - Impact: Lower GPU utilization, reduced throughput

3. **CPU Fallback** (last resort)
   - Model runs on CPU instead of GPU
   - Impact: Significantly slower inference (10-50x), but system remains operational

**Important**: Stage 2 classification (iNaturalist species identification) is NEVER disabled, even under extreme memory pressure. This ensures consistent wildlife identification quality.

### Recovery from Degradation

The system does NOT automatically recover to higher quality settings. This prevents oscillation between degraded/normal modes.

To manually reset degradation:
1. Restart the service: `sudo ./service.sh restart`
2. Or call `memory_manager.reset_degradation()` in code

## Monitoring

### Logs

Memory events are logged at appropriate levels:

```
INFO - GPU memory pressure changed: normal → high (usage: 78.3%, allocated: 15.6GB, reserved: 16.2GB, total: 20.0GB)
WARNING - Sustained GPU memory pressure: high (78.3% used, degradation level: 0)
ERROR - GPU OOM during inference for camera cam1: CUDA out of memory. Tried to allocate 512.00 MiB
WARNING - Reducing input size: (1280, 1280) → (640, 640)
WARNING - Disabling Stage 2 classification due to memory pressure
ERROR - Switching to CPU fallback due to persistent GPU OOM
INFO - GPU memory recovery #1 successful
```

### Web UI

Access the web UI at `http://<CAMERA_IP>:8000` to view:
- Real-time GPU memory usage gauge
- Memory pressure status (color-coded)
- Active degradation level and OOM count
- Alert banner for high/critical memory

### API Endpoint

Query `/stats` endpoint for programmatic monitoring:

```bash
curl http://localhost:8000/stats | jq '.memory'
```

Example response:
```json
{
  "cuda_available": true,
  "current_pressure": "high",
  "oom_events": 0,
  "recoveries": 0,
  "degradation_level": 0,
  "allocated_mb": 15872.5,
  "allocated_gb": 15.5,
  "reserved_mb": 16640.0,
  "reserved_gb": 16.3,
  "total_mb": 20480.0,
  "total_gb": 20.0,
  "usage_percent": 81.25,
  "allocated_percent": 77.5
}
```

## Testing

### Manual Testing

Run the OOM recovery test suite:

```bash
python tests/test_oom_recovery.py
```

This tests:
- MemoryManager functionality
- Artificial memory pressure creation
- InferenceEngine OOM recovery
- Memory stats API

### Simulating OOM

To artificially trigger OOM for testing:

```python
import torch
from src.memory_manager import MemoryManager

mm = MemoryManager()

# Allocate large tensor to trigger OOM
try:
    tensor = torch.zeros(100000000, device="cuda:0")
except torch.cuda.OutOfMemoryError:
    recommendations = mm.handle_oom_error()
    print(f"Recovery recommendations: {recommendations}")
```

### Monitoring During Testing

Watch logs in real-time:
```bash
./service.sh logs -f | grep -i "memory\|oom\|degradation"
```

Check GPU memory with `nvidia-smi`:
```bash
watch -n 1 nvidia-smi
```

## Performance Impact

### Normal Operation (No Degradation)

- **Overhead**: < 1ms per frame (memory check every 1 second)
- **Memory**: ~10MB for MemoryManager instance
- **CPU**: Negligible (< 0.1%)

### Degraded Operation

Impact varies by degradation level:

| Degradation Level | Input Size | Stage 2 | Inference Time | Accuracy Impact |
|------------------|------------|---------|----------------|-----------------|
| 0 (Normal)       | 1920x1920  | ✓       | 150-250ms      | Baseline        |
| 1 (Cache clear)  | 1920x1920  | ✓       | 150-250ms      | None            |
| 2 (Reduce batch) | 1920x1920  | ✓       | 160-270ms      | Minimal         |
| 3 (Reduce size)  | 640x640    | ✓       | 30-50ms        | Moderate        |
| 4 (CPU fallback) | 640x640    | ✓       | 500-2000ms     | Moderate        |

**Note**: Stage 2 classification is always enabled to maintain wildlife identification quality.

## Troubleshooting

### System Keeps Degrading

**Symptoms:**
- Degradation level keeps increasing
- Frequent OOM events
- Critical memory alerts

**Solutions:**
1. Reduce number of enabled cameras
2. Lower base input size in `config.yaml` (e.g., 1280x1280 → 640x640)
3. Disable Stage 2 classification: `use_two_stage: false`
4. Increase `buffer_size` to reduce frame buffering
5. Check for memory leaks with `nvidia-smi` over time

### Degradation Not Recovering

**Symptoms:**
- System stays in degraded mode even after memory pressure reduces
- Input size remains at 640x640

**Explanation:**
This is intentional to prevent oscillation. Manual restart required to restore full quality.

**Solutions:**
1. Restart service: `sudo ./service.sh restart`
2. Call `engine.memory_manager.reset_degradation()` in code

### CPU Fallback Too Slow

**Symptoms:**
- Inference time > 1 second
- System running on CPU instead of GPU

**Solutions:**
1. Fix underlying GPU OOM issue (reduce cameras, lower input size)
2. Restart service to attempt GPU reload
3. Increase GPU memory (hardware upgrade)
4. Run fewer cameras simultaneously

### Memory Leak Suspected

**Symptoms:**
- GPU memory slowly increases over time
- System eventually OOMs even with no activity

**Diagnosis:**
```bash
# Monitor memory over time
watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'

# Check for memory leaks in Python
python -c "
import torch
import gc
from src.memory_manager import MemoryManager
mm = MemoryManager()
for i in range(100):
    stats = mm.get_memory_stats()
    print(f'Iteration {i}: {stats[\"allocated_mb\"]:.1f}MB')
    gc.collect()
    torch.cuda.empty_cache()
"
```

**Solutions:**
- Ensure all tensors are properly deleted
- Use `del tensor` explicitly
- Call `torch.cuda.empty_cache()` periodically
- Report issue with reproduction steps

## Best Practices

### 1. Monitor Memory Proactively

- Check web UI regularly for memory usage trends
- Set up alerting for high memory usage (>80%)
- Monitor logs for OOM warnings

### 2. Size Input Appropriately

- Don't use larger input sizes than necessary
- Start with 640x640, increase only if accuracy insufficient
- Per-camera input size overrides for distant cameras

### 3. Stage 2 Always Enabled

- Stage 2 classification is never disabled during degradation
- System maintains wildlife identification quality even under memory pressure
- Input size reduction (Level 3) provides sufficient memory savings without sacrificing species ID

### 4. Test Under Load

- Run full system with all cameras before deployment
- Monitor memory usage for 24+ hours
- Simulate worst-case scenarios (all cameras detecting simultaneously)

### 5. Plan for Degradation

- Understand what happens when system degrades
- Acceptable quality reduction for reliability
- Test degraded performance meets minimum requirements

## Related Documentation

- [CONFIG_REFERENCE.md](../setup/CONFIG_REFERENCE.md) - Configuration options
- [PERFORMANCE.md](../PERFORMANCE.md) - Performance optimization guide
- [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) - General troubleshooting
- [API_REFERENCE.md](../api/API_REFERENCE.md) - Web API documentation

## References

- Issue #125: https://github.com/filthyrake/telescope_cam_detection/issues/125
- Issue #98: GPU memory leak in frame buffer
- Issue #99: Device string comparison
- Issue #110: Model loading failures
- Issue #122: Resource tracking
