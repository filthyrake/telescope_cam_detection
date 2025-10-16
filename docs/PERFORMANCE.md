# Performance Guide

Performance characteristics and optimization strategies for the Telescope Detection System.

## Current Performance

### Hardware
- **GPU**: NVIDIA A30 (24GB HBM2)
- **CPU**: AMD EPYC/Intel Xeon class
- **RAM**: 16GB+
- **Network**: Gigabit ethernet to cameras

### Metrics (Current System)

| Metric | Value |
|--------|-------|
| Inference time (YOLOX-S) | 11-21ms |
| Stage 2 classification (when triggered) | +20-30ms |
| Network latency (local) | 5-10ms |
| Total end-to-end (Stage 1 only) | 25-35ms |
| Total end-to-end (Stage 1+2) | 30-50ms |
| Sustained FPS | 25-30 FPS |
| RAM usage | ~500MB |
| VRAM usage | ~2GB per camera |

### Multi-Camera Performance

With 2 cameras:
- Total VRAM: ~4GB
- Total RAM: ~1GB
- FPS maintained: 25-30 per camera
- No GPU contention (A30 handles both easily)

## Model Comparison

### YOLOX Variants

| Model | Inference Time | Accuracy | VRAM | Use Case |
|-------|----------------|----------|------|----------|
| yolox-nano | 8-12ms | Good | ~1GB | Very fast, resource-constrained |
| yolox-tiny | 9-15ms | Good+ | ~1.5GB | Fast, balanced |
| **yolox-s** | **11-21ms** | **Better** | **~2GB** | **Current (recommended)** |
| yolox-m | 25-40ms | High | ~3GB | High accuracy needed |
| yolox-l | 50-80ms | Higher | ~4GB | Max accuracy |
| yolox-x | 80-120ms | Highest | ~6GB | Best possible accuracy |

### Input Size Impact

| Input Size | Inference Time | Small Object Detection |
|------------|----------------|------------------------|
| [640, 640] | ~8ms | Poor (misses distant wildlife) |
| [1280, 1280] | ~15ms | Good |
| **[1920, 1920]** | **~21ms** | **Excellent (current)** |
| [2560, 2560] | ~35ms | Excellent (overkill) |

**Recommendation**: Use [1920, 1920] for wildlife detection, can reduce to [640, 640] if only detecting larger/closer objects.

## Optimization Strategies

### For Lower GPU Load (New!)

**1. Enable empty frame filtering** (30-50% throughput improvement)
```yaml
performance:
  empty_frame_filter:
    enabled: true
    motion_threshold: 500   # Skip frames with <500px² motion
    min_motion_area: 100    # Filter small motion (shadows, leaves)
```

**How it works:**
- Uses lightweight frame differencing to detect motion
- Skips full YOLOX inference if no motion detected
- Expected 30-50% throughput improvement (70-90% of wildlife footage is empty)
- Adds only ~1-2ms per frame overhead

**2. Enable sparse detection** (3x GPU load reduction)
```yaml
performance:
  sparse_detection:
    enabled: true
    keyframe_interval: 3    # Run inference every 3rd frame
```

**How it works:**
- Runs full inference only on keyframes (every Nth frame)
- Reuses last detections on intermediate frames
- Expected 3x GPU load reduction with `keyframe_interval: 3`
- Trade-off: Detections appear "sticky" (30-60ms perceived lag)

**3. Combined optimization** (50-70% GPU load reduction!)
```yaml
performance:
  empty_frame_filter:
    enabled: true
  sparse_detection:
    enabled: true
    keyframe_interval: 3
```

**Benefits:**
- Dramatically reduces GPU utilization
- Enables more cameras on same GPU
- Reduces power consumption and heat
- Lower electricity costs for 24/7 operation

**When to use:**
- Static camera setups with occasional wildlife
- Multi-camera deployments where GPU is bottleneck
- Slow-moving wildlife (not birds in flight)
- 24/7 monitoring where most frames are empty

### For Lower Latency

**1. Switch to faster model**
```yaml
detection:
  model:
    name: "yolox-tiny"  # 9-15ms vs 11-21ms
```

**2. Reduce input size**
```yaml
detection:
  input_size: [640, 640]  # 8ms vs 21ms
```

**3. Use camera sub-stream**
```yaml
cameras:
  - stream: "sub"  # 640x480 vs 2560x1440
```

**4. Disable Stage 2**
```yaml
detection:
  use_two_stage: false  # Saves 20-30ms
```

**5. Disable performance optimizations if using sparse detection**
```yaml
performance:
  sparse_detection:
    enabled: false  # Lower latency, higher GPU load
```

### For Better Accuracy

**1. Use larger model**
```yaml
detection:
  model:
    name: "yolox-m"  # or yolox-l for even better
```

**2. Increase input size**
```yaml
detection:
  input_size: [2560, 2560]  # Better for tiny objects
```

**3. Lower confidence threshold**
```yaml
detection:
  conf_threshold: 0.10  # From 0.15, catches more detections
```

### For Lower Memory Usage

**Automatic Management**:
- System has **GPU OOM graceful degradation** (see [GPU OOM Graceful Degradation](features/OOM_GRACEFUL_DEGRADATION.md))
- Automatically detects memory pressure and applies progressive degradation
- Monitor real-time GPU memory in web UI
- Never disables Stage 2 classification

**Manual Tuning**:

**1. Use smaller model**
```yaml
detection:
  model:
    name: "yolox-nano"  # ~1GB VRAM vs ~2GB
```

**2. Reduce buffer sizes**
```yaml
cameras:
  - buffer_size: 1  # Already at minimum
```

**3. Disable snapshot saving**
```yaml
snapshots:
  enabled: false  # Saves disk I/O and RAM
```

## Benchmarking

### Test Inference Speed

```bash
python tests/test_inference.py
```

Sample output:
```
Testing YOLOX-S on 1920x1920 input:
Average inference time: 15.2ms
P50: 14.8ms
P95: 17.5ms
P99: 19.2ms
FPS: 65.8
```

### Test End-to-End Latency

```bash
python tests/test_latency.py
```

Sample output:
```
Component latencies:
- Frame capture: 3.2ms
- Inference: 15.2ms
- Post-processing: 2.1ms
- WebSocket send: 1.5ms
Total: 22.0ms (45.5 FPS)
```

### Test Camera Connection

```bash
python tests/test_camera_connection.py
```

## Bottleneck Analysis

### Typical Bottlenecks

1. **GPU inference** (11-21ms) - Largest component
   - Solution: Use smaller model or reduce input size

2. **Network latency** (5-10ms) - Second largest
   - Solution: Use wired connection, ensure same subnet

3. **Frame capture** (2-5ms) - Usually fine
   - Solution: Use sub-stream if needed

4. **Post-processing** (1-3ms) - Negligible
   - No optimization needed

### Identifying Your Bottleneck

Check the `/stats` endpoint while running:
```bash
curl http://localhost:8000/stats
```

Look at:
- `inference_time_ms` - If >30ms, reduce model size
- `total_latency_ms` - Should be <50ms
- `fps` - Should be >20

## GPU Utilization

### Check GPU Usage

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi
```

Look for:
- **Utilization**: Should be 30-50% with 2 cameras
- **Memory**: Should be ~4GB with 2 cameras
- **Temperature**: Should be <80°C
- **Power**: Typical 50-100W

### GPU Not Fully Utilized?

This is normal! The system is latency-optimized, not throughput-optimized. Low GPU utilization means:
- Fast response times
- Room for more cameras
- Efficient power usage

If you want higher GPU utilization (batch processing):
- Increase number of cameras
- Use larger model (yolox-x)
- Increase input size
- Enable Stage 2 classification

## Network Performance

### Minimize Network Latency

**1. Wired connection** (best)
- Cameras and server on same switch
- Expected: 1-2ms latency

**2. WiFi 6** (acceptable)
- Good signal strength needed
- Expected: 5-10ms latency

**3. WiFi 5 or older** (may struggle)
- Can cause frame drops
- Expected: 10-20ms latency

### Test Network

```bash
# Ping camera
ping 192.168.1.100

# Should see:
# 64 bytes from 192.168.1.100: icmp_seq=1 ttl=64 time=1.2 ms
```

## Future Optimizations

### TensorRT (Planned)

Converting YOLOX to TensorRT can reduce inference to 3-5ms:

```bash
# Will be added in future release
python scripts/convert_to_tensorrt.py --model yolox-s
```

Expected gains:
- Current: 11-21ms → TensorRT: 3-5ms
- **3-4x speedup**

**Note**: Not critical at current speed (11-21ms is excellent).

### Multi-Stream Batch Processing (Planned)

Process multiple camera frames in single GPU call:
- Current: 11-21ms per camera (sequential)
- Batched: 15-25ms for 2 cameras (parallel)
- **~30% speedup for multi-camera**

### FP16 Precision (Possible)

Already using mixed precision on compatible GPUs:
- Automatic on A30 (Ampere architecture)
- Negligible accuracy loss
- ~20% speedup already realized

## Recommended Configurations

### High-Performance (Current)
```yaml
detection:
  model:
    name: "yolox-s"
  input_size: [1920, 1920]
  conf_threshold: 0.15
```
**Result**: 25-35ms latency, excellent small object detection

### Balanced
```yaml
detection:
  model:
    name: "yolox-tiny"
  input_size: [1280, 1280]
  conf_threshold: 0.20
```
**Result**: 15-25ms latency, good detection

### Ultra-Fast
```yaml
detection:
  model:
    name: "yolox-nano"
  input_size: [640, 640]
  conf_threshold: 0.25
cameras:
  - stream: "sub"
```
**Result**: 8-15ms latency, basic detection

### Maximum Accuracy
```yaml
detection:
  model:
    name: "yolox-x"
  input_size: [2560, 2560]
  conf_threshold: 0.10
  use_two_stage: true
```
**Result**: 100-150ms latency, best possible detection

## Monitoring Performance

### Real-Time Stats

Access http://localhost:8000/stats for:
```json
{
  "fps": 28.5,
  "inference_time_ms": 15.2,
  "total_latency_ms": 32.1,
  "gpu_memory_mb": 2048,
  "detections_per_second": 12.3
}
```

### Long-Term Monitoring

Metrics are logged in `logs/performance.log`:
```bash
# View recent performance
tail -f logs/performance.log
```

### Alerts for Performance Issues

The system logs warnings when:
- Latency >100ms (sustained)
- FPS <15 (sustained)
- GPU memory pressure detected (see [OOM Graceful Degradation](features/OOM_GRACEFUL_DEGRADATION.md))
- Frame drops detected

GPU memory monitoring:
- Real-time memory gauge in web UI
- Alert banners for High/Critical/Extreme pressure levels
- Automatic degradation and recovery

Check logs: `./service.sh logs -f`
