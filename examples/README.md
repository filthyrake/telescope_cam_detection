# Example Configurations

This directory contains example configuration files for different use cases and hardware setups.

## Available Examples

### 1. Single Camera (`config_single_camera.yaml`)

**Best for:** Simple setup with one camera

**Features:**
- Single camera configuration
- Balanced performance
- YOLOX-S model (11-21ms inference)
- Standard 640x640 resolution
- No Stage 2 classification (for simplicity)

**Performance:**
- Latency: ~25-35ms
- FPS: 25-30
- VRAM: ~2GB

**Use when:**
- First time setup
- Testing the system
- Simple monitoring needs

---

### 2. Multi-Camera (`config_multi_camera.yaml`)

**Best for:** Multiple cameras with different views

**Features:**
- 2+ camera configuration
- Per-camera detection overrides
- Stage 2 species classification enabled
- Different settings for each camera (overview vs ground-level)

**Performance:**
- Latency: ~30-50ms per camera
- FPS: 20-25 per camera
- VRAM: ~4GB total (2 cameras)

**Use when:**
- Monitoring from multiple angles
- Need per-camera customization
- Want species identification

---

### 3. High Performance (`config_high_performance.yaml`)

**Best for:** Maximum accuracy, powerful GPU

**Features:**
- YOLOX-X model (highest accuracy)
- 1920x1920 resolution
- Very low confidence thresholds (catch everything)
- Stage 2 with image enhancement (Real-ESRGAN)
- Video clip saving

**Performance:**
- Latency: ~150-250ms
- FPS: 4-7
- VRAM: ~8-10GB

**Hardware Requirements:**
- RTX 3090, A30, A40, A100, or better
- 16GB+ VRAM recommended

**Use when:**
- Accuracy is critical
- Small/distant objects must be detected
- GPU power is not a concern
- Recording wildlife for research

---

### 4. Low Latency (`config_low_latency.yaml`)

**Best for:** Maximum speed, real-time response

**Features:**
- YOLOX-Nano model (fastest)
- 416x416 resolution
- Sub-stream from camera
- No Stage 2 classification
- Minimal queuing

**Performance:**
- Latency: ~10-20ms
- FPS: 50-100
- VRAM: ~500MB-1GB

**Use when:**
- Latency is critical
- Real-time collision detection needed
- Running on older/weaker GPU
- Fast response more important than accuracy

**Trade-offs:**
- Lower accuracy on small objects
- No species classification
- May miss distant wildlife

---

## How to Use

### 1. Copy Example to Config

```bash
# Single camera
cp examples/config_single_camera.yaml config/config.yaml

# Multi-camera
cp examples/config_multi_camera.yaml config/config.yaml

# High performance
cp examples/config_high_performance.yaml config/config.yaml

# Low latency
cp examples/config_low_latency.yaml config/config.yaml
```

### 2. Edit Configuration

Update these values in your chosen config:

```yaml
cameras:
  - ip: "192.168.1.100"  # UPDATE: Your camera IP
```

### 3. Set Up Credentials

Create `camera_credentials.yaml` with your passwords:

```bash
cp camera_credentials.example.yaml camera_credentials.yaml
nano camera_credentials.yaml
```

### 4. Run the System

```bash
python main.py
```

---

## Customization Guide

### Adjusting Detection Sensitivity

**More sensitive (catch more):**
```yaml
conf_threshold: 0.15  # Lower = more detections
min_box_area: 20      # Smaller = catch tiny objects
```

**Less sensitive (fewer false positives):**
```yaml
conf_threshold: 0.40  # Higher = fewer detections
min_box_area: 200     # Larger = only bigger objects
```

### Improving Performance

**Faster inference:**
- Use smaller model: `yolox-nano` or `yolox-tiny`
- Reduce input size: `[416, 416]` or `[640, 640]`
- Disable Stage 2: `use_two_stage: false`

**Better accuracy:**
- Use larger model: `yolox-l` or `yolox-x`
- Increase input size: `[1280, 1280]` or `[1920, 1920]`
- Enable Stage 2: `use_two_stage: true`

### Per-Class Thresholds

Adjust confidence for specific classes:

```yaml
class_confidence_overrides:
  person: 0.80  # Higher = fewer false person detections
  bird: 0.55    # Medium sensitivity for birds
  cat: 0.30     # Lower = catch more cats (may include wildlife)
```

---

## Comparison Table

| Config | Latency | FPS | VRAM | Accuracy | Use Case |
|--------|---------|-----|------|----------|----------|
| Single Camera | 25-35ms | 25-30 | 2GB | Good | General use |
| Multi-Camera | 30-50ms | 20-25 | 4GB | Good | Multiple views |
| High Performance | 150-250ms | 4-7 | 8-10GB | Excellent | Research |
| Low Latency | 10-20ms | 50-100 | 1GB | Fair | Real-time |

---

## Need Help?

- 📚 Full config reference: [docs/setup/CONFIG_REFERENCE.md](../docs/setup/CONFIG_REFERENCE.md)
- 💬 Ask questions: [GitHub Discussions](https://github.com/filthyrake/telescope_cam_detection/discussions)
- 🐛 Report issues: [GitHub Issues](https://github.com/filthyrake/telescope_cam_detection/issues)
