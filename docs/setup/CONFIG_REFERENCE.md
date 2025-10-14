# Configuration Reference

Complete configuration guide for the Telescope Detection System.

## Table of Contents

1. [Configuration Files](#configuration-files)
2. [Camera Configuration](#camera-configuration)
3. [Detection Configuration](#detection-configuration)
4. [Species Classification (Stage 2)](#species-classification-stage-2)
5. [Web Server Configuration](#web-server-configuration)
6. [Snapshot Configuration](#snapshot-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)

---

## Configuration Files

The system uses two separate configuration files:

### Main Configuration: `config/config.yaml`

Contains all system settings **except** camera credentials:
- Camera IPs and settings
- Detection thresholds
- Model selection
- Web server settings
- Snapshot configuration

**Tracked in git**: ✅ Safe to commit

### Camera Credentials: `camera_credentials.yaml`

Contains sensitive camera passwords:
- Camera usernames and passwords
- Never tracked in git (gitignored)

**Format:**
```yaml
cameras:
  cam1:
    username: "admin"
    password: "your_password_here"
  cam2:
    username: "admin"
    password: "your_password_here"
```

**Setup:**
```bash
cp camera_credentials.example.yaml camera_credentials.yaml
nano camera_credentials.yaml  # Edit with real passwords
```

---

## Camera Configuration

### Multi-Camera Support

The system supports multiple cameras with independent configurations:

```yaml
cameras:
  - id: "cam1"
    name: "Main Backyard View"
    ip: "192.168.1.100"
    stream: "main"
    enabled: true
    protocol: "rtsp-tcp"
    target_width: 1920
    target_height: 1080
    buffer_size: 1

  - id: "cam2"
    name: "Ground Level View"
    ip: "192.168.1.101"
    stream: "main"
    enabled: true
    protocol: "rtsp-tcp"
    target_width: 2560
    target_height: 1440
    buffer_size: 1

    # Optional: Per-camera detection overrides
    detection_overrides:
      conf_threshold: 0.25
      min_box_area: 100
      class_confidence_overrides:
        person: 0.80
        bird: 0.60

    # Optional: Per-camera Stage 2 preprocessing
    stage2_preprocessing:
      crop_padding_percent: 40
      min_crop_size: 100
```

### Camera Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | string | **required** | Unique camera identifier (e.g., "cam1", "cam2") |
| `name` | string | **required** | Human-readable camera name |
| `ip` | string | **required** | Camera IP address |
| `stream` | string | "main" | Stream quality: `"main"` (high) or `"sub"` (low) |
| `enabled` | bool | true | Enable/disable this camera |
| `protocol` | string | "rtsp" | Stream protocol (see below) |
| `target_width` | int | 1280 | Frame width for inference |
| `target_height` | int | 720 | Frame height for inference |
| `buffer_size` | int | 1 | OpenCV buffer (1 = lowest latency) |

### Protocol Options

| Protocol | Description | Use Case |
|----------|-------------|----------|
| `rtsp` or `rtsp-udp` | Standard RTSP over UDP | Default, good for local networks |
| `rtsp-tcp` | RTSP over TCP | **Recommended** - Fixes screen tearing, more reliable |
| `onvif` | ONVIF Profile S path | Alternative for ONVIF-compatible cameras |
| `h265` | H.265/HEVC encoding | If camera supports H.265 (smaller bandwidth) |

**Example:**
```yaml
cameras:
  - id: "cam1"
    protocol: "rtsp-tcp"  # TCP transport (no packet loss/tearing)
```

### Per-Camera Detection Overrides

Override global detection settings for specific cameras:

```yaml
cameras:
  - id: "cam2"
    detection_overrides:
      conf_threshold: 0.25        # Higher than global
      min_box_area: 100           # Filter smaller boxes

      class_confidence_overrides:
        person: 0.80              # Specific class thresholds
        bird: 0.60
        cat: 0.65
```

**Use cases:**
- Ground-level cameras see closer objects → higher thresholds
- Overhead cameras see distant objects → lower thresholds
- Different lighting conditions per camera

### Per-Camera Stage 2 Preprocessing

Customize Stage 2 classification per camera:

```yaml
cameras:
  - id: "cam2"
    stage2_preprocessing:
      crop_padding_percent: 40    # Add 40% padding around bbox
      min_crop_size: 100          # Skip Stage 2 if crop < 100x100
```

**Benefits:**
- Higher resolution cameras → larger crops for better classification
- Distant objects → more padding for context

---

## Detection Configuration

### YOLOX Model Settings

```yaml
detection:
  # Model selection
  model:
    name: "yolox-s"               # Model variant
    weights: "models/yolox/yolox_s.pth"

  device: "cuda:0"                # GPU device

  # Input resolution (affects speed vs accuracy)
  input_size: [1920, 1920]        # [height, width]

  # Detection thresholds
  conf_threshold: 0.15            # Confidence threshold (0.0-1.0)
  nms_threshold: 0.45             # Non-max suppression IoU

  # Filtering
  wildlife_only: true             # Filter to wildlife-relevant classes
  min_box_area: 20                # Minimum bounding box area (px²)
  max_detections: 300             # Max detections per frame

  # Per-class confidence overrides
  class_confidence_overrides:
    person: 0.60                  # Higher threshold for people
    bird: 0.55                    # Birds can be distant
    cat: 0.40
    dog: 0.40
```

### Model Variants

| Model | Inference Time | Accuracy | Use Case |
|-------|----------------|----------|----------|
| `yolox-nano` | 8-12ms | Lower | Constrained systems |
| `yolox-tiny` | 9-15ms | Good | Fast detection |
| `yolox-s` | **11-21ms** | **Balanced** | **Current default** |
| `yolox-m` | 25-40ms | Better | More accuracy needed |
| `yolox-l` | 50-80ms | High | Best accuracy |
| `yolox-x` | 80-120ms | Highest | Maximum accuracy |

### Input Size

Larger input = better small object detection, slower inference:

```yaml
detection:
  input_size: [640, 640]      # Fast (~11ms)
  input_size: [1280, 1280]    # Balanced (~17ms)
  input_size: [1920, 1920]    # Best quality (~21ms)
```

### Detection Classes

YOLOX detects 80 COCO classes. When `wildlife_only: true`, filters to:

**Wildlife-relevant classes:**
- **Mammals**: person, cat, dog, horse, cow, sheep, elephant, bear, zebra, giraffe
- **Birds**: bird (generic)
- **Other**: All standard COCO objects

Full COCO class list available at: https://github.com/Megvii-BaseDetection/YOLOX

### Class Confidence Overrides

Set different confidence thresholds per class:

```yaml
detection:
  conf_threshold: 0.15          # Global default

  class_confidence_overrides:
    person: 0.60                # People = higher threshold
    bird: 0.55                  # Birds = slightly higher
    cat: 0.40                   # Cats = standard
    dog: 0.40                   # Dogs = standard
```

**Why override?**
- **Higher values** = fewer false positives, may miss detections
- **Lower values** = more detections, more false positives
- **People often have more false positives** → use higher threshold

---

## Species Classification (Stage 2)

Optional fine-grained species identification using iNaturalist:

```yaml
detection:
  use_two_stage: true                    # Enable Stage 2
  enable_species_classification: true

species_classification:
  enabled: true
  confidence_threshold: 0.3              # Stage 2 confidence minimum

  inat_classifier:
    model_name: "timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"
    taxonomy_file: "models/inat2021_taxonomy_simple.json"
    input_size: 336
    confidence_threshold: 0.3

    # Filter which Stage 1 classes trigger Stage 2
    wildlife_classes:
      - "bird"
      - "cat"
      - "dog"
      - "horse"
      - "cow"
      - "sheep"
      - "bear"
      - "zebra"
      - "giraffe"
      - "elephant"
```

### Stage 2 Performance

- **Model**: EVA02-Large (iNaturalist 2021, 10,000 species)
- **Inference Time**: +20-30ms per detection
- **Accuracy**: 92% top-1 on validation set
- **VRAM**: ~1.5GB additional

### Stage 2 Preprocessing

Global settings (can be overridden per-camera):

```yaml
species_classification:
  crop_padding_percent: 20      # Add 20% padding around bbox
  min_crop_size: 50             # Skip if crop < 50x50 pixels
```

**Padding benefits:**
- Provides context around the animal
- Improves classification accuracy
- Especially important for birds (need to see habitat)

---

## Web Server Configuration

```yaml
web:
  host: "0.0.0.0"    # Bind address (0.0.0.0 = all interfaces)
  port: 8000         # HTTP port
```

### Access

- **Local**: http://localhost:8000
- **Network**: http://YOUR_SERVER_IP:8000

### Endpoints

- `GET /` - Web UI
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /video/feed` - MJPEG video stream
- `GET /clips_list` - List saved clips
- `WS /ws/detections` - WebSocket detection feed

See [API_REFERENCE.md](../api/API_REFERENCE.md) for complete API documentation.

---

## Snapshot Configuration

Automatic saving of detection clips:

```yaml
snapshots:
  enabled: true
  save_mode: "image"                # "image" or "clip"

  # Triggering
  trigger_classes:                  # Classes that trigger saves
    - "person"
    - "coyote"
    - "bobcat"
    - "bird"

  min_confidence: 0.5               # Minimum confidence to save
  cooldown_seconds: 45              # Cooldown between saves (same class)

  # Save options
  save_annotated: true              # Save with bounding boxes
  save_raw: false                   # Also save without boxes

  # Clip mode settings (if save_mode: "clip")
  clip_duration: 3                  # Seconds before/after detection
  clip_fps: 10                      # Frames per second

  # Storage
  clips_directory: "clips"          # Save location
  max_clips: 1000                   # Auto-cleanup (0 = unlimited)
```

### Save Modes

| Mode | Description | File Size | Use Case |
|------|-------------|-----------|----------|
| `image` | Single frame JPEG | ~100-300KB | Quick reference, low storage |
| `clip` | Short video (MP4) | ~1-5MB | Full context, behavioral analysis |

### Cooldown System

Prevents duplicate saves of the same animal:

```yaml
snapshots:
  cooldown_seconds: 45    # Wait 45s before saving same class again
```

**Example:**
- Coyote detected at 10:00:00 → saved
- Coyote detected at 10:00:20 → skipped (cooldown)
- Coyote detected at 10:00:50 → saved (cooldown expired)

---

## Performance Tuning

### Inference Speed vs Quality

Trade-off between speed and accuracy:

```yaml
# FAST (15-20 FPS, lower accuracy)
detection:
  model:
    name: "yolox-tiny"
  input_size: [640, 640]

# BALANCED (20-30 FPS, good accuracy) ← Current default
detection:
  model:
    name: "yolox-s"
  input_size: [1920, 1920]

# HIGH QUALITY (5-10 FPS, best accuracy)
detection:
  model:
    name: "yolox-x"
  input_size: [1920, 1920]
```

### Performance Metrics

**Current System (YOLOX-S @ 1920x1920):**
- **Inference**: 11-21ms per frame
- **Stage 2**: +20-30ms when triggered
- **Total Pipeline**: 25-35ms (Stage 1 only), 30-50ms (Stage 1+2)
- **FPS**: 25-30 sustained
- **GPU VRAM**: ~2GB per camera

### Memory Management

```yaml
cameras:
  - buffer_size: 1          # Keep at 1 for low latency

detection:
  max_detections: 300       # Limit detections per frame
```

### Multi-Camera Scaling

Each camera adds:
- ~2GB GPU VRAM
- ~500MB CPU RAM
- 1 stream capture thread
- 1 inference thread

**A30 GPU (24GB VRAM)**: Can support ~10 cameras simultaneously

---

## Advanced Features

### Motion Filtering

Reduces false positives from static objects using background subtraction:

```yaml
motion_filter:
  enabled: true                  # Enable motion-based filtering
  history: 500                   # Frames for background model (higher = slower adaptation)
  var_threshold: 16              # Background/foreground threshold (higher = less sensitive)
  detect_shadows: true           # Detect and filter shadows
  min_motion_area: 100           # Minimum motion area in pixels²
  motion_blur_size: 21           # Gaussian blur kernel (must be odd)
```

**How it works:**
- Builds background model over time
- Detections without motion are filtered out
- Reduces false positives from telescope covers, furniture, etc.

**Performance:**
- Adds ~2-5ms per frame
- Reduces false positives by 40-60%
- Works best with static cameras

### Time-of-Day Filtering

Intelligently filters detections based on species activity patterns:

```yaml
time_of_day_filter:
  enabled: true                  # Enable time-based filtering
  confidence_penalty: 0.3        # Multiplier for out-of-pattern detections (0.0-1.0)
  hard_filter: false             # If true, completely remove unlikely detections
  use_system_timezone: true      # Automatically use system timezone (handles DST)
```

**Activity Patterns:**
- **Diurnal species** (birds, lizards, squirrels): Active dawn-day
  - Nighttime detections penalized 70% (likely bugs/bats)
- **Nocturnal species** (bats, owls, geckos): Active dusk-night
  - Daytime detections unaffected
- **Crepuscular species** (coyotes, rabbits, deer): Active dawn/dusk/night
  - Most times unaffected
- **Cathemeral species** (humans, cats, dogs): Active anytime
  - Never penalized

**Example:**
- Bird detected at 11pm with 0.85 confidence
- Filter reduces to 0.26 (0.85 × 0.3 = 0.26)
- Likely filtered out by confidence threshold
- Reason: Birds are diurnal, probably a bug or bat

**Stage 2 Enhancement:**
When Stage 2 is enabled, time-of-day filtering also re-ranks species classifications:

```yaml
# Time-of-day re-ranking (Stage 2)
time_of_day_top_k: 5            # Get top-5 species for re-ranking (default: 5)
time_of_day_penalty: 0.3        # Penalty for unlikely species (default: 0.3)
```

**Example:**
- Stage 1: "bird" at 2am
- Stage 2 gets top-5 species:
  1. "Gambel's Quail" (diurnal) - 0.90 → 0.27 (penalized)
  2. "Great Horned Owl" (nocturnal) - 0.60 → 0.60 (unchanged)
- Result: System picks owl over quail (correct!)

**Species Database:**
- 128+ species with known activity patterns
- Automatic fallback for unknown species
- No configuration needed - works out of the box

**Configuration:**
- `confidence_penalty: 0.3` → 70% reduction for unlikely detections
- `hard_filter: true` → Completely remove unlikely detections (strict mode)
- `use_system_timezone: true` → Handles DST automatically (recommended)

### Detection Zones (Planned)

Define spatial regions for detection:

```yaml
detection_zones:
  - name: "telescope_area"
    type: "polygon"
    points: [[100, 100], [500, 100], [500, 500], [100, 500]]
    enabled: false
```

### Collision Detection (Planned)

Proximity alerts for telescope safety:

```yaml
collision_detection:
  enabled: false
  min_distance: 50        # Minimum distance (pixels)
  alert_threshold: 30     # Alert if closer than this
```

---

## Troubleshooting

### No Detections

**Check:**
1. **Confidence too high**: Lower `conf_threshold` (try 0.10)
2. **Min box area too high**: Lower `min_box_area` (try 20)
3. **Wildlife filter**: Set `wildlife_only: false` temporarily
4. **Class overrides**: Check `class_confidence_overrides`

```yaml
detection:
  conf_threshold: 0.10           # Lower threshold
  min_box_area: 20               # Smaller minimum
  wildlife_only: false           # Test with all classes
```

### Too Many False Positives

**Solutions:**
1. **Raise confidence**: Increase `conf_threshold` (try 0.25)
2. **Raise class thresholds**: Increase specific classes
3. **Increase min box area**: Filter small detections

```yaml
detection:
  conf_threshold: 0.25
  min_box_area: 100

  class_confidence_overrides:
    person: 0.70                 # Higher for false positives
```

### High Latency (>50ms)

**Optimizations:**
1. **Smaller model**: Switch to `yolox-tiny` or `yolox-nano`
2. **Lower resolution**: Reduce `input_size` to [640, 640]
3. **Disable Stage 2**: Set `use_two_stage: false`
4. **Use sub stream**: Change camera `stream: "sub"`

```yaml
detection:
  model:
    name: "yolox-tiny"           # Faster model
  input_size: [640, 640]         # Lower resolution

cameras:
  - stream: "sub"                # Lower quality stream
```

### Camera Connection Issues

**RTSP-TCP Issues:**
```yaml
cameras:
  - protocol: "rtsp"             # Try standard UDP
  # or
  - protocol: "onvif"            # Try ONVIF path
```

**Test connection:**
```bash
python tests/test_camera_connection.py
```

### GPU Memory Issues

**Reduce VRAM usage:**
1. Disable cameras: `enabled: false`
2. Smaller model: `yolox-nano`
3. Lower input size: `[640, 640]`
4. Disable Stage 2: `use_two_stage: false`

```yaml
cameras:
  - id: "cam2"
    enabled: false               # Disable extra cameras

detection:
  model:
    name: "yolox-nano"
  input_size: [640, 640]
  use_two_stage: false
```

---

## Configuration Examples

### Minimal (Single Camera, Fast)

```yaml
cameras:
  - id: "cam1"
    ip: "192.168.1.100"
    enabled: true

detection:
  model:
    name: "yolox-s"
  device: "cuda:0"
  conf_threshold: 0.15

web:
  port: 8000
```

### Production (Multi-Camera, Stage 2)

```yaml
cameras:
  - id: "cam1"
    name: "Main View"
    ip: "192.168.1.100"
    protocol: "rtsp-tcp"
    enabled: true
    target_width: 1920
    target_height: 1080

  - id: "cam2"
    name: "Ground Level"
    ip: "192.168.1.101"
    protocol: "rtsp-tcp"
    enabled: true
    target_width: 2560
    target_height: 1440
    detection_overrides:
      conf_threshold: 0.25
      class_confidence_overrides:
        person: 0.80

detection:
  model:
    name: "yolox-s"
  device: "cuda:0"
  input_size: [1920, 1920]
  conf_threshold: 0.15
  use_two_stage: true
  enable_species_classification: true

species_classification:
  enabled: true

snapshots:
  enabled: true
  trigger_classes: ["bird", "cat", "dog"]
  cooldown_seconds: 45

web:
  host: "0.0.0.0"
  port: 8000
```

---

## Related Documentation

- [Architecture Overview](../architecture/ARCHITECTURE.md)
- [API Reference](../api/API_REFERENCE.md)
- [Stage 2 Setup Guide](../features/STAGE2_SETUP.md)
- [Service Setup](SERVICE_SETUP.md)

---

**Document Version**: 2.0
**Last Updated**: 2025-10-06
**System Version**: v1.2 (YOLOX)
