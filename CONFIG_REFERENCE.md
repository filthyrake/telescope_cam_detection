# Configuration Reference

Complete configuration guide for `config/config.yaml`.

## Configuration File Structure

```yaml
camera:        # Camera connection settings
detection:     # Detection model settings
species_classification:  # Stage 2 classification
web:           # Web server settings
performance:   # Performance tuning
snapshots:     # Clip saving settings
detection_zones:  # Collision zones (future)
collision_detection:  # Collision detection (future)
```

---

## Camera Configuration

```yaml
camera:
  ip: "10.0.8.18"
  username: "admin"
  password: "5326jbbD"
  stream: "main"
  target_width: 1280
  target_height: 720
  buffer_size: 1
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ip` | string | **required** | Camera IP address |
| `username` | string | **required** | RTSP username |
| `password` | string | **required** | RTSP password |
| `stream` | string | "main" | Stream type: "main" (2K) or "sub" (lower res) |
| `target_width` | int | 1280 | Resize frame width (inference resolution) |
| `target_height` | int | 720 | Resize frame height (inference resolution) |
| `buffer_size` | int | 1 | OpenCV buffer size (1 = lowest latency) |

### Notes

- **Main stream**: 2560x1440, higher quality, higher latency
- **Sub stream**: Lower resolution, faster, less bandwidth
- **Resize**: Always resized to target size for inference (maintains aspect ratio)
- **Buffer size**: Keep at 1 for real-time detection

### RTSP URL Format

Constructed as: `rtsp://{username}:{password}@{ip}:554/h264Preview_01_{stream}`

---

## Detection Configuration

```yaml
detection:
  model: "yolov8x-worldv2.pt"
  device: "cuda:0"

  use_yolo_world: true
  yolo_world_classes:
    - "person"
    - "coyote"
    - "rabbit"
    # ... more classes

  use_two_stage: true
  enable_species_classification: true

  confidence: 0.25
  iou_threshold: 0.45
  min_box_area: 50
  max_detections: 300
  target_classes: []
```

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | **required** | Model weights file (auto-downloads if not present) |
| `device` | string | "cuda:0" | PyTorch device: "cuda:0", "cuda:1", "cpu" |

**Available Models**:
- `yolov8x-worldv2.pt` - YOLO-World (current, open-vocabulary)
- `yolov8x.pt` - Standard YOLOv8 (80 COCO classes)
- `yolov8n.pt` - Nano (fastest, lower accuracy)
- `models/telescope_custom.pt` - Custom trained (future)

---

### YOLO-World Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_yolo_world` | bool | true | Enable YOLO-World mode |
| `yolo_world_classes` | list[str] | **required** | Text prompts for detection classes |

**Class Prompts**:
- Plain text descriptions (e.g., "coyote", "bird")
- Can be specific (e.g., "desert lizard", "small lizard")
- Maximum ~20-30 classes for best performance
- More specific = better accuracy

**Example Classes**:
```yaml
yolo_world_classes:
  # People
  - "person"

  # Mammals
  - "coyote"
  - "rabbit"
  - "ground squirrel"
  - "javelina"
  - "deer"
  - "bobcat"
  - "fox"

  # Birds
  - "roadrunner"
  - "quail"
  - "hawk"
  - "raven"
  - "dove"

  # Reptiles
  - "lizard"
  - "desert lizard"
  - "small lizard"
  - "reptile"
  - "iguana"
  - "tortoise"
  - "snake"
```

---

### Two-Stage Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_two_stage` | bool | false | Enable two-stage detection pipeline |
| `enable_species_classification` | bool | false | Enable Stage 2 species classification |

**When to Enable**:
- `use_two_stage: true` + `enable_species_classification: true` = Full pipeline (Stage 1+2)
- `use_two_stage: false` = Stage 1 only (faster, no species ID)

**Performance Impact**:
- Stage 1 only: ~20-25ms
- Stage 1+2: ~30-35ms (+5-10ms per detection)

---

### Detection Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence` | float | 0.25 | Minimum detection confidence (0.0-1.0) |
| `iou_threshold` | float | 0.45 | IoU threshold for NMS (0.0-1.0) |
| `min_box_area` | int | 100 | Minimum bounding box area (pixels²) |
| `max_detections` | int | 300 | Maximum detections per frame |

**Tuning Guide**:

**Confidence (0.0 - 1.0)**:
- Lower = more detections, more false positives
- Higher = fewer detections, misses distant/small objects
- Recommended ranges:
  - Distant wildlife: 0.20-0.30
  - Close objects: 0.40-0.60
  - High precision: 0.60+

**min_box_area (pixels²)**:
- Frame size: 1280x720 = 921,600 total pixels
- Typical sizes at this resolution:
  - Tiny (50-200px²): Small distant animals, ~7x7 to 14x14 pixels
  - Small (200-1000px²): Birds, rabbits at distance
  - Medium (1000-5000px²): Coyote, larger animals
  - Large (5000+px²): Close objects, people
- Lower values catch smaller objects but increase false positives
- For 40ft distance:
  - 6" lizard: ~75-150px²
  - 14" lizard: ~600-1000px²
  - Rabbit: ~800-1500px²
  - Coyote: ~2000-4000px²

**Recommendations**:
```yaml
# For distant, small wildlife (current setup)
confidence: 0.25
min_box_area: 50

# For close, large animals only
confidence: 0.40
min_box_area: 500

# For high precision (few false positives)
confidence: 0.50
min_box_area: 200
```

---

### Legacy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_classes` | list[str] | [] | Filter to specific classes (ignored if `use_yolo_world: true`) |

**Note**: Only used with standard YOLO models, not YOLO-World.

---

## Species Classification Configuration

```yaml
species_classification:
  enabled: true
  confidence_threshold: 0.3

  inat_classifier:
    model_name: "timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"
    taxonomy_file: "models/inat2021_taxonomy_simple.json"
    input_size: 336
    confidence_threshold: 0.3
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable species classification (requires models) |
| `confidence_threshold` | float | 0.3 | Minimum confidence for species ID |

### iNaturalist Classifier

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | string | **required** | timm model name (HuggingFace Hub) |
| `taxonomy_file` | string | **required** | JSON file mapping class IDs to species names |
| `input_size` | int | 336 | Model input size (224, 336, etc.) |
| `confidence_threshold` | float | 0.3 | Species classification threshold |

**Model**: EVA02-Large
- **Classes**: 10,000 species
- **Accuracy**: 92% top-1, 98% top-5
- **Input**: 336x336 RGB
- **VRAM**: ~1.5GB
- **Inference**: ~5-10ms per detection

**Taxonomy File**: 
- Format: `{"class_id": "Common Name", ...}`
- Auto-downloaded via `scripts/download_inat_taxonomy.py`
- Location: `models/inat2021_taxonomy_simple.json`

---

## Web Server Configuration

```yaml
web:
  host: "0.0.0.0"
  port: 8000
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | "0.0.0.0" | Bind address (0.0.0.0 = all interfaces) |
| `port` | int | 8000 | HTTP port |

**Bind Addresses**:
- `0.0.0.0` - All interfaces (accessible from network)
- `127.0.0.1` - Localhost only (more secure)
- `192.168.1.x` - Specific interface

---

## Performance Configuration

```yaml
performance:
  frame_queue_size: 2
  detection_queue_size: 10
  history_size: 30
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame_queue_size` | int | 2 | Frame buffer size (lower = less latency) |
| `detection_queue_size` | int | 10 | Detection result buffer |
| `history_size` | int | 30 | Number of recent detections to keep |

**Tuning**:
- `frame_queue_size`: Keep at 1-2 for real-time (drops old frames)
- `detection_queue_size`: Larger = smoother but more memory
- `history_size`: For web UI statistics only

---

## Snapshots Configuration

```yaml
snapshots:
  enabled: true
  save_mode: "image"
  output_dir: "clips"

  trigger_classes:
    - "person"
    - "bird"
    # ...

  min_confidence: 0.30
  cooldown_seconds: 45
  save_annotated: true

  clip_duration: 10
  pre_buffer_seconds: 5
  fps: 30
```

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable snapshot saving |
| `save_mode` | string | "image" | Mode: "image" or "clip" (clip not implemented) |
| `output_dir` | string | "clips" | Directory to save snapshots |

---

### Trigger Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trigger_classes` | list[str] | [] | Classes that trigger saves (empty = all) |
| `min_confidence` | float | 0.6 | Minimum confidence to trigger save |
| `cooldown_seconds` | int | 30 | Seconds between saves for same class |

**Cooldown System**:
- Per-class timer prevents duplicate saves
- Example: Bird detected → Save → 45s cooldown → Can save bird again
- Different classes have independent cooldowns
- Prevents spam from static objects

**Example Trigger Classes**:
```yaml
trigger_classes:
  # Important detections only
  - "person"
  - "coyote"
  - "rabbit"
  - "bird"
  - "roadrunner"
  - "quail"

  # Or empty list to save everything
  # trigger_classes: []
```

---

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_annotated` | bool | true | Save version with bounding boxes drawn |

**Output Files**:
```
clips/
├── bird_20251005_071214_813087_conf0.52.jpg          # Raw frame
├── bird_20251005_071214_813087_conf0.52_annotated.jpg # With boxes
└── bird_20251005_071214_813087_conf0.52.json         # Metadata
```

---

### Video Clip Parameters (Future)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_duration` | int | 10 | Total clip length (seconds) |
| `pre_buffer_seconds` | int | 5 | Seconds before detection |
| `fps` | int | 30 | Video frame rate |

**Note**: Video clip mode not yet implemented.

---

## Collision Detection Configuration (Future)

```yaml
detection_zones: []
  # Future: Define zones around telescope equipment
  # - name: "telescope_danger_zone"
  #   type: "polygon"
  #   points: [[100,200], [300,200], [300,400], [100,400]]
  #   alert_on_entry: ["person", "coyote"]

collision_detection:
  enabled: false
  danger_threshold: 50
```

**Status**: Planned feature, not yet implemented.

---

## Configuration Examples

### Example 1: Wildlife Monitoring (Current)

```yaml
camera:
  ip: "10.0.8.18"
  stream: "main"
  target_width: 1280
  target_height: 720

detection:
  model: "yolov8x-worldv2.pt"
  device: "cuda:0"
  use_yolo_world: true
  yolo_world_classes: ["person", "coyote", "rabbit", "bird", "lizard", ...]
  use_two_stage: true
  enable_species_classification: true
  confidence: 0.25
  min_box_area: 50

species_classification:
  enabled: true
  inat_classifier:
    model_name: "timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"

snapshots:
  enabled: true
  trigger_classes: ["person", "bird", "coyote", "rabbit"]
  cooldown_seconds: 45
```

---

### Example 2: Fast Detection (No Species ID)

```yaml
detection:
  model: "yolov8n.pt"  # Faster, smaller model
  use_two_stage: false
  enable_species_classification: false
  confidence: 0.40
  min_box_area: 200

# Stage 2 disabled for speed
species_classification:
  enabled: false
```

**Performance**: ~10-15ms (vs ~30-35ms with Stage 2)

---

### Example 3: High Precision (Few False Positives)

```yaml
detection:
  confidence: 0.50  # Higher threshold
  min_box_area: 300  # Larger objects only

snapshots:
  min_confidence: 0.60  # Only save high-confidence
  cooldown_seconds: 60  # Less frequent saves
```

---

### Example 4: Telescope Collision Detection (Future)

```yaml
detection:
  model: "models/telescope_custom.pt"  # Custom trained
  use_yolo_world: false
  target_classes:
    - "telescope_ota"
    - "telescope_mount"
    - "tripod_leg"
    - "counterweight"

detection_zones:
  - name: "telescope_1_danger_zone"
    type: "circle"
    center: [640, 480]
    radius: 200
    alert_on_entry: ["person", "tripod_leg"]

collision_detection:
  enabled: true
  danger_threshold: 100
```

---

## Configuration Validation

### Required Fields

All these must be present:
- `camera.ip`
- `camera.username`
- `camera.password`
- `detection.model`
- `detection.device`

### Valid Ranges

- `detection.confidence`: 0.0 - 1.0
- `detection.iou_threshold`: 0.0 - 1.0
- `detection.min_box_area`: 0 - 100000
- `snapshots.min_confidence`: 0.0 - 1.0
- `snapshots.cooldown_seconds`: 0 - 3600

### Validation Errors

Config errors are logged on startup:
```
ERROR: Configuration error: camera.ip is required
ERROR: Invalid confidence value: 1.5 (must be 0.0-1.0)
```

---

## Environment Variables

Override config with environment variables (not currently implemented):

```bash
export CAMERA_IP="10.0.8.19"
export DETECTION_CONFIDENCE="0.30"
python main.py
```

**Priority**: ENV vars > config file > defaults

---

## Configuration Best Practices

1. **Start with defaults** - Use provided config as baseline
2. **Tune gradually** - Adjust one parameter at a time
3. **Test changes** - Restart service and monitor performance
4. **Back up config** - Keep copies before major changes
5. **Document changes** - Add comments for custom settings

---

## Troubleshooting

### No Detections

**Check**:
- `confidence` too high? Try 0.25
- `min_box_area` too high? Try 50
- `yolo_world_classes` missing target? Add class

### Too Many False Positives

**Check**:
- `confidence` too low? Try 0.40
- `min_box_area` too low? Try 200
- Too many YOLO-World classes? Reduce to 10-15

### Poor Performance

**Check**:
- Disable Stage 2: `use_two_stage: false`
- Use smaller model: `yolov8n.pt`
- Reduce frame size: `target_width: 640`

### Lizards Not Detected

**Check**:
- Lower `min_box_area` to 50 or 25
- Lower `confidence` to 0.20
- Add specific prompts: "small lizard", "desert lizard"
- Camera too far away (limit of detection at 40-50ft)

---

## Changelog

### v1.2 (2025-10-05)
- Added lizard detection prompts
- Lowered `min_box_area` to 50
- Enabled Stage 2 species classification

### v1.1
- Added iNaturalist species classification
- Two-stage detection pipeline

### v1.0
- Initial configuration
- YOLO-World detection
- Snapshot saving

---

**Document Version**: 1.0
**Last Updated**: 2025-10-05
