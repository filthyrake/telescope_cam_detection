# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time object detection system for **telescope collision prevention** and **wildlife monitoring** using a Reolink RLC-410W camera and NVIDIA A30 GPU.

**Dual Purpose System**:
1. **Telescope Safety** (Primary Goal): Detect telescope equipment parts (tube, mount, tripod legs) and potential collision hazards (people, animals). Prevent accidents with telescope equipment by alerting when objects are too close to tripod legs or moving telescope components.

2. **Wildlife Monitoring** (Secondary Goal): Monitor desert animals (coyotes, rabbits, quail, roadrunners, lizards, etc.) using YOLO-World open-vocabulary detection.

**Current Status**:
- Wildlife detection: **ACTIVE** (using YOLO-World)
- Telescope detection: **IN TRAINING** (custom model requires annotated dataset)
- Collision detection logic: **FRAMEWORK READY** (config placeholders exist)

**Camera Setup**: Mounted overlooking backyard telescope setup and surrounding desert terrain

## Development Commands

### Running the System
```bash
# Activate virtual environment (always required)
source venv/bin/activate

# Run main application
python main.py

# Access web interface at http://localhost:8000
```

### Testing
```bash
# Test camera connection
python tests/test_camera_connection.py

# Test GPU inference performance
python tests/test_inference.py

# Test end-to-end latency
python tests/test_latency.py
```

### Training Custom Telescope Detection Model
```bash
# STEP 1: Capture training images from live stream
# Walk around telescopes during capture to get diverse angles
python training/scripts/capture_training_images.py --num 150 --interval 2

# Or headless mode for remote servers
python training/scripts/capture_training_images_headless.py

# STEP 2: Split dataset into train/val (80/20)
python training/scripts/prepare_dataset.py

# STEP 3: Annotate images with bounding boxes
# Use LabelImg or Roboflow to draw boxes around telescope parts
# See training/TRAINING_GUIDE.md for detailed instructions
pip install labelImg
labelImg training/datasets/telescope_equipment/images/train

# STEP 4: Train custom YOLOv8 model (15-30 mins on A30)
python training/scripts/train_custom_model.py --epochs 100 --batch 16

# STEP 5: Evaluate trained model
python training/scripts/evaluate_model.py --mode live  # Test on camera
python training/scripts/evaluate_model.py --mode images  # Test on validation set
```

### Utilities
```bash
# View saved wildlife clips with metadata
python scripts/view_snapshots.py

# Check GPU status
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Architecture Overview

### Component Pipeline
```
RTSP Camera → Stream Capture → Frame Queue → Inference Engine → Detection Processor → Web Server
                                                     ↓
                                          (Optional) Snapshot Saver
                                                     ↓
                                               clips/ directory
```

### Threading Model
- **Stream Capture Thread**: Continuously grabs frames from RTSP, drops old frames (Queue size=2)
- **Inference Thread**: GPU inference on queued frames (runs in InferenceEngine)
- **Detection Processor Thread**: Post-processes detections, handles snapshot saving
- **Web Server (Main Thread)**: FastAPI/Uvicorn serving video feed + WebSocket detections

### Key Design Patterns

**Low-Latency Frame Pipeline**: Uses Queue size=1-2 and aggressive frame dropping to process only the latest frames, target <100ms end-to-end latency.

**Two-Stage Detection (Optional)**:
- Stage 1: YOLO-World detects broad categories ("bird", "mammal", "reptile")
- Stage 2: Species classifier identifies specific species ("Gambel's Quail")
- Currently: Stage 1 only (Stage 2 framework exists but needs trained models)

**Snapshot Cooldown System**: Prevents duplicate saves using per-class cooldown timers (default 45s).

## Configuration System

**Main Config**: `config/config.yaml`

Critical settings:
- `detection.model`: Currently using `yolov8x-worldv2.pt` (YOLO-World for wildlife)
  - **Future**: Switch to `models/telescope_custom.pt` once trained
- `detection.use_yolo_world`: true (enables open-vocabulary detection)
- `detection.yolo_world_classes`: List of 17 desert species text prompts
- `detection.confidence`: 0.25 (lower for distant animals/telescope parts)
- `detection.min_box_area`: 100px² (filters tiny false positives)
- `snapshots.enabled`: true (saves detections to clips/)
- `snapshots.trigger_classes`: Which animals trigger saves
- `detection_zones`: **PLANNED** - Define danger zones around telescope equipment
- `collision_detection.enabled`: **PLANNED** - Enable proximity alerts

**Model Selection Strategy**:
- **YOLO-World** (`yolov8x-worldv2.pt`): For wildlife using text prompts (no training needed)
- **Custom YOLOv8** (`telescope_custom.pt`): For telescope parts (requires training on 7 classes)
- **Dual Model Setup** (future): Run both models simultaneously - YOLO-World for wildlife, custom model for telescope equipment

**Telescope Detection Classes** (training/datasets/telescope_equipment/classes.yaml):
```yaml
0: telescope_ota        # Optical Tube Assembly (main telescope tube)
1: telescope_mount      # Mount head that holds telescope
2: tripod_leg           # Tripod legs (CRITICAL for collision detection!)
3: counterweight        # Counterweight bar and weights
4: telescope_cover      # Telescope covers/caps
5: finder_scope         # Finder scope
6: focuser              # Focusing mechanism
```

## Important File Locations

### Core Modules (src/)
- `stream_capture.py`: RTSP stream handling with OpenCV
- `inference_engine.py`: GPU inference with YOLOv8/YOLO-World
- `detection_processor.py`: Post-processing, filtering, snapshot coordination
- `web_server.py`: FastAPI web interface with WebSocket + MJPEG stream
- `snapshot_saver.py`: Saves detection events as images/clips
- `two_stage_pipeline.py`: Two-stage detection framework (Stage 2 inactive)
- `species_classifier.py`: Species classification models (Stage 2 inactive)

### Entry Point
- `main.py`: Orchestrates all components, handles shutdown

### Training Infrastructure (training/)
- `training/datasets/telescope_equipment/`: 900+ training images collected
- `training/scripts/train_custom_model.py`: YOLOv8 training script
- `training/TRAINING_GUIDE.md`: Complete training workflow

### Documentation
- `README.md`: User-facing documentation
- `STAGE2_SETUP.md`: Guide for enabling species classification
- `SNAPSHOT_FEATURE.md`: Snapshot saving feature documentation
- `ANNOTATION_GUIDE.md`: How to annotate training images

## Development Workflow

### Training Custom Telescope Detection Model

**Why?** To detect telescope equipment parts for collision prevention. YOLO-World detects wildlife, but can't reliably identify telescope-specific parts like tripod legs.

**Training Process** (see training/TRAINING_GUIDE.md):
1. **Collect 100-200 images**: Use `capture_training_images.py` to grab frames from camera
   - Walk around telescopes to get different angles
   - Include covered and uncovered states
   - Vary distances (2-20 feet)
   - Different lighting conditions
2. **Annotate images**: Draw bounding boxes around 7 telescope parts using LabelImg or Roboflow
   - **Critical**: Precisely label all three tripod legs (needed for collision detection)
   - Be consistent with box placement
3. **Train model**: Run `train_custom_model.py` (15-30 mins on A30)
   - Target: mAP50 > 0.85
   - Output: `models/telescope_custom.pt`
4. **Deploy**: Update config to use custom model, restart system

**Expected Results**:
- 90%+ accuracy on telescope parts
- 10-15ms inference time
- No more false "person" detections on covered telescopes

### Adding New Wildlife Detection Classes
1. Edit `config/config.yaml` → `detection.yolo_world_classes`
2. Add text prompt for new animal (e.g., "javelina", "bobcat")
3. Restart system (no training needed with YOLO-World)

### Implementing Collision Detection Logic

**Framework exists but not active**. To implement:

1. **Enable collision detection** in config:
```yaml
collision_detection:
  enabled: true
  danger_threshold: 50  # pixels - minimum safe distance
```

2. **Define detection zones** around telescope equipment:
```yaml
detection_zones:
  - name: "telescope_1_danger_zone"
    type: "polygon"
    points: [[100,200], [300,200], [300,400], [100,400]]
    alert_on_entry: ["person", "coyote", "dog"]

  - name: "tripod_collision_zone"
    type: "radius"
    center: [640, 480]  # Center of tripod base
    radius: 150  # pixels
    alert_on_proximity: ["person", "bird", "counterweight"]
```

3. **Implement collision logic** in `src/detection_processor.py`:
   - Calculate distances between detected objects
   - Check if objects are within danger zones
   - Track velocity vectors for moving objects
   - Predict collision paths
   - Trigger alerts/sounds when collision risk detected

4. **Add alert system**:
   - WebSocket messages with collision warnings
   - Visual red bounding boxes on web UI
   - Audio alerts (optional)
   - Integration with telescope mount for emergency stop (future)

### Modifying Detection Thresholds
Common adjustments for distant wildlife:
- Lower `detection.confidence` (0.2-0.3 for small/distant animals)
- Adjust `detection.min_box_area` (100-200px² filters noise)
- Modify `snapshots.cooldown_seconds` (30-60s prevents spam)

### Debugging Detection Issues
1. Check logs for inference times (should be ~20-30ms)
2. View `clips/` directory for saved detections
3. Use `scripts/view_snapshots.py` to review detections with metadata
4. Lower confidence threshold if missing animals
5. Raise min_box_area if too many false positives

### Camera Configuration
**RTSP URL Format**: `rtsp://username:password@ip/stream`
- Main stream: 2560x1440 (high quality, higher latency)
- Sub stream: lower resolution (faster, used for real-time)
- Current: Using main stream resized to 1280x720

## GPU Optimization Notes

**NVIDIA A30 Performance**:
- Expected inference: 20-30ms (yolov8x-world)
- VRAM usage: ~2GB
- Runs at fp16 precision automatically
- CUDA 11.8+ required

**Performance Monitoring**:
- Web interface shows live latency metrics
- Check `/stats` endpoint for detailed performance
- Use `tests/test_latency.py` for benchmarking

## Model Management

**Auto-downloaded models** (stored in `models/`):
- `yolov8x-worldv2.pt`: Current model (YOLO-World v2)
- Downloaded automatically on first run via Ultralytics

**Custom trained models**:
- Train using `training/scripts/train_custom_model.py`
- Output: `training/runs/detect/train/weights/best.pt`
- Update config to point to custom model

## WebSocket API

**Connect**: `ws://localhost:8000/ws/detections`

**Message Format**:
```json
{
  "type": "detections",
  "frame_id": 12345,
  "timestamp": 1634567890.123,
  "latency_ms": 25.5,
  "inference_time_ms": 20.2,
  "total_detections": 3,
  "detection_counts": {"bird": 2, "rabbit": 1},
  "detections": [
    {
      "class_name": "bird",
      "confidence": 0.85,
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
    }
  ]
}
```

## Common Issues

### General Issues

**"No module named 'src.stream_capture'"**: Run from repo root, not from src/

**Camera connection fails**: Check IP, credentials, firewall. Run `tests/test_camera_connection.py`

**High latency (>100ms)**: Switch to sub stream, reduce resolution, or use yolov8s instead of yolov8x

### Wildlife Detection Issues

**False positives on distant objects**: Increase `min_box_area` to 200-300px²

**Missing wildlife detections**: Lower `confidence` threshold to 0.2, verify YOLO-World classes include the animal

**YOLO-World detects telescope cover as "person"**: This is why custom telescope model is needed! Train custom model to properly identify telescope parts.

### Telescope Detection Issues

**Custom model not detecting telescope parts**:
- Lower confidence threshold to 0.3-0.4
- Check if model was trained properly (mAP50 > 0.80)
- Verify you're using correct model path in config
- Collect more diverse training images and retrain

**Tripod legs not detected**:
- Most critical for collision detection
- Ensure training images include clear views of all three legs
- Annotate legs precisely during training
- May need separate "tripod_leg_close" vs "tripod_leg_far" classes

**Training fails with OOM error**:
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `--base-model yolov8n.pt`
- Reduce image size: `--imgsz 320`

**Poor training results (mAP50 < 0.70)**:
- Collect more diverse training images (100-200+ recommended)
- Improve annotation quality (tight, consistent boxes)
- Train for more epochs (200-300)
- Try larger base model (yolov8m or yolov8x)

## Code Style Notes

- Logging: Use `logging.getLogger(__name__)` in each module
- Queues: Use `Queue(maxsize=N)` for backpressure control
- Threading: All components implement `.start()` and `.stop()` methods
- Config: Load from YAML, validate in each component's `__init__`
- Error handling: Components return False on init failure, log errors clearly

## Advanced Features (Framework Ready, Not Active)

### Stage 2 Species Classification (Wildlife)

Fine-grained species identification for wildlife (e.g., "Gambel's Quail" instead of just "bird").

**Status**: Framework exists but requires trained models
**Files**: `src/species_classifier.py`, `src/two_stage_pipeline.py`
**Documentation**: `STAGE2_SETUP.md`
**Requirements**: 3 trained models (bird_classifier.pth, mammal_classifier.pth, reptile_classifier.pth)
**Activation**: Set `detection.use_two_stage: true` in config + provide model weights
**Performance**: Adds ~5-10ms per detection

### Collision Detection System (Telescope)

Prevent accidents by detecting proximity between telescope parts and hazards.

**Status**: Config placeholders exist, logic needs implementation
**Key Components**:
- Detection zones (polygons/circles around telescope equipment)
- Distance calculation between detected objects
- Velocity tracking for moving objects
- Alert system for collision warnings
- Emergency stop integration (future)

**Implementation Checklist**:
- [ ] Train custom telescope detection model
- [ ] Define detection zones in config
- [ ] Implement distance calculation logic
- [ ] Add collision risk scoring algorithm
- [ ] Create alert system (visual/audio)
- [ ] Test with simulated collision scenarios
- [ ] Add velocity tracking for prediction
- [ ] Integrate with telescope mount control (optional)

### Multi-Camera Support (Future)

Monitor telescopes from multiple angles for comprehensive coverage.

**Requirements**:
- Multiple Reolink cameras
- Modified `stream_capture.py` for multi-stream handling
- Updated inference engine for batch processing
- Composite view in web UI

### Telescope Mount Integration (Future)

Direct integration with INDI/ASCOM telescope control software for emergency stop commands.

**Use Case**: Auto-stop telescope slew if collision risk detected
**Platforms**: INDI (Linux), ASCOM (Windows)
**Safety**: Requires hardware safety interlocks
