# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time object detection system for **telescope collision prevention** and **wildlife monitoring** using Reolink cameras and NVIDIA A30 GPU.

**Dual Purpose System**:
1. **Telescope Safety** (Primary Goal): Detect telescope equipment parts (tube, mount, tripod legs) and potential collision hazards (people, animals). Prevent accidents with telescope equipment by alerting when objects are too close to tripod legs or moving telescope components.

2. **Wildlife Monitoring** (Secondary Goal): Monitor desert animals using YOLOX COCO detection + optional iNaturalist species classification.

**Current Status**:
- Wildlife detection: **ACTIVE** (using YOLOX-S at 11-21ms inference)
- Multi-camera: **ACTIVE** (2 cameras: cam1 + cam2)
- Species classification: **FRAMEWORK READY** (iNaturalist Stage 2, 10,000 species)
- Telescope detection: **PLANNED** (custom model requires annotated dataset)
- Collision detection logic: **FRAMEWORK READY** (config placeholders exist)
- License: **MIT** (all dependencies Apache 2.0/BSD/MIT compatible)

**Camera Setup**: Two Reolink cameras (RLC-410W + E1 Pro) overlooking backyard telescope and desert terrain

## Development Commands

### Initial Setup: Camera Credentials

**IMPORTANT**: Camera credentials are stored separately from config files and never committed to git.

```bash
# First time setup: Copy the example credentials file
cp camera_credentials.example.yaml camera_credentials.yaml

# Edit with your actual camera credentials
nano camera_credentials.yaml
```

The `camera_credentials.yaml` file format:
```yaml
cameras:
  cam1:
    username: "admin"
    password: "your_password_here"
  cam2:
    username: "admin"
    password: "your_password_here"
```

**Security Notes**:
- `camera_credentials.yaml` is gitignored and will NEVER be committed
- `config/config.yaml` contains NO credentials (only camera IPs and settings)
- Credentials are loaded at runtime and merged with camera configs
- Never commit real credentials to version control

### Running the System
```bash
# Activate virtual environment (always required)
source venv/bin/activate

# Run main application (will load credentials automatically)
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
Multi-Camera RTSP → Stream Captures → Frame Queues → Inference Engines → Detection Processors → Web Server
                                                              ↓
                                                   (Optional) Stage 2 Classifier
                                                              ↓
                                                       Snapshot Savers
                                                              ↓
                                                        clips/ directory
```

### Threading Model
- **Stream Capture Threads**: One per camera, continuously grabs frames from RTSP (Queue size=1)
- **Inference Threads**: One per camera, YOLOX GPU inference on queued frames
- **Stage 2 Thread (Optional)**: iNaturalist species classification when enabled
- **Detection Processor Threads**: Post-process detections, coordinate snapshot saving
- **Web Server (Main Thread)**: FastAPI/Uvicorn serving video feeds + WebSocket detections

### Key Design Patterns

**Ultra-Low-Latency Pipeline**: Achieves 25-35ms end-to-end with YOLOX inference at 11-21ms.

**Detection Pipeline**:
- **Stage 1**: YOLOX object detection (80 COCO classes)
  - Wildlife filtering: person, cat, dog, bird, horse, cow, bear, etc.
  - Configurable per-camera thresholds and overrides
  - Input size: [1920, 1920] for small distant wildlife detection
- **Stage 2 (Optional)**: iNaturalist EVA02 species classification
  - 10,000 species from iNaturalist 2021
  - Per-camera preprocessing settings (crop padding, min size)
  - Only runs when Stage 1 detects wildlife-relevant classes

**Snapshot Cooldown System**: Prevents duplicate saves using per-class cooldown timers (default 45s).

## Configuration System

**Main Config**: `config/config.yaml`
**Credentials**: `camera_credentials.yaml` (gitignored, never committed)

Critical settings:
- `cameras[]`: Multi-camera configuration array
  - Per-camera detection overrides (thresholds, min_box_area)
  - Per-camera Stage 2 preprocessing settings
- `detection.model.name`: `yolox-s` (balanced speed/accuracy)
- `detection.model.weights`: `models/yolox/yolox_s.pth` (69MB)
- `detection.input_size`: [1920, 1920] (larger for small wildlife)
- `detection.conf_threshold`: 0.15 (low for distant animals)
- `detection.wildlife_only`: true (filters to relevant COCO classes)
- `detection.class_confidence_overrides`: Per-class thresholds (e.g., person: 0.60)
- `snapshots.enabled`: true (saves detections to clips/)
- `snapshots.trigger_classes`: Which animals trigger saves
- `detection_zones`: **PLANNED** - Define danger zones around telescope equipment
- `collision_detection.enabled`: **PLANNED** - Enable proximity alerts

**Model Selection Strategy**:
- **YOLOX** (Apache 2.0): Stage 1 detection (80 COCO classes, 11-21ms)
  - `yolox-s`: Current (balanced)
  - `yolox-tiny`: Faster for constrained systems
  - `yolox-x`: Higher accuracy when needed
- **iNaturalist** (Apache 2.0): Stage 2 classification (10,000 species, +20-30ms)
  - Optional fine-grained species ID
  - EVA02-Large-Patch14-CLIP-336 model
- **Custom Model** (future): For telescope parts (requires training on 7 classes)
  - Will train YOLOX on telescope dataset
- **License**: All components MIT/Apache 2.0/BSD compatible for open source release

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
- `stream_capture.py`: Multi-camera RTSP stream handling
- `inference_engine_yolox.py`: YOLOX GPU inference (Apache 2.0) **[CURRENT]**
- `two_stage_pipeline_yolox.py`: Stage 2 iNaturalist classifier (optional)
- `species_classifier.py`: Species classification (iNaturalist EVA02)
- `yolox_detector.py`: YOLOX model wrapper
- `detection_processor.py`: Post-processing, snapshot coordination
- `web_server.py`: FastAPI + WebSocket interface
- `snapshot_saver.py`: Saves detection clips
- `image_enhancement.py`: Optional preprocessing (CLAHE, Real-ESRGAN)

### Backup Files (for reference)
- `inference_engine.py.groundingdino.backup`: Previous GroundingDINO implementation
- `inference_engine.py.ultralytics.backup`: Original Ultralytics YOLO implementation
- `config/config.yaml.groundingdino.backup`: Previous config

### Key Files
- `main.py`: Entry point (multi-camera orchestration)
- `config/config.yaml`: Main configuration
- `camera_credentials.yaml`: Camera passwords (gitignored)
- `models/yolox/`: YOLOX model weights (69MB, auto-downloads)

## Development Workflow

### Training Custom Telescope Detection Model (Future)

**Note**: Currently using GroundingDINO for wildlife. Telescope detection is planned for future.

If you're asked to help with telescope detection:
1. Training images are in `training/datasets/telescope_equipment/` (1,050+ images)
2. See `training/TRAINING_GUIDE.md` for complete workflow
3. Will need to train separate model or fine-tune GroundingDINO
4. Target: Detect telescope tube, mount, tripod legs for collision prevention

### Adding New Wildlife Detection Classes

**Stage 1 (YOLOX)**: Limited to 80 COCO classes (pre-trained)
- Cannot add new classes without retraining
- Current classes cover most wildlife: person, cat, dog, bird, horse, cow, bear, etc.

**Stage 2 (iNaturalist)**: 10,000 species already supported
- No configuration needed - automatically classifies detected wildlife
- Enable in `config/config.yaml` → `detection.use_two_stage: true`

**Custom Training** (future): Train YOLOX on custom classes
- See `docs/training/` for training workflow
- Can add telescope parts, specific animals, etc.

### Implementing Collision Detection Logic

**Framework exists but not active**. Config placeholders are in `config/config.yaml`:
- `detection_zones`: Define danger zones around telescope
- `collision_detection.enabled`: Enable proximity alerts

If asked to implement, see comments in `detection_processor.py` for approach.

### Common Adjustments

**Detection thresholds** (in `config/config.yaml`):
- `box_threshold`: 0.25 (confidence for bounding boxes)
- `text_threshold`: 0.25 (text-image matching confidence)
- `min_box_area`: 50px² (filters tiny false positives)
- `class_confidence_overrides`: Per-class thresholds (e.g., person: 0.60)

**Debugging**:
- Check `clips/` directory for saved detections
- Use `scripts/view_snapshots.py` to review with metadata
- Logs show inference times (currently 11-21ms YOLOX, excellent performance)
- Use `./service.sh logs -f` to monitor live system logs

## GPU Optimization

**Current Performance (A30)**:
- YOLOX-S: ~11-21ms inference (excellent)
- Stage 2 (iNaturalist): +20-30ms when triggered
- VRAM usage: ~2GB per camera (~4GB total for 2 cameras)
- Use `/stats` endpoint for detailed metrics

**Future Optimization**:
- TensorRT: Can reduce to 3-5ms (not critical, current speed excellent)
- FP16: Already optimal for this workload
- Batch inference: Could process multiple cameras simultaneously

## API Quick Reference

- **WebSocket**: `ws://localhost:8000/ws/detections` (real-time detections)
- **HTTP**: `GET /stats` (performance metrics)
- See `docs/api/API_REFERENCE.md` for complete documentation

## Common Issues

**"No module named 'src.stream_capture'"**: Run from repo root, not from src/

**Camera connection fails**: Check IP, credentials, firewall. Run `tests/test_camera_connection.py`

**High latency (>50ms)**:
- Reduce `input_size` from [1920, 1920] to [640, 640]
- Switch to `yolox-tiny` or `yolox-nano`
- Check per-camera `detection_overrides`

**False positives**:
- Increase `conf_threshold` or per-camera overrides
- Increase `min_box_area`
- Adjust `class_confidence_overrides` (e.g., person: 0.80)

**Missing detections**:
- Lower `conf_threshold` (currently 0.15, very sensitive)
- Increase `input_size` for better small object detection
- Check `wildlife_only` filter isn't excluding needed classes

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

---

## Credits & Licenses

This project is licensed under the **MIT License**. All dependencies use permissive licenses compatible with MIT.

### Core Dependencies

- **YOLOX** (Apache 2.0) - https://github.com/Megvii-BaseDetection/YOLOX
  - Ultra-fast object detection (11-21ms inference)
- **iNaturalist/EVA02** (Apache 2.0) - https://github.com/huggingface/pytorch-image-models
  - 10,000 species classification (Stage 2)
- **PyTorch** (BSD-3-Clause) - https://pytorch.org/
  - Deep learning framework
- **OpenCV** (Apache 2.0) - https://opencv.org/
  - Computer vision library
- **FastAPI** (MIT) - https://fastapi.tiangolo.com/
  - Web framework

All components are MIT/Apache 2.0/BSD compatible.

### Migration History

- **v1.0**: Ultralytics YOLO (AGPL-3.0, license incompatible)
- **v1.1**: GroundingDINO (Apache 2.0, 120ms inference, too slow)
- **v1.2** (Current): YOLOX (Apache 2.0, 11-21ms inference, 47x faster)
