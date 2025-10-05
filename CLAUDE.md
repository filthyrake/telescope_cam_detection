# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time object detection system for **telescope collision prevention** and **wildlife monitoring** using a Reolink RLC-410W camera and NVIDIA A30 GPU.

**Dual Purpose System**:
1. **Telescope Safety** (Primary Goal): Detect telescope equipment parts (tube, mount, tripod legs) and potential collision hazards (people, animals). Prevent accidents with telescope equipment by alerting when objects are too close to tripod legs or moving telescope components.

2. **Wildlife Monitoring** (Secondary Goal): Monitor desert animals (coyotes, rabbits, quail, roadrunners, lizards, etc.) using GroundingDINO open-vocabulary detection with 93 comprehensive text prompts.

**Current Status**:
- Wildlife detection: **ACTIVE** (using GroundingDINO - Apache 2.0 license)
- Telescope detection: **PLANNED** (custom model requires annotated dataset)
- Collision detection logic: **FRAMEWORK READY** (config placeholders exist)
- License: **MIT** (all dependencies Apache 2.0 or MIT compatible)

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

**Detection Pipeline**:
- **Stage 1**: GroundingDINO open-vocabulary detection with 93 text prompts
  - Comprehensive desert wildlife: mammals, birds, reptiles, amphibians
  - Specific species: "zebra-tailed lizard", "gambel's quail", "desert iguana", etc.
- **Stage 2 (Optional)**: iNaturalist species classifier for fine-grained ID
  - Framework ready but disabled for initial testing
  - Can be re-enabled after GroundingDINO validation

**Snapshot Cooldown System**: Prevents duplicate saves using per-class cooldown timers (default 45s).

## Configuration System

**Main Config**: `config/config.yaml`

Critical settings:
- `detection.model.config`: `models/GroundingDINO_SwinT_OGC.py` (Apache 2.0)
- `detection.model.weights`: `models/groundingdino_swint_ogc.pth` (662MB)
- `detection.text_prompts`: List of 93 comprehensive wildlife + human prompts
- `detection.box_threshold`: 0.25 (confidence for bounding boxes)
- `detection.text_threshold`: 0.25 (confidence for text-image matching)
- `detection.min_box_area`: 50px² (catches small distant lizards)
- `snapshots.enabled`: true (saves detections to clips/)
- `snapshots.trigger_classes`: Which animals trigger saves
- `detection_zones`: **PLANNED** - Define danger zones around telescope equipment
- `collision_detection.enabled`: **PLANNED** - Enable proximity alerts

**Model Selection Strategy**:
- **GroundingDINO** (Apache 2.0): For wildlife using 93 text prompts (no training needed)
  - Open-vocabulary detection with natural language
  - Supports specific species names: "desert iguana", "zebra-tailed lizard"
- **Custom Model** (future): For telescope parts (requires training on 7 classes)
  - Will need to train separate model or fine-tune GroundingDINO
- **License**: All components MIT/Apache 2.0 compatible for open source release

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
- `stream_capture.py`: RTSP stream handling
- `inference_engine.py`: GroundingDINO inference (Apache 2.0)
- `detection_processor.py`: Post-processing, snapshot coordination
- `web_server.py`: FastAPI + WebSocket interface
- `snapshot_saver.py`: Saves detection clips
- `two_stage_pipeline.py`: Optional iNaturalist classifier (disabled)
- `species_classifier.py`: Species classification (disabled)

### Key Files
- `main.py`: Entry point
- `config/config.yaml`: Main configuration (93 text prompts here)
- `models/`: GroundingDINO weights (662MB) + config

## Development Workflow

### Training Custom Telescope Detection Model (Future)

**Note**: Currently using GroundingDINO for wildlife. Telescope detection is planned for future.

If you're asked to help with telescope detection:
1. Training images are in `training/datasets/telescope_equipment/` (1,050+ images)
2. See `training/TRAINING_GUIDE.md` for complete workflow
3. Will need to train separate model or fine-tune GroundingDINO
4. Target: Detect telescope tube, mount, tripod legs for collision prevention

### Adding New Wildlife Detection Classes
1. Edit `config/config.yaml` → `detection.text_prompts`
2. Add text prompt for new animal (e.g., "coati", "ringtail")
3. Restart system (no training needed with GroundingDINO open-vocabulary)

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
- Logs show inference times (currently ~120ms, target <50ms with TensorRT)

## GPU Optimization

**Current Performance (A30)**:
- GroundingDINO: ~120ms inference (unoptimized)
- Target with TensorRT: ~13ms (Phase 4 - planned)
- VRAM usage: ~2-3GB
- Use `/stats` endpoint for detailed metrics

## API Quick Reference

- **WebSocket**: `ws://localhost:8000/ws/detections` (real-time detections)
- **HTTP**: `GET /stats` (performance metrics)
- See `API_REFERENCE.md` for complete documentation

## Common Issues

**"No module named 'src.stream_capture'"**: Run from repo root, not from src/

**Camera connection fails**: Check IP, credentials, firewall. Run `tests/test_camera_connection.py`

**High latency (>160ms)**: Expected with current GroundingDINO. Will improve with TensorRT (Phase 4)

**False positives**: Increase `min_box_area` or adjust `class_confidence_overrides`

**Missing detections**: Lower `box_threshold` or add more specific text prompts

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

- **GroundingDINO** (Apache 2.0) - https://github.com/IDEA-Research/GroundingDINO
  - Open-vocabulary object detection
- **iNaturalist/EVA02** (Apache 2.0) - https://github.com/huggingface/pytorch-image-models  
  - 10,000 species classification
- **PyTorch** (BSD-3-Clause) - https://pytorch.org/
  - Deep learning framework
- **OpenCV** (Apache 2.0) - https://opencv.org/
  - Computer vision library
- **FastAPI** (MIT) - https://fastapi.tiangolo.com/
  - Web framework

All components are MIT/Apache 2.0/BSD compatible.
