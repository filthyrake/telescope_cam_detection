# Telescope Detection System - Complete Documentation

Welcome to the complete documentation for the Telescope Detection System - a real-time wildlife monitoring and telescope collision prevention platform.

## 📚 Documentation Index

### Getting Started

| Document | Description | Audience |
|----------|-------------|----------|
| **[README.md](README.md)** | Project overview and quick start | Everyone |
| **[QUICKSTART.md](QUICKSTART.md)** | Fastest path from zero to running | New users |
| **[SERVICE_SETUP.md](SERVICE_SETUP.md)** | Complete systemd service guide | System admins |

### System Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture, components, data flow | Developers |
| **[API_REFERENCE.md](API_REFERENCE.md)** | Complete API documentation | Developers, integrators |
| **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)** | Configuration parameter guide | Users, admins |

### Features & Setup

| Document | Description | Audience |
|----------|-------------|----------|
| **[STAGE2_SETUP.md](STAGE2_SETUP.md)** | Species classification (iNaturalist) | Advanced users |
| **[SNAPSHOT_FEATURE.md](SNAPSHOT_FEATURE.md)** | Automatic clip saving | Users |

### Training & Development

| Document | Description | Audience |
|----------|-------------|----------|
| **[training/TRAINING_WORKFLOW.md](training/TRAINING_WORKFLOW.md)** | Complete training pipeline | ML engineers |
| **[training/COLLISION_SCENARIOS.md](training/COLLISION_SCENARIOS.md)** | Collision scenario capture guide | Telescope operators |
| **[ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)** | Image annotation instructions | Data annotators |
| **[CUSTOM_TRAINING_READY.md](CUSTOM_TRAINING_READY.md)** | Custom model training | ML engineers |

### Developer Resources

| Document | Description | Audience |
|----------|-------------|----------|
| **[CLAUDE.md](CLAUDE.md)** | Project context for AI assistants | Claude Code users |
| **[telescope_cam_detection_spec.md](telescope_cam_detection_spec.md)** | Original project spec | Developers |

---

## 🚀 Quick Navigation by Task

### I want to...

#### Install and Run
1. [Install the system](README.md#quick-start) (first time setup)
2. [Run as a service](SERVICE_SETUP.md#quick-start) (production deployment)
3. [Quick commands](QUICKSTART.md) (cheat sheet)

#### Configure the System
1. [Camera settings](CONFIG_REFERENCE.md#camera-configuration)
2. [Detection thresholds](CONFIG_REFERENCE.md#detection-thresholds)
3. [Enable species ID](STAGE2_SETUP.md)
4. [Configure snapshots](CONFIG_REFERENCE.md#snapshots-configuration)

#### Understand the System
1. [Architecture overview](ARCHITECTURE.md#system-overview)
2. [Component details](ARCHITECTURE.md#component-details)
3. [Data flow](ARCHITECTURE.md#data-flow)
4. [Performance characteristics](ARCHITECTURE.md#performance-characteristics)

#### Integrate with the System
1. [HTTP API endpoints](API_REFERENCE.md#http-endpoints)
2. [WebSocket API](API_REFERENCE.md#websocket-endpoint)
3. [Client libraries](API_REFERENCE.md#client-libraries)
4. [Data models](API_REFERENCE.md#data-models)

#### Train Custom Models
1. [Capture training data](training/TRAINING_WORKFLOW.md#phase-1-continue-static-captures)
2. [Extract frames from video](training/TRAINING_WORKFLOW.md#phase-2-dynamic-video-capture)
3. [Capture collision scenarios](training/COLLISION_SCENARIOS.md)
4. [Annotate images](ANNOTATION_GUIDE.md)
5. [Train model](CUSTOM_TRAINING_READY.md)

#### Troubleshoot Issues
1. [Service won't start](SERVICE_SETUP.md#troubleshooting)
2. [No detections](CONFIG_REFERENCE.md#no-detections)
3. [Poor performance](CONFIG_REFERENCE.md#poor-performance)
4. [Check logs](SERVICE_SETUP.md#viewing-logs)

---

## 📖 Documentation by Audience

### For End Users

**Goal**: Install, configure, and use the system

1. Start: [README.md](README.md) - Overview and quick start
2. Install: [QUICKSTART.md](QUICKSTART.md) - Fast installation
3. Service: [SERVICE_SETUP.md](SERVICE_SETUP.md) - Run as service
4. Configure: [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) - Tune settings
5. Features: [STAGE2_SETUP.md](STAGE2_SETUP.md), [SNAPSHOT_FEATURE.md](SNAPSHOT_FEATURE.md)

**Time Investment**: 2-4 hours to full operational system

---

### For Developers

**Goal**: Understand, modify, and integrate with the system

1. Architecture: [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system design
2. API: [API_REFERENCE.md](API_REFERENCE.md) - All endpoints and protocols
3. Config: [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) - Configuration system
4. Code: Browse [src/](src/) - Component implementations
5. Tests: [tests/](tests/) - Test scripts

**Key Files to Read**:
- `src/inference_engine.py` - Detection pipeline
- `src/two_stage_pipeline.py` - Two-stage architecture
- `src/web_server.py` - API implementation
- `main.py` - System orchestration

---

### For ML Engineers

**Goal**: Train custom models for telescope equipment detection

1. Workflow: [training/TRAINING_WORKFLOW.md](training/TRAINING_WORKFLOW.md) - Complete pipeline
2. Data Capture: 
   - Static: [training/scripts/capture_training_images_headless.py](training/scripts/capture_training_images_headless.py)
   - Video: [training/scripts/extract_frames_from_stream.py](training/scripts/extract_frames_from_stream.py)
   - Collision: [training/COLLISION_SCENARIOS.md](training/COLLISION_SCENARIOS.md)
3. Annotation: [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)
4. Training: [training/scripts/train_custom_model.py](training/scripts/train_custom_model.py)
5. Evaluation: [training/scripts/evaluate_model.py](training/scripts/evaluate_model.py)

**Dataset Requirements**:
- 2,000-3,000 images minimum
- 7 classes (telescope parts)
- Diverse lighting/positions
- Critical: Collision scenarios

---

### For System Administrators

**Goal**: Deploy, monitor, and maintain the system

1. Installation: [SERVICE_SETUP.md](SERVICE_SETUP.md) - systemd service
2. Monitoring: [SERVICE_SETUP.md#monitoring](SERVICE_SETUP.md#monitoring)
3. Logs: [SERVICE_SETUP.md#viewing-logs](SERVICE_SETUP.md#viewing-logs)
4. Troubleshooting: [SERVICE_SETUP.md#troubleshooting](SERVICE_SETUP.md#troubleshooting)
5. Performance: [ARCHITECTURE.md#performance-characteristics](ARCHITECTURE.md#performance-characteristics)

**Key Commands**:
```bash
./service.sh status      # Check service
./service.sh logs -f     # Monitor logs
./service.sh restart     # Restart after config change
nvidia-smi              # Check GPU
```

---

## 🎯 Common Use Cases

### Use Case 1: Wildlife Monitoring

**Goal**: Detect and identify desert animals

**Setup**:
1. Install system: [QUICKSTART.md](QUICKSTART.md)
2. Enable Stage 2: [STAGE2_SETUP.md](STAGE2_SETUP.md)
3. Configure classes: [CONFIG_REFERENCE.md#yolo-world-parameters](CONFIG_REFERENCE.md#yolo-world-parameters)
4. Enable snapshots: [CONFIG_REFERENCE.md#snapshots-configuration](CONFIG_REFERENCE.md#snapshots-configuration)

**Configuration**:
```yaml
detection:
  use_yolo_world: true
  yolo_world_classes: ["coyote", "rabbit", "bird", "lizard", ...]
  use_two_stage: true
  enable_species_classification: true
  confidence: 0.25
  min_box_area: 50

species_classification:
  enabled: true

snapshots:
  enabled: true
  trigger_classes: ["coyote", "rabbit", "bird"]
```

**Expected Results**:
- Real-time detection with species ID
- Automatic clip saving
- Web UI with live feed

---

### Use Case 2: Telescope Collision Prevention (Future)

**Goal**: Prevent telescope equipment collisions

**Setup**:
1. Capture training data: [training/TRAINING_WORKFLOW.md](training/TRAINING_WORKFLOW.md)
2. Capture collision scenarios: [training/COLLISION_SCENARIOS.md](training/COLLISION_SCENARIOS.md)
3. Annotate images: [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)
4. Train custom model: [training/scripts/train_custom_model.py](training/scripts/train_custom_model.py)
5. Deploy model: [CONFIG_REFERENCE.md#model-parameters](CONFIG_REFERENCE.md#model-parameters)

**Status**: In development, training data collection phase

---

### Use Case 3: API Integration

**Goal**: Build custom applications on top of detection system

**Resources**:
1. API Reference: [API_REFERENCE.md](API_REFERENCE.md)
2. WebSocket Protocol: [API_REFERENCE.md#websocket-endpoint](API_REFERENCE.md#websocket-endpoint)
3. Client Examples: [API_REFERENCE.md#client-libraries](API_REFERENCE.md#client-libraries)

**Example**: Python monitoring script
```python
import asyncio
import websockets

async def monitor():
    async with websockets.connect('ws://localhost:8000/ws/detections') as ws:
        async for msg in ws:
            data = json.loads(msg)
            print(f"Detected: {data['total_detections']}")

asyncio.run(monitor())
```

---

## 🔧 Technical Specifications

### System Requirements

| Component | Specification |
|-----------|---------------|
| OS | Ubuntu 22.04 LTS (or similar) |
| GPU | NVIDIA A30 (or any CUDA GPU) |
| CUDA | 11.8+ |
| Python | 3.11+ |
| RAM | 8GB+ |
| Storage | 50GB+ (for clips) |
| Network | 10+ Mbps (camera stream) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Latency (Stage 1 only) | 20-25ms |
| Latency (Stage 1+2) | 30-35ms |
| FPS | 25-30 |
| GPU VRAM | ~2GB |
| CPU RAM | ~500MB |
| Detection range | 0-50ft effectively |

### Supported Cameras

- **Reolink RLC-410W** (tested, recommended)
- Any RTSP-compatible camera
- Resolution: 720p - 4K (auto-resized)

### Detection Capabilities

**Stage 1 (YOLO-World)**:
- 20 wildlife classes (configurable)
- Open-vocabulary detection
- Custom text prompts

**Stage 2 (iNaturalist)**:
- 10,000 species
- 92% top-1 accuracy
- Detailed taxonomy

---

## 📝 Version History

### v1.2 (Current - 2025-10-05)
- ✅ Stage 2 species classification (iNaturalist)
- ✅ Two-stage detection pipeline
- ✅ Enhanced lizard detection (4 prompts, lower threshold)
- ✅ systemd service deployment
- ✅ Comprehensive documentation

### v1.1 (2025-10-04)
- Stage 2 framework
- iNaturalist model integration
- Training infrastructure

### v1.0 (Initial)
- YOLO-World detection
- Real-time video streaming
- Web interface
- Snapshot saving

---

## 🤝 Contributing

### Reporting Issues

Found a bug or have a suggestion?

1. Check service logs: `./service.sh logs`
2. Review troubleshooting: [SERVICE_SETUP.md#troubleshooting](SERVICE_SETUP.md#troubleshooting)
3. Document the issue with logs and config

### Documentation Updates

To update documentation:

1. Edit relevant `.md` file
2. Follow existing format/style
3. Update this index if adding new docs
4. Test all code examples

---

## 📞 Support Resources

### Documentation Files

All documentation is in markdown format in this repository:
- `/` - Main docs (this file, README, etc.)
- `/training/` - Training-related docs
- `/tests/` - Test scripts and examples

### Log Files

```bash
# Service logs
./service.sh logs
./service.sh logs -f

# System logs
journalctl -u telescope_detection.service

# Application logs
# (Sent to systemd journal)
```

### Configuration

- Main config: `config/config.yaml`
- Service: `telescope_detection.service`
- Management: `service.sh`

---

## 🗺️ Project Structure

```
telescope_cam_detection/
├── DOCUMENTATION.md          # This file (master index)
├── README.md                 # Project overview
├── QUICKSTART.md            # Fast start guide
├── ARCHITECTURE.md          # System architecture
├── API_REFERENCE.md         # API documentation
├── CONFIG_REFERENCE.md      # Configuration guide
├── SERVICE_SETUP.md         # Service deployment
├── STAGE2_SETUP.md          # Species classification
├── SNAPSHOT_FEATURE.md      # Clip saving feature
├── ANNOTATION_GUIDE.md      # Data annotation
├── CUSTOM_TRAINING_READY.md # Model training
├── CLAUDE.md                # AI assistant context
│
├── main.py                  # Entry point
├── config/                  # Configuration
├── src/                     # Core modules
├── web/                     # Web UI
├── models/                  # Model weights
├── clips/                   # Saved detections
├── training/                # Training infrastructure
│   ├── TRAINING_WORKFLOW.md
│   ├── COLLISION_SCENARIOS.md
│   ├── datasets/
│   └── scripts/
├── tests/                   # Test scripts
└── scripts/                 # Utility scripts
```

---

## 🎓 Learning Path

### Beginner (2-4 hours)

1. Read [README.md](README.md) - 15 min
2. Follow [QUICKSTART.md](QUICKSTART.md) - 30 min
3. Set up service [SERVICE_SETUP.md](SERVICE_SETUP.md) - 60 min
4. Explore web UI - 30 min
5. Tune config [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) - 60 min

**Outcome**: Running system with basic understanding

---

### Intermediate (1-2 days)

1. Study [ARCHITECTURE.md](ARCHITECTURE.md) - 2 hours
2. Enable Stage 2 [STAGE2_SETUP.md](STAGE2_SETUP.md) - 2 hours
3. Explore [API_REFERENCE.md](API_REFERENCE.md) - 2 hours
4. Build simple integration - 4 hours
5. Optimize performance - 2 hours

**Outcome**: Deep understanding, can integrate and customize

---

### Advanced (1-2 weeks)

1. Collect training data [training/TRAINING_WORKFLOW.md](training/TRAINING_WORKFLOW.md) - 2 days
2. Annotate images [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) - 3-5 days
3. Train custom model [training/scripts/train_custom_model.py](training/scripts/train_custom_model.py) - 1 day
4. Deploy and test - 1 day
5. Fine-tune and iterate - 2-3 days

**Outcome**: Custom trained model for your specific use case

---

## 📄 License

See main [README.md](README.md) for license information.

---

## 🙏 Acknowledgments

- **YOLOv8 & YOLO-World**: Ultralytics
- **iNaturalist 2021**: Visipedia
- **EVA02**: BAAI Vision
- **timm**: HuggingFace
- **FastAPI**: Sebastián Ramírez

---

**Documentation Version**: 1.0  
**Last Updated**: 2025-10-05  
**Maintainer**: Damen + Claude (AI Assistant)

---

## 📚 Quick Links

- **Repository**: (Add your repo URL)
- **Issues**: (Add issues URL)
- **Discussions**: (Add discussions URL)
- **Web Interface**: http://localhost:8000
- **API Base**: http://localhost:8000/

---

*For questions, issues, or suggestions, check the troubleshooting guides or review the service logs.*
