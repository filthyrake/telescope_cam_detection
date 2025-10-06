# Telescope Detection System - Documentation

Complete documentation for the real-time wildlife monitoring and telescope collision prevention system.

## 📖 Quick Navigation

### Getting Started

| Document | Description | Time |
|----------|-------------|------|
| **[../README.md](../README.md)** | Project overview and quick start | 10 min |
| **[../QUICKSTART.md](../QUICKSTART.md)** | Fastest path to running system | 5 min |
| **[setup/SERVICE_SETUP.md](setup/SERVICE_SETUP.md)** | Run as systemd service (production) | 20 min |

### System Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)** | Complete system design & data flow | Developers |
| **[setup/CONFIG_REFERENCE.md](setup/CONFIG_REFERENCE.md)** | Configuration parameters guide | All users |
| **[api/API_REFERENCE.md](api/API_REFERENCE.md)** | HTTP & WebSocket APIs | Developers |

### Features

| Document | Description | Audience |
|----------|-------------|----------|
| **[features/SNAPSHOT_FEATURE.md](features/SNAPSHOT_FEATURE.md)** | Automatic clip saving | Users |
| **[features/STAGE2_SETUP.md](features/STAGE2_SETUP.md)** | Species classification (10,000 species) | Advanced users |
| **[features/STAGE2_ENHANCEMENT_OPTIONS.md](features/STAGE2_ENHANCEMENT_OPTIONS.md)** | Image preprocessing options | ML engineers |

### Training & Development

| Document | Description | Audience |
|----------|-------------|----------|
| **[training/TRAINING_WORKFLOW.md](training/TRAINING_WORKFLOW.md)** | Complete training pipeline | ML engineers |
| **[training/COLLISION_SCENARIOS.md](training/COLLISION_SCENARIOS.md)** | Capture collision scenarios | Telescope operators |
| **[training/ANNOTATION_GUIDE.md](training/ANNOTATION_GUIDE.md)** | Image annotation instructions | Data annotators |
| **[training/CUSTOM_TRAINING_READY.md](training/CUSTOM_TRAINING_READY.md)** | Train custom YOLOX models | ML engineers |

### Historical Reference

| Document | Description |
|----------|-------------|
| **[archive/YOLOX_INTEGRATION_SUMMARY.md](archive/YOLOX_INTEGRATION_SUMMARY.md)** | YOLOX migration summary (v1.2) |
| **[archive/MIGRATION_PLAN.md](archive/MIGRATION_PLAN.md)** | Full migration history (Ultralytics → GroundingDINO → YOLOX) |
| **[archive/telescope_cam_detection_spec.md](archive/telescope_cam_detection_spec.md)** | Original project specification |

---

## 🚀 Common Tasks

### I want to...

#### Install and Run
1. **First time setup**: [../README.md](../README.md#quick-start)
2. **Run as service**: [setup/SERVICE_SETUP.md](setup/SERVICE_SETUP.md)
3. **Quick commands**: [../QUICKSTART.md](../QUICKSTART.md)

#### Configure the System
1. **Camera settings**: [setup/CONFIG_REFERENCE.md](setup/CONFIG_REFERENCE.md#camera-configuration)
2. **Detection thresholds**: [setup/CONFIG_REFERENCE.md](setup/CONFIG_REFERENCE.md#detection-thresholds)
3. **Enable species ID**: [features/STAGE2_SETUP.md](features/STAGE2_SETUP.md)
4. **Configure snapshots**: [setup/CONFIG_REFERENCE.md](setup/CONFIG_REFERENCE.md#snapshots-configuration)

#### Understand the System
1. **Architecture overview**: [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)
2. **How it works**: [../README.md](../README.md#architecture)
3. **Performance characteristics**: [../README.md](../README.md#performance-notes)

#### Integrate with the System
1. **WebSocket API**: [api/API_REFERENCE.md](api/API_REFERENCE.md#websocket-endpoint)
2. **HTTP endpoints**: [api/API_REFERENCE.md](api/API_REFERENCE.md#http-endpoints)
3. **Data formats**: [api/API_REFERENCE.md](api/API_REFERENCE.md#data-models)

#### Train Custom Models
1. **Capture training data**: [training/TRAINING_WORKFLOW.md](training/TRAINING_WORKFLOW.md)
2. **Annotate images**: [training/ANNOTATION_GUIDE.md](training/ANNOTATION_GUIDE.md)
3. **Train model**: [training/CUSTOM_TRAINING_READY.md](training/CUSTOM_TRAINING_READY.md)

#### Troubleshoot Issues
1. **Service issues**: [setup/SERVICE_SETUP.md](setup/SERVICE_SETUP.md#troubleshooting)
2. **Configuration issues**: [setup/CONFIG_REFERENCE.md](setup/CONFIG_REFERENCE.md#troubleshooting)
3. **Check logs**: `./service.sh logs -f`

---

## 📚 Documentation by Audience

### For End Users

**Goal**: Install, configure, and use the system

**Reading Order**:
1. [../README.md](../README.md) - Project overview (10 min)
2. [../QUICKSTART.md](../QUICKSTART.md) - Quick installation (5 min)
3. [setup/SERVICE_SETUP.md](setup/SERVICE_SETUP.md) - Production deployment (20 min)
4. [setup/CONFIG_REFERENCE.md](setup/CONFIG_REFERENCE.md) - Tune settings (30 min)
5. [features/](features/) - Optional features

**Time Investment**: 1-2 hours to operational system

---

### For Developers

**Goal**: Understand, modify, and integrate

**Reading Order**:
1. [../CLAUDE.md](../CLAUDE.md) - Quick project context (15 min)
2. [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md) - System design (30 min)
3. [api/API_REFERENCE.md](api/API_REFERENCE.md) - Integration (20 min)
4. [../src/](../src/) - Source code review

**Key Files**:
- `src/inference_engine_yolox.py` - Detection engine
- `src/two_stage_pipeline_yolox.py` - Species classification
- `src/web_server.py` - API implementation
- `main.py` - System orchestration

---

### For ML Engineers

**Goal**: Train custom models

**Reading Order**:
1. [training/TRAINING_WORKFLOW.md](training/TRAINING_WORKFLOW.md) - Complete pipeline (1 hour)
2. [training/ANNOTATION_GUIDE.md](training/ANNOTATION_GUIDE.md) - Data annotation (30 min)
3. [training/CUSTOM_TRAINING_READY.md](training/CUSTOM_TRAINING_READY.md) - Model training (1 hour)
4. [training/COLLISION_SCENARIOS.md](training/COLLISION_SCENARIOS.md) - Scenario capture (30 min)

**Dataset Requirements**:
- 2,000-3,000 images minimum
- 7 telescope classes
- Diverse lighting/angles
- Critical: Collision scenarios

---

### For System Administrators

**Goal**: Deploy and maintain

**Reading Order**:
1. [setup/SERVICE_SETUP.md](setup/SERVICE_SETUP.md) - systemd service (30 min)
2. [setup/CONFIG_REFERENCE.md](setup/CONFIG_REFERENCE.md) - Configuration (30 min)
3. [../README.md](../README.md#troubleshooting) - Troubleshooting (15 min)

**Key Commands**:
```bash
./service.sh status      # Check service health
./service.sh logs -f     # Monitor logs
./service.sh restart     # Restart after config change
nvidia-smi              # Check GPU utilization
```

---

## 🎯 System Overview

### Current Technology Stack

**Detection (Stage 1):**
- **YOLOX-S**: 11-21ms inference on NVIDIA A30
- **80 COCO classes**: Filtered to wildlife-relevant
- **Multi-camera**: 2 cameras (RLC-410W + E1 Pro)

**Classification (Stage 2 - Optional):**
- **iNaturalist EVA02**: 10,000 species
- **+20-30ms**: When triggered
- **92% accuracy**: Top-1 on validation set

**Performance:**
- **25-30 FPS**: Real-time sustained
- **25-35ms**: End-to-end latency (Stage 1 only)
- **30-50ms**: With Stage 2 classification
- **~4GB VRAM**: Total for 2 cameras

### License

**MIT License** - Fully open source

All dependencies use permissive licenses:
- YOLOX: Apache 2.0
- iNaturalist/EVA02: Apache 2.0
- PyTorch: BSD-3-Clause
- OpenCV: Apache 2.0
- FastAPI: MIT

---

## 📞 Support

### Getting Help

1. **Check logs**: `./service.sh logs -f`
2. **Review config**: `config/config.yaml`
3. **Run tests**: `tests/test_*.py`
4. **Read docs**: You're here! 📖

### Common Issues

| Issue | Solution |
|-------|----------|
| Service won't start | Check logs, verify credentials file exists |
| No detections | Lower `conf_threshold`, check `wildlife_only` filter |
| High latency | Reduce `input_size`, switch to `yolox-tiny` |
| False positives | Increase thresholds, adjust `class_confidence_overrides` |

See [setup/SERVICE_SETUP.md](setup/SERVICE_SETUP.md#troubleshooting) for detailed troubleshooting.

---

## 🗺️ Project Structure

```
telescope_cam_detection/
├── README.md                    # Main entry point
├── CLAUDE.md                    # AI assistant context
├── QUICKSTART.md               # Fast setup guide
│
├── docs/                        # This directory!
│   ├── README.md               # This file
│   ├── architecture/           # System design
│   ├── setup/                  # Installation & config
│   ├── features/               # Feature guides
│   ├── api/                    # API reference
│   ├── training/               # Model training
│   └── archive/                # Historical docs
│
├── config/                     # Configuration
│   ├── config.yaml            # Main config
│   └── camera_credentials.yaml # Passwords (gitignored)
│
├── src/                        # Source code
│   ├── inference_engine_yolox.py
│   ├── two_stage_pipeline_yolox.py
│   ├── stream_capture.py
│   ├── detection_processor.py
│   └── web_server.py
│
├── tests/                      # Test scripts
├── training/                   # Training infrastructure
├── web/                        # Web UI
├── models/                     # Model weights
└── clips/                      # Saved detections
```

---

## 📝 Documentation Version

- **Version**: 2.0
- **Last Updated**: 2025-10-06
- **System Version**: v1.2 (YOLOX)
- **Maintainer**: Damen + Claude Code

---

## 🙏 Credits

- **YOLOX**: Megvii-BaseDetection
- **iNaturalist**: Visipedia / BAAI Vision
- **PyTorch**: Meta AI
- **FastAPI**: Sebastián Ramírez

---

*For the latest updates, check the main [README.md](../README.md)*
