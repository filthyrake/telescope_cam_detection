# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Coming Soon
- Docker containerization for easy deployment
- GitHub Actions CI/CD pipeline
- Installation automation script
- Additional example configurations

---

## [1.0.0] - 2025-01-06

### Initial Public Release 🎉

First public release of the Telescope Detection System - a real-time object detection platform for wildlife monitoring and telescope collision prevention.

### Added

#### Core Detection System
- **YOLOX-S object detection** (Apache 2.0 license)
  - 11-21ms inference time on NVIDIA A30
  - 80 COCO classes with wildlife filtering
  - Configurable confidence thresholds
  - Per-class confidence overrides
- **Multi-camera support** (tested with 2 cameras)
  - Per-camera configuration and overrides
  - Independent inference engines
  - Synchronized web UI display
- **Optional iNaturalist Stage 2 classification**
  - 10,000 species fine-grained identification
  - EVA02-Large model integration
  - Hierarchical taxonomic fallback
  - Geographic filtering (Mojave Desert species whitelist)
  - Per-camera preprocessing settings

#### Camera Integration
- **RTSP stream capture** with multiple protocol support
  - rtsp (UDP), rtsp-tcp (TCP), onvif, h265
  - Automatic reconnection on failures
  - Low-latency buffering (1-frame buffer)
  - TCP transport for reliability (prevents screen tearing)
- **Reolink camera support**
  - RLC-410W (2560x1440)
  - E1 Pro (2560x1440)
  - Main and sub stream support

#### Web Interface
- **Real-time video streaming** via HTTP
- **WebSocket detection feed** with live updates
- **Multi-camera view** with synchronized streams
- **Detection overlays** with bounding boxes and labels
- **Performance metrics** (FPS, latency, GPU usage)
- **Detection history** and statistics
- **Responsive design** for desktop and mobile

#### Snapshot System
- **Automatic clip saving** on detection events
- **Configurable triggers** (by class, confidence)
- **Cooldown system** prevents duplicate saves
- **Annotated + raw versions** saved
- **Gallery view** for reviewing captures
- **Metadata tracking** (timestamp, confidence, class)

#### Performance
- **25-35ms end-to-end latency** (Stage 1 only)
- **30-50ms with Stage 2 classification**
- **25-30 FPS sustained** on dual cameras
- **~2GB VRAM per camera** on NVIDIA A30
- **~500MB RAM** per camera process

#### Configuration
- **YAML-based configuration** system
- **Separate credentials file** (gitignored)
- **Per-camera detection overrides**
- **Per-camera Stage 2 preprocessing**
- **Flexible model selection** (YOLOX variants)
- **Geographic species filtering**

#### Documentation
- **Comprehensive README** with quick start guide
- **Complete API reference** (REST + WebSocket)
- **Architecture documentation** with diagrams
- **Config reference** with all parameters
- **Training guides** for custom models
- **Setup guides** (systemd service, production deployment)
- **Security policy** (SECURITY.md)
- **Contribution guidelines** (CONTRIBUTING.md)

#### Testing
- Camera connection tests
- Inference performance tests
- End-to-end latency tests
- Stage 2 integration tests
- System startup tests

#### Deployment
- **systemd service** configuration
- **Service management script** (service.sh)
- **Auto-restart** on failures
- **Log management** and rotation
- **Production-ready** hardening options

### Changed
- Migrated from Ultralytics YOLO (AGPL-3.0) to YOLOX (Apache 2.0) for license compatibility
- Replaced GroundingDINO (120ms inference) with YOLOX (11-21ms, 47x faster)
- Updated all documentation for current system architecture

### Technical Details

**System Stack:**
- Python 3.11+
- PyTorch 2.0+ with CUDA 11.8+
- YOLOX (Apache 2.0)
- iNaturalist EVA02 (Apache 2.0)
- OpenCV 4.x (Apache 2.0)
- FastAPI + Uvicorn (MIT)

**Hardware Tested:**
- NVIDIA A30 GPU (24GB HBM2)
- Ubuntu 22.04 LTS
- Reolink RLC-410W cameras
- Reolink E1 Pro cameras

**License:**
- Project: MIT License
- All dependencies: Apache 2.0, BSD-3-Clause, or MIT compatible

### Known Limitations

- No built-in authentication for web UI (use reverse proxy)
- RTSP streams unencrypted (credentials sent in clear text on LAN)
- No input validation on YAML config
- Manual setup required (automated installation coming soon)

### Security

- Credentials stored in gitignored `camera_credentials.yaml`
- Config file separation (IP addresses in separate gitignored file)
- Git history cleaned of all leaked credentials
- No hardcoded passwords or sensitive data

### Contributors

Special thanks to:
- Claude Code by Anthropic for development assistance
- YOLOX team (Megvii, Inc.) for the detection model
- iNaturalist for species classification model
- PyTorch, OpenCV, FastAPI teams for core libraries

---

## Release Notes

### v1.0.0 - Initial Public Release

This is the first public release! The system has been in private development and testing since late 2024. Key milestones:

- **2024-12**: Initial prototype with Ultralytics YOLO
- **2025-01-04**: Migration to YOLOX for license compatibility
- **2025-01-05**: Multi-camera support added
- **2025-01-06**: Stage 2 classification integration
- **2025-01-06**: Documentation overhaul and security hardening
- **2025-01-06**: Community health files and open-source preparation
- **2025-01-06**: Public release

### Upgrading

This is the initial release - no upgrade path needed.

### Breaking Changes

None - initial release.

---

## Links

- [Project Repository](https://github.com/filthyrake/telescope_cam_detection)
- [Issue Tracker](https://github.com/filthyrake/telescope_cam_detection/issues)
- [Discussions](https://github.com/filthyrake/telescope_cam_detection/discussions)
- [Security Policy](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)

---

**Legend:**
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for security-related changes
