# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Performance optimizations** (#133, #134, #138) - 2025-10-15
  - WebSocket message optimization: Skip empty detection frames, reducing network traffic by 80-95%
  - Real-ESRGAN LRU cache: 95-99% reduction in enhancement time for repeated detections
  - Batched GPU inference foundation: `detect_batch()` method for future multi-camera batching
  - 8x8 grayscale thumbnail MD5-based cache (similarity heuristic) with configurable size (default: 100 entries)
  - Cache hit rate tracking and performance statistics
  - Status updates sent every 5 seconds when no wildlife detected
  - Enables Real-ESRGAN in production (previously too slow at 1000ms per enhancement)

- **Grid view enhancements for multi-camera monitoring** (#94) - 2025-10-15
  - Layout selector dropdown (auto, 1×1, 2×1, 2×2, 3×2, 3×3)
  - Auto layout based on number of visible cameras
  - Camera visibility toggles (show/hide specific cameras)
  - Live FPS counter per-camera in top-left corner
  - Detection count badges (orange badge, shows for 3s after detection)
  - Flash border animation on new detections (orange glow effect)
  - Connection status indicator (green/red dot in top-right)
  - Click any camera to fullscreen and switch to single view
  - Hover effects with scale and glow
  - Smooth CSS transitions and animations

- **Privacy-preserving face masking** (#88) - 2025-10-15
  - Automatic face detection and masking in live feeds
  - Multiple masking styles: gaussian blur, pixelate, black box, adaptive blur
  - Configurable detection intervals for performance optimization
  - Per-camera face masking overrides
  - Backend retains unmasked versions for security investigation
  - OpenCV Haar Cascade and MediaPipe backends supported

- **Camera health monitoring and automatic restart** (#83) - 2025-10-14
  - Automatic health monitoring for all cameras
  - Configurable health checks: FPS, frame age, connection status, error rate
  - Automatic restart with exponential backoff on failures
  - Isolated camera restart (others continue running)
  - Health score calculation (0-100)
  - REST API endpoints for health status and manual restart
  - Configurable restart attempts, cooldown periods, and backoff multipliers

- **Configuration hot-reload** (#85) - 2025-10-14
  - Reload configuration without system restart via API endpoint
  - Hot-reloadable settings: detection thresholds, class overrides, snapshot settings, filters
  - Identifies settings that require restart
  - REST API: `POST /api/config/reload`
  - Thread-safe reload with validation

- **Time-of-day filtering system** (#22)
  - Detection-level confidence adjustment based on species activity patterns
  - Stage 2 species re-ranking using time-of-day context
  - Species activity database (128+ species with diurnal/nocturnal/crepuscular/cathemeral patterns)
  - Automatic timezone detection with DST support
  - Configurable confidence penalties for out-of-pattern detections
  - Alternative suggestions (e.g., "bird" at night → suggest "bat")
  - Examples: Birds at night penalized 70% (likely bugs/bats), reptiles at night penalized (need warmth)

### Changed
- **GPU tensor pipeline optimization** (#92) - 2025-10-15
  - Zero-copy GPU tensor pipeline from video decode through Stage 2 classification
  - Frames kept as GPU tensors throughout pipeline (no CPU transfers until web streaming)
  - PyTorch GPU preprocessing in Stage 2 classifier (replaces cv2 CPU operations)
  - ~30-50% CPU reduction from Phase 1 GPU tensor storage
  - ~10-20% additional CPU reduction from Phase 2 GPU preprocessing
  - Combined with PR #91: Total CPU usage reduced from 1062% to ~400% (62% reduction)
  - Updated yolox_detector, species_classifier, visualization_utils, detection_processor, web_server, and snapshot_saver to handle GPU tensors

- **Code quality improvements** (#89) - 2025-10-15
  - Removed hardcoded magic numbers throughout codebase
  - Created centralized constants module (`src/constants.py`)
  - Made RTSP retry settings configurable (`rtsp_max_failures`, `rtsp_retry_delay`)
  - Made web streaming settings configurable (`mjpeg_fps`, `jpeg_quality`)
  - Improved code maintainability and readability

### Fixed
- **GPU video decode performance regression on A30** (#152) - 2025-10-15
  - Reverted GPU video decode (h264_cuvid via FFmpeg) to CPU decode (OpenCV)
  - A30 lacks dedicated NVDEC hardware → GPU decode was inefficient (261% CPU via FFmpeg)
  - Kept GPU image resize (torch.nn.functional.interpolate) - working well!
  - FFmpeg CPU usage: 261% → 0% (eliminated entirely)
  - Hybrid approach (CPU decode + GPU resize) optimal for A30 compute GPU architecture
  - Partially reverts PR #91 (GPU decode), keeps GPU resize from same PR
  - Performance: ~40% better than GPU decode, stable 26-27 fps throughput

- **JPEG encoding bottleneck causing 72% result drops** (#150, #147, #148) - 2025-10-15
  - Skip frame buffering in "image" snapshot mode (only buffer for "clip" mode)
  - Only draw detection annotations when actually saving (not every frame)
  - Eliminated CPU-intensive JPEG encoding on every frame
  - Result drop rate: 72% → <5%
  - CPU usage reduced significantly

- **GPU memory and performance optimizations** (#146, #145) - 2025-10-15
  - Eliminated redundant GPU tensor copy in detection processor (#115)
  - Single GPU copy (1.5ms) instead of double copy (3ms) - 50% improvement
  - Optimized buffer memory tracking with incremental calculation (#116)
  - Prevents O(n) memory sum on every frame add (30-60ms → <0.1ms)

- **Client timeout and stability fixes** (#144, #143) - 2025-10-15
  - Prevent client timeout on JPEG encoding failures (#113)
  - Return cached frame on encoding error instead of breaking stream
  - Medium priority bug fixes batch 2 (#107, #110, #111, #112):
    - YOLOX model loading with retry logic (exponential backoff)
    - Safe cleanup of failed cameras (queue emptying, resource cleanup)
    - Active camera validation (prevents startup with 0 cameras)
    - Shared coordinator memory leak (bounded deque for metrics)

- **High-priority bug fixes** (#129) - 2025-10-15
  - GPU device string comparison causing unnecessary GPU→GPU transfers (10-20ms improvement)
  - Missing frame validation after cv2.resize() preventing web server crashes
  - Queue overflow observability with drop rate tracking (drops/sec and overall %)
  - VideoWriter failure resource leaks (cleanup of partial video files)
  - Input size validation for aspect ratio and model compatibility
  - Config hot-reload race condition with copy-on-write pattern
  - Per-class confidence threshold constant (MIN_TIME_DELTA) for better maintainability

- **Critical bug fixes: GPU race conditions and resource leaks** (#128) - 2025-10-15
  - GPU tensor race condition in queue operations (cloning before queuing)
  - Thread orphaning with enhanced timeout error logging (WARNING→ERROR level)
  - FFmpeg zombie processes on failed termination (proper wait() after kill())
  - GPU memory leak in snapshot frame buffer (explicit tensor cleanup)
  - Prevents CUDA errors, system crashes, GPU OOM, and zombie processes

- **Critical bug fixes** (#84) - 2025-10-14
  - Fixed VideoWriter initialization and frame writing
  - Fixed thread cleanup on shutdown
  - Fixed YOLOX model validation
  - Fixed configuration validation edge cases

- **Missing type imports** (#86) - 2025-10-14
  - Added missing Dict and Any imports from typing module
  - Fixes type hints compatibility

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
