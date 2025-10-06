# Telescope Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Real-time object detection system for monitoring astronomical telescopes and desert wildlife using Reolink cameras and NVIDIA A30 GPU.

## Features

- **Ultra-fast detection**: YOLOX inference at 11-21ms
- **Multi-camera support**: Monitor from multiple angles simultaneously
- **80 COCO classes**: Including all wildlife-relevant categories (person, bird, cat, dog, bear, etc.)
- **Optional species classification**: iNaturalist Stage 2 for fine-grained species ID (10,000 species)
- **GPU accelerated**: Optimized for NVIDIA A30
- **Real-time performance**: 25-30 FPS with detection overlays
- **Web interface**: Live video stream with detection overlays
- **WebSocket streaming**: Real-time detection results
- **Snapshot saving**: Automatic image/clip saving on detection events
- **MIT License**: Fully open source with permissive licensing (Apache 2.0 dependencies)

## Screenshots & Demo

> **Note**: The web interface includes:
> - Live RTSP video streams from multiple cameras
> - Real-time detection overlays with confidence scores
> - Species classification labels (when Stage 2 enabled)
> - Performance metrics (FPS, latency, GPU usage)
> - Detection history and statistics

## System Requirements

- **OS**: Ubuntu 22.04 LTS (or similar)
- **GPU**: NVIDIA A30 (or any CUDA-capable GPU)
- **CUDA**: 11.8+ (already installed on this system)
- **Python**: 3.11+ (3.12 installed)
- **RAM**: 8GB+ recommended
- **Network**: Access to Reolink camera on local network

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Configure System

**IMPORTANT**: Camera credentials are stored separately in `camera_credentials.yaml` (not tracked in git).

```bash
# First time: Copy example credentials file
cp camera_credentials.example.yaml camera_credentials.yaml

# Edit with your camera passwords
nano camera_credentials.yaml
```

Main configuration in `config/config.yaml`:

```yaml
cameras:
  - id: "cam1"
    name: "Main Backyard View"
    ip: "192.168.1.100"
    stream: "main"
    enabled: true
  - id: "cam2"
    name: "Secondary View"
    ip: "192.168.1.101"
    stream: "main"
    enabled: true

detection:
  model:
    name: "yolox-s"  # YOLOX-S (fast, balanced)
    weights: "models/yolox/yolox_s.pth"
  device: "cuda:0"
  input_size: [1920, 1920]  # Larger for small wildlife detection
  conf_threshold: 0.15  # Low threshold for distant animals
  wildlife_only: true  # Filter to wildlife-relevant COCO classes

web:
  host: "0.0.0.0"
  port: 8000
```

### 3. Run the System

```bash
# Make main script executable
chmod +x main.py

# Run the system
python main.py
```

The system will:
1. Connect to cameras (2 configured: cam1 + cam2)
2. Load YOLOX model (69MB, downloads automatically)
3. Start inference engines (one per camera)
4. Launch web server with live feeds

Access the web interface at: **http://localhost:8000**

### 4. Run as a Service (Recommended)

For production use, run the system as a systemd service that starts automatically on boot:

```bash
# Install the service
sudo ./service.sh install

# Start the service
sudo ./service.sh start

# Check status
./service.sh status

# View logs
./service.sh logs -f
```

See [SERVICE_SETUP.md](SERVICE_SETUP.md) for complete documentation.

**Quick Reference:**
```bash
sudo ./service.sh start    # Start service
sudo ./service.sh stop     # Stop service
sudo ./service.sh restart  # Restart service
./service.sh status        # Check status
./service.sh logs          # View recent logs
./service.sh logs -f       # Follow logs live
```

## Testing

### Test Camera Connection

```bash
python tests/test_camera_connection.py
```

This will verify:
- RTSP connection to camera
- Frame capture capability
- Stream properties

### Test GPU Inference

```bash
python tests/test_inference.py
```

This will benchmark:
- GPU inference speed
- Different resolutions
- Model comparison (nano, small, medium)

### Test End-to-End Latency

```bash
python tests/test_latency.py
```

This will measure:
- Complete pipeline latency
- Individual component times
- Percentile statistics (P50, P95, P99)

## Project Structure

```
telescope_cam_detection/
├── config/
│   ├── config.yaml              # Main configuration
│   └── camera_credentials.yaml  # Camera passwords (gitignored)
├── src/
│   ├── stream_capture.py        # Multi-camera RTSP handling
│   ├── inference_engine_yolox.py# YOLOX GPU inference
│   ├── two_stage_pipeline_yolox.py # Stage 2 species classification
│   ├── detection_processor.py   # Post-processing detections
│   ├── snapshot_saver.py        # Image/clip saving module
│   └── web_server.py            # FastAPI + WebSocket server
├── web/
│   ├── index.html               # Web interface
│   └── app.js                   # Frontend JavaScript
├── docs/                        # Complete documentation
│   ├── setup/                   # Installation & configuration
│   ├── features/                # Feature guides
│   ├── api/                     # API reference
│   ├── training/                # Custom model training
│   └── archive/                 # Historical docs
├── scripts/
│   └── view_snapshots.py        # Browse saved snapshots
├── tests/
│   ├── test_camera_connection.py
│   ├── test_inference.py
│   ├── test_performance.py
│   └── test_stage2_integration.py
├── models/
│   └── yolox/                   # YOLOX model weights
├── training/                    # Training infrastructure
│   ├── datasets/                # Training data
│   └── scripts/                 # Training scripts
├── logs/                        # System logs
├── clips/                       # Saved detection snapshots/clips
├── main.py                      # Main application entry point
└── requirements.txt             # Python dependencies
```

## Architecture

```
[Camera] --RTSP--> [Stream Capture] --Queue--> [Inference Engine]
                                                       |
                                                       v
                                               [Detection Processor]
                                                       |
                                                       v
                                                  [Web Server]
                                                       |
                                              +--------+--------+
                                              |                 |
                                          WebSocket         MJPEG
                                              |                 |
                                       [Detection Data]    [Video Feed]
                                              |                 |
                                              +--------+--------+
                                                       |
                                                  [Browser UI]
```

## Performance (Current)

- ✅ Multi-camera RTSP capture
- ✅ YOLOX GPU inference at **11-21ms** per frame
- ✅ Web interface with real-time overlays
- ✅ **Achieved latency**: 25-35ms end-to-end (Stage 1 only)
- ✅ **Achieved FPS**: 25-30 FPS sustained
- ✅ **Memory usage**: ~500MB RAM, ~2GB VRAM per camera
- 🎯 **With Stage 2**: 30-50ms total (20-40 FPS)

## Configuration Options

### Camera Settings

- **stream**: `main` (2560x1440) or `sub` (lower resolution)
- **target_width/height**: Resize frames before inference (default: 1280x720)
- **buffer_size**: Keep at 1 for lowest latency

### Detection Settings

- **model**: YOLOX variant selection
  - `yolox-nano`: Fastest (~8-12ms inference)
  - `yolox-tiny`: Very fast (~9-15ms inference)
  - `yolox-s`: **Balanced (current)** (~11-21ms inference)
  - `yolox-m`: Medium (~25-40ms inference)
  - `yolox-l`: Large (~50-80ms inference)
  - `yolox-x`: Highest accuracy (~80-120ms inference)
- **conf_threshold**: Minimum detection confidence (0.0-1.0)
- **wildlife_only**: Filter to wildlife-relevant COCO classes
- **input_size**: Larger = better for small/distant objects (e.g., [1920, 1920])

### Snapshot Settings

See [docs/features/SNAPSHOT_FEATURE.md](docs/features/SNAPSHOT_FEATURE.md) for complete documentation.

- **enabled**: Enable/disable snapshot saving
- **save_mode**: `"image"` (single frames) or `"clip"` (video clips)
- **trigger_classes**: Classes that trigger saves (empty = all detections)
- **min_confidence**: Minimum confidence to trigger save
- **cooldown_seconds**: Prevent duplicate saves
- **save_annotated**: Save with/without bounding boxes

### Web Settings

- **host**: `0.0.0.0` (all interfaces) or `127.0.0.1` (local only)
- **port**: Web server port (default: 8000)

## Troubleshooting

### Camera Connection Issues

```bash
# Test camera connection
python tests/test_camera_connection.py

# Common issues:
# - Wrong IP address: Check camera IP in router
# - Wrong credentials: Verify username/password
# - Network issues: Ping camera IP
# - Firewall: Ensure port 554 (RTSP) is open
```

### GPU/CUDA Issues

```bash
# Verify GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA not available:
# - Check NVIDIA driver installation
# - Reinstall PyTorch with CUDA support
```

### High Latency

If latency is >50ms:

1. **Reduce model size**: Switch from `yolox-s` to `yolox-tiny` or `yolox-nano`
2. **Lower input_size**: Reduce from [1920, 1920] to [640, 640]
3. **Use sub stream**: Change camera stream from `main` to `sub`
4. **Disable Stage 2**: Set `use_two_stage: false` if enabled
5. **Check network**: Ensure camera is on same subnet as server

### Web Interface Not Loading

```bash
# Check if server is running
curl http://localhost:8000/health

# Check firewall
sudo ufw status

# Allow port if needed
sudo ufw allow 8000
```

## Development Roadmap

### Phase 1: Core System ✅ (Complete)
- [x] Multi-camera RTSP stream capture
- [x] YOLOX inference on NVIDIA A30
- [x] Web interface with live video feeds
- [x] Real-time detection overlays
- [x] Snapshot/clip saving feature
- [x] Wildlife-only class filtering
- [x] Per-camera configuration overrides

### Phase 2: Species Classification ✅ (Complete)
- [x] Two-stage detection pipeline
- [x] iNaturalist EVA02 integration (10,000 species)
- [x] Stage 2 preprocessing optimizations
- [x] Per-camera Stage 2 settings

### Phase 3: Custom Training (In Progress)
- [x] Training infrastructure ready
- [x] Captured 1,050+ telescope training images
- [ ] Annotate telescope parts (7 classes)
- [ ] Train custom YOLOX model
- [ ] Deploy telescope detection model

### Phase 4: Advanced Features (Planned)
- [ ] Collision detection logic (proximity alerts)
- [ ] Zone-based detection areas
- [ ] TensorRT optimization (3-5ms inference)
- [ ] Integration with telescope mount control
- [ ] Email/SMS alerts
- [ ] Configuration web UI

## API Reference

### WebSocket Endpoint

Connect to: `ws://localhost:8000/ws/detections`

**Message Format:**
```json
{
  "type": "detections",
  "frame_id": 12345,
  "timestamp": 1634567890.123,
  "latency_ms": 85.5,
  "inference_time_ms": 15.2,
  "total_detections": 2,
  "detection_counts": {
    "person": 1,
    "cat": 1
  },
  "detections": [
    {
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 500
      }
    }
  ]
}
```

### HTTP Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /video/feed` - MJPEG video stream

## License

MIT License - fully open source. See [LICENSE](LICENSE) for details.

All dependencies use permissive licenses (Apache 2.0, BSD, MIT).

## Support

For issues or questions:
1. Check logs: `./service.sh logs` or `logs/` directory
2. Run test scripts to diagnose problems (see `tests/`)
3. Review configuration in `config/config.yaml`
4. See complete documentation in `docs/` directory

## Performance Notes

- **A30 GPU**: 24GB HBM2, excellent for this workload
- **YOLOX-S on 1920x1920**: ~11-21ms inference (current)
- **Stage 2 (iNaturalist)**: +20-30ms when classification triggered
- **Network latency**: ~5-10ms typical for local cameras
- **Total pipeline**: 25-35ms (Stage 1 only), 30-50ms (Stage 1+2)
- **Future optimization**: TensorRT can reduce to 3-5ms (not critical at current speed)

## Credits

This project uses the following open-source components:

- **YOLOX** (Apache 2.0) - https://github.com/Megvii-BaseDetection/YOLOX
- **iNaturalist / EVA02** (Apache 2.0) - https://github.com/huggingface/pytorch-image-models
- **PyTorch** (BSD-3-Clause) - https://pytorch.org/
- **OpenCV** (Apache 2.0) - https://opencv.org/
- **FastAPI** (MIT) - https://fastapi.tiangolo.com/

## Wildlife Detection Coverage

**Stage 1 (YOLOX)**: 80 COCO classes filtered to wildlife-relevant:
- **Mammals**: person, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Birds**: bird (generic detection)
- **Other**: All standard COCO object classes

**Stage 2 (iNaturalist)**: Optional fine-grained species classification:
- **10,000 species** from iNaturalist 2021 dataset
- **Detailed taxonomy**: genus, family, order, class
- **High accuracy**: 92% top-1 on validation set

See [docs/features/STAGE2_SETUP.md](docs/features/STAGE2_SETUP.md) for Stage 2 setup.
