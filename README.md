# Telescope Detection System

Real-time object detection system for monitoring astronomical telescopes using Reolink RLC-410W camera and NVIDIA A30 GPU.

## Features

- **Ultra-low latency**: Target <100ms end-to-end latency
- **Real-time detection**: People, animals, and custom telescope equipment
- **GPU accelerated**: Optimized for NVIDIA A30
- **Web interface**: Live video stream with detection overlays
- **WebSocket streaming**: Real-time detection results
- **Snapshot saving**: Automatic image/clip saving on detection events

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

Edit `config/config.yaml` if needed. Default configuration is already set up for your camera:

```yaml
camera:
  ip: "10.0.8.18"
  username: "admin"
  password: "5326jbbD"
  stream: "main"

detection:
  model: "yolov8x.pt"
  device: "cuda:0"
  confidence: 0.5

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
1. Connect to the camera
2. Download YOLOv8 model (if not already present)
3. Start the inference engine
4. Launch the web server

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
│   └── config.yaml           # System configuration
├── src/
│   ├── stream_capture.py     # RTSP stream handling
│   ├── inference_engine.py   # GPU inference with YOLOv8
│   ├── detection_processor.py# Post-processing detections
│   ├── snapshot_saver.py     # Image/clip saving module
│   └── web_server.py         # FastAPI + WebSocket server
├── web/
│   ├── index.html            # Web interface
│   └── app.js                # Frontend JavaScript
├── scripts/
│   └── view_snapshots.py     # Browse and manage saved snapshots
├── tests/
│   ├── test_camera_connection.py
│   ├── test_inference.py
│   └── test_latency.py
├── models/                   # YOLOv8 models (auto-downloaded)
├── logs/                     # System logs
├── clips/                    # Saved detection snapshots/clips
├── main.py                   # Main application entry point
└── requirements.txt          # Python dependencies
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

## Performance Targets (Phase 1)

- ✓ RTSP stream capture working
- ✓ GPU inference running
- ✓ Web interface with overlays
- ⏱ **Target latency**: <200ms (optimized to <100ms in Phase 2)
- 🎯 **Target FPS**: 15-30 FPS
- 💾 **Memory usage**: <4GB RAM, <2GB VRAM

## Configuration Options

### Camera Settings

- **stream**: `main` (2560x1440) or `sub` (lower resolution)
- **target_width/height**: Resize frames before inference (default: 1280x720)
- **buffer_size**: Keep at 1 for lowest latency

### Detection Settings

- **model**: Choose YOLOv8 variant
  - `yolov8n.pt`: Fastest, lowest accuracy (~10ms inference)
  - `yolov8s.pt`: Balanced (~15ms inference)
  - `yolov8m.pt`: Medium (~25ms inference)
  - `yolov8x.pt`: Highest accuracy, slower (~50ms inference)
- **confidence**: Minimum detection confidence (0.0-1.0)
- **target_classes**: List specific classes to detect (or empty for all)

### Snapshot Settings

See [SNAPSHOT_FEATURE.md](SNAPSHOT_FEATURE.md) for complete documentation.

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

If latency is >200ms:

1. **Reduce model size**: Switch from `yolov8x` to `yolov8s` or `yolov8n`
2. **Lower resolution**: Reduce `target_width/height` in config
3. **Use sub stream**: Change camera stream from `main` to `sub`
4. **Check network**: Ensure camera is on same subnet as server

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

### Phase 1: Basic Pipeline ✓ (Complete)
- [x] RTSP stream capture
- [x] YOLOv8 inference on A30
- [x] Web interface with video feed
- [x] Basic detection overlay
- [x] Snapshot/clip saving feature

### Phase 2: Custom Training (In Progress)
- [x] Training infrastructure ready
- [ ] Collect telescope training images
- [ ] Annotate telescope parts
- [ ] Train custom YOLOv8 model
- [ ] Deploy custom model

### Phase 3: Optimization
- [ ] TensorRT model optimization
- [ ] GStreamer pipeline for lower latency
- [ ] Optimize to <100ms end-to-end
- [ ] Performance monitoring dashboard

### Phase 4: Advanced Features
- [ ] Collision detection logic
- [ ] Zone-based alerts
- [ ] Multi-camera support
- [ ] Integration with telescope control software
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

This project is for personal use. Modify and extend as needed for your telescope monitoring setup.

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Run test scripts to diagnose problems
3. Review configuration in `config/config.yaml`

## Performance Notes

- **A30 GPU**: 24GB HBM2, excellent for this workload
- **YOLOv8x on 1280x720**: ~30-50ms inference time expected
- **YOLOv8n on 1280x720**: ~10-15ms inference time expected
- **Network latency**: ~20-40ms typical for local camera
- **Total pipeline**: ~50-100ms achievable with optimization

## Credits

- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **FastAPI**: https://fastapi.tiangolo.com/
- **OpenCV**: https://opencv.org/
- **PyTorch**: https://pytorch.org/
