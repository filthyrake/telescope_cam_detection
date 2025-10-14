# Telescope Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

Real-time object detection system for monitoring astronomical telescopes and desert wildlife using Reolink cameras and NVIDIA GPUs.

## Features

- **Ultra-fast detection**: 11-21ms inference with YOLOX
- **Multi-camera support**: Monitor multiple angles simultaneously with fault-tolerant startup
- **Motion filtering**: Background subtraction to eliminate false positives from static objects
- **Automatic reconnection**: Cameras reconnect automatically if connection is lost
- **80 COCO classes**: Wildlife-relevant categories (person, bird, cat, dog, etc.)
- **Per-class filtering**: Customizable confidence thresholds and size constraints per detection class
- **Optional species classification**: iNaturalist Stage 2 (10,000 species) with geographic filtering
- **Web interface**: Live video streams with real-time detection overlays
- **Automatic snapshots**: Save interesting detections to disk with configurable cooldown
- **MIT License**: Fully open source

## Quick Start

### 1. Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Cameras

Copy the credentials template and add your camera passwords:

```bash
cp camera_credentials.example.yaml camera_credentials.yaml
nano camera_credentials.yaml  # Add your passwords
```

Edit `config/config.yaml` to set camera IPs and detection preferences.

### 3. Run the System

```bash
python main.py
```

Access the web interface at **http://localhost:8000**

### 4. Run as a Service (Recommended)

For production use with auto-start on boot:

```bash
sudo ./service.sh install
sudo ./service.sh start
./service.sh logs -f  # Watch logs
```

See [SERVICE_SETUP.md](docs/setup/SERVICE_SETUP.md) for complete service documentation.

## Documentation

### Setup & Configuration
- **[Configuration Reference](docs/setup/CONFIG_REFERENCE.md)** - All config options explained
- **[Service Setup](docs/setup/SERVICE_SETUP.md)** - Running as systemd service
- **[Camera Credentials](camera_credentials.example.yaml)** - Secure credential storage

### Features & Usage
- **[Snapshot Feature](docs/features/SNAPSHOT_FEATURE.md)** - Automatic image/video saving
- **[Species Classification (Stage 2)](docs/features/STAGE2_SETUP.md)** - Fine-grained species ID
- **[API Reference](docs/api/API_REFERENCE.md)** - WebSocket and HTTP endpoints

### Performance & Troubleshooting
- **[Performance Guide](docs/PERFORMANCE.md)** - Benchmarks and optimization
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Architecture](docs/architecture/ARCHITECTURE.md)** - System design and components

### Training & Development
- **[Training Guide](docs/training/TRAINING_GUIDE.md)** - Train custom models
- **[Annotation Guide](docs/training/ANNOTATION_GUIDE.md)** - Label your own dataset

## Testing

```bash
# Test camera connection
python tests/test_camera_connection.py

# Benchmark GPU inference
python tests/test_inference.py

# Measure end-to-end latency
python tests/test_latency.py
```

## Project Structure

```
telescope_cam_detection/
├── config/                      # Configuration files
├── src/                         # Core application modules
├── web/                         # Web interface (HTML/JS)
├── docs/                        # Complete documentation
├── tests/                       # Test scripts
├── models/                      # Model weights
├── training/                    # Training infrastructure
├── clips/                       # Saved detection snapshots
└── main.py                      # Application entry point
```

## System Requirements

- **OS**: Ubuntu 22.04+ (or similar Linux)
- **GPU**: NVIDIA GPU with CUDA support (A30 recommended)
- **Python**: 3.11+
- **RAM**: 8GB+ recommended
- **Network**: Local network access to Reolink cameras

## Performance

With NVIDIA A30:
- **Inference**: 11-21ms per frame
- **FPS**: 25-30 sustained
- **Latency**: 25-35ms end-to-end
- **Memory**: ~2GB VRAM per camera

See [Performance Guide](docs/PERFORMANCE.md) for optimization strategies and benchmarks.

## Development Roadmap

- ✅ **Phase 1**: Core detection system (complete)
- ✅ **Phase 2**: Species classification (complete)
- 🔨 **Phase 3**: Custom telescope training (in progress)
- 📋 **Phase 4**: Collision detection and alerts (planned)

## Troubleshooting

Having issues? Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for common problems and solutions.

Quick fixes:
- **Camera not connecting?** Run `python tests/test_camera_connection.py`
- **GPU not working?** Check with `nvidia-smi` and verify CUDA is available
- **High latency?** See [Performance Guide](docs/PERFORMANCE.md) for optimization tips

## API

### WebSocket
Connect to `ws://localhost:8000/ws/detections` for real-time detection events.

### HTTP
- `GET /` - Web interface
- `GET /health` - Health check
- `GET /stats` - Performance metrics
- `GET /video/feed` - MJPEG video stream

See [API Reference](docs/api/API_REFERENCE.md) for complete documentation.

## Credits

Built with these excellent open-source projects:

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Apache 2.0) - Object detection
- **[iNaturalist/EVA02](https://github.com/huggingface/pytorch-image-models)** (Apache 2.0) - Species classification
- **[PyTorch](https://pytorch.org/)** (BSD-3) - Deep learning framework
- **[OpenCV](https://opencv.org/)** (Apache 2.0) - Computer vision
- **[FastAPI](https://fastapi.tiangolo.com/)** (MIT) - Web framework

## License

MIT License - see [LICENSE](LICENSE) for details. All dependencies use permissive licenses (Apache 2.0, BSD, MIT).

## Support

- 📖 **Documentation**: See `docs/` directory
- 🐛 **Issues**: [GitHub Issues](https://github.com/filthyrake/telescope_cam_detection/issues)
- 📊 **Logs**: `./service.sh logs` or check `logs/` directory
