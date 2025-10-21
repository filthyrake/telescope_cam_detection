# Copilot Instructions for Telescope Detection System

This repository contains a real-time object detection system for monitoring astronomical telescopes and desert wildlife using Reolink cameras and NVIDIA GPUs.

## Project Overview

- **Language**: Python 3.11+
- **Framework**: PyTorch 2.0+ with CUDA 11.8+
- **Detection Model**: YOLOX (Apache 2.0) - 11-21ms inference time
- **Species Classification**: iNaturalist EVA02 (10,000 species)
- **Web Framework**: FastAPI with WebSocket support
- **Computer Vision**: OpenCV with PyAV for GPU-accelerated video decode

## Project Structure

```
telescope_cam_detection/
├── config/                      # Configuration files (YAML)
├── src/                         # Core application modules
│   ├── stream_capture.py       # RTSP camera capture
│   ├── inference_engine_yolox.py # YOLOX detection
│   ├── detection_processor.py  # Post-processing
│   ├── web_server.py           # FastAPI server
│   ├── snapshot_saver.py       # Image/video saving
│   └── species_classifier.py   # Stage 2 classification
├── web/                         # Web interface (HTML/JS)
├── tests/                       # Test scripts
├── models/                      # Model weights
├── clips/                       # Saved detection snapshots
└── main.py                      # Application entry point
```

## Build and Test

### Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration examples
cp config/config.yaml.example config/config.yaml
cp camera_credentials.example.yaml camera_credentials.yaml
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Test specific component
python tests/test_inference.py      # GPU inference benchmark
python tests/test_camera_connection.py  # Camera connectivity
python tests/test_latency.py        # End-to-end latency
python tests/test_performance.py    # Performance metrics
python tests/test_oom_recovery.py   # OOM recovery testing
```

**Important**: Many tests require GPU and camera hardware. Tests may be skipped if:
- CUDA is not available (`torch.cuda.is_available()`)
- Camera credentials are not configured
- Model weights are not downloaded

### Linting and Code Quality

This project follows PEP 8 style guide with:
- 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Type hints where possible

```python
# Example function signature
def process_frame(
    frame: np.ndarray,
    confidence: float = 0.5
) -> List[Detection]:
    """Process a frame and return detections."""
    pass
```

### Running the Application

```bash
# Development mode
python main.py

# Production with systemd service
sudo ./service.sh install
sudo ./service.sh start
./service.sh logs -f
```

Access web interface at **http://localhost:8000**

## Coding Standards

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `StreamCapture`, `InferenceEngine`)
- **Functions/methods**: `snake_case` (e.g., `process_frame`, `connect_camera`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_CONFIDENCE`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_method`)

### Logging

Use Python's `logging` module consistently:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Starting detection processing")
logger.warning("Low confidence detection: %.2f", conf)
logger.error("Failed to connect to camera: %s", error)
```

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors before raising
- Handle GPU OOM gracefully (see `src/inference_engine_yolox.py`)

```python
try:
    result = risky_operation()
except ConnectionError as e:
    logger.error(f"Camera connection failed: {e}")
    raise
```

## Configuration Guidelines

### Camera Configuration

Cameras are configured in `config/config.yaml`:
- Each camera has unique ID, IP address, and stream type
- Support for `rtsp`, `rtsp-tcp`, `onvif`, `h265` protocols
- Per-camera detection overrides available
- Never commit `config/config.yaml` - it's gitignored

### Credentials Security

**CRITICAL**: Credentials are stored separately:
- `camera_credentials.yaml` contains usernames/passwords
- This file is `.gitignored` and must NEVER be committed
- Use `camera_credentials.example.yaml` as template
- Web API Bearer tokens also in this file

### Detection Configuration

- YOLOX model supports 80 COCO classes
- Confidence thresholds: Default 0.15 (wildlife), can override per-class
- Input size affects speed/accuracy: 640x640 (fast), 1920x1920 (accurate)
- Wildlife-only mode filters to relevant classes

## GPU and CUDA Requirements

### GPU Testing

This project requires NVIDIA GPU with CUDA support:

```python
import torch
assert torch.cuda.is_available(), "CUDA not available"
logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
```

### Memory Management

- System includes OOM graceful degradation
- Automatic quality reduction on GPU memory pressure
- See `docs/features/OOM_GRACEFUL_DEGRADATION.md`
- Test with: `python tests/test_oom_recovery.py`

### Performance Targets

With NVIDIA A30:
- Inference: 11-21ms per frame
- FPS: 25-30 sustained
- Latency: 25-35ms end-to-end
- Memory: ~2GB VRAM per camera

## Documentation

### When to Update Documentation

Update relevant docs when changes affect:
- **Installation**: Update README.md Quick Start
- **Configuration**: Update `docs/setup/CONFIG_REFERENCE.md`
- **API**: Update `docs/api/API_REFERENCE.md`
- **Performance**: Update `docs/PERFORMANCE.md`
- **Features**: Update or create feature docs in `docs/features/`

### Documentation Style

- Use Markdown format
- Provide code examples for complex concepts
- Include setup/configuration steps
- Test all code examples before committing

## Common Development Tasks

### Adding New Camera Support

1. Update `config/config.yaml` with camera entry
2. Test connection: `python tests/test_camera_connection.py`
3. Verify stream format and resolution
4. Update documentation if new protocol

### Modifying Detection Thresholds

1. Edit `config/config.yaml` detection section
2. Test with: `python tests/test_inference.py`
3. Validate with real camera feed
4. Document reasoning in commit message

### Adding New Object Classes

1. Verify class exists in YOLOX COCO (80 classes)
2. Add to `config.yaml` class_confidence_overrides if needed
3. Test detection accuracy
4. Update wildlife filtering if relevant

### Performance Optimization

1. Benchmark baseline: `python tests/test_performance.py`
2. Make targeted changes to inference or preprocessing
3. Re-benchmark and compare metrics
4. Document improvements in `docs/PERFORMANCE.md`

## Security Considerations

### Credentials

- Never commit `camera_credentials.yaml`
- Never commit `config/config.yaml` with real IPs/settings
- Use Bearer tokens for web API authentication
- Rotate tokens periodically

### Face Privacy

- Face masking available via `FaceMasker` class
- Blurs detected faces in saved clips
- See `src/face_masker.py` and `tests/test_face_masking.py`

### Dependencies

- All dependencies use permissive licenses (Apache 2.0, BSD, MIT)
- Pin versions in `requirements.txt`
- Test updates in isolation before deploying

## Troubleshooting

### Camera Connection Issues

```bash
# Test camera connectivity
python tests/test_camera_connection.py

# Check RTSP URL generation
python -c "from src.stream_capture import create_rtsp_url; print(create_rtsp_url('cam1'))"
```

### GPU Issues

```bash
# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi

# Test inference
python tests/test_inference.py
```

### High Latency

1. Check `docs/PERFORMANCE.md` for optimization tips
2. Reduce input size in `config/config.yaml`
3. Enable empty frame filtering
4. Use sparse detection mode

## Testing Philosophy

- Unit tests for individual components
- Integration tests for multi-component workflows
- Performance tests for benchmarking
- Hardware tests may require GPU and cameras
- Mock external dependencies when possible
- All tests should be idempotent

## Commit Message Format

Follow Conventional Commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
- `feat(detection): Add support for YOLOv9 model`
- `fix(camera): Handle connection timeout gracefully`
- `docs(api): Update WebSocket endpoint documentation`

## Additional Resources

- [README.md](../README.md) - Project overview and quick start
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [docs/](../docs/) - Comprehensive documentation
- [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) - Common issues
- [PERFORMANCE.md](../docs/PERFORMANCE.md) - Optimization guide
