# GitHub Copilot Instructions - Telescope Detection System

## Project Overview

This is a **real-time object detection system** for monitoring astronomical telescopes and desert wildlife using Reolink cameras and NVIDIA GPUs. The system leverages PyTorch-based object detection models (YOLOX/RT-DETR) and optional iNaturalist species classification for fine-grained wildlife identification.

**Key Features:**
- Multi-camera RTSP stream processing with fault-tolerant startup
- GPU-accelerated inference (11-21ms per frame with YOLOX)
- Two-stage detection pipeline: Stage 1 (object detection) → Stage 2 (species classification)
- Web interface with live streams and real-time detection overlays
- Automatic snapshot/video clip saving for wildlife events
- Privacy-preserving face masking (optional)
- Graceful GPU OOM degradation with progressive quality reduction
- Systemd service integration for production deployment

## Tech Stack

**Core Technologies:**
- Python 3.11+
- PyTorch 2.1+ / torchvision
- CUDA 11.8+
- FastAPI (web framework)
- OpenCV (computer vision)
- PyAV (GPU-accelerated video decoding)
- WebSockets (real-time communication)

**Models:**
- YOLOX (Apache 2.0) - Fast object detection (11-21ms inference)
- RT-DETR (Apache 2.0) - Transformer-based detector for small objects
- iNaturalist/EVA02 (Apache 2.0) - 10,000 species classification (optional)

**Infrastructure:**
- RTSP camera streams (Reolink cameras)
- Systemd services for production
- Docker support (optional)
- YAML-based configuration

## Coding Standards

### Python Style
- **Follow PEP 8** style guide strictly
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters**
- **Type hints are required** for all function parameters and return values
- Use docstrings (Google/NumPy style) for all public functions and classes

**Example:**
```python
def process_frame(
    frame: np.ndarray,
    confidence: float = 0.5
) -> List[Detection]:
    """
    Process a frame and return detections.

    Args:
        frame: Input image as numpy array (H, W, C)
        confidence: Minimum confidence threshold (0.0-1.0)

    Returns:
        List of Detection objects with bboxes and confidences
    """
    pass
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `StreamCapture`, `InferenceEngine`)
- **Functions/methods**: `snake_case` (e.g., `process_frame`, `get_detections`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_CONFIDENCE`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_helper`)

### Logging
Use Python's `logging` module consistently:
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Starting detection processing")
logger.warning("Low confidence detection: %.2f", confidence)
logger.error("Failed to connect to camera: %s", error)
```

**Log Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (e.g., low FPS, low confidence)
- `ERROR`: Error messages (e.g., camera connection failures)
- `CRITICAL`: Critical errors that may crash the application

### Error Handling
- Use **specific exception types** (ConnectionError, ValueError, etc.)
- Provide **meaningful error messages** with context
- **Log errors before raising** for debugging
- Use try-except for operations that may fail (camera connections, GPU operations)

```python
try:
    result = risky_operation()
except ConnectionError as e:
    logger.error(f"Camera connection failed: {e}")
    raise
```

## Project Structure

```
telescope_cam_detection/
├── config/                      # Configuration files
│   └── config.yaml             # Main system configuration
├── src/                         # Core application modules
│   ├── stream_capture.py       # RTSP stream capture
│   ├── inference_engine*.py    # Object detection engines
│   ├── species_classifier.py   # iNaturalist Stage 2 classifier
│   ├── detection_processor.py  # Detection filtering/processing
│   ├── web_server.py           # FastAPI web server
│   ├── snapshot_saver.py       # Snapshot/clip saving
│   └── ...                     # Other modules
├── web/                         # Web interface (HTML/JS/CSS)
├── tests/                       # Test scripts
├── docs/                        # Documentation
├── models/                      # Model weights (gitignored)
├── clips/                       # Saved detection snapshots
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
└── service.sh                   # Systemd service management
```

## Configuration System

**Configuration is managed through YAML files:**
- `config/config.yaml` - Main system configuration (cameras, detection, web server)
- `camera_credentials.yaml` - Camera credentials (gitignored for security)

**Key Configuration Patterns:**
- Multi-camera setup with per-camera overrides
- Per-class confidence thresholds and size constraints
- Stage 2 species classification settings
- Privacy/face masking settings
- Snapshot/clip saving triggers
- Camera health monitoring and auto-restart

**Environment Variables:**
- `TELESCOPE_CLIPS_TOKEN` - Bearer token for clips endpoint authentication
- `TELESCOPE_UNMASKED_TOKEN` - Token for unmasked faces access

**Always validate configuration:**
- Use `config/config.yaml` as the source of truth
- Never commit `camera_credentials.yaml` (contains passwords)
- Document all new configuration options in `docs/setup/CONFIG_REFERENCE.md`

## Testing

### Test Framework
- **pytest** for all tests
- Tests are in `tests/` directory
- Test files follow `test_*.py` naming convention

### Test Categories
1. **Unit tests**: Component-level testing (e.g., `test_bbox_utils.py`)
2. **Integration tests**: Multi-component testing (e.g., `test_stage2_integration.py`)
3. **Performance tests**: Benchmarking (e.g., `test_inference.py`, `test_latency.py`)
4. **Hardware tests**: Camera/GPU testing (e.g., `test_camera_connection.py`)

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_inference.py

# Run with verbose output
pytest -v tests/

# Run tests matching pattern
pytest -k "test_camera"
```

### Writing Tests
- Use descriptive test names: `test_<component>_<behavior>_<condition>`
- Test both success and failure cases
- Mock external dependencies (cameras, GPU) when appropriate
- Include docstrings explaining test purpose

**Example:**
```python
def test_stream_capture_handles_invalid_url():
    """Test that StreamCapture handles invalid RTSP URL gracefully."""
    capture = StreamCapture("rtsp://invalid")
    assert capture.connect() is False
    assert capture.is_connected is False
```

## Building and Running

### Development Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy example configurations
cp camera_credentials.example.yaml camera_credentials.yaml
# Edit camera_credentials.yaml with your camera passwords

# Run the system
python main.py
```

### Production Deployment
```bash
# Install as systemd service
sudo ./service.sh install

# Start the service
sudo ./service.sh start

# View logs
./service.sh logs -f

# Check status
./service.sh status
```

### Web Interface
- Access at **http://localhost:8000**
- WebSocket endpoint: `ws://localhost:8000/ws/detections`
- API documentation: `http://localhost:8000/docs` (FastAPI auto-generated)

## GPU and Performance

### GPU Requirements
- NVIDIA GPU with CUDA support (A30 recommended)
- CUDA 11.8+
- ~2GB VRAM per camera
- GPU OOM graceful degradation automatically reduces quality if memory is low

### Performance Optimization
- **Batched inference**: Process multiple camera frames in one GPU forward pass (3-4x throughput)
- **Empty frame filtering**: Skip frames with no motion (30-50% throughput gain)
- **Sparse detection**: Only run detection every N frames (3x GPU load reduction)
- **TensorRT optimization**: 1.5-2.4x speedup for YOLOX (optional, requires conversion)

### Checking GPU
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi -l 1
```

## Common Development Tasks

### Adding a New Camera
1. Edit `config/config.yaml` - add camera to `cameras:` list
2. Add credentials to `camera_credentials.yaml`
3. Test connection: `python tests/test_camera_connection.py`

### Adding a New Detection Class
1. Classes are inherited from COCO dataset (80 classes)
2. Add per-class overrides in `config/config.yaml`:
   - `class_confidence_overrides`
   - `class_size_constraints`
3. Update `trigger_classes` in `snapshots:` section if needed

### Adding New Features
1. Create feature module in `src/`
2. Add tests in `tests/test_<feature>.py`
3. Update `config/config.yaml` with new configuration options
4. Document in `docs/features/<FEATURE_NAME>.md`
5. Update `README.md` if user-facing

### Debugging Common Issues
- **Camera not connecting**: Run `python tests/test_camera_connection.py`
- **GPU not working**: Check `nvidia-smi` and verify CUDA availability
- **High latency**: See `docs/PERFORMANCE.md` for optimization tips
- **OOM errors**: System should auto-recover; check `docs/features/OOM_GRACEFUL_DEGRADATION.md`

## Documentation

### Documentation Standards
- Use **Markdown** format for all docs
- Place docs in `docs/` directory, organized by category:
  - `docs/setup/` - Setup and configuration
  - `docs/features/` - Feature documentation
  - `docs/api/` - API reference
  - `docs/training/` - Training and annotation guides
- Include **code examples** where helpful
- Add **screenshots** for UI features
- Update `README.md` table of contents when adding new docs

### Key Documents
- `README.md` - Project overview and quick start
- `CONTRIBUTING.md` - Contribution guidelines
- `docs/setup/CONFIG_REFERENCE.md` - Complete configuration reference
- `docs/setup/SERVICE_SETUP.md` - Systemd service setup
- `docs/TROUBLESHOOTING.md` - Common issues and solutions
- `docs/PERFORMANCE.md` - Performance benchmarks and optimization

## Security and Privacy

### Security Best Practices
- **Never commit credentials** - use `camera_credentials.yaml` (gitignored)
- **Use environment variables** for sensitive tokens (e.g., `TELESCOPE_CLIPS_TOKEN`)
- **Enable authentication** for clips endpoint in production
- **Face masking** available for privacy (optional, configurable per-camera)

### Privacy Features
- Face masking with multiple styles (gaussian_blur, pixelate, black_box)
- Backend retains unmasked versions for security investigation
- Configurable per-camera face masking overrides
- Bearer token authentication for unmasked access

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples:**
```
feat(detection): Add support for RT-DETR model

- Add RT-DETR model loader
- Update config schema for new model
- Add tests for RT-DETR inference

Closes #456
```

```
fix(camera): Handle connection timeout gracefully

Previously, connection timeouts would crash the application.
Now they are caught and logged, with automatic retry.

Fixes #789
```

## License

MIT License - All dependencies use permissive licenses (Apache 2.0, BSD, MIT).

## Additional Notes

- This is a **hobby project** with sporadic maintenance
- Focus on **minimal, surgical changes** when contributing
- **Test thoroughly** before submitting PRs
- Check existing issues before creating new ones
- Review `CONTRIBUTING.md` for detailed contribution guidelines
