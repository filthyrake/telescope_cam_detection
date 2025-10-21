# GitHub Copilot Instructions

This document provides guidelines for GitHub Copilot to work effectively with the Telescope Detection System codebase.

## Project Overview

This is a real-time object detection system for monitoring astronomical telescopes and desert wildlife using Reolink cameras and NVIDIA GPUs. It uses PyTorch-based models (YOLOX/RT-DETR for object detection and iNaturalist for species classification) with a FastAPI web interface.

## Code Style and Linting

### Python Style Guide

- Follow **PEP 8** style guide strictly
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters**
- Use type hints for function parameters and return values
- Format code with `black --line-length 100`
- Sort imports with `isort --profile black`

### Linting Tools

- **flake8**: For syntax errors and basic style checks
- **black**: For code formatting (line length 100)
- **isort**: For import sorting (black profile)
- **bandit**: For security issue detection

Run linting before committing:
```bash
flake8 src/ --max-line-length=100
black --line-length 100 src/ main.py
isort --profile black src/ main.py
bandit -r src/ -ll
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `StreamCapture`, `InferenceEngine`)
- **Functions/methods**: `snake_case` (e.g., `process_frame`, `get_detections`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_FPS`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_method`)
- **Module-level**: `snake_case` (e.g., `stream_capture.py`, `detection_processor.py`)

## Architecture and Components

### Core Components

- **Stream Capture** (`src/stream_capture.py`): RTSP camera connection and frame capture
- **Inference Engine** (`src/inference_engine_yolox.py`, `src/rtdetr_detector.py`): GPU-accelerated object detection
- **Detection Processor** (`src/detection_processor.py`): Post-processing and filtering of detections
- **Web Server** (`src/web_server.py`): FastAPI-based web interface with WebSocket support
- **Species Classifier** (`src/species_classifier.py`): Optional Stage 2 fine-grained classification
- **Snapshot Saver** (`src/snapshot_saver.py`): Automatic saving of detection clips

### Multi-Camera Architecture

- System supports multiple cameras with fault-tolerant startup
- Each camera has its own stream capture, but shares inference engine via `SharedInferenceCoordinator`
- Cameras can have per-camera detection overrides and preprocessing settings
- All detections are merged into a single queue for web server

### Configuration Management

- Main config: `config/config.yaml` (all settings except credentials)
- Credentials: `camera_credentials.yaml` (gitignored, see `camera_credentials.example.yaml`)
- Hot-reload support via watchdog monitoring
- Camera-level overrides for detection thresholds and preprocessing

## Testing

### Test Framework

- Use **pytest** for all tests
- Test files: `tests/test_*.py`
- Run tests: `pytest tests/ -v`
- Coverage: `pytest --cov=src tests/`

### Test Categories

1. **Unit tests** (`test_bbox_utils.py`, `test_enhancement.py`): No GPU or camera required
2. **Integration tests** (`test_stage2_integration.py`): May require GPU
3. **Performance tests** (`test_inference.py`, `test_latency.py`): Require GPU
4. **Hardware tests** (`test_camera_connection.py`): Require camera access

### Test Guidelines

- Write tests for new features and bug fixes
- Use descriptive test names: `test_<component>_<behavior>_<expected_result>`
- Test both success and failure cases
- Mock expensive operations (camera connections, GPU inference) when appropriate
- Include docstrings explaining what the test validates

Example:
```python
def test_stream_capture_connects_successfully():
    """Test that StreamCapture connects to valid RTSP URL."""
    capture = StreamCapture(valid_rtsp_url)
    assert capture.connect() is True
    assert capture.is_connected is True
```

## Documentation

### Docstring Format

Use Google-style docstrings with type hints:

```python
def process_frame(
    frame: np.ndarray,
    confidence: float = 0.5
) -> List[Detection]:
    """
    Process a frame and return detections.

    Args:
        frame: Input image as numpy array (H, W, C) in BGR format
        confidence: Minimum confidence threshold (0.0-1.0)

    Returns:
        List of Detection objects with bounding boxes and labels

    Raises:
        ValueError: If frame is empty or invalid shape
    """
    pass
```

### Documentation Updates

When changing functionality, update:
- Function/class docstrings
- `README.md` if it affects user-facing features
- Relevant documentation in `docs/` directory
- Configuration examples if adding new config options

## Logging

### Logging Standards

Use Python's `logging` module (never `print` statements):

```python
import logging
logger = logging.getLogger(__name__)

# Log levels
logger.debug("Detailed debug information")
logger.info("General information about normal operation")
logger.warning("Warning about potential issues")
logger.error("Error occurred but application continues")
logger.critical("Critical error, application may crash")
```

### Logging Best Practices

- Use appropriate log levels
- Include context in log messages (camera ID, frame number, etc.)
- Use f-strings or lazy formatting: `logger.info("Processing frame %d", frame_num)`
- Log errors with exception info: `logger.error("Failed to connect", exc_info=True)`

## Error Handling

### Exception Handling

- Use specific exception types (not bare `except:`)
- Provide meaningful error messages
- Log errors before raising or returning
- Handle GPU OOM gracefully (see `src/memory_manager.py`)

```python
try:
    result = risky_operation()
except ConnectionError as e:
    logger.error(f"Camera connection failed: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Graceful degradation if possible
```

### Graceful Degradation

- GPU OOM: Progressively reduce quality (see `docs/features/OOM_GRACEFUL_DEGRADATION.md`)
- Camera disconnection: Auto-reconnect with exponential backoff
- Missing optional features: Warn and continue without (e.g., Stage 2 classification)

## Security

### Security Guidelines

- **Never** commit credentials or API keys
- Use `camera_credentials.yaml` (gitignored) for passwords
- Validate all external inputs (camera URLs, config values)
- Use Bearer token authentication for sensitive endpoints (clips directory)
- Run security checks with `bandit -r src/ -ll`
- Check dependencies for vulnerabilities

### Secure Coding Practices

- Sanitize user inputs before logging (no passwords in logs)
- Use parameterized queries if adding database support
- Validate configuration values with appropriate bounds checks
- Use secure defaults (HTTPS, secure cookies, etc.)

## Dependencies

### Allowed Dependencies

Core dependencies are listed in `requirements.txt`:
- **torch/torchvision**: Deep learning framework (BSD-3)
- **opencv-python**: Computer vision (Apache 2.0)
- **fastapi/uvicorn**: Web framework (MIT)
- **pyyaml**: Configuration (MIT)
- **av**: GPU-accelerated video decode (BSD)
- **timm**: Model library for iNaturalist (Apache 2.0)

### Adding New Dependencies

- Check license compatibility (prefer MIT, Apache 2.0, BSD)
- Add to `requirements.txt` with minimum version
- Document why the dependency is needed
- Consider performance and size impact
- Update Docker images if needed

## Performance

### Performance Guidelines

- **Optimize for latency**: 25-35ms end-to-end is target
- **GPU efficiency**: Use batched inference via `SharedInferenceCoordinator`
- **Memory management**: Monitor GPU memory, implement graceful degradation
- **Empty frame filtering**: Skip inference on static frames (30-50% throughput gain)
- **Sparse detection**: Reduce inference frequency when no motion detected

### Profiling

- Use `test_latency.py` to measure end-to-end latency
- Use `test_performance.py` for throughput benchmarks
- Profile GPU usage with `nvidia-smi` or PyTorch profiler
- Monitor memory with `src/memory_manager.py` utilities

## Web Interface

### Web Server Guidelines

- Use FastAPI for all HTTP endpoints
- Use WebSockets for real-time updates (`/ws/detections`)
- Serve static files from `web/` directory
- Implement proper CORS headers for API endpoints
- Add Bearer token auth for sensitive resources (clips)

### API Conventions

- RESTful endpoints: `/api/v1/<resource>`
- Health check: `GET /health`
- Metrics: `GET /stats`
- Video streams: `GET /video/feed` (MJPEG)
- WebSocket: `ws://localhost:8000/ws/detections`

## Commit Messages

Follow **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(detection): Add support for RT-DETR model

- Add RT-DETR model loader
- Update config schema for new model
- Add tests for RT-DETR inference

Closes #456
```

## CI/CD

### Continuous Integration

Workflows in `.github/workflows/`:
- **lint.yml**: Code quality checks (flake8, black, isort, bandit)
- **tests.yml**: Unit tests on Python 3.11 and 3.12
- **docker-build.yml**: Docker image building

### Pre-commit Checks

Before pushing code:
1. Run linting: `flake8 src/ --max-line-length=100`
2. Format code: `black --line-length 100 src/ main.py`
3. Sort imports: `isort --profile black src/ main.py`
4. Run tests: `pytest tests/ -v`
5. Check security: `bandit -r src/ -ll`

## Additional Resources

- **Contributing Guide**: See `CONTRIBUTING.md` for detailed contribution guidelines
- **Architecture**: See `docs/architecture/ARCHITECTURE.md` for system design
- **Performance**: See `docs/PERFORMANCE.md` for optimization strategies
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md` for common issues

## Common Patterns

### Adding a New Component

1. Create module in `src/` with descriptive name
2. Add class with docstring explaining purpose
3. Implement initialization with config validation
4. Add proper error handling and logging
5. Write unit tests in `tests/test_<component>.py`
6. Update documentation if user-facing
7. Add to `main.py` if needed in main loop

### Adding Configuration Options

1. Add to `config/config.yaml` with comment explaining purpose
2. Add validation in component that uses it
3. Update `docs/setup/CONFIG_REFERENCE.md`
4. Add default value for backward compatibility
5. Test with and without the new option

### Adding New Detection Features

1. Consider performance impact (latency target: <35ms)
2. Make features optional via config flags
3. Add logging for feature activation/deactivation
4. Test with multiple cameras
5. Document in appropriate `docs/features/` file
