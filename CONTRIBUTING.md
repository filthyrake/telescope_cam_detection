# Contributing to Telescope Detection System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior by opening an issue.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **System information** (OS, Python version, GPU model)
- **Log files** from `logs/` directory if applicable
- **Configuration** (sanitized, no credentials)

Use the bug report template when creating issues.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear descriptive title**
- **Provide detailed description** of the proposed feature
- **Explain why this enhancement would be useful**
- **List alternatives** you've considered
- **Include mockups/examples** if applicable

### Pull Requests

We actively welcome pull requests! Areas where we'd love contributions:

- 🐛 Bug fixes
- 📚 Documentation improvements
- ✨ New camera model support
- 🎨 Web UI enhancements
- 🧪 Additional tests
- 🚀 Performance optimizations
- 🌍 Internationalization

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/telescope_cam_detection.git
cd telescope_cam_detection

# Add upstream remote
git remote add upstream https://github.com/filthyrake/telescope_cam_detection.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if you add them)
# pip install -r requirements-dev.txt
```

### 3. Set Up Configuration

```bash
# Copy example files
cp config/config.yaml.example config/config.yaml
cp camera_credentials.example.yaml camera_credentials.yaml

# Edit with your camera credentials (never commit these!)
nano camera_credentials.yaml
```

### 4. Verify Setup

```bash
# Run tests
python -m pytest tests/

# Test camera connection (optional, requires camera)
python tests/test_camera_connection.py

# Test inference (requires GPU)
python tests/test_performance.py
```

## Pull Request Process

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout master
git merge upstream/master

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, concise commit messages
- Follow the coding standards (see below)
- Add tests for new features
- Update documentation as needed
- Keep commits focused (one logical change per commit)

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/

# Test specific component
python tests/test_your_new_feature.py

# Check code style (if we add linters)
# flake8 src/
# black --check src/
```

### 4. Commit and Push

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: Add support for new camera model XYZ

- Add RTSP URL generation for XYZ cameras
- Update documentation with setup instructions
- Add tests for XYZ camera connection

Fixes #123"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Create Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Fill out the PR template
4. Link related issues
5. Wait for review

### 6. Review Process

- Maintainers will review your PR
- Address feedback by pushing new commits
- Once approved, your PR will be merged
- Your contribution will be credited in release notes!

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters**
- Use type hints where possible

```python
def process_frame(
    frame: np.ndarray,
    confidence: float = 0.5
) -> List[Detection]:
    """
    Process a frame and return detections.

    Args:
        frame: Input image as numpy array
        confidence: Minimum confidence threshold

    Returns:
        List of Detection objects
    """
    pass
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `StreamCapture`)
- **Functions/methods**: `snake_case` (e.g., `process_frame`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_method`)

### File Organization

```
src/
├── component_name.py        # Main component
├── component_name_test.py   # Tests (if not in tests/)
└── utils/                   # Utility functions
```

### Logging

Use Python's `logging` module:

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

```python
try:
    result = risky_operation()
except ConnectionError as e:
    logger.error(f"Camera connection failed: {e}")
    raise
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test both success and failure cases

```python
def test_stream_capture_connects_successfully():
    """Test that StreamCapture connects to valid RTSP URL."""
    capture = StreamCapture(valid_rtsp_url)
    assert capture.connect() is True
    assert capture.is_connected is True

def test_stream_capture_handles_invalid_url():
    """Test that StreamCapture handles invalid RTSP URL gracefully."""
    capture = StreamCapture("rtsp://invalid")
    assert capture.connect() is False
    assert capture.is_connected is False
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_inference.py

# Run with coverage
pytest --cov=src tests/

# Run tests matching pattern
pytest -k "test_camera"
```

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Use Google-style or NumPy-style docstrings
- Include usage examples for complex functionality

### README Updates

If your change affects:
- Installation process → Update Quick Start
- Configuration → Update Configuration Options
- API → Update API Reference
- Performance → Update Performance Notes

### Creating New Documentation

- Place docs in `docs/` directory
- Use Markdown format
- Add to `docs/README.md` table of contents
- Include code examples where helpful

### Documentation Style

- Use clear, concise language
- Provide examples for complex concepts
- Include screenshots/diagrams where helpful
- Test all code examples

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

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

**Examples:**

```
feat(detection): Add support for YOLOv9 model

- Add YOLOv9 model loader
- Update config schema for new model
- Add tests for YOLOv9 inference

Closes #456
```

```
fix(camera): Handle connection timeout gracefully

Previously, connection timeouts would crash the application.
Now they are caught and logged, with automatic retry.

Fixes #789
```

## Questions?

- 💬 **General questions**: Open a [Discussion](https://github.com/filthyrake/telescope_cam_detection/discussions)
- 🐛 **Bug reports**: Open an [Issue](https://github.com/filthyrake/telescope_cam_detection/issues)
- 💡 **Feature ideas**: Open an [Issue](https://github.com/filthyrake/telescope_cam_detection/issues) with enhancement label

## Recognition

Contributors will be:
- Credited in release notes
- Listed in CONTRIBUTORS.md (if we create one)
- Thanked in commit messages

Thank you for making this project better! 🎉
