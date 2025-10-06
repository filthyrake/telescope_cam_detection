---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Go to '...'
2. Run command '...'
3. See error

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

What actually happened instead.

## System Information

- **OS**: [e.g., Ubuntu 22.04]
- **Python Version**: [e.g., 3.12]
- **GPU**: [e.g., NVIDIA A30, RTX 3090]
- **CUDA Version**: [e.g., 11.8]
- **Camera Model**: [e.g., Reolink RLC-410W]

## Configuration

```yaml
# Paste relevant config.yaml sections (REMOVE CREDENTIALS!)
detection:
  model:
    name: "yolox-s"
  conf_threshold: 0.15
```

## Logs

```
# Paste relevant log output from logs/ directory
[2025-01-06 10:30:45] ERROR - Failed to connect to camera
```

## Screenshots

If applicable, add screenshots to help explain your problem.

## Additional Context

Add any other context about the problem here.
