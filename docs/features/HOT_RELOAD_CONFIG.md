# Configuration Hot-Reload

The telescope detection system supports hot-reloading configuration changes without requiring a full system restart. This allows you to tune detection parameters, adjust filters, and modify snapshot settings on the fly.

## Overview

Configuration hot-reload enables you to:
- **Adjust detection thresholds** without interrupting camera feeds
- **Tune per-class confidence overrides** to reduce false positives
- **Modify size constraints** to filter detections by bounding box area
- **Update snapshot settings** (cooldown, trigger classes, confidence threshold)
- **Change filter parameters** (motion filter, time-of-day filter)
- **Faster iteration** during setup and testing
- **No sudo required** for configuration changes
- **Zero downtime** for supported settings

## Hot-Reloadable Settings

The following settings can be reloaded without restarting the system:

###  Detection Settings
- `detection.conf_threshold` - Global confidence threshold
- `detection.nms_threshold` - NMS IoU threshold
- `detection.min_box_area` - Minimum bounding box area (pixels²)
- `detection.max_detections` - Maximum detections per frame
- `detection.class_confidence_overrides` - Per-class confidence thresholds
- `detection.class_size_constraints` - Per-class min/max box area constraints
- Per-camera `detection_overrides` for all of the above

### Snapshot Settings
- `snapshots.cooldown_seconds` - Seconds between saves for same class
- `snapshots.trigger_classes` - List of classes that trigger saves
- `snapshots.min_confidence` - Minimum confidence to trigger save
- `snapshots.save_annotated` - Whether to save annotated frames

### Motion Filter Settings
- `motion_filter.history` - Number of frames for background model
- `motion_filter.var_threshold` - Threshold for background/foreground classification
- `motion_filter.detect_shadows` - Whether to detect and filter shadows
- `motion_filter.min_motion_area` - Minimum motion area in pixels²
- `motion_filter.motion_blur_size` - Gaussian blur kernel size
- `motion_filter.min_motion_ratio` - Minimum motion ratio (0.0-1.0)

### Time-of-Day Filter Settings
- `time_of_day_filter.enabled` - Enable/disable time-based filtering
- `time_of_day_filter.confidence_penalty` - Confidence multiplier for out-of-pattern detections
- `time_of_day_filter.hard_filter` - Whether to remove out-of-pattern detections entirely

## Settings Requiring Restart

The following settings **require a full system restart** to take effect:

### Camera Settings
- Adding or removing cameras
- Changing camera IP addresses
- Changing camera stream types or protocols
- Changing camera resolution settings

### Model Settings
- `detection.model.weights` - Model weights path
- `detection.input_size` - Input image size for inference
- `detection.device` - GPU/CPU device selection
- `detection.use_two_stage` - Enabling/disabling two-stage detection
- `detection.enable_species_classification` - Enabling/disabling species classification

### Web Server Settings
- `web.host` - Web server host
- `web.port` - Web server port

## Usage

### Manual Reload via API

The simplest way to reload configuration is via the REST API:

```bash
# Reload configuration from config.yaml
curl -X POST http://localhost:8000/api/config/reload

# Get current active configuration
curl http://localhost:8000/api/config/current
```

**Response Example:**
```json
{
  "success": true,
  "reloaded": [
    "detection.conf_threshold",
    "detection.class_confidence_overrides",
    "snapshots.cooldown_seconds"
  ],
  "requires_restart": [
    "detection.input_size"
  ],
  "warnings": [],
  "errors": []
}
```

### Manual Reload from Python

You can also trigger reload from Python code:

```python
from main import TelescopeDetectionSystem

system = TelescopeDetectionSystem()
system.load_config()
system.validate_config()
system.initialize_components()

# Later, reload configuration
result = system.reload_config()
print(f"Reload successful: {result['success']}")
print(f"Settings reloaded: {result['reloaded']}")
print(f"Requires restart: {result['requires_restart']}")
```

## Example Workflow

### Scenario: Reducing False Positives

You notice too many false "bird" detections from telescope covers. Here's how to adjust without restart:

1. **Edit config.yaml:**
   ```yaml
   detection:
     class_confidence_overrides:
       bird: 0.75  # Increased from 0.70
     class_size_constraints:
       bird:
         max: 6000  # Reduced from 8000 to filter large objects
   ```

2. **Reload configuration:**
   ```bash
   curl -X POST http://localhost:8000/api/config/reload
   ```

3. **Verify changes:**
   - System continues running without interruption
   - New bird detections use the updated thresholds immediately
   - Existing camera feeds remain connected
   - No inference state is lost

### Scenario: Adjusting Snapshot Cooldown

You're getting too many duplicate snapshots. Increase the cooldown:

1. **Edit config.yaml:**
   ```yaml
   snapshots:
     cooldown_seconds: 180  # Increased from 120
   ```

2. **Reload:**
   ```bash
   curl -X POST http://localhost:8000/api/config/reload
   ```

3. **Result:**
   - New cooldown takes effect immediately
   - No snapshots are lost
   - System continues running

## Thread Safety

All hot-reload operations are thread-safe:
- Uses locks to prevent concurrent reloads
- Atomic updates where possible
- Validates new configuration before applying
- Rolls back on validation errors

## Validation

Configuration is validated before reloading:
- Schema validation (data types, value ranges)
- Cross-field validation (e.g., min < max for size constraints)
- If validation fails, old configuration is retained
- Detailed error messages in response

## Limitations

### What Hot-Reload Cannot Do

1. **Add/Remove Cameras**: Requires restart to initialize new pipelines
2. **Change Model Architecture**: Model must be reloaded (requires restart)
3. **Change Input Resolution**: Requires reloading model with new input size
4. **Switch GPU/CPU**: Device must be reinitialized (requires restart)
5. **Enable/Disable Stage 2**: Requires loading/unloading species classifier (requires restart)

### Performance Impact

- **Reload time**: < 100ms for typical configuration changes
- **No dropped frames**: Cameras continue capturing during reload
- **No detection loss**: Processing pipeline continues without interruption
- **Thread-safe**: Multiple components updated atomically

## Troubleshooting

### Reload Returns Errors

**Problem:** `/api/config/reload` returns validation errors

**Solution:**
1. Check logs for detailed error message
2. Verify config.yaml syntax is valid YAML
3. Ensure all values are within valid ranges
4. Test with `/api/config/current` to see current values

### Changes Not Applied

**Problem:** Configuration reloaded successfully but changes not visible

**Solution:**
1. Check if setting requires restart (see list above)
2. Verify you're editing the correct config.yaml file
3. Check response from `/api/config/reload` to see what was reloaded
4. Monitor system logs for reload confirmation messages

### Validation Fails

**Problem:** Configuration validation fails after edit

**Solution:**
1. Revert to last known good configuration
2. Check for typos in class names (case-sensitive)
3. Ensure numeric values are in correct ranges:
   - Confidence thresholds: 0.0-1.0
   - Box areas: >= 0
   - Cooldown seconds: >= 0
4. Run validation manually before reload (see tests)

## Best Practices

### Before Reloading

1. **Backup Configuration:**
   ```bash
   cp config/config.yaml config/config.yaml.backup
   ```

2. **Test Changes Gradually:**
   - Change one setting at a time
   - Observe results before making additional changes
   - Keep notes on what works

3. **Monitor System Logs:**
   ```bash
   ./service.sh logs -f  # If running as service
   # OR
   # Watch console output if running directly
   ```

### After Reloading

1. **Verify Application:**
   - Check web UI for new detection behavior
   - Monitor logs for reload confirmation
   - Verify no errors in response

2. **Document Changes:**
   - Keep a changelog of configuration tweaks
   - Note which settings improved detection accuracy
   - Track false positive/negative rates

3. **Commit Working Configurations:**
   - Once you find optimal settings, commit to git
   - Use descriptive commit messages
   - Tag stable configurations

## Future Enhancements

### Automatic File Watching (Planned)

Future versions may support automatic reload on file change:

```yaml
system:
  hot_reload:
    enabled: true
    watch_files: true  # Auto-reload on config.yaml change
    auto_reload_interval_seconds: 60  # Periodic reload (0 = disabled)
```

This feature requires the `watchdog` library (already in requirements.txt).

### Web UI Config Editor (Planned)

A future web UI may allow editing configuration directly in the browser:
- Live preview of changes
- Validation feedback
- One-click reload button
- Diff viewer showing what changed

See [Issue #81](https://github.com/filthyrake/telescope_cam_detection/issues/81) for progress.

## Related Documentation

- [Configuration Reference](../setup/CONFIG_REFERENCE.md) - Full list of all configuration options
- [API Reference](../api/API_REFERENCE.md) - Complete REST API documentation
- [Performance Guide](../PERFORMANCE.md) - Optimizing detection settings
- [Troubleshooting](../TROUBLESHOOTING.md) - Common issues and solutions

## Testing

Run the hot-reload tests to verify functionality:

```bash
# Activate virtual environment
source venv/bin/activate

# Run hot-reload tests
python tests/test_config_reload.py
```

Expected output:
```
================================================================================
Configuration Hot-Reload Tests
================================================================================
INFO - Testing InferenceEngine hot-reload...
INFO - ✓ InferenceEngine hot-reload test passed
INFO - Testing SnapshotSaver hot-reload...
INFO - ✓ SnapshotSaver hot-reload test passed
INFO - Testing MotionFilter hot-reload...
INFO - ✓ MotionFilter hot-reload test passed
INFO - Testing TimeOfDayFilter hot-reload...
INFO - ✓ TimeOfDayFilter hot-reload test passed
================================================================================
✅ All hot-reload tests passed!
================================================================================
```

## Contributing

If you encounter issues with hot-reload or have suggestions:

1. Check existing issues: https://github.com/filthyrake/telescope_cam_detection/issues
2. Open a new issue with:
   - Configuration file (redacted credentials)
   - Reload API response
   - System logs
   - Expected vs actual behavior

Pull requests welcome for:
- Additional hot-reloadable settings
- Improved validation
- Automatic file watching
- Web UI config editor

---

**Related Issues:**
- [#78 - Feature: Hot-reload configuration](https://github.com/filthyrake/telescope_cam_detection/issues/78)
- [#81 - Feature: Web-based configuration editor](https://github.com/filthyrake/telescope_cam_detection/issues/81)
