# 📸 Snapshot/Clip Saving Feature

The telescope detection system can automatically save images or video clips when specific objects are detected.

## Features

- **Image Mode**: Save single frames with detections
- **Clip Mode**: Save video clips with pre-detection buffer
- **Trigger Classes**: Only save when specific classes are detected
- **Cooldown**: Prevent duplicate saves for the same class
- **Metadata**: JSON files with detection info, timestamps, and latency
- **Annotations**: Optional bounding box overlays

## Configuration

Edit `config/config.yaml`:

```yaml
snapshots:
  enabled: true                    # Enable snapshot saving
  save_mode: "image"              # "image" or "clip"
  output_dir: "clips"             # Directory to save files

  # Trigger settings
  trigger_classes:                # Classes that trigger saves (empty = all)
    - "person"
    - "cat"
    - "dog"
  min_confidence: 0.6             # Minimum confidence to trigger
  cooldown_seconds: 30            # Seconds between saves per class
  save_annotated: true            # Save with bounding boxes

  # Video clip settings (only used if save_mode="clip")
  clip_duration: 10               # Total clip duration in seconds
  pre_buffer_seconds: 5           # Seconds before detection
  fps: 30                         # Frames per second
```

## Usage Examples

### Image Mode (Default)

Saves single JPG images when triggered:

```yaml
snapshots:
  enabled: true
  save_mode: "image"
  trigger_classes: ["person"]
  min_confidence: 0.6
  cooldown_seconds: 30
```

**Result**: When a person is detected with >60% confidence, saves:
- `clips/person_20250610_143052_123456_conf0.95.jpg` - Image file
- `clips/person_20250610_143052_123456_conf0.95.json` - Metadata

### Clip Mode

Saves MP4 video clips with pre-detection buffer:

```yaml
snapshots:
  enabled: true
  save_mode: "clip"
  trigger_classes: ["person"]
  clip_duration: 10
  pre_buffer_seconds: 5
  fps: 30
```

**Result**: Saves 10-second clips (5s before + 5s after detection)

### Save All Detections

Leave `trigger_classes` empty to save all detections:

```yaml
snapshots:
  trigger_classes: []  # Save everything
```

### Multiple Classes

Save when any of several classes are detected:

```yaml
snapshots:
  trigger_classes:
    - "person"
    - "cat"
    - "dog"
    - "telescope_ota"
    - "tripod_leg"
```

## Managing Snapshots

### View Saved Snapshots

```bash
# Show all snapshots with details
python scripts/view_snapshots.py

# List mode
python scripts/view_snapshots.py --list

# Statistics only
python scripts/view_snapshots.py --stats

# Interactive menu
python scripts/view_snapshots.py --interactive
```

### Browse Files

```bash
# Open clips directory
xdg-open clips/  # Linux
open clips/      # macOS
explorer clips\  # Windows
```

### Delete Old Files

```bash
# Delete snapshots older than 7 days
python scripts/view_snapshots.py --delete-older-than 7
```

## Snapshot Metadata

Each snapshot has an accompanying JSON file with detection information:

```json
{
  "timestamp": 1749453052.123,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 600
      }
    }
  ],
  "detection_counts": {
    "person": 1
  },
  "latency_ms": 45.2
}
```

## Advanced Configuration

### Adjust Cooldown

Prevent spam by adjusting cooldown between saves:

```yaml
cooldown_seconds: 60  # Only save every 60 seconds per class
```

### Confidence Threshold

Only save high-confidence detections:

```yaml
min_confidence: 0.8  # 80% confidence required
```

### Disable Annotations

Save raw frames without bounding boxes:

```yaml
save_annotated: false
```

### Change Output Directory

```yaml
output_dir: "/mnt/storage/telescope_detections"
```

## Storage Considerations

### Image Mode
- ~200-500 KB per image
- With 30s cooldown: ~2-3 images per class per minute
- Daily storage: ~0.5-2 GB (depends on activity)

### Clip Mode
- ~10-50 MB per 10-second clip
- With 30s cooldown: ~2 clips per class per minute
- Daily storage: ~5-20 GB (depends on activity)

### Automatic Cleanup

Add cron job to clean old files:

```bash
# Delete files older than 7 days (runs daily at 3 AM)
0 3 * * * /home/damen/telescope_cam_detection/venv/bin/python \
  /home/damen/telescope_cam_detection/scripts/view_snapshots.py \
  --delete-older-than 7
```

## Troubleshooting

### No Snapshots Being Saved

1. **Check if enabled**:
   ```yaml
   snapshots:
     enabled: true  # Must be true
   ```

2. **Check trigger classes**:
   - If `trigger_classes` is set, only those classes will trigger saves
   - Try setting to empty list `[]` to save all detections

3. **Check confidence threshold**:
   - Lower `min_confidence` to capture more detections
   - Try `0.3` for testing

4. **Check logs**:
   - Look for `📸 Saved snapshot:` messages in console
   - Indicates successful saves

5. **Check cooldown**:
   - Snapshots only save once per cooldown period per class
   - If testing, set `cooldown_seconds: 5` (short cooldown)

### Clips Directory Not Created

The `clips/` directory is created automatically on first save. If not created:

```bash
mkdir -p clips
```

### Permission Errors

Ensure write permissions:

```bash
chmod 755 clips
```

## Integration with Custom Training

After training a custom telescope model, update config to save telescope detections:

```yaml
detection:
  model: "models/telescope_custom.pt"

snapshots:
  enabled: true
  trigger_classes:
    - "telescope_ota"
    - "telescope_mount"
    - "tripod_leg"
    - "counterweight"
  min_confidence: 0.5
  cooldown_seconds: 60
```

This creates a dataset of real-world telescope detections for further refinement.

## Performance Impact

- **Image Mode**: Minimal (<1ms per save)
- **Clip Mode**: Moderate (~50ms for video encoding)
- Saves happen asynchronously in detection processor thread
- No impact on inference or stream capture performance

## Future Enhancements

Potential additions:
- Cloud upload (S3, Google Drive)
- Email/SMS notifications on trigger
- Web gallery viewer
- Motion detection for clip post-buffer
- Multi-camera support
- Alert zones (save only when detection in specific area)

---

**The snapshot feature is now fully operational!** Check the `clips/` directory after detections to see saved files.
