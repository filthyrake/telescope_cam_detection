# Object Tracking Feature

## Overview

The object tracking feature assigns unique IDs to detected objects and tracks them across multiple frames. This enables:

- **Unique animal counts** - Distinguish "5 detections of same coyote" from "5 different coyotes"
- **Dwell time measurement** - Track how long an animal stayed in view
- **Movement analysis** - Calculate velocity, direction, and distance traveled
- **Behavior analytics** - Identify patterns like loitering or rapid movement
- **Cross-frame continuity** - Maintain identity even with temporary occlusions

## How It Works

### Tracking Algorithm

The system uses **IoU (Intersection over Union)** based tracking:

1. **Detection Association**: For each new frame, detections are matched to existing tracks using IoU overlap
2. **Track Creation**: Unmatched detections create new tracks with unique UUIDs
3. **Track Update**: Matched detections update the track's position, confidence, and trajectory
4. **Track Aging**: Tracks not seen for `max_age` frames are marked as completed
5. **Validation**: Only tracks with at least `min_hits` detections are kept in history

### Track Lifecycle

```
┌─────────────┐    Detection    ┌──────────┐    No Detection    ┌──────────┐
│   Created   │ ────────────────>│  Active  │ ──────────────────>│   Lost   │
│ (frame 1)   │                  │(tracking)│   (age counter)    │ (aged++)  │
└─────────────┘                  └──────────┘                    └──────────┘
                                       │                                │
                                       │                                │
                                       │ Matched detection              │ age > max_age
                                       └────────────────────────────────┘
                                                                         │
                                                                         ▼
                                                                   ┌──────────┐
                                                                   │Completed │
                                                                   │ (stored) │
                                                                   └──────────┘
```

## Configuration

Add to `config/config.yaml`:

```yaml
tracking:
  enabled: true  # Enable object tracking
  algorithm: "iou"  # Tracking algorithm (currently only "iou" supported)
  
  # Tracking parameters
  max_age: 30  # Delete track after 30 frames without detection
  min_hits: 3  # Require 3 detections before considering track valid
  iou_threshold: 0.3  # Minimum IoU for association (0.0-1.0)
  
  # Per-camera tracking
  per_camera: true  # Track separately per camera (no cross-camera tracking)
```

### Parameters Explained

- **`max_age`**: How many frames to wait before deleting a lost track
  - Higher values = more tolerance for occlusions/missed detections
  - Lower values = faster cleanup of old tracks
  - Recommended: 30 frames (~1 second at 30 FPS)

- **`min_hits`**: Minimum detections required before track is considered valid
  - Higher values = fewer false positives from spurious detections
  - Lower values = faster track establishment
  - Recommended: 3 detections

- **`iou_threshold`**: Minimum overlap (IoU) to match detection to track
  - Higher values = stricter matching (fewer ID switches, more new tracks)
  - Lower values = looser matching (more ID switches, fewer new tracks)
  - Recommended: 0.3 (30% overlap)

- **`per_camera`**: Whether to track separately per camera
  - `true`: Each camera has independent tracking (no cross-camera IDs)
  - `false`: Global tracking across all cameras (experimental)

## API Endpoints

### Get Active Tracks

```
GET /api/tracks/active?camera_id=cam1
```

Returns currently active tracks (objects in view).

**Query Parameters:**
- `camera_id` (optional): Filter by camera ID

**Response:**
```json
{
  "tracks": [
    {
      "track_id": "550e8400-e29b-41d4-a716-446655440000",
      "class_name": "coyote",
      "species": "Canis latrans",
      "camera_id": "cam1",
      "first_seen": 1729538625.5,
      "last_seen": 1729538627.2,
      "frames_detected": 45,
      "current_bbox": {
        "x1": 500,
        "y1": 300,
        "x2": 700,
        "y2": 500
      },
      "current_confidence": 0.87,
      "avg_confidence": 0.85,
      "distance_traveled": 234.5,
      "speed": 117.2,
      "dwell_time": 1.7,
      "status": "active",
      "trajectory": [
        [500, 400, 1729538625.5],
        [505, 405, 1729538625.55],
        ...
      ]
    }
  ],
  "count": 1
}
```

### Get Track History

```
GET /api/tracks/{track_id}/history
```

Returns full trajectory and history for a specific track.

**Path Parameters:**
- `track_id`: Track UUID

**Response:**
```json
{
  "track_id": "550e8400-e29b-41d4-a716-446655440000",
  "class_name": "coyote",
  "species": "Canis latrans",
  "camera_id": "cam1",
  "first_seen": 1729538625.5,
  "last_seen": 1729538627.2,
  "frames_detected": 45,
  "avg_confidence": 0.85,
  "distance_traveled": 234.5,
  "speed": 117.2,
  "dwell_time": 1.7,
  "status": "completed",
  "trajectory": [
    [500, 400, 1729538625.5],
    [505, 405, 1729538625.55],
    ...
  ],
  "bbox_history": [
    {"x1": 500, "y1": 300, "x2": 700, "y2": 500},
    {"x1": 505, "y1": 305, "x2": 705, "y2": 505},
    ...
  ]
}
```

### Get Tracking Statistics

```
GET /api/tracks/stats?camera_id=cam1&start=1729538000
```

Returns tracking statistics.

**Query Parameters:**
- `camera_id` (optional): Filter by camera ID
- `start` (optional): Unix timestamp to filter completed tracks

**Response:**
```json
{
  "total_unique_tracks": 234,
  "total_active_tracks": 3,
  "total_completed_tracks": 231,
  "by_class": {
    "coyote": 12,
    "rabbit": 89,
    "bird": 133
  },
  "active_by_class": {
    "coyote": 1,
    "rabbit": 2
  },
  "completed_by_class": {
    "coyote": 11,
    "rabbit": 87,
    "bird": 133
  },
  "avg_dwell_time_seconds": 15.3,
  "longest_track": {
    "track_id": "550e8400-e29b-41d4-a716-446655440000",
    "class_name": "coyote",
    "duration_seconds": 120.5,
    "distance_traveled": 1234.5,
    "frames_detected": 3615
  }
}
```

## WebSocket Updates

Tracking information is automatically included in WebSocket messages when enabled:

```json
{
  "type": "detections",
  "camera_id": "cam1",
  "camera_name": "Main Backyard View",
  "frame_id": 12345,
  "timestamp": 1729538627.5,
  "total_detections": 2,
  "detections": [...],
  "tracks": {
    "550e8400-e29b-41d4-a716-446655440000": {
      "track_id": "550e8400-e29b-41d4-a716-446655440000",
      "class_name": "coyote",
      "status": "active",
      "current_bbox": {...},
      "trajectory": [[x, y, t], ...]
    }
  },
  "total_active_tracks": 1
}
```

## Track Data Structure

Each track maintains:

### Identity
- `track_id`: Unique UUID
- `class_name`: Object class (e.g., "coyote", "bird")
- `species`: iNaturalist species name (if Stage 2 enabled)
- `camera_id`: Source camera

### Lifecycle
- `first_seen`: Timestamp when track was created
- `last_seen`: Timestamp of last detection
- `frames_detected`: Total frames where object was detected
- `status`: "active", "lost", or "completed"

### Current State
- `current_bbox`: Latest bounding box {x1, y1, x2, y2}
- `current_confidence`: Latest detection confidence

### History (limited to last 100 frames)
- `trajectory`: List of (x, y, timestamp) center points
- `bbox_history`: List of bounding boxes
- `confidence_history`: List of confidence values

### Metrics
- `avg_confidence`: Average confidence across all detections
- `distance_traveled`: Total pixels traveled
- `speed`: Current velocity (pixels/second)
- `dwell_time`: Time between first and last seen (seconds)

## Use Cases

### 1. Population Counting

**Problem:** "How many unique coyotes visited last week?"

**Solution:**
```bash
# Get stats for last 7 days
curl "http://localhost:8000/api/tracks/stats?camera_id=cam1&start=$(date -d '7 days ago' +%s)"
```

This gives you **unique individuals** rather than total detection count.

### 2. Behavior Analysis

**Problem:** "Did that coyote linger or just pass through?"

**Solution:** Check `dwell_time` in track data:
- Dwell time > 60s = Loitering behavior (investigating area)
- Dwell time < 10s = Passing through
- High `distance_traveled` + short dwell = Running/chasing

### 3. Movement Patterns

**Problem:** "What direction do rabbits typically travel?"

**Solution:** Analyze `trajectory` data:
```python
# Get trajectory from track
trajectory = track['trajectory']
start_x, start_y, _ = trajectory[0]
end_x, end_y, _ = trajectory[-1]

# Calculate direction
direction_x = end_x - start_x
direction_y = end_y - start_y

if direction_x > 0:
    print("Traveling east")
else:
    print("Traveling west")
```

### 4. Zone Analytics

**Problem:** "Track entered danger zone and stayed 45 seconds"

**Future Enhancement:** Combine tracks with detection zones to detect:
- Entry/exit events
- Dwell time in specific zones
- Approach speed and direction

### 5. Rare Species Detection

**Problem:** "Bobcat last seen 2 weeks ago, returned today"

**Solution:** Query historical tracks:
```bash
# Find all bobcat tracks
curl "http://localhost:8000/api/tracks/stats" | jq '.by_class.bobcat'

# Get specific track history
curl "http://localhost:8000/api/tracks/{track_id}/history"
```

## Performance Considerations

### Memory Usage

- Each track stores up to 100 trajectory points
- Completed tracks are kept in memory (not persisted to disk)
- Consider periodic cleanup for long-running systems

### CPU Impact

- IoU calculation is fast (~0.1ms per detection pair)
- Minimal overhead at <10 detections per frame
- Scales linearly with number of detections

### Accuracy Factors

1. **Detection Quality**
   - More stable detections = better tracking
   - Higher confidence thresholds reduce false tracks

2. **Frame Rate**
   - Higher FPS = better temporal continuity
   - Lower FPS = more gaps, harder to maintain IDs

3. **Object Motion**
   - Slow-moving objects = easier to track
   - Fast-moving objects may create new tracks if they move >70% between frames

4. **Occlusions**
   - `max_age` parameter handles temporary occlusions
   - Long occlusions (>max_age frames) will create new tracks

## Limitations

1. **No Re-identification**: Once a track is lost, the same object will get a new ID if it returns
2. **Single-camera**: Tracks don't persist across cameras (no cross-camera tracking)
3. **Class-based matching**: Objects must have same class to match (prevents bird→cat switches)
4. **No appearance features**: Uses only position/IoU (no visual similarity matching)

## Future Enhancements

### DeepSORT Integration
- Add appearance feature extraction
- Better re-identification after occlusions
- More robust to fast motion

### Cross-camera Tracking
- Track animals across multiple cameras
- Requires camera pose/location information
- Useful for property-wide monitoring

### Track Persistence
- Save completed tracks to database
- Historical analytics over weeks/months
- Track revisit detection

### Advanced Analytics
- Speed/acceleration analysis
- Abnormal behavior detection
- Zone-based event triggers

## Troubleshooting

### Tracks Keep Switching IDs

**Cause:** `iou_threshold` too high, detections don't overlap enough

**Solution:** Lower `iou_threshold` from 0.3 to 0.2

### Too Many Short-lived Tracks

**Cause:** `min_hits` too low, spurious detections create tracks

**Solution:** Increase `min_hits` from 3 to 5

### Tracks Disappear Too Quickly

**Cause:** `max_age` too low, temporary occlusions lose tracks

**Solution:** Increase `max_age` from 30 to 60 frames

### Multiple Tracks for Same Object

**Cause:** Large gaps between detections or fast motion

**Solution:**
- Increase detection frame rate
- Lower confidence threshold to get more detections
- Increase `max_age` to tolerate longer gaps

## Example Workflow

1. **Enable tracking** in `config/config.yaml`
2. **Start system** and monitor WebSocket for track IDs
3. **View active tracks** via API to see current objects
4. **Query statistics** to analyze patterns over time
5. **Retrieve track history** for interesting individuals

## Testing

Run tracking tests:

```bash
# Unit tests (no dependencies)
python3 tests/test_tracking.py

# All tests
python3 -m pytest tests/test_tracking.py -v
```

15 tests cover:
- Track initialization and updates
- Velocity and distance calculations
- Multi-object tracking
- Track aging and completion
- Per-camera isolation
- Statistics generation
