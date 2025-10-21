# Object Tracking - Quick Start

## Enable Tracking

Edit `config/config.yaml`:

```yaml
tracking:
  enabled: true
  algorithm: "iou"
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  per_camera: true
```

## API Endpoints

### Active Tracks
```bash
# All cameras
curl http://localhost:8000/api/tracks/active

# Specific camera
curl http://localhost:8000/api/tracks/active?camera_id=cam1
```

### Track History
```bash
curl http://localhost:8000/api/tracks/{track-uuid}/history
```

### Statistics
```bash
# All time
curl http://localhost:8000/api/tracks/stats

# Last 24 hours
curl "http://localhost:8000/api/tracks/stats?start=$(date -d '1 day ago' +%s)"

# Specific camera
curl "http://localhost:8000/api/tracks/stats?camera_id=cam1"
```

## WebSocket Data

Track data automatically included in `/ws/detections`:

```json
{
  "type": "detections",
  "tracks": {
    "track-uuid": {
      "track_id": "...",
      "class_name": "coyote",
      "status": "active",
      "dwell_time": 5.2,
      "distance_traveled": 124.5,
      "speed": 23.9,
      "trajectory": [[x, y, t], ...]
    }
  },
  "total_active_tracks": 1
}
```

## Key Metrics

- **`dwell_time`**: How long object has been in view (seconds)
- **`distance_traveled`**: Total pixels moved
- **`speed`**: Current velocity (pixels/second)
- **`frames_detected`**: Number of frames where detected
- **`avg_confidence`**: Average detection confidence

## Tuning Parameters

### Too many ID switches?
Lower `iou_threshold`: 0.3 → 0.2

### Too many false tracks?
Increase `min_hits`: 3 → 5

### Tracks lost too quickly?
Increase `max_age`: 30 → 60

## Common Use Cases

### Count unique individuals
```bash
curl http://localhost:8000/api/tracks/stats | jq '.by_class'
```

### Find longest track
```bash
curl http://localhost:8000/api/tracks/stats | jq '.longest_track'
```

### Monitor active tracks
```javascript
// WebSocket client
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.tracks) {
    console.log(`Active tracks: ${data.total_active_tracks}`);
    Object.values(data.tracks).forEach(track => {
      console.log(`${track.class_name}: ${track.dwell_time}s`);
    });
  }
};
```

## See Also

- [Full Documentation](OBJECT_TRACKING.md)
- [API Reference](../api/API_REFERENCE.md)
