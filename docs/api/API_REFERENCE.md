# API Reference

Complete API documentation for the Telescope Detection System.

## Base URL

```
http://localhost:8000
```

## HTTP Endpoints

### GET `/`

**Description**: Main web interface

**Response**: HTML page with live detection view

**Features**:
- Real-time video feed with MJPEG
- WebSocket connection for detections
- Live bounding box overlay
- Statistics panel
- Link to clips gallery

---

### GET `/video_feed`

**Description**: Motion JPEG video stream

**Response**:
```
Content-Type: multipart/x-mixed-replace; boundary=frame
```

**Format**: Continuous JPEG frames

**Frame Rate**: ~20-30 FPS (depends on camera capture rate)

**Example Usage**:
```html
<img src="http://localhost:8000/video_feed" />
```

---

### GET `/stats`

**Description**: System performance statistics

**Response**: `application/json`

```json
{
  "stream_capture": {
    "is_connected": true,
    "fps": 29.5,
    "dropped_frames": 12,
    "total_frames": 15234
  },
  "inference_engine": {
    "is_loaded": true,
    "device": "cuda:0",
    "fps": 28.3,
    "avg_inference_time_ms": 32.1,
    "total_inferences": 15100,
    "conf_threshold": 0.25
  },
  "detection_processor": {
    "processed_count": 15100,
    "history_size": 30
  },
  "uptime_seconds": 3600,
  "timestamp": 1696512345.123
}
```

**HTTP Status**:
- `200 OK`: Success
- `500 Internal Server Error`: System error

---

### GET `/api/cameras/{camera_id}/health`

**Description**: Get health status for a specific camera

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `camera_id` | string | Unique camera identifier (e.g., `cam1`, `cam2`) |

**Response**: `application/json`

```json
{
  "camera_id": "cam1",
  "name": "Main Backyard View",
  "is_connected": true,
  "fps": 28.5,
  "dropped_frames": 12,
  "total_frames": 45620,
  "last_frame_time": 1697654321.123,
  "uptime_seconds": 3600.0
}
```

**Fields**:
- `camera_id`: Unique camera identifier
- `name`: Human-readable camera name
- `is_connected`: Whether camera is currently connected
- `fps`: Current frames per second
- `dropped_frames`: Total frames dropped due to full queue
- `total_frames`: Total frames captured since start
- `last_frame_time`: Timestamp of last frame capture
- `uptime_seconds`: Time since camera started (seconds)

**HTTP Status**:
- `200 OK`: Success
- `404 Not Found`: Camera not found

**Example**:
```bash
curl http://localhost:8000/api/cameras/cam1/health
```

**Use Cases**:
- Health monitoring and alerting
- Load balancer health checks
- Automated diagnostics
- Grafana/Prometheus integration

---

### GET `/api/cameras/{camera_id}/stats`

**Description**: Get detailed statistics for a specific camera

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `camera_id` | string | Unique camera identifier (e.g., `cam1`, `cam2`) |

**Response**: `application/json`

```json
{
  "camera_id": "cam1",
  "capture": {
    "is_connected": true,
    "fps": 28.5,
    "total_frames": 45620,
    "dropped_frames": 12,
    "queue_size": 1
  },
  "inference": {
    "device": "cuda:0",
    "fps": 27.3,
    "avg_inference_time_ms": 18.5,
    "total_inferences": 45000,
    "dropped_results": 3,
    "drop_rate": 0.0001
  },
  "detections": {
    "total_processed": 45000,
    "history_size": 30,
    "last_detection_time": 1697654300.0,
    "dropped_results": 0,
    "drop_rate": 0.0,
    "by_class": {
      "bird": 456,
      "person": 12,
      "cat": 8
    }
  },
  "filters": {
    "motion_filter_enabled": true,
    "time_of_day_filter_enabled": true
  }
}
```

**Fields**:
- `capture`: Stream capture statistics
  - `is_connected`: Camera connection status
  - `fps`: Capture frames per second
  - `total_frames`: Total frames captured
  - `dropped_frames`: Frames dropped due to full queue
  - `queue_size`: Current frame queue depth
- `inference`: Inference engine statistics
  - `device`: GPU device (e.g., `cuda:0`, `cpu`)
  - `fps`: Inference frames per second
  - `avg_inference_time_ms`: Average inference time in milliseconds
  - `total_inferences`: Total inferences performed
  - `dropped_results`: Results dropped due to full queue
  - `drop_rate`: Percentage of dropped results (0.0-1.0)
- `detections`: Detection processor statistics
  - `total_processed`: Total detections processed
  - `history_size`: Number of detections in history buffer
  - `last_detection_time`: Timestamp of last detection
  - `dropped_results`: Results dropped by processor
  - `drop_rate`: Percentage of dropped results
  - `by_class`: Detection counts by class name
- `filters`: Filter configuration
  - `motion_filter_enabled`: Whether motion filter is active
  - `time_of_day_filter_enabled`: Whether time-of-day filter is active

**HTTP Status**:
- `200 OK`: Success
- `404 Not Found`: Camera not found

**Example**:
```bash
curl http://localhost:8000/api/cameras/cam1/stats | jq
```

**Use Cases**:
- Performance monitoring per camera
- Debugging camera-specific issues
- Tuning detection settings per camera
- Historical data collection

---

### GET `/api/system/stats`

**Description**: Get system-wide statistics including GPU memory and aggregate metrics

**Response**: `application/json`

```json
{
  "timestamp": 1697654400.0,
  "cameras": {
    "total": 2,
    "connected": 2,
    "active_detections": 2,
    "per_camera": [
      {
        "id": "cam1",
        "name": "Main Backyard View",
        "is_connected": true,
        "fps": 28.5,
        "last_detection_time": 1697654390.0
      },
      {
        "id": "cam2",
        "name": "Secondary View",
        "is_connected": true,
        "fps": 25.3,
        "last_detection_time": 1697654385.0
      }
    ]
  },
  "websocket": {
    "active_connections": 2
  },
  "queues": {
    "frame_queue_total_depth": 3,
    "detection_queue_depth": 1
  },
  "gpu": {
    "memory_allocated_mb": 1250.5,
    "memory_reserved_mb": 1536.0,
    "device": "cuda:0"
  },
  "performance": {
    "avg_fps": 26.9,
    "avg_inference_time_ms": 19.2,
    "total_inferences": 98000
  }
}
```

**Fields**:
- `timestamp`: Current server timestamp
- `cameras`: Camera statistics
  - `total`: Total configured cameras
  - `connected`: Number of connected cameras
  - `active_detections`: Cameras with recent detections
  - `per_camera`: Array of per-camera summaries
- `websocket`: WebSocket statistics
  - `active_connections`: Number of active WebSocket clients
- `queues`: Queue depth statistics
  - `frame_queue_total_depth`: Combined depth of all frame queues
  - `detection_queue_depth`: Depth of shared detection queue
- `gpu`: GPU memory statistics
  - `memory_allocated_mb`: Current GPU memory allocated (MB)
  - `memory_reserved_mb`: Total GPU memory reserved (MB)
  - `device`: GPU device identifier
- `performance`: Aggregate performance metrics
  - `avg_fps`: Average FPS across all cameras
  - `avg_inference_time_ms`: Average inference time across all cameras
  - `total_inferences`: Total inferences across all cameras

**HTTP Status**:
- `200 OK`: Success

**Example**:
```bash
curl http://localhost:8000/api/system/stats | jq '.gpu'
```

**Use Cases**:
- System-wide monitoring dashboards
- GPU memory monitoring and alerts
- Performance baseline tracking
- Capacity planning

---

### GET `/api/clips`

**Description**: List all saved detection clips (requires authentication if token configured)

**Authentication**: Bearer token (optional, see Configuration)

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 100 | Maximum clips to return |

**Request Headers**:
```
Authorization: Bearer <your-token-here>
```

**Response**: `application/json`

```json
[
  {
    "filename": "bird_20251005_071214_813087_conf0.52.jpg",
    "timestamp": "2025-10-05T07:12:14.813087",
    "class": "bird",
    "confidence": 0.52,
    "size_bytes": 245678,
    "has_annotated": true,
    "has_metadata": true,
    "metadata_file": "bird_20251005_071214_813087_conf0.52.json"
  },
  ...
]
```

**HTTP Status**:
- `200 OK`: Success
- `401 Unauthorized`: Missing or invalid authentication token (when `TELESCOPE_CLIPS_TOKEN` is set)
- `500 Internal Server Error`: Error reading clips directory

**Example (authenticated)**:
```bash
curl -H "Authorization: Bearer your-token-here" http://localhost:8000/api/clips?limit=10
```

**Example (public access, if no token configured)**:
```bash
curl http://localhost:8000/api/clips?limit=10
```

---

### GET `/clips_list`

**Description**: Legacy endpoint, redirects to `/api/clips` (requires authentication if token configured)

**Status**: Deprecated, use `/api/clips` instead

**Authentication**: Bearer token (optional, see Configuration)

**HTTP Status**:
- `307 Temporary Redirect`: Redirects to `/api/clips`
- `401 Unauthorized`: Missing or invalid authentication token (when `TELESCOPE_CLIPS_TOKEN` is set)

---

### GET `/api/clips/{filename}`

**Description**: Serve a specific clip file (requires authentication if token configured)

**Authentication**: Bearer token (optional, see Configuration)

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | string | Clip filename (e.g., `bird_20251005_071214_813087_conf0.52.jpg`) |

**Request Headers**:
```
Authorization: Bearer <your-token-here>
```

**Response**: Binary file (JPEG, MP4, or JSON depending on file type)

**HTTP Status**:
- `200 OK`: File found and served
- `401 Unauthorized`: Missing or invalid authentication token (when `TELESCOPE_CLIPS_TOKEN` is set)
- `404 Not Found`: File does not exist
- `403 Forbidden`: File outside clips directory

**Example (authenticated)**:
```bash
curl -H "Authorization: Bearer your-token-here" \
  http://localhost:8000/api/clips/bird_20251005_071214_813087_conf0.52.jpg \
  -o bird.jpg
```

**Example (public access, if no token configured)**:
```bash
curl http://localhost:8000/api/clips/bird_20251005_071214_813087_conf0.52.jpg -o bird.jpg
```

---

### GET `/clips_browser`

**Description**: Web-based clips gallery

**Response**: HTML page

**Features**:
- Thumbnail grid of all clips
- Lightbox view for full images
- Filter by class
- Sort by date
- Shows confidence scores
- Shows species names (Stage 2)

---

## WebSocket Endpoint

### WS `/ws/detections`

**Description**: Real-time detection stream

**Protocol**: WebSocket (RFC 6455)

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/detections');
```

**Message Type**: JSON text frames

**Message Format**:
```json
{
  "type": "detections",
  "frame_id": 12345,
  "timestamp": 1696512345.123,
  "latency_ms": 32.5,
  "inference_time_ms": 28.2,
  "total_detections": 2,
  "detection_counts": {
    "bird": 1,
    "rabbit": 1
  },
  "detections": [
    {
      "class_name": "bird",
      "confidence": 0.85,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400,
        "area": 80000
      },
      "stage1_class": "bird",
      "stage2_enabled": true,
      "species": "Gambel's Quail",
      "species_confidence": 0.92,
      "species_alternatives": [
        {
          "species": "California Quail",
          "confidence": 0.78,
          "class_id": 3621
        },
        {
          "species": "Scaled Quail",
          "confidence": 0.65,
          "class_id": 3623
        }
      ],
      "taxonomy_group": "bird"
    },
    {
      "class_name": "rabbit",
      "confidence": 0.72,
      "bbox": {
        "x1": 500,
        "y1": 300,
        "x2": 650,
        "y2": 450,
        "area": 22500
      },
      "stage1_class": "rabbit",
      "stage2_enabled": true,
      "species": "Desert Cottontail",
      "species_confidence": 0.88,
      "taxonomy_group": "mammal"
    }
  ]
}
```

**Message Rate**: ~20-30 messages/second

**Connection Lifecycle**:
1. **Connect**: Client initiates WebSocket connection
2. **Accept**: Server accepts and logs connection
3. **Stream**: Server sends detection messages continuously
4. **Disconnect**: Client closes or connection lost
5. **Reconnect**: Client should reconnect on disconnect

**Error Handling**:
```javascript
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket closed, reconnecting...');
  setTimeout(() => connectWebSocket(), 1000);
};
```

**Connection Limits**: No hard limit, but performance degrades with many clients

---

## Data Models

### Detection Object

```typescript
interface Detection {
  class_name: string;              // YOLOX COCO class
  confidence: number;              // 0.0 - 1.0
  bbox: BoundingBox;
  stage1_class: string;            // Original detection class
  stage2_enabled: boolean;         // Whether Stage 2 ran
  species?: string;                // iNaturalist species name
  species_confidence?: number;     // 0.0 - 1.0
  species_alternatives?: AlternativeSpecies[];
  taxonomy_group?: string;         // "bird" | "mammal" | "reptile"
}

interface BoundingBox {
  x1: number;  // Top-left X
  y1: number;  // Top-left Y
  x2: number;  // Bottom-right X
  y2: number;  // Bottom-right Y
  area: number; // Pixels²
}

interface AlternativeSpecies {
  species: string;
  confidence: number;
  class_id: number;
}
```

### Clip Metadata

```typescript
interface ClipMetadata {
  timestamp: string;              // ISO 8601
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
  species?: string;
  species_confidence?: number;
  frame_shape: [number, number, number];  // [height, width, channels]
}
```

---

## Client Libraries

### JavaScript/TypeScript

**Connect to WebSocket**:
```javascript
class DetectionClient {
  constructor(url = 'ws://localhost:8000/ws/detections') {
    this.url = url;
    this.ws = null;
    this.onDetection = null;
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (this.onDetection) {
        this.onDetection(data);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('Disconnected, reconnecting in 1s...');
      setTimeout(() => this.connect(), 1000);
    };
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage
const client = new DetectionClient();
client.onDetection = (data) => {
  console.log(`Detected ${data.total_detections} objects`);
  data.detections.forEach(det => {
    console.log(`- ${det.class_name}: ${det.confidence.toFixed(2)}`);
    if (det.species) {
      console.log(`  Species: ${det.species} (${det.species_confidence.toFixed(2)})`);
    }
  });
};
client.connect();
```

**Fetch Statistics**:
```javascript
async function getStats() {
  const response = await fetch('http://localhost:8000/stats');
  const stats = await response.json();
  console.log(`FPS: ${stats.inference_engine.fps.toFixed(1)}`);
  console.log(`Latency: ${stats.inference_engine.avg_inference_time_ms.toFixed(1)}ms`);
  return stats;
}
```

**List Clips**:
```javascript
async function getClips(limit = 50, token = null) {
  const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
  const response = await fetch(`http://localhost:8000/api/clips?limit=${limit}`, { headers });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const clips = await response.json();
  return clips;
}

// Usage (with authentication)
const clips = await getClips(50, 'your-token-here');

// Usage (public access, if no token configured)
const clips = await getClips(50);
```

---

### Python

**WebSocket Client**:
```python
import asyncio
import websockets
import json

async def listen_detections():
    uri = "ws://localhost:8000/ws/detections"

    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            print(f"Detected {data['total_detections']} objects")
            for det in data['detections']:
                print(f"- {det['class_name']}: {det['confidence']:.2f}")
                if 'species' in det:
                    print(f"  Species: {det['species']} ({det['species_confidence']:.2f})")

# Run
asyncio.run(listen_detections())
```

**HTTP Client**:
```python
import requests

def get_stats():
    response = requests.get('http://localhost:8000/stats')
    return response.json()

def get_clips(limit=50, token=None):
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    response = requests.get(f'http://localhost:8000/api/clips?limit={limit}', headers=headers)
    response.raise_for_status()  # Raise exception for 401, 404, etc.
    return response.json()

# Usage
stats = get_stats()
print(f"FPS: {stats['inference_engine']['fps']:.1f}")

# With authentication
clips = get_clips(limit=10, token='your-token-here')
for clip in clips:
    print(f"{clip['timestamp']}: {clip['class']} ({clip['confidence']:.2f})")

# Or without authentication (if no token configured)
clips = get_clips(limit=10)
```

---

## Rate Limits

**No hard rate limits**, but consider:

- **WebSocket**: Each client consumes ~5 KB/s
- **MJPEG Stream**: Each client consumes ~2-4 Mbps
- **HTTP Requests**: No limit, but be reasonable

**Recommendations**:
- Limit WebSocket clients to 5-10 for best performance
- Use HTTP polling sparingly (max 1 req/second)
- MJPEG streams are resource-intensive

---

## Authentication

### Clips Directory Authentication (Optional)

The system supports optional Bearer token authentication for saved clips:

**Configuration**:
```bash
# Set environment variable
export TELESCOPE_CLIPS_TOKEN="your-secure-random-token"

# Generate secure token
openssl rand -base64 32
```

**Protected Endpoints**:
- `/api/clips` - List clips
- `/api/clips/{filename}` - Download clips
- `/clips_list` - Legacy endpoint (redirects)

**Usage**:
```bash
# Authenticated request
curl -H "Authorization: Bearer your-token-here" http://localhost:8000/api/clips

# Returns 401 if token is wrong or missing (when TELESCOPE_CLIPS_TOKEN is set)
```

**Backward Compatibility**:
- If `TELESCOPE_CLIPS_TOKEN` is not set, clips are publicly accessible
- Warning logged on first unauthenticated access

### Other Endpoints

**Current**: No authentication (open access)

**Recommendations**:
- Use nginx reverse proxy with HTTP basic auth
- Add API key support for programmatic access
- Implement OAuth2 for production deployments

---

## CORS

**Current**: No CORS headers (same-origin only)

**To enable CORS**, modify `src/web_server.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Error Responses

### HTTP Errors

```json
{
  "detail": "Error message here"
}
```

**Status Codes**:
- `200 OK`: Success
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

### WebSocket Errors

WebSocket errors are logged but not sent to clients. Check server logs for details.

---

## Performance Tips

1. **Use WebSocket for real-time data** (not HTTP polling)
2. **Limit concurrent MJPEG streams** (max 2-3 clients)
3. **Fetch clips list with limit** to avoid large responses
4. **Cache statistics** if polling frequently
5. **Use thumbnail images** when displaying many clips

---

## Examples

### Full Web Client Example

See `web/index.html` for complete working example with:
- MJPEG video display
- WebSocket detection overlay
- Real-time bounding boxes
- Species labels
- Statistics display

### Python Monitoring Script

```python
#!/usr/bin/env python3
"""Monitor telescope detection system."""

import asyncio
import websockets
import json
from datetime import datetime

async def monitor():
    uri = "ws://localhost:8000/ws/detections"

    async with websockets.connect(uri) as ws:
        print("Connected to detection stream")

        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            if data['total_detections'] > 0:
                timestamp = datetime.fromtimestamp(data['timestamp'])
                print(f"\n[{timestamp}] {data['total_detections']} detections:")

                for det in data['detections']:
                    species = det.get('species', det['class_name'])
                    conf = det.get('species_confidence', det['confidence'])
                    print(f"  • {species} ({conf:.1%})")

if __name__ == "__main__":
    asyncio.run(monitor())
```

---

## Changelog

### v1.0 (2025-10-05)
- Initial API documentation
- WebSocket detection stream
- HTTP endpoints for stats and clips
- Two-stage detection with species classification

---

**Document Version**: 1.0
**Last Updated**: 2025-10-05
