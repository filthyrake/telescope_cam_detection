# Telescope Monitoring & Collision Detection System
## Technical Specification v1.0

---

## Executive Summary

Real-time object detection system for monitoring backyard telescope equipment using Reolink RLC-410W camera, processing on Linux server with NVIDIA A30 GPU. Primary focus on ultra-low latency detection of people, animals, and telescope collision prevention.

---

## System Architecture

### High-Level Architecture

```
[RLC-410W Camera] --RTSP--> [Stream Processor] ---> [GPU Inference Engine]
                                   |                         |
                                   v                         v
                            [Frame Buffer]           [Detection Results]
                                   |                         |
                                   v                         v
                            [Web Server] <---WebSocket--- [Browser UI]
```

### Core Components

1. **RTSP Stream Handler**: Direct RTSP connection to camera
2. **Inference Pipeline**: NVIDIA A30 optimized detection
3. **WebSocket Server**: Real-time detection result streaming
4. **Web Interface**: Live video with detection overlays
5. **Model Manager**: Custom model training/loading system

---

## Technical Stack

### Backend Core
- **Language**: Python 3.11+
- **RTSP Processing**: OpenCV with GStreamer backend or PyAV for lower latency
- **Deep Learning Framework**: PyTorch 2.0+ with TensorRT optimization
- **Object Detection Model**: YOLOv8/YOLOv9 (primary) or RT-DETR for transformer-based approach
- **Web Framework**: FastAPI with Uvicorn (ASGI)
- **WebSocket**: Native FastAPI WebSocket support
- **Process Manager**: Supervisord or systemd services

### Frontend
- **Framework**: React or vanilla JS with WebRTC
- **Video Rendering**: WebRTC for sub-100ms latency or MSE (Media Source Extensions) as fallback
- **UI Components**: Minimal UI - focus on video feed with overlay canvas
- **Detection Overlays**: Canvas-based bounding boxes with labels

### GPU Optimization
- **CUDA**: 11.8+ for A30 support
- **TensorRT**: 8.6+ for optimized inference
- **cuDNN**: 8.9+
- **Mixed Precision**: FP16 inference for 2x speedup on A30

---

## Implementation Details

### 1. RTSP Stream Capture

```python
# Connection string for RLC-410W
rtsp_url = "rtsp://admin:password@camera_ip:554/h264Preview_01_main"

# GStreamer pipeline for hardware acceleration
pipeline = (
    f"rtspsrc location={rtsp_url} latency=0 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvideoconvert ! video/x-raw(memory:NVMM),format=BGRx ! "
    "nvvideoconvert ! video/x-raw,format=BGR ! appsink"
)
```

### 2. Frame Processing Pipeline

```python
class StreamProcessor:
    def __init__(self):
        self.frame_queue = Queue(maxsize=2)  # Minimal buffer
        self.detection_queue = Queue()
        self.model = load_optimized_model()
    
    def process_stream(self):
        # Drop frames if processing can't keep up
        # Prioritize latest frame over completeness
```

### 3. Object Detection Models

#### Base Models
- **People/Animals**: YOLOv8x pretrained on COCO
- **Telescopes**: Custom trained YOLOv8 on your equipment

#### Custom Training Pipeline
```python
# Dataset structure for telescope detection
dataset/
├── images/
│   ├── telescope_001.jpg
│   ├── telescope_002.jpg
├── labels/
│   ├── telescope_001.txt  # YOLO format annotations
│   ├── telescope_002.txt
├── classes.txt  # telescope, mount, tripod_leg, counterweight
```

#### Collision Detection Logic
```python
class CollisionDetector:
    def __init__(self, danger_threshold=50):  # pixels
        self.danger_zones = []  # Define around tripod legs
        
    def check_collision_risk(self, detections):
        # Calculate distances between telescope parts and tripod legs
        # Track velocity vectors for moving components
        # Predict intersection points
```

### 4. Real-time Web Interface

#### WebSocket Communication
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Send detection results as JSON
        detections = await detection_queue.get()
        await websocket.send_json({
            "timestamp": time.time(),
            "detections": detections,
            "collision_risk": collision_score
        })
```

#### Frontend Display
```javascript
// Overlay detections on video stream
class DetectionOverlay {
    constructor(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
    }
    
    drawDetections(detections) {
        // Clear previous frame
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        detections.forEach(det => {
            if (det.collision_risk > 0.7) {
                this.ctx.strokeStyle = '#FF0000';  // Red for collision risk
                this.ctx.lineWidth = 4;
            } else {
                this.ctx.strokeStyle = '#00FF00';  // Green for normal
                this.ctx.lineWidth = 2;
            }
            
            this.ctx.strokeRect(det.x, det.y, det.width, det.height);
            this.ctx.fillText(det.label, det.x, det.y - 5);
        });
    }
}
```

### 5. Performance Optimization

#### A30 GPU Utilization
- **Batch Size**: 1 (for minimal latency)
- **Input Resolution**: 1280x720 (downscale from 2K for speed)
- **TensorRT Optimization**: Convert ONNX to TRT engine
- **Dynamic Batching**: Disabled for consistent latency

#### Latency Targets
- Camera to capture: ~50ms
- Preprocessing: ~5ms
- Inference: ~10-15ms on A30
- Post-processing: ~5ms
- WebSocket transmission: ~5ms
- **Total: <100ms end-to-end**

### 6. Custom Detection Configuration

#### Configuration File (YAML)
```yaml
detection_zones:
  - name: "telescope_1_danger_zone"
    type: "polygon"
    points: [[100,200], [300,200], [300,400], [100,400]]
    alert_on_entry: ["person", "large_animal"]
    
  - name: "tripod_collision_zone"
    type: "radius"
    center: [640, 480]
    radius: 150
    alert_on_proximity: ["telescope_part", "counterweight"]

detection_classes:
  people:
    model: "yolov8x"
    confidence: 0.5
    
  wildlife:
    model: "yolov8x"
    classes: ["bird", "cat", "dog", "bear", "deer"]
    confidence: 0.3
    
  telescope_equipment:
    model: "custom_telescope_model.pt"
    classes: ["ota", "mount_head", "counterweight", "tripod_leg"]
    confidence: 0.4
```

### 7. Optional Clip Saving

```python
class ClipSaver:
    def __init__(self, buffer_seconds=10):
        self.ring_buffer = deque(maxlen=buffer_seconds * 30)  # 30fps
        
    def save_on_detection(self, detection_type):
        if detection_type in self.save_triggers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clips/{detection_type}_{timestamp}.mp4"
            # Save ring buffer to file
```

---

## Development Phases

### Phase 1: Basic Pipeline (Week 1)
1. RTSP stream capture working
2. Basic YOLOv8 inference on A30
3. Simple web interface showing video feed

### Phase 2: Real-time Optimization (Week 2)
1. TensorRT model optimization
2. WebSocket detection streaming
3. Canvas-based overlay rendering

### Phase 3: Custom Detection (Week 3-4)
1. Collect telescope training data
2. Train custom models
3. Implement collision detection logic

### Phase 4: Advanced Features (Week 5+)
1. Zone-based detection
2. Velocity tracking for collision prediction
3. Configuration UI
4. Clip saving system

---

## Performance Metrics

### Target Specifications
- **Latency**: <100ms from camera to display
- **Frame Rate**: 30fps processing capability
- **GPU Utilization**: 40-60% for single stream
- **CPU Usage**: <20% 
- **Memory**: <4GB RAM, <2GB VRAM
- **Network**: <10Mbps bandwidth usage

---

## Deployment

### Docker Deployment
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY app/ /app
WORKDIR /app

CMD ["python", "main.py"]
```

### System Service
```ini
[Unit]
Description=Telescope Detection System
After=network.target

[Service]
Type=simple
User=telescope
WorkingDirectory=/opt/telescope-detector
ExecStart=/usr/bin/python3 /opt/telescope-detector/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Configuration Examples

### Night Vision Adjustments
```python
# Adjust detection confidence for IR mode
if is_night_vision_active():
    model.conf_threshold *= 0.8  # Lower confidence for IR
    preprocessing.enable_histogram_equalization()
```

### Multi-Telescope Setup
```python
telescopes = {
    "scope_1": {
        "bbox": [100, 100, 500, 500],
        "mount_type": "equatorial",
        "max_slew_speed": 4.0  # degrees/second
    },
    "scope_2": {
        "bbox": [600, 100, 1000, 500],
        "mount_type": "alt-az",
        "max_slew_speed": 6.0
    }
}
```

---

## Testing Strategy

1. **Latency Testing**: Measure frame timestamps through pipeline
2. **Load Testing**: Simulate rapid telescope movement
3. **Detection Accuracy**: Test with various lighting conditions
4. **Collision Detection**: Simulate near-miss scenarios
5. **Network Resilience**: Test with WiFi interference

---

## Future Enhancements

- Multi-camera support for different angles
- Integration with telescope control software (INDI/ASCOM)
- Auto-tracking of specific objects
- ML-based behavior prediction
- Integration with weather stations for environmental awareness
- Emergency stop command to telescope mount on collision detection