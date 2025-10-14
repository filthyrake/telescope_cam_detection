# Telescope Detection System - Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Threading Model](#threading-model)
6. [Detection Pipeline](#detection-pipeline)
7. [Configuration System](#configuration-system)
8. [API Endpoints](#api-endpoints)
9. [Performance Characteristics](#performance-characteristics)
10. [Deployment Architecture](#deployment-architecture)

---

## System Overview

The Telescope Detection System is a real-time object detection and species classification platform designed for:

1. **Wildlife Monitoring**: Detect and identify desert animals using YOLOX + iNaturalist
2. **Telescope Collision Prevention**: Detect telescope equipment and prevent collisions (future)

### Key Features

- **Ultra-low latency**: <35ms end-to-end (Stage 1+2)
- **Two-stage detection**: Broad detection → Fine-grained species ID
- **Real-time video**: RTSP camera stream processing
- **GPU accelerated**: NVIDIA A30 with CUDA
- **Web interface**: Live video + WebSocket detection feed
- **Auto-restart**: Systemd service with automatic recovery

### Technology Stack

| Component | Technology |
|-----------|----------|
| Language | Python 3.12 |
| Detection (Stage 1) | YOLOX-S (Megvii, Apache 2.0) |
| Classification (Stage 2) | EVA02-Large (timm) + iNaturalist 2021 |
| Video Streaming | OpenCV + RTSP/RTSP-TCP |
| Web Framework | FastAPI + Uvicorn |
| Frontend | HTML5 + JavaScript + WebSockets |
| GPU | CUDA 11.8+ / PyTorch 2.1+ |
| Service Management | systemd |
| Cameras | Reolink RLC-410W (2560x1440) + E1 Pro (2560x1440) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TELESCOPE DETECTION SYSTEM                        │
└─────────────────────────────────────────────────────────────────────┘


┌──────────────┐         ┌─────────────────────────────────────────┐
│   Reolink    │  RTSP   │       Stream Capture Threads            │
│   Cameras    │  /TCP   │  (One per camera)                       │
│  - RLC-410W  ├────────▶│  - OpenCV VideoCapture                  │
│  - E1 Pro    │         │  - Per-camera frame sizing              │
│ (2560x1440)  │         │  - Frame queue management               │
└──────────────┘         └───────────────┬─────────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   Frame Queues      │
                              │   (maxsize=1)       │
                              │  - Drops old frames │
                              └──────────┬──────────┘
                                         │
                                         ▼
                       ┌────────────────────────────────────┐
                       │   Inference Engine Threads         │
                       │  (One per camera)                  │
                       │  ┌──────────────────────────────┐  │
                       │  │  Two-Stage Pipeline          │  │
                       │  │                              │  │
                       │  │  ┌────────────────────────┐ │  │
                       │  │  │ STAGE 1: YOLOX         │ │  │
                       │  │  │ - Object detection     │ │  │
                       │  │  │ - Bounding boxes       │ │  │
                       │  │  │ - 80 COCO classes      │ │  │
                       │  │  │ - ~11-21ms inference   │ │  │
                       │  │  └──────────┬─────────────┘ │  │
                       │  │             ▼               │  │
                       │  │  ┌────────────────────────┐ │  │
                       │  │  │ STAGE 2: iNaturalist   │ │  │
                       │  │  │ - Crop detections      │ │  │
                       │  │  │ - Species classifier   │ │  │
                       │  │  │ - 10,000 species       │ │  │
                       │  │  │ - ~20-30ms per detect  │ │  │
                       │  │  └────────────────────────┘ │  │
                       │  │                              │  │
                       │  └──────────────────────────────┘  │
                       │         NVIDIA A30 GPU             │
                       └──────────────┬─────────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │  Inference Queue    │
                           │  (maxsize=10)       │
                           └──────────┬──────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────────┐
                    │  Detection Processor Thread        │
                    │  - Post-processing                 │
                    │  - Filtering (confidence, size)    │
                    │  - Snapshot trigger logic          │
                    │  - Detection history               │
                    └─────────────┬──────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
         ┌─────────────────────┐    ┌─────────────────────┐
         │  Snapshot Saver     │    │  Detection Queue    │
         │  - Save images      │    │  (maxsize=10)       │
         │  - Cooldown logic   │    └──────────┬──────────┘
         │  - Annotated copies │               │
         └─────────────────────┘               ▼
                 │                  ┌─────────────────────┐
                 │                  │   Web Server        │
                 │                  │   (FastAPI)         │
                 │                  │                     │
                 ▼                  │  ┌──────────────┐   │
         ┌─────────────┐            │  │ HTTP Routes  │   │
         │   clips/    │            │  │ - /          │   │
         │  Directory  │            │  │ - /feed      │   │
         └─────────────┘            │  │ - /stats     │   │
                                    │  └──────────────┘   │
                                    │                     │
                                    │  ┌──────────────┐   │
                                    │  │  WebSocket   │   │
                                    │  │ /ws/detects  │   │
                                    │  └──────────────┘   │
                                    │                     │
                                    │  ┌──────────────┐   │
                                    │  │ MJPEG Stream │   │
                                    │  │ /video_feed  │   │
                                    │  └──────────────┘   │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │   Web Browser       │
                                    │   (Client)          │
                                    │  - Live video       │
                                    │  - Detection boxes  │
                                    │  - Species labels   │
                                    │  - Clips gallery    │
                                    └─────────────────────┘
```

---

## Component Details

### 1. Stream Capture (`src/stream_capture.py`)

**Purpose**: Capture frames from RTSP camera stream

**Key Features**:
- OpenCV VideoCapture for RTSP
- Frame resizing (2560x1440 → 1280x720)
- Aggressive frame dropping (keeps only latest)
- Runs in dedicated thread

**Configuration**:
```yaml
camera:
  ip: "192.168.1.100"
  username: "admin"
  password: "..."
  stream: "main"
  target_width: 1280
  target_height: 720
  buffer_size: 1
```

**Threading**: Single capture thread, non-blocking queue puts

**Performance**:
- Target FPS: 30
- Drops frames if queue full (prioritizes latest)

---

### 2. Inference Engine (`src/inference_engine.py`)

**Purpose**: Run GPU-accelerated object detection

**Architecture**:
```
┌──────────────────────────────────────┐
│       InferenceEngine                │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Standard Mode:                │ │
│  │  - Direct YOLO inference       │ │
│  │  - Single model                │ │
│  │  - ~20-25ms                    │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Two-Stage Mode:               │ │
│  │  ┌──────────────────────────┐  │ │
│  │  │ TwoStageDetectionPipeline│  │ │
│  │  │  - YOLOX (Stage 1)  │  │ │
│  │  │  - SpeciesClassifier     │  │ │
│  │  │    (Stage 2)             │  │ │
│  │  └──────────────────────────┘  │ │
│  │  - ~30-35ms total             │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
```

**Configuration**:
```yaml
detection:
  model: "yolov8x-worldv2.pt"
  device: "cuda:0"
  confidence: 0.25
  iou_threshold: 0.45
  min_box_area: 50
  max_detections: 300
  use_two_stage: true
  enable_species_classification: true
```

**Threading**: Single inference thread, processes from frame queue

**Performance**:
- Stage 1 only: ~20-25ms
- Stage 1+2: ~30-35ms
- FPS: 25-30 (real-time)

---

### 3. Two-Stage Pipeline (`src/two_stage_pipeline.py`)

**Purpose**: Orchestrate detection + classification

**Stage 1: YOLOX Detection**
- Open-vocabulary object detection
- Text prompts for classes
- Generates bounding boxes
- Confidence threshold: 0.25

**Stage 2: iNaturalist Classification**
- Crops detected objects
- Routes to appropriate classifier (bird/mammal/reptile)
- EVA02-Large model (10K species)
- Confidence threshold: 0.3

**Data Flow**:
```
Frame → YOLOX → Detections (bbox + class)
                ↓
        For each detection:
                ↓
        Crop image → SpeciesClassifier → Species name
                ↓
        Enhanced detection (class + species)
```

---

### 4. Species Classifier (`src/species_classifier.py`)

**Purpose**: Fine-grained species identification

**Model**: `timm/eva02_large_patch14_clip_336.merged2b_ft_inat21`
- Architecture: EVA02-Large (Vision Transformer)
- Training: iNaturalist 2021 (10,000 species)
- Accuracy: 92% top-1, 98% top-5
- Input: 336x336 RGB
- Output: Species name + confidence

**Taxonomy Mapping**:
- File: `models/inat2021_taxonomy_simple.json`
- Format: `{class_id: "Common Name"}`
- Includes: Kingdom, Phylum, Class, Order, Family, Genus

**Configuration**:
```yaml
species_classification:
  enabled: true
  confidence_threshold: 0.3
  inat_classifier:
    model_name: "timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"
    taxonomy_file: "models/inat2021_taxonomy_simple.json"
    input_size: 336
```

---

### 5. Detection Processor (`src/detection_processor.py`)

**Purpose**: Post-process detections and coordinate snapshots

**Responsibilities**:
- Filter by confidence/size
- Maintain detection history
- Trigger snapshot saves
- Manage cooldown timers
- Forward to web server

**Detection History**:
```python
history = {
    'frame_id': int,
    'timestamp': float,
    'detections': [
        {
            'class_name': str,
            'confidence': float,
            'bbox': {...},
            'species': str,  # Stage 2
            'species_confidence': float  # Stage 2
        }
    ]
}
```

---

### 6. Snapshot Saver (`src/snapshot_saver.py`)

**Purpose**: Save detection events to disk

**Modes**:
- **Image**: Single frame snapshot (current)
- **Clip**: Video clip with pre/post buffer (future)

**Cooldown System**:
- Per-class cooldown timers
- Default: 45 seconds
- Prevents duplicate saves

**Output Format**:
```
clips/
├── bird_20251005_071214_813087_conf0.52.jpg          # Raw
├── bird_20251005_071214_813087_conf0.52_annotated.jpg # Annotated
└── bird_20251005_071214_813087_conf0.52.json         # Metadata
```

**Metadata JSON**:
```json
{
  "timestamp": "2025-10-05T07:12:14.813087",
  "class_name": "bird",
  "confidence": 0.52,
  "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
  "species": "Gambel's Quail",
  "species_confidence": 0.89,
  "frame_shape": [720, 1280, 3]
}
```

**Configuration**:
```yaml
snapshots:
  enabled: true
  save_mode: "image"
  output_dir: "clips"
  trigger_classes: ["person", "bird", ...]
  min_confidence: 0.30
  cooldown_seconds: 45
  save_annotated: true
```

---

### 7. Web Server (`src/web_server.py`)

**Purpose**: Provide web interface and API

**Framework**: FastAPI + Uvicorn

**Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | MJPEG video stream |
| `/ws/detections` | WebSocket | Real-time detection stream |
| `/stats` | GET | System statistics JSON |
| `/clips_list` | GET | List saved clips |
| `/clips/{filename}` | GET | Serve clip file |
| `/clips_browser` | GET | Clips gallery UI |

**WebSocket Protocol**:
```json
{
  "type": "detections",
  "frame_id": 12345,
  "timestamp": 1696512345.123,
  "latency_ms": 32.5,
  "inference_time_ms": 28.2,
  "total_detections": 2,
  "detection_counts": {"bird": 1, "rabbit": 1},
  "detections": [
    {
      "class_name": "bird",
      "confidence": 0.85,
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
      "species": "Gambel's Quail",
      "species_confidence": 0.92
    }
  ]
}
```

---

## Data Flow

### Complete Detection Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DETECTION FLOW                             │
└─────────────────────────────────────────────────────────────────────┘

1. CAPTURE
   Camera → RTSP → OpenCV → Resize → Frame
                                      │
2. QUEUE                              │
   Frame → frame_queue.put() ─────────┘
                │
3. INFERENCE    │
                ├──→ frame_queue.get() → Frame
                │                         │
                │                         ▼
                │              ┌──────────────────────┐
                │              │   STAGE 1: YOLO      │
                │              │   - Detect objects   │
                │              │   - Generate boxes   │
                │              │   - Class labels     │
                │              └──────────┬───────────┘
                │                         │
                │                         ▼
                │              ┌──────────────────────┐
                │              │   STAGE 2: iNat      │
                │              │   - Crop detections  │
                │              │   - Classify species │
                │              │   - Add species name │
                │              └──────────┬───────────┘
                │                         │
                │                         ▼
                │              {class, bbox, species, conf}
                │                         │
4. POST-PROCESS                           │
                ├──→ inference_queue.put()┘
                │                         │
                │                         ▼
                │              detection_processor.get()
                │                         │
                │              ┌──────────▼───────────┐
                │              │  Filter & Process   │
                │              │  - Size filter      │
                │              │  - Confidence       │
                │              │  - History          │
                │              └──────────┬───────────┘
                │                         │
                │              ┌──────────┴───────────┐
                │              ▼                      ▼
                │     snapshot_saver.process()  detection_queue.put()
                │              │                      │
5. OUTPUT       │              ▼                      │
                │      ┌─────────────┐                │
                │      │  Save to    │                │
                │      │  clips/     │                │
                │      └─────────────┘                │
                │                                     │
6. DISPLAY                                            │
                └──────────────────────────────────────┘
                                                      │
                                                      ▼
                                            web_server.send()
                                                      │
                                           ┌──────────┴──────────┐
                                           ▼                     ▼
                                     WebSocket              MJPEG Stream
                                        │                        │
                                        └────────┬───────────────┘
                                                 ▼
                                           Web Browser
                                        (Live visualization)
```

---

## Threading Model

### Thread Overview

| Thread | Component | Purpose | Blocking? |
|--------|-----------|---------|-----------|
| Main | `main.py` | Orchestration, web server | Yes (Uvicorn) |
| Capture | `RTSPStreamCapture` | Camera frame capture | No |
| Inference | `InferenceEngine` | GPU detection/classification | No |
| Detection | `DetectionProcessor` | Post-processing | No |

### Inter-Thread Communication

```
┌──────────────┐
│ Main Thread  │  (Coordinates everything)
└──────┬───────┘
       │
       ├──▶ [Start Capture Thread] ────▶ frame_queue
       │                                      │
       ├──▶ [Start Inference Thread] ◀───────┘
       │          │
       │          └──────────────────▶ inference_queue
       │                                      │
       ├──▶ [Start Detection Thread] ◀───────┘
       │          │
       │          └──────────────────▶ detection_queue
       │                                      │
       └──▶ [Run Web Server] ◀────────────────┘
              (Blocking)
```

### Queue Characteristics

| Queue | Size | Drop Policy | Purpose |
|-------|------|-------------|---------|
| `frame_queue` | 2 | Drop oldest | Latest frames only |
| `inference_queue` | 10 | Block | Preserve detections |
| `detection_queue` | 10 | Block | Preserve for display |

### Synchronization

- **Lock-free**: Queues handle synchronization
- **Non-blocking**: Threads check queues with timeouts
- **Graceful shutdown**: Event flags for clean stop

---

## Detection Pipeline

### Stage 1: YOLOX Detection

**Input**: Frame (1280x720x3 BGR)

**Process**:
1. Preprocess image (normalize, resize if needed)
2. Run YOLOX inference
3. Apply NMS (IoU threshold: 0.45)
4. Filter by confidence (0.25)
5. Filter by size (50px² minimum)

**Output**: List of detections
```python
[
  {
    'class_name': 'bird',
    'confidence': 0.85,
    'bbox': {'x1': 100, 'y1': 200, 'x2': 300, 'y2': 400}
  }
]
```

**Performance**: ~20-25ms on NVIDIA A30

---

### Stage 2: Species Classification

**Input**: Detection from Stage 1

**Process**:
1. Map class to taxonomy group (bird/mammal/reptile)
2. Crop bounding box from frame (with padding)
3. Resize to 336x336
4. Normalize (ImageNet stats)
5. Run EVA02 inference
6. Get top-K predictions
7. Filter by confidence (0.3)

**Output**: Enhanced detection
```python
{
  'class_name': 'bird',
  'confidence': 0.85,
  'bbox': {...},
  'species': "Gambel's Quail",
  'species_confidence': 0.92,
  'species_alternatives': [
    {'species': 'California Quail', 'confidence': 0.78},
    {'species': 'Scaled Quail', 'confidence': 0.65}
  ]
}
```

**Performance**: ~5-10ms per detection on NVIDIA A30

---

### Combined Pipeline Performance

| Scenario | Stage 1 | Stage 2 | Total | FPS |
|----------|---------|---------|-------|-----|
| No detections | 20ms | 0ms | 20ms | 50 |
| 1 detection | 20ms | 8ms | 28ms | 35 |
| 3 detections | 20ms | 24ms | 44ms | 23 |
| 5 detections | 20ms | 40ms | 60ms | 16 |

**Note**: Stage 2 runs sequentially per detection (can be parallelized)

---

## Configuration System

### Configuration File: `config/config.yaml`

**Structure**:
```yaml
camera:
  # Camera connection
  ip: "192.168.1.100"
  username: "admin"
  password: "..."
  stream: "main"  # "main" or "sub"

  # Frame processing
  target_width: 1280
  target_height: 720
  buffer_size: 1

detection:
  # Model
  model: "yolov8x-worldv2.pt"
  device: "cuda:0"

  # YOLOX settings
  use_yolo_world: true
  yolo_world_classes: [...]

  # Two-stage pipeline
  use_two_stage: true
  enable_species_classification: true

  # Thresholds
  confidence: 0.25
  iou_threshold: 0.45
  min_box_area: 50
  max_detections: 300

species_classification:
  enabled: true
  confidence_threshold: 0.3
  inat_classifier:
    model_name: "timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"
    taxonomy_file: "models/inat2021_taxonomy_simple.json"
    input_size: 336

web:
  host: "0.0.0.0"
  port: 8000

performance:
  frame_queue_size: 2
  detection_queue_size: 10
  history_size: 30

snapshots:
  enabled: true
  save_mode: "image"
  output_dir: "clips"
  trigger_classes: [...]
  min_confidence: 0.30
  cooldown_seconds: 45
  save_annotated: true
```

### Configuration Loading

**Location**: `main.py:load_config()`

**Process**:
1. Read YAML file
2. Validate required fields
3. Apply defaults for missing values
4. Pass to components during initialization

**Validation**: Each component validates its config section in `__init__`

---

## API Endpoints

### HTTP Endpoints

#### GET `/`
**Purpose**: Main web interface

**Response**: HTML page with:
- Live video feed
- Real-time detection overlay
- Statistics panel
- Clips gallery link

---

#### GET `/video_feed`
**Purpose**: MJPEG video stream

**Content-Type**: `multipart/x-mixed-replace`

**Format**: Motion JPEG (frame-by-frame)

**FPS**: ~20-30 (based on capture rate)

---

#### GET `/stats`
**Purpose**: System statistics

**Response**:
```json
{
  "stream_capture": {
    "is_connected": true,
    "fps": 29.5,
    "dropped_frames": 12
  },
  "inference_engine": {
    "fps": 28.3,
    "avg_inference_time_ms": 32.1,
    "device": "cuda:0"
  },
  "detection_processor": {
    "processed_count": 15234,
    "history_size": 30
  },
  "uptime_seconds": 3600
}
```

---

#### GET `/clips_list`
**Purpose**: List saved clips

**Query Params**:
- `limit`: Max clips to return (default: 100)

**Response**:
```json
[
  {
    "filename": "bird_20251005_071214_813087_conf0.52.jpg",
    "timestamp": "2025-10-05T07:12:14.813087",
    "class": "bird",
    "confidence": 0.52,
    "has_annotated": true,
    "has_metadata": true
  }
]
```

---

#### GET `/clips/{filename}`
**Purpose**: Serve clip file

**Response**: Image file (JPEG)

---

### WebSocket Endpoint

#### WS `/ws/detections`
**Purpose**: Real-time detection stream

**Connection**: Persistent WebSocket

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
        "y2": 400
      },
      "species": "Gambel's Quail",
      "species_confidence": 0.92
    }
  ]
}
```

**Rate**: ~20-30 messages/second

---

## Performance Characteristics

### Latency Breakdown

```
┌────────────────────────────────────────────────────────────┐
│                     LATENCY BUDGET                          │
└────────────────────────────────────────────────────────────┘

Camera → Network → Capture → Queue → Inference → Queue → Display
  ~5ms     ~5ms     ~2ms      ~1ms     ~30ms      ~1ms    ~2ms

Total: ~46ms end-to-end (conservative estimate)
       ~30-35ms typical (under good conditions)
```

### Component Performance

| Component | Time | Notes |
|-----------|------|-------|
| RTSP capture | 1-2ms | Network dependent |
| Frame resize | 1ms | OpenCV optimized |
| Queue overhead | <1ms | Python Queue |
| YOLO inference | 20-25ms | GPU (A30) |
| Species classify | 5-10ms | Per detection |
| Post-processing | 1-2ms | CPU |
| WebSocket send | 1ms | Local network |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| YOLOX model | ~69MB | GPU VRAM |
| iNaturalist model | ~1.5GB | GPU VRAM |
| Frame buffers | ~50MB | CPU RAM |
| Python overhead | ~500MB | CPU RAM |
| **Total** | **~2.5GB** | Peak usage |

### GPU Utilization

- **Idle**: 5-10%
- **Detection only**: 30-40%
- **Detection + Classification**: 50-60%
- **Memory**: ~2GB / 24GB (8%)

### Network Bandwidth

- **RTSP Stream**: ~4-8 Mbps (2560x1440)
- **WebSocket**: ~5 KB/s (detection messages)
- **MJPEG Feed**: ~2-4 Mbps (1280x720)

---

## Deployment Architecture

### systemd Service

**Service File**: `/etc/systemd/system/telescope_detection.service`

**Architecture**:
```
┌──────────────────────────────────────────┐
│         systemd (PID 1)                  │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  telescope_detection.service             │
│                                          │
│  User: damen                             │
│  WorkingDirectory: /home/damen/...      │
│  ExecStart: venv/bin/python main.py     │
│                                          │
│  Restart: always                         │
│  RestartSec: 10                          │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│         main.py (PID 12345)              │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Thread: StreamCapture             │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  Thread: InferenceEngine           │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  Thread: DetectionProcessor        │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  Thread: Uvicorn (Web Server)      │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### Logging

**Method**: journald (systemd journal)

**View logs**:
```bash
./service.sh logs
./service.sh logs -f
journalctl -u telescope_detection.service
```

**Log Levels**:
- INFO: Normal operation
- WARNING: Recoverable issues
- ERROR: Failures requiring attention

### Auto-Restart

**Policy**: Always restart on failure

**Restart Delay**: 10 seconds

**Scenarios**:
- Python crash → Restart
- GPU error → Restart
- Camera disconnect → Restart (may fail repeatedly)
- OOM kill → Restart

### Monitoring

**Service Status**:
```bash
./service.sh status
systemctl status telescope_detection.service
```

**Performance**:
- Web UI: http://localhost:8000/stats
- GPU: `nvidia-smi`
- Logs: `./service.sh logs -f`

---

## File Structure

```
telescope_cam_detection/
├── main.py                         # Entry point
├── config/
│   └── config.yaml                 # System configuration
├── src/                            # Core modules
│   ├── stream_capture.py
│   ├── inference_engine.py
│   ├── two_stage_pipeline.py
│   ├── species_classifier.py
│   ├── detection_processor.py
│   ├── snapshot_saver.py
│   ├── web_server.py
│   └── visualization_utils.py
├── web/                            # Web UI assets
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/                         # Model weights & taxonomy
│   ├── yolov8x-worldv2.pt         # Auto-downloaded
│   ├── inat2021_taxonomy.json     # Species mapping
│   └── inat2021_taxonomy_simple.json
├── clips/                          # Saved detections
│   ├── *.jpg                       # Raw captures
│   ├── *_annotated.jpg            # With bounding boxes
│   └── *.json                      # Metadata
├── training/                       # Training infrastructure
│   ├── datasets/
│   │   └── telescope_equipment/
│   │       └── images/raw/
│   └── scripts/
│       ├── capture_training_images_headless.py
│       ├── extract_frames_from_stream.py
│       ├── capture_collision_scenarios.py
│       ├── prepare_dataset.py
│       ├── train_custom_model.py
│       └── evaluate_model.py
├── tests/                          # Test scripts
│   ├── test_camera_connection.py
│   ├── test_inference.py
│   ├── test_latency.py
│   └── test_stage2_inat.py
├── scripts/                        # Utility scripts
│   ├── download_inat_taxonomy.py
│   └── view_snapshots.py
├── service.sh                      # Service management
├── telescope_detection.service    # systemd unit file
└── docs/                           # Documentation
    ├── README.md
    ├── ARCHITECTURE.md             # This file
    ├── SERVICE_SETUP.md
    ├── STAGE2_SETUP.md
    ├── QUICKSTART.md
    ├── COLLISION_SCENARIOS.md
    └── TRAINING_WORKFLOW.md
```

---

## Security Considerations

### Network Security

- Web server binds to `0.0.0.0` (all interfaces)
- No authentication on web interface
- Camera credentials in plaintext config

**Recommendations**:
- Use firewall to restrict port 8000
- Consider nginx reverse proxy with auth
- Encrypt config file or use secrets management

### Camera Access

- RTSP credentials stored in config
- No encryption on RTSP stream
- Camera on local network only

### File Permissions

```bash
chmod 600 config/config.yaml  # Protect credentials
chmod 755 service.sh          # Executable
chmod 644 *.md                # Documentation
```

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Service won't start | Camera unreachable | Check network, credentials |
| High latency | GPU overloaded | Reduce detections, disable Stage 2 |
| No detections | Threshold too high | Lower confidence, min_box_area |
| Constant restarts | Persistent failure | Check logs, test components |
| Memory leak | Resource not released | Restart service, check code |

### Debug Commands

```bash
# Service status
./service.sh status

# Recent errors
./service.sh logs | grep ERROR

# GPU status
nvidia-smi

# Test camera
python tests/test_camera_connection.py

# Test inference
python tests/test_inference.py

# Full logs
journalctl -u telescope_detection.service -f
```

---

## Future Enhancements

### Planned Features

1. **Telescope Collision Detection**
   - Train custom model on telescope equipment
   - Implement collision detection logic
   - Add danger zone configuration

2. **Video Clip Mode**
   - Replace image snapshots with video clips
   - Pre/post-event buffering
   - H.264 encoding

3. **Multi-Camera Support**
   - Support multiple camera streams
   - Composite view
   - Per-camera configuration

4. **Advanced Analytics**
   - Species occurrence tracking
   - Time-of-day patterns
   - Heat maps

5. **Mobile App**
   - Push notifications
   - Remote viewing
   - Clip downloads

---

## References

- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
- iNaturalist 2021: https://github.com/visipedia/inat_comp/tree/master/2021
- EVA02: https://github.com/baaivision/EVA
- timm: https://github.com/huggingface/pytorch-image-models
- FastAPI: https://fastapi.tiangolo.com/
- OpenCV: https://opencv.org/

---

**Document Version**: 1.0
**Last Updated**: 2025-10-05
**Author**: Claude (AI Assistant) + Damen
