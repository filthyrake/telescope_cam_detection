# Quick Start Guide

## 🚀 Phase 1 Implementation Complete!

Your telescope detection system is ready to run. Here's what was built:

### ✅ What's Working

1. **RTSP Stream Capture** - Connects to your Reolink camera (10.0.8.18)
2. **GPU Inference Engine** - YOLOv8 running on NVIDIA A30
3. **Detection Processing** - Post-processes detections and tracks latency
4. **Web Interface** - Real-time video with detection overlays
5. **WebSocket Streaming** - Live detection data to browser

### 🎯 Quick Test

Before running the full system, test individual components:

```bash
# Activate environment
source venv/bin/activate

# Test 1: Camera Connection (~30 seconds)
python tests/test_camera_connection.py

# Test 2: GPU Inference (~2 minutes)
python tests/test_inference.py

# Test 3: End-to-End Latency (~30 seconds)
python tests/test_latency.py
```

### 🏃 Run the System

```bash
# Option 1: Use start script
./start.sh

# Option 2: Manual
source venv/bin/activate
python main.py
```

Then open browser to: **http://localhost:8000**

### 📊 What You'll See

- **Live video stream** from camera (1280x720 downscaled)
- **Bounding boxes** around detected objects:
  - Red boxes = People
  - Orange boxes = Animals (cats, dogs, birds, etc.)
  - Green boxes = Other objects
- **Real-time statistics**:
  - Total detections
  - Inference time
  - End-to-end latency
  - Detection counts by type

### 🎛️ Configuration

Edit `config/config.yaml` to adjust:

- **Model size**: Switch between yolov8n (fast) to yolov8x (accurate)
- **Confidence**: Adjust detection threshold (0.0-1.0)
- **Target classes**: Filter specific objects to detect
- **Resolution**: Change target_width/height for speed/quality tradeoff

### 🔧 Performance Expectations (Phase 1)

- **Latency**: Target <200ms (will optimize to <100ms in Phase 2)
- **FPS**: 15-30 FPS
- **GPU Usage**: 40-60%
- **RAM**: <4GB

### 📝 Next Steps

**Immediate:**
1. Run tests to verify camera and GPU
2. Start main application
3. Check web interface
4. Monitor latency in browser

**Phase 2 (Future):**
- TensorRT optimization for <100ms latency
- GStreamer pipeline for lower capture latency
- Performance dashboard

**Phase 3 (Future):**
- Custom telescope detection models
- Collision detection logic
- Zone-based alerts

### 🐛 Troubleshooting

**Camera won't connect:**
```bash
# Verify camera is reachable
ping 10.0.8.18

# Check RTSP port
nc -zv 10.0.8.18 554
```

**High latency (>200ms):**
- Switch to smaller model: `yolov8n.pt` instead of `yolov8x.pt`
- Lower resolution in config: `target_width: 640, target_height: 480`

**GPU not detected:**
```bash
# Check NVIDIA driver
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 📁 Project Structure

```
telescope_cam_detection/
├── main.py              # Start here
├── start.sh             # Quick start script
├── config/
│   └── config.yaml      # Main configuration
├── src/                 # Core modules
├── web/                 # Frontend files
├── tests/               # Test scripts
└── README.md            # Full documentation
```

### 🎬 Example Run

```bash
$ ./start.sh
🔭 Starting Telescope Detection System...

INFO - Configuration loaded successfully
INFO - Stream capture initialized
INFO - Inference engine initialized
INFO - Detection processor initialized
INFO - Starting web server on http://0.0.0.0:8000
================================================================================
System is running!
Open browser to: http://localhost:8000
Press Ctrl+C to stop
================================================================================
```

### 💡 Tips

- **First run**: YOLOv8 model will auto-download (~50MB for yolov8n, ~150MB for yolov8x)
- **Latency check**: Look at the green indicator in top-right of web UI
- **Detection tuning**: Adjust confidence threshold if too many/few detections
- **Night mode**: Camera's IR mode should work, may need lower confidence threshold

---

**Ready to start? Run:** `./start.sh`

**Questions? Check:** `README.md` for full documentation
