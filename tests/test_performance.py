#!/usr/bin/env python3
"""
Quick performance diagnostic to identify latency bottleneck.
"""
import time
import yaml
import cv2
import numpy as np
from pathlib import Path

# Load config
config_path = Path("config/config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

print("=" * 80)
print("PERFORMANCE DIAGNOSTIC")
print("=" * 80)

# Test 1: Camera connection and frame capture
print("\n1. Testing camera connection and frame capture...")
camera_config = config['camera']
stream_path = "h264Preview_01_main" if camera_config['stream'] == "main" else "h264Preview_01_sub"
rtsp_url = f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['ip']}:554/{stream_path}"
print(f"   RTSP URL: rtsp://{camera_config['username']}:***@{camera_config['ip']}:554/{stream_path}")

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("❌ FAILED: Cannot connect to camera")
    exit(1)

# Capture a few frames and measure time
frame_times = []
for i in range(5):
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        print(f"❌ FAILED: Cannot read frame {i+1}")
        continue
    elapsed = (time.time() - start) * 1000
    frame_times.append(elapsed)
    print(f"   Frame {i+1}: {elapsed:.1f}ms, shape={frame.shape}")

cap.release()
avg_frame_time = np.mean(frame_times)
print(f"✅ Camera OK: Average frame capture time: {avg_frame_time:.1f}ms")

# Test 2: Inference engine
print("\n2. Testing inference engine with GroundingDINO...")
print("   Loading model (this may take a few seconds)...")

from src.inference_engine import InferenceEngine

detection_config = config['detection']
model_config_dict = detection_config.get('model', {})
text_prompts = detection_config.get('text_prompts', [])

print(f"   Text prompts: {len(text_prompts)} classes")
print(f"   Caption length: {len(' . '.join(text_prompts))} characters")

engine = InferenceEngine(
    model_config=model_config_dict.get('config', 'models/GroundingDINO_SwinT_OGC.py'),
    model_weights=model_config_dict.get('weights', 'models/groundingdino_swint_ogc.pth'),
    device=detection_config['device'],
    box_threshold=detection_config.get('box_threshold', 0.25),
    text_threshold=detection_config.get('text_threshold', 0.25),
    text_prompts=text_prompts,
    min_box_area=detection_config.get('min_box_area', 0),
    max_det=detection_config.get('max_detections', 300),
    class_confidence_overrides=detection_config.get('class_confidence_overrides', {})
)

if not engine.load_model():
    print("❌ FAILED: Cannot load model")
    exit(1)

print("   Model loaded successfully!")

# Test inference on a blank frame
print("\n3. Testing inference speed...")
test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Warmup (first inference is always slower)
_ = engine.infer_single(test_frame)

# Measure actual inference time
inference_times = []
for i in range(10):
    start = time.time()
    detections = engine.infer_single(test_frame)
    elapsed = (time.time() - start) * 1000
    inference_times.append(elapsed)
    print(f"   Inference {i+1}: {elapsed:.1f}ms, {len(detections)} detections")

avg_inference = np.mean(inference_times)
print(f"\n✅ Average inference time: {avg_inference:.1f}ms")

# Test 3: Total pipeline estimate
print("\n4. Estimated total pipeline latency:")
print(f"   - Frame capture:  {avg_frame_time:.1f}ms")
print(f"   - Inference:      {avg_inference:.1f}ms")
print(f"   - Processing:     ~5-10ms (estimated)")
print(f"   - Queue delays:   ~0-50ms (estimated)")
print(f"   - TOTAL ESTIMATE: {avg_frame_time + avg_inference + 10:.1f}ms")

print("\n" + "=" * 80)
if avg_inference > 500:
    print("⚠️  WARNING: Inference is VERY SLOW (>500ms)")
    print("   This is expected without TensorRT optimization.")
    print("   GroundingDINO needs TensorRT to achieve <50ms inference.")
elif avg_inference > 200:
    print("⚠️  Inference is slower than expected (>200ms)")
    print("   Expected ~120ms unoptimized. This might be due to:")
    print("   - Long caption with 93 text prompts")
    print("   - GPU memory pressure")
    print("   - Model not fully optimized")
elif avg_inference > 100:
    print("✅ Inference speed is within expected range (100-200ms)")
    print("   TensorRT optimization will reduce this to <50ms")
else:
    print("✅ Inference speed is excellent (<100ms)")

print("=" * 80)
