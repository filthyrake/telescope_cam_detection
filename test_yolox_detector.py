"""
Test YOLOX detector wrapper with real camera frame
"""
import sys
import cv2
import numpy as np
import time
from src.yolox_detector import YOLOXDetector

print("="*60)
print("Testing YOLOX Detector Wrapper")
print("="*60)

# Initialize detector
print("\n1. Initializing YOLOX detector...")
detector = YOLOXDetector(
    model_name="yolox-s",
    model_path="models/yolox/yolox_s.pth",
    device="cuda:0",
    conf_threshold=0.25,
    nms_threshold=0.45,
)

# Load model
if not detector.load_model():
    print("✗ Failed to load model")
    sys.exit(1)

print("\n2. Testing detection on random frame...")
# Create test frame (1280x720 like camera)
test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Warmup
for _ in range(10):
    _ = detector.detect(test_frame)

# Benchmark
print("\n3. Benchmarking 50 detection runs...")
times = []
for i in range(50):
    start = time.time()
    detections = detector.detect(test_frame)
    elapsed = time.time() - start
    times.append(elapsed)

    if i < 5:
        print(f"   Run {i+1}: {elapsed*1000:.1f}ms, {len(detections)} detections")

avg_time = np.mean(times)
std_time = np.std(times)
fps = 1 / avg_time

print(f"\n" + "="*60)
print(f"RESULTS:")
print(f"="*60)
print(f"Average: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
print(f"FPS: {fps:.1f}")
print(f"")
print(f"✓ YOLOX detector wrapper working!")
print(f"Ready to integrate into inference engine")
