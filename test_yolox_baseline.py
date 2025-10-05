"""
Test YOLOX baseline inference speed (before TensorRT)
"""
import sys
sys.path.insert(0, 'YOLOX')

import torch
import cv2
import numpy as np
import time
from yolox.exp import get_exp
from yolox.utils import postprocess

print("="*60)
print("Testing YOLOX Baseline Inference")
print("="*60)

# Load YOLOX model
print("\n1. Loading YOLOX-S model...")
exp = get_exp(None, "yolox-s")
model = exp.get_model()

# Load weights
ckpt = torch.load("models/yolox/yolox_s.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.cuda()
model.eval()

print(f"✓ YOLOX-S loaded")
print(f"  Input size: {exp.test_size}")
print(f"  Num classes: {exp.num_classes} (COCO)")

# Test inference
print(f"\n2. Testing baseline inference...")
test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Preprocess
img_resized = cv2.resize(test_img, tuple(exp.test_size))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().cuda()

# Warmup
with torch.no_grad():
    for _ in range(10):
        _ = model(img_tensor)

# Benchmark
print(f"\n3. Benchmarking 50 inference runs...")
times = []
with torch.no_grad():
    for i in range(50):
        start = time.time()
        outputs = model(img_tensor)
        torch.cuda.synchronize()  # Wait for GPU
        elapsed = time.time() - start
        times.append(elapsed)
        if i < 10:
            print(f"   Run {i+1}: {elapsed*1000:.1f}ms")

avg_time = np.mean(times)
std_time = np.std(times)
fps = 1 / avg_time

print(f"\n" + "="*60)
print(f"BASELINE RESULTS:")
print(f"="*60)
print(f"Average: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
print(f"FPS: {fps:.1f}")
print(f"")
print(f"Compare to GroundingDINO: 560ms (1.8 FPS)")
print(f"Speedup: {560/avg_time/1000:.1f}x faster")
print(f"")
print(f"Next: Convert to TensorRT for ~2-5x additional speedup")
print(f"Expected after TensorRT: 5-15ms (65-200 FPS)")
