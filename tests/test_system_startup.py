"""
Test system startup with YOLOX
Quick test to verify all components initialize correctly
"""
import sys
import time
import logging
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from inference_engine_yolox import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*60)
print("Testing System Startup with YOLOX")
print("="*60)

# Load config
print("\n1. Loading config...")
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
logger.info("✓ Config loaded")

# Test inference engine initialization
print("\n2. Initializing YOLOX inference engine...")
detection_config = config['detection']
model_config = detection_config.get('model', {})

engine = InferenceEngine(
    model_name=model_config.get('name', 'yolox-s'),
    model_path=model_config.get('weights', 'models/yolox/yolox_s.pth'),
    device=detection_config['device'],
    conf_threshold=detection_config.get('conf_threshold', 0.25),
    nms_threshold=detection_config.get('nms_threshold', 0.45),
    min_box_area=detection_config.get('min_box_area', 0),
    max_det=detection_config.get('max_detections', 300),
    class_confidence_overrides=detection_config.get('class_confidence_overrides', {})
)

# Load model
if not engine.load_model():
    logger.error("✗ Failed to load model")
    sys.exit(1)

logger.info("✓ Inference engine initialized")

# Test inference
print("\n3. Testing inference...")
import numpy as np
test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

start = time.time()
detections = engine.infer_single(test_frame)
elapsed = time.time() - start

logger.info(f"✓ Inference successful: {elapsed*1000:.1f}ms, {len(detections)} detections")

print("\n" + "="*60)
print("SUCCESS!")
print("="*60)
print(f"")
print(f"✓ YOLOX integration complete")
print(f"✓ Inference time: {elapsed*1000:.1f}ms (~{1/elapsed:.0f} FPS)")
print(f"✓ Ready to start full system with live camera")
print(f"")
print(f"To start the full system:")
print(f"  python main.py")
print(f"")
print(f"Then open browser to: http://localhost:8000")
