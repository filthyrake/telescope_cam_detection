#!/usr/bin/env python3
"""
Test script to measure end-to-end latency.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import time
from queue import Queue
import numpy as np

from stream_capture import RTSPStreamCapture, create_rtsp_url
from inference_engine import InferenceEngine
from detection_processor import DetectionProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_end_to_end_latency(
    camera_ip: str,
    username: str,
    password: str,
    duration: int = 30
):
    """
    Measure end-to-end latency from camera to detection results.

    Args:
        camera_ip: IP address of the camera
        username: Camera username
        password: Camera password
        duration: Test duration in seconds
    """
    logger.info("=" * 80)
    logger.info("End-to-End Latency Test")
    logger.info("=" * 80)

    # Create queues
    frame_queue = Queue(maxsize=2)
    inference_queue = Queue(maxsize=10)
    detection_queue = Queue(maxsize=10)

    # Create RTSP URL
    rtsp_url = create_rtsp_url(camera_ip, username, password, "main")

    # Initialize components
    logger.info("Initializing components...")

    stream_capture = RTSPStreamCapture(
        rtsp_url=rtsp_url,
        frame_queue=frame_queue,
        target_width=1280,
        target_height=720
    )

    inference_engine = InferenceEngine(
        model_path="yolov8n.pt",
        device="cuda:0",
        conf_threshold=0.5,
        input_queue=frame_queue,
        output_queue=inference_queue
    )

    detection_processor = DetectionProcessor(
        input_queue=inference_queue,
        output_queue=detection_queue
    )

    # Start components
    logger.info("Starting components...")

    if not stream_capture.start():
        logger.error("Failed to start stream capture")
        return False

    if not inference_engine.start():
        logger.error("Failed to start inference engine")
        stream_capture.stop()
        return False

    if not detection_processor.start():
        logger.error("Failed to start detection processor")
        stream_capture.stop()
        inference_engine.stop()
        return False

    logger.info(f"All components started. Collecting data for {duration} seconds...")
    logger.info("(This includes warm-up time)")

    # Collect latency measurements
    latencies = []
    inference_times = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            if not detection_queue.empty():
                result = detection_queue.get()

                latency = result.get('total_latency_ms', 0)
                inference_time = result.get('inference_time_ms', 0)

                latencies.append(latency)
                inference_times.append(inference_time)

                if len(latencies) % 10 == 0:
                    logger.info(f"  Samples collected: {len(latencies)}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")

    finally:
        # Stop components
        logger.info("Stopping components...")
        detection_processor.stop()
        inference_engine.stop()
        stream_capture.stop()

    # Analyze results
    if not latencies:
        logger.error("No latency measurements collected")
        return False

    logger.info("\n" + "=" * 80)
    logger.info("Latency Analysis")
    logger.info("=" * 80)

    logger.info(f"\nTotal samples: {len(latencies)}")

    # Latency statistics
    logger.info("\nEnd-to-End Latency (camera to detection):")
    logger.info(f"  - Average: {np.mean(latencies):.1f}ms")
    logger.info(f"  - Std dev: {np.std(latencies):.1f}ms")
    logger.info(f"  - Min: {np.min(latencies):.1f}ms")
    logger.info(f"  - Max: {np.max(latencies):.1f}ms")
    logger.info(f"  - P50: {np.percentile(latencies, 50):.1f}ms")
    logger.info(f"  - P95: {np.percentile(latencies, 95):.1f}ms")
    logger.info(f"  - P99: {np.percentile(latencies, 99):.1f}ms")

    # Inference time statistics
    logger.info("\nInference Time:")
    logger.info(f"  - Average: {np.mean(inference_times):.1f}ms")
    logger.info(f"  - Min: {np.min(inference_times):.1f}ms")
    logger.info(f"  - Max: {np.max(inference_times):.1f}ms")

    # Breakdown
    avg_latency = np.mean(latencies)
    avg_inference = np.mean(inference_times)
    other_latency = avg_latency - avg_inference

    logger.info("\nLatency Breakdown:")
    logger.info(f"  - Inference: {avg_inference:.1f}ms ({avg_inference/avg_latency*100:.1f}%)")
    logger.info(f"  - Other (capture + network + processing): {other_latency:.1f}ms ({other_latency/avg_latency*100:.1f}%)")

    # Performance assessment
    logger.info("\n" + "=" * 80)
    logger.info("Performance Assessment:")
    logger.info("=" * 80)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    if avg_latency < 100:
        logger.info("✓ EXCELLENT: Average latency < 100ms (target achieved)")
    elif avg_latency < 200:
        logger.info("✓ GOOD: Average latency < 200ms (acceptable for Phase 1)")
    else:
        logger.warning("⚠ NEEDS OPTIMIZATION: Average latency > 200ms")

    if p95_latency < 150:
        logger.info("✓ P95 latency is good (< 150ms)")
    elif p95_latency < 250:
        logger.info("⚠ P95 latency acceptable but could be improved")
    else:
        logger.warning("✗ P95 latency needs optimization")

    # Recommendations
    logger.info("\nRecommendations:")
    if avg_inference > 30:
        logger.info("  - Consider using a smaller YOLOv8 model (n or s)")
        logger.info("  - Or reduce input resolution")

    if other_latency > 100:
        logger.info("  - Network/capture latency is high")
        logger.info("  - Check network connection to camera")
        logger.info("  - Consider using GStreamer pipeline")

    logger.info("\n" + "=" * 80)
    return True


def test_component_latencies():
    """
    Test individual component latencies.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Component-Level Latency Test")
    logger.info("=" * 80)

    # Test inference only
    logger.info("\nTesting inference latency...")
    inference_engine = InferenceEngine(
        model_path="yolov8n.pt",
        device="cuda:0"
    )

    if not inference_engine.load_model():
        logger.error("Failed to load model")
        return False

    # Create test frame
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        _ = inference_engine.infer_single(test_frame)

    # Measure
    times = []
    for _ in range(50):
        start = time.time()
        _ = inference_engine.infer_single(test_frame)
        times.append((time.time() - start) * 1000)

    logger.info(f"Inference latency (1280x720):")
    logger.info(f"  - Average: {np.mean(times):.1f}ms")
    logger.info(f"  - P95: {np.percentile(times, 95):.1f}ms")

    return True


if __name__ == "__main__":
    # Camera configuration - UPDATE WITH YOUR CAMERA INFO
    CAMERA_IP = "192.168.1.100"  # Replace with your camera IP
    USERNAME = "admin"
    PASSWORD = "your_password_here"  # Update with your camera password

    # Test component latencies first
    test_component_latencies()

    print()

    # Test end-to-end latency
    logger.info("Starting end-to-end latency test...")
    logger.info("This will take ~30 seconds")
    print()

    test_end_to_end_latency(
        camera_ip=CAMERA_IP,
        username=USERNAME,
        password=PASSWORD,
        duration=30
    )
