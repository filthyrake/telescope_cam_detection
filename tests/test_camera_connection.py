#!/usr/bin/env python3
"""
Test script to verify RTSP camera connection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import time
import cv2
from stream_capture import create_rtsp_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_camera_connection(camera_ip: str, username: str, password: str, stream_type: str = "main"):
    """
    Test camera connection and display basic info.

    Args:
        camera_ip: IP address of the camera
        username: Camera username
        password: Camera password
        stream_type: 'main' or 'sub'
    """
    logger.info("=" * 80)
    logger.info("Camera Connection Test")
    logger.info("=" * 80)

    # Create RTSP URL
    rtsp_url = create_rtsp_url(camera_ip, username, password, stream_type)
    logger.info(f"RTSP URL: rtsp://{username}:***@{camera_ip}:554/...")

    # Try to connect
    logger.info("Attempting to connect...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logger.error("Failed to open RTSP stream")
        logger.error("Possible issues:")
        logger.error("  - Incorrect IP address")
        logger.error("  - Wrong credentials")
        logger.error("  - Camera not reachable on network")
        logger.error("  - RTSP port 554 blocked")
        return False

    logger.info("Successfully connected to camera!")

    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Stream properties:")
    logger.info(f"  - Resolution: {width}x{height}")
    logger.info(f"  - FPS: {fps}")

    # Try to read a few frames
    logger.info("\nTesting frame capture...")
    success_count = 0
    fail_count = 0
    frame_times = []

    for i in range(30):
        start = time.time()
        ret, frame = cap.read()
        elapsed = time.time() - start

        if ret:
            success_count += 1
            frame_times.append(elapsed)
        else:
            fail_count += 1

        time.sleep(0.033)  # ~30 FPS

    cap.release()

    # Report results
    logger.info("\nTest Results:")
    logger.info(f"  - Successful reads: {success_count}/30")
    logger.info(f"  - Failed reads: {fail_count}/30")

    if frame_times:
        avg_time = sum(frame_times) / len(frame_times)
        logger.info(f"  - Avg frame read time: {avg_time*1000:.1f}ms")

    logger.info("=" * 80)

    if success_count >= 25:
        logger.info("✓ Camera connection test PASSED")
        return True
    else:
        logger.error("✗ Camera connection test FAILED")
        return False


if __name__ == "__main__":
    # Camera configuration - UPDATE WITH YOUR CAMERA INFO
    CAMERA_IP = "192.168.1.100"  # Replace with your camera IP
    USERNAME = "admin"
    PASSWORD = "your_password_here"  # Update with your camera password

    # Test main stream
    logger.info("Testing MAIN stream (high quality)...")
    main_ok = test_camera_connection(CAMERA_IP, USERNAME, PASSWORD, "main")

    print()

    # Test sub stream
    logger.info("Testing SUB stream (lower quality)...")
    sub_ok = test_camera_connection(CAMERA_IP, USERNAME, PASSWORD, "sub")

    print()

    if main_ok:
        logger.info("✓ Main stream is working correctly")
    else:
        logger.error("✗ Main stream has issues")

    if sub_ok:
        logger.info("✓ Sub stream is working correctly")
    else:
        logger.error("✗ Sub stream has issues")
