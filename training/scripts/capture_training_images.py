#!/usr/bin/env python3
"""
Capture training images from the live camera stream.
This script helps collect diverse training data for custom telescope detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import time
import argparse
from datetime import datetime
from stream_capture import create_rtsp_url, RTSPStreamCapture
from queue import Queue


def capture_training_images(
    camera_ip: str,
    username: str,
    password: str,
    output_dir: str,
    num_images: int = 100,
    interval: int = 2
):
    """
    Capture training images from camera.

    Args:
        camera_ip: Camera IP address
        username: Camera username
        password: Camera password
        output_dir: Directory to save images
        num_images: Number of images to capture
        interval: Seconds between captures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📸 Capturing {num_images} training images to {output_dir}")
    print(f"⏱️  Capturing every {interval} seconds")
    print("\nInstructions:")
    print("  - Move around the telescopes")
    print("  - Change angles and distances")
    print("  - Include covered and uncovered states")
    print("  - Press 'q' to quit early")
    print("  - Press 's' to save current frame manually")
    print("\nStarting in 3 seconds...\n")

    time.sleep(3)

    # Connect to camera
    rtsp_url = create_rtsp_url(camera_ip, username, password, "main")
    frame_queue = Queue(maxsize=2)

    capture = RTSPStreamCapture(rtsp_url, frame_queue)

    if not capture.start():
        print("❌ Failed to connect to camera")
        return False

    print("✅ Connected to camera\n")

    captured = 0
    last_auto_capture = time.time()

    try:
        while captured < num_images:
            if not frame_queue.empty():
                frame_data = frame_queue.get()
                frame = frame_data['frame']

                # Display frame
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f"Captured: {captured}/{num_images}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    display_frame,
                    "Press 's' to save, 'q' to quit",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow('Training Image Capture', display_frame)

                # Auto-capture at intervals
                if time.time() - last_auto_capture >= interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = output_path / f"telescope_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    captured += 1
                    last_auto_capture = time.time()
                    print(f"📸 Captured {captured}/{num_images}: {filename.name}")

                # Manual capture with 's' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = output_path / f"telescope_manual_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    captured += 1
                    print(f"📸 Manual capture {captured}/{num_images}: {filename.name}")
                elif key == ord('q'):
                    print("\n⚠️  Capture stopped by user")
                    break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⚠️  Capture interrupted by user")

    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print(f"\n✅ Capture complete! Saved {captured} images to {output_dir}")
        print(f"\n📝 Next step: Annotate images with:")
        print(f"   python training/scripts/prepare_dataset.py")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture training images from camera")
    parser.add_argument("--ip", default="192.168.1.100", help="Camera IP address")
    parser.add_argument("--user", default="admin", help="Camera username")
    parser.add_argument("--password", required=True, help="Camera password")
    parser.add_argument("--output", default="training/datasets/telescope_equipment/images/raw",
                       help="Output directory")
    parser.add_argument("--num", type=int, default=100, help="Number of images to capture")
    parser.add_argument("--interval", type=int, default=2, help="Seconds between captures")

    args = parser.parse_args()

    capture_training_images(
        camera_ip=args.ip,
        username=args.user,
        password=args.password,
        output_dir=args.output,
        num_images=args.num,
        interval=args.interval
    )
