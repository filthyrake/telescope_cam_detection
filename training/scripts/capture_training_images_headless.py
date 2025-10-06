#!/usr/bin/env python3
"""
Capture training images from the live camera stream (headless mode).
Perfect for capturing samples via SSH without X11 display.
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


def capture_training_images_headless(
    camera_ip: str,
    username: str,
    password: str,
    output_dir: str,
    num_images: int = 100,
    interval: int = 2,
    description: str = "covered"
):
    """
    Capture training images from camera (headless mode).

    Args:
        camera_ip: Camera IP address
        username: Camera username
        password: Camera password
        output_dir: Directory to save images
        num_images: Number of images to capture
        interval: Seconds between captures
        description: Description for filename (e.g., "covered", "uncovered", "dark")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"📸 Training Image Capture (Headless Mode)")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Images: {num_images}")
    print(f"Interval: {interval} seconds")
    print(f"Description: {description}")
    print()
    print("This will capture images automatically at the specified interval.")
    print("Press Ctrl+C to stop early.")
    print()
    print("Starting in 3 seconds...")
    print("=" * 80)
    print()

    time.sleep(3)

    # Connect to camera
    rtsp_url = create_rtsp_url(camera_ip, username, password, "main")
    frame_queue = Queue(maxsize=2)

    capture = RTSPStreamCapture(rtsp_url, frame_queue)

    if not capture.start():
        print("❌ Failed to connect to camera")
        return False

    print("✅ Connected to camera")
    print()

    captured = 0
    last_capture = time.time()
    start_time = time.time()

    try:
        while captured < num_images:
            if not frame_queue.empty():
                frame_data = frame_queue.get()
                frame = frame_data['frame']

                # Auto-capture at intervals
                if time.time() - last_capture >= interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = output_path / f"telescope_{description}_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    captured += 1
                    last_capture = time.time()

                    elapsed = time.time() - start_time
                    remaining = (num_images - captured) * interval

                    print(f"[{captured:3d}/{num_images}] {filename.name} "
                          f"(Elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s)")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("⚠️  Capture stopped by user")

    finally:
        capture.stop()

        elapsed = time.time() - start_time
        print()
        print("=" * 80)
        print(f"✅ Capture complete!")
        print(f"   Saved: {captured} images")
        print(f"   Location: {output_dir}")
        print(f"   Time: {elapsed:.0f}s")
        print("=" * 80)
        print()
        print("📝 Next steps:")
        print("   1. Review images in the output directory")
        print("   2. Annotate with LabelImg or Roboflow")
        print("   3. Run: python training/scripts/train_custom_model.py")
        print()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture training images from camera (headless mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture 50 covered telescope images
  python capture_training_images_headless.py --num 50 --desc covered

  # Capture 100 images every 5 seconds
  python capture_training_images_headless.py --num 100 --interval 5 --desc dark

  # Custom output directory
  python capture_training_images_headless.py --output /data/telescope_images --desc test
        """
    )
    parser.add_argument("--ip", default="192.168.1.100", help="Camera IP address")
    parser.add_argument("--user", default="admin", help="Camera username")
    parser.add_argument("--password", required=True, help="Camera password")
    parser.add_argument("--output",
                       default="training/datasets/telescope_equipment/images/raw",
                       help="Output directory")
    parser.add_argument("--num", type=int, default=100,
                       help="Number of images to capture")
    parser.add_argument("--interval", type=int, default=2,
                       help="Seconds between captures")
    parser.add_argument("--desc", default="covered",
                       help="Description for filename (e.g., covered, uncovered, dark)")

    args = parser.parse_args()

    capture_training_images_headless(
        camera_ip=args.ip,
        username=args.user,
        password=args.password,
        output_dir=args.output,
        num_images=args.num,
        interval=args.interval,
        description=args.desc
    )
