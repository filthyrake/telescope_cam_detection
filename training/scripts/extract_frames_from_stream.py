#!/usr/bin/env python3
"""
Extract training frames from live RTSP camera stream.
Perfect for capturing dynamic telescope movements over time.
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


def extract_frames_from_stream(
    camera_ip: str,
    username: str,
    password: str,
    output_dir: str,
    duration: int = 300,  # 5 minutes default
    interval: float = 1.0,  # 1 frame per second
    description: str = "dynamic"
):
    """
    Extract training frames from live camera stream.

    Args:
        camera_ip: Camera IP address
        username: Camera username
        password: Camera password
        output_dir: Directory to save images
        duration: Total recording duration in seconds
        interval: Seconds between frame captures
        description: Description for filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"📹 Video Frame Extraction")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
    print(f"Interval: {interval} seconds")
    print(f"Expected frames: ~{int(duration / interval)}")
    print(f"Description: {description}")
    print()
    print("INSTRUCTIONS:")
    print("  1. Press Enter to start recording")
    print("  2. Slew your telescopes naturally through various positions")
    print("  3. Try different altitudes, azimuths, and orientations")
    print("  4. Move both scopes independently and together")
    print("  5. Press Ctrl+C to stop early")
    print()
    print("=" * 80)
    print()

    input("Press Enter to start recording...")
    print()

    # Connect to camera
    rtsp_url = create_rtsp_url(camera_ip, username, password, "main")
    frame_queue = Queue(maxsize=2)

    capture = RTSPStreamCapture(rtsp_url, frame_queue)

    if not capture.start():
        print("❌ Failed to connect to camera")
        return False

    print("✅ Connected to camera")
    print()
    print("🎬 RECORDING STARTED - Slew your telescopes now!")
    print()

    captured = 0
    last_capture = time.time()
    start_time = time.time()
    end_time = start_time + duration

    try:
        while time.time() < end_time:
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
                    remaining = end_time - time.time()
                    progress = (elapsed / duration) * 100

                    print(f"[{captured:4d} frames] {filename.name} "
                          f"({progress:.0f}% - {remaining:.0f}s remaining)")

            time.sleep(0.01)

        print()
        print("⏱️  Time limit reached!")

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("⚠️  Recording stopped by user")

    finally:
        capture.stop()

        elapsed = time.time() - start_time
        fps = captured / elapsed if elapsed > 0 else 0

        print()
        print("=" * 80)
        print(f"✅ Extraction complete!")
        print(f"   Saved: {captured} frames")
        print(f"   Location: {output_dir}")
        print(f"   Duration: {elapsed:.0f}s")
        print(f"   Average: {fps:.2f} frames/sec")
        print("=" * 80)
        print()
        print("📝 Next steps:")
        print("   1. Review extracted frames")
        print("   2. Delete any blurry or redundant images")
        print("   3. Continue with collision scenario captures")
        print("   4. Annotate all images with LabelImg")
        print()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract training frames from live camera stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5 minute recording, 1 frame/sec (~300 frames)
  python extract_frames_from_stream.py --duration 300 --interval 1.0

  # 10 minute recording, 1 frame every 2 seconds (~300 frames)
  python extract_frames_from_stream.py --duration 600 --interval 2.0

  # Quick 2 minute test
  python extract_frames_from_stream.py --duration 120 --interval 0.5 --desc test

TIPS:
  - Interval 0.5-1.0 sec works well (not too redundant, good coverage)
  - 5-10 minutes gives 300-1200 diverse frames
  - Move scopes slowly and deliberately for best results
  - Avoid very fast slewing (causes motion blur)
        """
    )
    parser.add_argument("--ip", default="10.0.8.18", help="Camera IP address")
    parser.add_argument("--user", default="admin", help="Camera username")
    parser.add_argument("--password", default="5326jbbD", help="Camera password")
    parser.add_argument("--output",
                       default="training/datasets/telescope_equipment/images/raw",
                       help="Output directory")
    parser.add_argument("--duration", type=int, default=300,
                       help="Recording duration in seconds (default: 300 = 5 minutes)")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Seconds between frame captures (default: 1.0)")
    parser.add_argument("--desc", default="dynamic",
                       help="Description for filename prefix")

    args = parser.parse_args()

    extract_frames_from_stream(
        camera_ip=args.ip,
        username=args.user,
        password=args.password,
        output_dir=args.output,
        duration=args.duration,
        interval=args.interval,
        description=args.desc
    )
