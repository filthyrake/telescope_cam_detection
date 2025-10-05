#!/usr/bin/env python3
"""
Guided collision scenario capture - walks you through each critical scenario.
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


# Collision scenarios to capture
SCENARIOS = [
    {
        "id": 1,
        "name": "Tripod Legs - Adjacent Touching",
        "description": "Position tripod legs of both scopes 1-2 inches apart or touching",
        "images": 40,
        "priority": "⭐⭐⭐ CRITICAL",
        "filename": "collision_tripod_adjacent"
    },
    {
        "id": 2,
        "name": "Tripod Legs - Crossing Paths",
        "description": "Position scopes so tripod legs would cross if moved",
        "images": 40,
        "priority": "⭐⭐⭐ CRITICAL",
        "filename": "collision_tripod_crossing"
    },
    {
        "id": 3,
        "name": "Tripod Legs - Near Miss",
        "description": "Position tripod legs 6-12 inches apart (warning zone)",
        "images": 40,
        "priority": "⭐⭐⭐ CRITICAL",
        "filename": "collision_tripod_nearmiss"
    },
    {
        "id": 4,
        "name": "Optical Tubes - Pointed at Each Other",
        "description": "Point both telescope tubes directly at each other (6-24 inches apart)",
        "images": 50,
        "priority": "⭐⭐⭐ CRITICAL",
        "filename": "collision_tubes_pointed"
    },
    {
        "id": 5,
        "name": "Optical Tubes - Slew Path Collision",
        "description": "Position Telescope 2 in the path where Telescope 1 would slew through",
        "images": 40,
        "priority": "⭐⭐ HIGH",
        "filename": "collision_tubes_slew_path"
    },
    {
        "id": 6,
        "name": "Counterweight - Near Tripod Leg",
        "description": "Slew scope so counterweight bar comes within 6 inches of other tripod leg",
        "images": 50,
        "priority": "⭐⭐ HIGH",
        "filename": "collision_counterweight_tripod"
    },
    {
        "id": 7,
        "name": "Counterweights - Crossing Paths",
        "description": "Position both scopes so counterweights would collide mid-slew",
        "images": 35,
        "priority": "⭐⭐ HIGH",
        "filename": "collision_counterweight_crossing"
    },
    {
        "id": 8,
        "name": "Mount Heads - Too Close",
        "description": "Position both scopes at high altitude (70-85°), mount heads 6-12 inches apart",
        "images": 35,
        "priority": "⭐⭐ HIGH",
        "filename": "collision_mount_heads"
    },
    {
        "id": 9,
        "name": "Finder Scope - Near Other Tube",
        "description": "Position finder scope pointing toward other scope's tube (4-8 inches away)",
        "images": 25,
        "priority": "⭐ MEDIUM",
        "filename": "collision_finder_scope"
    },
    {
        "id": 10,
        "name": "Both at Zenith",
        "description": "Position both scopes near zenith (85-90°), high collision risk",
        "images": 45,
        "priority": "⭐⭐ HIGH",
        "filename": "collision_both_zenith"
    },
    {
        "id": 11,
        "name": "Opposite Directions with Overlap",
        "description": "Scope 1 at 30° alt/0° az, Scope 2 at 30° alt/180° az, overlap in middle",
        "images": 35,
        "priority": "⭐⭐ HIGH",
        "filename": "collision_opposite_overlap"
    },
    {
        "id": 12,
        "name": "Covered Telescopes in Collision",
        "description": "Cover both telescopes, set up tripod leg collision scenario",
        "images": 35,
        "priority": "⭐⭐ HIGH",
        "filename": "collision_covered"
    }
]


def capture_scenario(scenario, camera_ip, username, password, output_dir, interval=2):
    """Capture images for a single collision scenario."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"SCENARIO {scenario['id']}: {scenario['name']}")
    print("=" * 80)
    print(f"Priority: {scenario['priority']}")
    print(f"Images to capture: {scenario['images']}")
    print()
    print("SETUP:")
    print(f"  {scenario['description']}")
    print()
    print("=" * 80)
    print()

    choice = input("Ready to capture this scenario? (y/n/s=skip): ").strip().lower()

    if choice == 's':
        print("⏭️  Skipped")
        return 0

    if choice != 'y':
        print("⏭️  Skipped")
        return 0

    print()
    print(f"Starting capture in 3 seconds...")
    time.sleep(3)

    # Connect to camera
    rtsp_url = create_rtsp_url(camera_ip, username, password, "main")
    frame_queue = Queue(maxsize=2)

    capture = RTSPStreamCapture(rtsp_url, frame_queue)

    if not capture.start():
        print("❌ Failed to connect to camera")
        return 0

    print("✅ Connected to camera")
    print()

    captured = 0
    last_capture = time.time()
    start_time = time.time()

    try:
        while captured < scenario['images']:
            if not frame_queue.empty():
                frame_data = frame_queue.get()
                frame = frame_data['frame']

                # Auto-capture at intervals
                if time.time() - last_capture >= interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = output_path / f"telescope_{scenario['filename']}_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    captured += 1
                    last_capture = time.time()

                    remaining = scenario['images'] - captured
                    print(f"[{captured:3d}/{scenario['images']}] {filename.name} ({remaining} remaining)")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print()
        print("⚠️  Capture interrupted")

    finally:
        capture.stop()
        elapsed = time.time() - start_time

        print()
        print(f"✅ Captured {captured} images in {elapsed:.0f}s")

    return captured


def main():
    parser = argparse.ArgumentParser(
        description="Guided collision scenario capture",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ip", default="10.0.8.18", help="Camera IP")
    parser.add_argument("--user", default="admin", help="Camera username")
    parser.add_argument("--password", default="5326jbbD", help="Camera password")
    parser.add_argument("--output",
                       default="training/datasets/telescope_equipment/images/raw",
                       help="Output directory")
    parser.add_argument("--interval", type=int, default=2,
                       help="Seconds between captures")
    parser.add_argument("--start-from", type=int, default=1,
                       help="Start from scenario number")

    args = parser.parse_args()

    print("=" * 80)
    print("🎯 GUIDED COLLISION SCENARIO CAPTURE")
    print("=" * 80)
    print()
    print("This script will guide you through capturing critical collision scenarios.")
    print()
    print(f"Total scenarios: {len(SCENARIOS)}")
    print(f"Total images: ~{sum(s['images'] for s in SCENARIOS)}")
    print(f"Estimated time: 2-3 hours")
    print()
    print("You can:")
    print("  - Skip scenarios by typing 's'")
    print("  - Stop anytime with Ctrl+C")
    print("  - Resume later with --start-from N")
    print()
    print("=" * 80)
    print()

    input("Press Enter to begin...")

    total_captured = 0
    scenarios_completed = 0

    for scenario in SCENARIOS:
        if scenario['id'] < args.start_from:
            continue

        captured = capture_scenario(
            scenario,
            args.ip,
            args.user,
            args.password,
            args.output,
            args.interval
        )

        if captured > 0:
            total_captured += captured
            scenarios_completed += 1

        # Offer to continue
        if scenario['id'] < len(SCENARIOS):
            print()
            cont = input("Continue to next scenario? (y/n): ").strip().lower()
            if cont != 'y':
                break

    # Summary
    print()
    print("=" * 80)
    print("📊 CAPTURE SUMMARY")
    print("=" * 80)
    print(f"Scenarios completed: {scenarios_completed}/{len(SCENARIOS)}")
    print(f"Total images captured: {total_captured}")
    print(f"Output directory: {args.output}")
    print()
    print("📝 Next steps:")
    print("  1. Review captured images")
    print("  2. Delete any blurry or redundant frames")
    print("  3. Annotate with LabelImg")
    print("  4. Train the model!")
    print()


if __name__ == "__main__":
    main()
