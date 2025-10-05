#!/usr/bin/env python3
"""
Snapshot Viewer
Browse and manage saved snapshots/clips from detection events.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def list_snapshots(clips_dir: Path = Path("clips")) -> List[Dict[str, Any]]:
    """
    List all saved snapshots with their metadata.

    Args:
        clips_dir: Directory containing saved clips

    Returns:
        List of snapshot info dictionaries
    """
    if not clips_dir.exists():
        print(f"Clips directory not found: {clips_dir}")
        return []

    snapshots = []

    # Find all image and video files
    for file_path in sorted(clips_dir.glob("*"), reverse=True):
        if file_path.suffix in ['.jpg', '.jpeg', '.png', '.mp4', '.avi']:
            # Try to load metadata
            metadata_file = file_path.with_suffix('.json')
            metadata = None

            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load metadata for {file_path.name}: {e}")

            # Get file info
            stat = file_path.stat()

            snapshot_info = {
                'filename': file_path.name,
                'path': str(file_path),
                'size_kb': stat.st_size / 1024,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'metadata': metadata
            }

            snapshots.append(snapshot_info)

    return snapshots


def print_snapshot_summary(snapshots: List[Dict[str, Any]]):
    """
    Print a summary of all snapshots.

    Args:
        snapshots: List of snapshot info dictionaries
    """
    if not snapshots:
        print("No snapshots found.")
        return

    print("=" * 80)
    print(f"Found {len(snapshots)} snapshots")
    print("=" * 80)
    print()

    for i, snapshot in enumerate(snapshots, 1):
        print(f"{i}. {snapshot['filename']}")
        print(f"   Path: {snapshot['path']}")
        print(f"   Size: {snapshot['size_kb']:.1f} KB")
        print(f"   Created: {snapshot['created'].strftime('%Y-%m-%d %H:%M:%S')}")

        if snapshot['metadata']:
            metadata = snapshot['metadata']
            detections = metadata.get('detections', [])

            if detections:
                print(f"   Detections:")
                for det in detections:
                    print(f"      - {det['class_name']}: {det['confidence']:.2f}")

            detection_counts = metadata.get('detection_counts', {})
            if detection_counts:
                counts_str = ", ".join([f"{k}: {v}" for k, v in detection_counts.items()])
                print(f"   Counts: {counts_str}")

            latency = metadata.get('latency_ms', 0)
            if latency:
                print(f"   Latency: {latency:.1f}ms")

        print()


def print_statistics(snapshots: List[Dict[str, Any]]):
    """
    Print statistics about saved snapshots.

    Args:
        snapshots: List of snapshot info dictionaries
    """
    if not snapshots:
        return

    print("=" * 80)
    print("Statistics")
    print("=" * 80)

    # Count by class
    class_counts = {}
    total_size = 0

    for snapshot in snapshots:
        total_size += snapshot['size_kb']

        if snapshot['metadata']:
            detections = snapshot['metadata'].get('detections', [])
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"Total snapshots: {len(snapshots)}")
    print(f"Total size: {total_size / 1024:.1f} MB")
    print()

    if class_counts:
        print("Detections by class:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")

    print()


def delete_old_snapshots(clips_dir: Path, days: int):
    """
    Delete snapshots older than specified days.

    Args:
        clips_dir: Directory containing clips
        days: Delete files older than this many days
    """
    import time

    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted = 0

    for file_path in clips_dir.glob("*"):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            try:
                file_path.unlink()
                deleted += 1
                print(f"Deleted: {file_path.name}")
            except Exception as e:
                print(f"Failed to delete {file_path.name}: {e}")

    print(f"\nDeleted {deleted} old files")


def interactive_menu():
    """Interactive menu for managing snapshots."""
    clips_dir = Path("clips")

    while True:
        print("\n" + "=" * 80)
        print("Snapshot Viewer - Menu")
        print("=" * 80)
        print("1. List all snapshots")
        print("2. Show statistics")
        print("3. Delete snapshots older than X days")
        print("4. Open clips directory")
        print("5. Exit")
        print()

        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            snapshots = list_snapshots(clips_dir)
            print_snapshot_summary(snapshots)

        elif choice == "2":
            snapshots = list_snapshots(clips_dir)
            print_statistics(snapshots)

        elif choice == "3":
            days = input("Delete files older than how many days? ").strip()
            try:
                days = int(days)
                confirm = input(f"Delete all files older than {days} days? (yes/no): ").strip().lower()
                if confirm == "yes":
                    delete_old_snapshots(clips_dir, days)
            except ValueError:
                print("Invalid number")

        elif choice == "4":
            import subprocess
            import platform

            if platform.system() == "Linux":
                subprocess.run(["xdg-open", str(clips_dir)])
            elif platform.system() == "Darwin":
                subprocess.run(["open", str(clips_dir)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", str(clips_dir)])
            else:
                print(f"Please open: {clips_dir.absolute()}")

        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid option")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="View and manage detection snapshots")
    parser.add_argument('--dir', type=str, default='clips', help='Clips directory')
    parser.add_argument('--list', action='store_true', help='List all snapshots')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--delete-older-than', type=int, metavar='DAYS',
                       help='Delete snapshots older than N days')
    parser.add_argument('--interactive', action='store_true', help='Interactive menu mode')

    args = parser.parse_args()

    clips_dir = Path(args.dir)

    if args.interactive:
        interactive_menu()
    elif args.list:
        snapshots = list_snapshots(clips_dir)
        print_snapshot_summary(snapshots)
    elif args.stats:
        snapshots = list_snapshots(clips_dir)
        print_statistics(snapshots)
    elif args.delete_older_than is not None:
        delete_old_snapshots(clips_dir, args.delete_older_than)
    else:
        # Default: show summary and stats
        snapshots = list_snapshots(clips_dir)
        print_snapshot_summary(snapshots)
        print_statistics(snapshots)


if __name__ == "__main__":
    main()
