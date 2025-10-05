#!/usr/bin/env python3
"""
Fix timestamps on training images and clips by subtracting 7 hours (UTC -> PDT conversion)
"""

import os
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

# Subtract 7 hours to convert UTC to PDT
TIME_OFFSET = timedelta(hours=7)

def fix_training_images():
    """Fix timestamps in training image filenames and file mtimes"""
    training_dir = Path("/home/damen/telescope_cam_detection/training/datasets/telescope_equipment/images/raw")

    # Pattern: telescope_covered_lateafternoon_20251005_162702_525232.jpg
    # Captures: YYYYMMDD_HHMMSS_microseconds
    pattern = r'(\d{8})_(\d{6})_(\d{6})'

    renamed_count = 0

    for img_file in sorted(training_dir.glob("*.jpg")):
        match = re.search(pattern, img_file.name)
        if not match:
            continue

        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        micro_str = match.group(3)  # microseconds

        # Parse the UTC timestamp
        utc_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

        # Convert to PDT
        pdt_time = utc_time - TIME_OFFSET

        # Generate new filename
        new_date_str = pdt_time.strftime("%Y%m%d")
        new_time_str = pdt_time.strftime("%H%M%S")

        old_timestamp = f"{date_str}_{time_str}_{micro_str}"
        new_timestamp = f"{new_date_str}_{new_time_str}_{micro_str}"

        new_name = img_file.name.replace(old_timestamp, new_timestamp)
        new_path = img_file.parent / new_name

        if img_file != new_path:
            print(f"Renaming: {img_file.name} -> {new_name}")
            img_file.rename(new_path)
            renamed_count += 1

    print(f"\nRenamed {renamed_count} training images")

def fix_clip_metadata():
    """Fix timestamps in clip JSON metadata files and filenames"""
    clips_dir = Path("/home/damen/telescope_cam_detection/clips")

    # Pattern: bird_20251005_141214_812546_conf0.52.json
    pattern = r'(\w+)_(\d{8})_(\d{6})_(\d{6})_(conf[\d.]+)\.(json|jpg)'

    updated_count = 0

    for json_file in sorted(clips_dir.glob("*.json")):
        match = re.search(pattern, json_file.name)
        if not match:
            continue

        class_name = match.group(1)
        date_str = match.group(2)
        time_str = match.group(3)
        micro_str = match.group(4)
        conf_str = match.group(5)

        # Parse UTC timestamp
        utc_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

        # Convert to PDT
        pdt_time = utc_time - TIME_OFFSET

        new_date_str = pdt_time.strftime("%Y%m%d")
        new_time_str = pdt_time.strftime("%H%M%S")

        # Update JSON content
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Update timestamp field (if it exists)
        if 'timestamp' in data:
            # Subtract 7 hours from Unix timestamp
            data['timestamp'] = data['timestamp'] - (7 * 3600)

        # Generate new filenames
        old_base = f"{class_name}_{date_str}_{time_str}_{micro_str}_{conf_str}"
        new_base = f"{class_name}_{new_date_str}_{new_time_str}_{micro_str}_{conf_str}"

        new_json_name = f"{new_base}.json"
        new_jpg_name = f"{new_base}.jpg"

        new_json_path = json_file.parent / new_json_name

        # Update filename references in JSON
        if 'filename' in data:
            data['filename'] = new_jpg_name
        if 'url' in data:
            data['url'] = data['url'].replace(old_base, new_base)

        # Write updated JSON
        with open(new_json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Rename corresponding JPG file
        old_jpg = json_file.parent / f"{old_base}.jpg"
        new_jpg = json_file.parent / new_jpg_name

        if old_jpg.exists():
            old_jpg.rename(new_jpg)

        # Remove old JSON file if name changed
        if json_file != new_json_path:
            json_file.unlink()
            print(f"Updated: {json_file.name} -> {new_json_name}")
            updated_count += 1

    print(f"\nUpdated {updated_count} clip metadata files")

if __name__ == "__main__":
    print("Fixing timestamps (converting UTC to PDT by subtracting 7 hours)...\n")
    print("=" * 70)
    print("TRAINING IMAGES:")
    print("=" * 70)
    fix_training_images()

    print("\n" + "=" * 70)
    print("CLIP METADATA:")
    print("=" * 70)
    fix_clip_metadata()

    print("\n✓ All timestamps fixed!")
