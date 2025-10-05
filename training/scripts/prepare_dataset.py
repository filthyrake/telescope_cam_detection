#!/usr/bin/env python3
"""
Prepare dataset for training by splitting images into train/val sets.
"""

import shutil
import random
from pathlib import Path
import argparse


def prepare_dataset(
    raw_images_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    seed: int = 42
):
    """
    Split images into training and validation sets.

    Args:
        raw_images_dir: Directory containing raw images
        output_dir: Output dataset directory
        train_split: Fraction of images for training (rest for validation)
        seed: Random seed for reproducibility
    """
    raw_path = Path(raw_images_dir)
    output_path = Path(output_dir)

    train_images = output_path / "images" / "train"
    val_images = output_path / "images" / "val"
    train_labels = output_path / "labels" / "train"
    val_labels = output_path / "labels" / "val"

    # Create directories
    for dir_path in [train_images, val_images, train_labels, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = list(raw_path.glob("*.jpg")) + list(raw_path.glob("*.png"))

    if not image_files:
        print(f"❌ No images found in {raw_images_dir}")
        return False

    print(f"📊 Found {len(image_files)} images")

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(image_files)

    # Split
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"📦 Train set: {len(train_files)} images")
    print(f"📦 Val set: {len(val_files)} images")

    # Copy train files
    for img_file in train_files:
        shutil.copy(img_file, train_images / img_file.name)

    # Copy val files
    for img_file in val_files:
        shutil.copy(img_file, val_images / img_file.name)

    print(f"\n✅ Dataset prepared at {output_dir}")
    print(f"\n📝 Next steps:")
    print(f"   1. Annotate images using LabelImg or Roboflow:")
    print(f"      - Train images: {train_images}")
    print(f"      - Val images: {val_images}")
    print(f"   2. Export labels in YOLO format to:")
    print(f"      - Train labels: {train_labels}")
    print(f"      - Val labels: {val_labels}")
    print(f"   3. Update classes.yaml with your class names")
    print(f"   4. Run training: python training/scripts/train_custom_model.py")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--raw", default="training/datasets/telescope_equipment/images/raw",
                       help="Raw images directory")
    parser.add_argument("--output", default="training/datasets/telescope_equipment",
                       help="Output dataset directory")
    parser.add_argument("--split", type=float, default=0.8,
                       help="Training split fraction (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    prepare_dataset(
        raw_images_dir=args.raw,
        output_dir=args.output,
        train_split=args.split,
        seed=args.seed
    )
