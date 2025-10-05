#!/usr/bin/env python3
"""
Train custom YOLOv8 model for telescope equipment detection.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def train_telescope_model(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cuda:0",
    project: str = "training/runs",
    name: str = "telescope_detection"
):
    """
    Train custom telescope detection model.

    Args:
        data_yaml: Path to dataset YAML file
        base_model: Base model to fine-tune from
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to train on (cuda:0, cpu, etc.)
        project: Project directory for runs
        name: Run name
    """
    print("=" * 80)
    print("🔭 Telescope Equipment Detection - Custom Model Training")
    print("=" * 80)

    # Check CUDA
    if "cuda" in device and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"
    elif "cuda" in device:
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # Load base model
    print(f"\n📦 Loading base model: {base_model}")
    model = YOLO(base_model)

    # Training configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"   - Dataset: {data_yaml}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Image size: {imgsz}")
    print(f"   - Batch size: {batch}")
    print(f"   - Device: {device}")
    print(f"   - Output: {project}/{name}")

    # Start training
    print(f"\n🚀 Starting training...\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=20,  # Early stopping patience
        save=True,
        plots=True,
        val=True,
        # Augmentation settings
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=10.0,  # Rotation
        translate=0.1,  # Translation
        scale=0.5,  # Scale
        flipud=0.0,  # Flip up-down
        fliplr=0.5,  # Flip left-right
        mosaic=1.0,  # Mosaic augmentation
    )

    print(f"\n✅ Training complete!")
    print(f"📊 Results saved to: {project}/{name}")
    print(f"🎯 Best model: {project}/{name}/weights/best.pt")

    # Validation metrics
    print(f"\n📈 Final Metrics:")
    print(f"   - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # Export model
    print(f"\n💾 Exporting model...")
    model_path = Path(project) / name / "weights" / "best.pt"
    export_path = Path("models") / "telescope_custom.pt"
    export_path.parent.mkdir(exist_ok=True)

    import shutil
    shutil.copy(model_path, export_path)

    print(f"✅ Model exported to: {export_path}")
    print(f"\n📝 Next steps:")
    print(f"   1. Evaluate model: python training/scripts/evaluate_model.py")
    print(f"   2. Update config to use custom model:")
    print(f"      detection:")
    print(f"        model: \"models/telescope_custom.pt\"")
    print(f"   3. Restart detection system")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom telescope detection model")
    parser.add_argument("--data", default="training/datasets/telescope_equipment/classes.yaml",
                       help="Path to dataset YAML file")
    parser.add_argument("--base-model", default="yolov8n.pt",
                       help="Base model to fine-tune (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size (reduce if OOM)")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to train on")
    parser.add_argument("--project", default="training/runs",
                       help="Project directory")
    parser.add_argument("--name", default="telescope_detection",
                       help="Run name")

    args = parser.parse_args()

    train_telescope_model(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )
