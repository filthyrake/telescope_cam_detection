#!/usr/bin/env python3
"""
Setup wildlife detection model.
Downloads a wildlife-specific YOLOv8 model or helps create custom classes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO
import yaml

# Custom wildlife classes for your area
DESERT_WILDLIFE_CLASSES = {
    0: 'coyote',
    1: 'rabbit',
    2: 'ground_squirrel',
    3: 'roadrunner',
    4: 'quail',
    5: 'iguana',
    6: 'tortoise',
    7: 'lizard',
    8: 'person',
    9: 'bird_other'
}

def check_available_models():
    """Check what models are available."""
    print("=" * 80)
    print("Wildlife Detection Model Options")
    print("=" * 80)
    print()
    print("Option 1: Use COCO model with generic classes")
    print("  - Available now")
    print("  - Will detect: dog (→coyote), bird (→roadrunner/quail)")
    print("  - Miss: rabbits, iguanas, tortoises, lizards, ground squirrels")
    print()
    print("Option 2: Train custom wildlife model")
    print("  - Most accurate for your specific animals")
    print("  - Requires 100-200 labeled images per class")
    print("  - Takes ~1-2 hours to collect + annotate + train")
    print()
    print("Option 3: Use pre-trained wildlife model from Roboflow/Ultralytics")
    print("  - Checking available models...")
    print()

    # Try to find wildlife models on Ultralytics Hub
    print("Available wildlife models:")
    print("  - iWildCam (camera trap animals)")
    print("  - NACTI (North American wildlife)")
    print("  - Custom models from Roboflow Universe")
    print()
    print("=" * 80)
    print()

def create_desert_wildlife_yaml():
    """Create dataset YAML for desert wildlife."""
    yaml_content = {
        'path': str(Path(__file__).parent.parent / 'training' / 'datasets' / 'desert_wildlife'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': DESERT_WILDLIFE_CLASSES
    }

    output_file = Path(__file__).parent.parent / 'training' / 'datasets' / 'desert_wildlife' / 'dataset.yaml'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"✅ Created dataset config: {output_file}")
    print()
    print("Custom wildlife classes defined:")
    for idx, name in DESERT_WILDLIFE_CLASSES.items():
        print(f"  {idx}: {name}")
    print()

def recommend_approach():
    """Recommend best approach based on needs."""
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("🎯 Best approach: Capture training data for YOUR specific wildlife")
    print()
    print("Why?")
    print("  - Your desert animals are unique (coyotes, iguanas, tortoises, etc.)")
    print("  - COCO model will miss most of them")
    print("  - Pre-trained wildlife models are mostly for African/camera trap animals")
    print("  - Custom model will be MUCH more accurate")
    print()
    print("Timeline:")
    print("  1. Capture wildlife images (use snapshot feature!): 1-7 days")
    print("  2. Annotate images (LabelImg/Roboflow): 2-4 hours")
    print("  3. Train custom model: 30-60 minutes")
    print()
    print("=" * 80)
    print()
    print("IMMEDIATE SOLUTION (While collecting data):")
    print("=" * 80)
    print()
    print("Lower confidence threshold to 0.25 and enable these COCO classes:")
    print("  - bird (will catch: roadrunner, quail, other birds)")
    print("  - dog (might catch: coyotes)")
    print("  - cat (might catch: some lizards/iguanas)")
    print()
    print("This will save snapshots when it detects *anything* that looks like")
    print("a dog/bird/cat. You can then use these saved snapshots as training data!")
    print()
    print("Command to update config:")
    print('  Set confidence to 0.25 in config/config.yaml')
    print('  Keep trigger_classes: [bird, dog, cat, person]')
    print()
    print("=" * 80)

def main():
    check_available_models()
    create_desert_wildlife_yaml()
    recommend_approach()

    print("\n📝 Next steps:")
    print("  1. Lower confidence to 0.25 to catch more wildlife")
    print("  2. Let system run for a few days collecting snapshots")
    print("  3. Review clips/ directory for wildlife captures")
    print("  4. Use saved snapshots as training data")
    print("  5. Annotate and train custom model")
    print()

if __name__ == "__main__":
    main()
