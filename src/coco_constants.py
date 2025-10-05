"""
COCO Dataset Constants
Shared constants for YOLOX detection and species classification
"""

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Wildlife-relevant COCO classes (class_id -> name)
WILDLIFE_CLASSES = {
    0: "person",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

# Mapping from COCO class_id to classifier category for Stage 2
CLASS_ID_TO_CATEGORY = {
    14: "bird",         # bird
    15: "mammal",       # cat
    16: "mammal",       # dog
    17: "mammal",       # horse
    18: "mammal",       # sheep
    19: "mammal",       # cow
    20: "mammal",       # elephant
    21: "mammal",       # bear
    22: "mammal",       # zebra
    23: "mammal",       # giraffe
}

# Mammal class IDs (extracted from WILDLIFE_CLASSES for easy filtering)
MAMMAL_CLASS_IDS = [15, 16, 17, 18, 19, 20, 21, 22, 23]

# Performance baseline constants
GROUNDINGDINO_BASELINE_MS = 560  # Original GroundingDINO inference time
YOLOX_TARGET_MS = 15  # Target YOLOX inference time
