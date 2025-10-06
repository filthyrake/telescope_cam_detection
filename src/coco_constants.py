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
# Mojave desert wildlife only: person, bird, cat (bobcat), dog (coyote/fox), bear
# Removed farm animals: horse, sheep, cow (false positives on cacti/bushes)
# Removed African animals: elephant, zebra, giraffe (not in North America)
WILDLIFE_CLASSES = {
    0: "person",
    14: "bird",
    15: "cat",      # Bobcat, wild cats
    16: "dog",      # Coyote, fox, wild dogs
    21: "bear",     # Black bear (rare but possible)
}

# Mapping from COCO class_id to classifier category for Stage 2
CLASS_ID_TO_CATEGORY = {
    14: "bird",         # bird
    15: "mammal",       # cat (bobcat, wild cats)
    16: "mammal",       # dog (coyote, fox)
    21: "mammal",       # bear
}

# Mammal class IDs (extracted from WILDLIFE_CLASSES for easy filtering)
MAMMAL_CLASS_IDS = [15, 16, 21]

# Performance baseline constants
GROUNDINGDINO_BASELINE_MS = 560  # Original GroundingDINO inference time
YOLOX_TARGET_MS = 15  # Target YOLOX inference time
