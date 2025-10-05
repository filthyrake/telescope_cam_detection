"""
Visualization Utilities
Functions for drawing bounding boxes and annotations on frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# Color palette for different classes (BGR format for OpenCV)
CLASS_COLORS = {
    'person': (0, 102, 255),      # Orange
    'bird': (0, 170, 255),         # Yellow/Gold
    'cat': (255, 68, 255),         # Pink/Magenta
    'dog': (255, 187, 68),         # Light Blue
    'horse': (147, 20, 255),       # Purple
    'sheep': (0, 215, 255),        # Gold
    'cow': (114, 128, 250),        # Light Coral
    'elephant': (170, 178, 32),    # Teal
    'bear': (45, 82, 160),         # Brown
    'zebra': (128, 128, 128),      # Gray
    'giraffe': (0, 255, 255),      # Yellow
}

# Default color for unknown classes
DEFAULT_COLOR = (0, 255, 136)  # Green (matches UI theme)


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """
    Get color for a class name.

    Args:
        class_name: Name of the detected class

    Returns:
        BGR color tuple
    """
    return CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Dict[str, float],
    class_name: str,
    confidence: float,
    color: Tuple[int, int, int] = None,
    thickness: int = 3,
    font_scale: float = 0.7,
    draw_label: bool = True,
    species: str = None,
    species_confidence: float = None
) -> np.ndarray:
    """
    Draw a single bounding box on a frame.

    Args:
        frame: Image frame (numpy array)
        bbox: Bounding box dict with x1, y1, x2, y2
        class_name: Name of detected class (Stage 1)
        confidence: Detection confidence (0-1, Stage 1)
        color: BGR color tuple (None = auto-select by class)
        thickness: Line thickness
        font_scale: Font size scale
        draw_label: Whether to draw label text
        species: Species name from Stage 2 classifier (optional)
        species_confidence: Species confidence from Stage 2 (optional)

    Returns:
        Frame with bounding box drawn (modifies in-place and returns)
    """
    # Get coordinates
    x1 = int(bbox['x1'])
    y1 = int(bbox['y1'])
    x2 = int(bbox['x2'])
    y2 = int(bbox['y2'])

    # Get color
    if color is None:
        color = get_class_color(class_name)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw label if requested
    if draw_label:
        # Fallback logic: Show species if available, otherwise show Stage 1 class
        if species is not None and species_confidence is not None:
            label = f"{species} {species_confidence:.2f}"
        else:
            label = f"{class_name} {confidence:.2f}"

        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness=2
        )

        # Draw background rectangle for text
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - baseline),
            (x1 + text_width, label_y + baseline),
            color,
            -1  # Filled
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return frame


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    thickness: int = 3,
    font_scale: float = 0.7,
    draw_labels: bool = True
) -> np.ndarray:
    """
    Draw all detections on a frame.

    Args:
        frame: Image frame (numpy array)
        detections: List of detection dictionaries
        thickness: Line thickness
        font_scale: Font size scale
        draw_labels: Whether to draw label text

    Returns:
        Frame with all bounding boxes drawn (creates a copy)
    """
    # Create a copy to avoid modifying original
    annotated_frame = frame.copy()

    # Draw each detection
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']

        # Extract Stage 2 species info if available
        species = detection.get('species')
        species_confidence = detection.get('species_confidence')

        draw_bounding_box(
            annotated_frame,
            bbox,
            class_name,
            confidence,
            thickness=thickness,
            font_scale=font_scale,
            draw_label=draw_labels,
            species=species,
            species_confidence=species_confidence
        )

    return annotated_frame


def add_info_overlay(
    frame: np.ndarray,
    info_text: List[str],
    position: str = "top-left",
    font_scale: float = 0.6,
    thickness: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (0, 255, 136)
) -> np.ndarray:
    """
    Add text overlay to frame.

    Args:
        frame: Image frame
        info_text: List of text lines to display
        position: "top-left", "top-right", "bottom-left", "bottom-right"
        font_scale: Font size scale
        thickness: Text thickness
        bg_color: Background color (BGR)
        text_color: Text color (BGR)

    Returns:
        Frame with overlay added
    """
    if not info_text:
        return frame

    h, w = frame.shape[:2]
    margin = 10
    line_height = 30

    # Calculate position
    if position == "top-left":
        x, y = margin, margin + 20
    elif position == "top-right":
        x, y = w - 300, margin + 20
    elif position == "bottom-left":
        x, y = margin, h - len(info_text) * line_height - margin
    else:  # bottom-right
        x, y = w - 300, h - len(info_text) * line_height - margin

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 5, y - 20),
        (x + 290, y + len(info_text) * line_height),
        bg_color,
        -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text lines
    for i, line in enumerate(info_text):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA
        )

    return frame


if __name__ == "__main__":
    # Test visualization utilities
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_frame[:] = (50, 50, 50)  # Dark gray background

    # Test detections
    test_detections = [
        {
            'class_name': 'bird',
            'confidence': 0.92,
            'bbox': {'x1': 100, 'y1': 100, 'x2': 300, 'y2': 250}
        },
        {
            'class_name': 'person',
            'confidence': 0.87,
            'bbox': {'x1': 400, 'y1': 200, 'x2': 550, 'y2': 500}
        },
        {
            'class_name': 'cat',
            'confidence': 0.75,
            'bbox': {'x1': 700, 'y1': 150, 'x2': 850, 'y2': 300}
        }
    ]

    # Draw detections
    annotated_frame = draw_detections(test_frame, test_detections)

    # Add info overlay
    info_text = [
        "Detection System Active",
        "3 objects detected",
        "Confidence: 0.85 avg"
    ]
    annotated_frame = add_info_overlay(annotated_frame, info_text)

    # Save test image
    output_path = "test_visualization.jpg"
    cv2.imwrite(output_path, annotated_frame)

    logger.info(f"✅ Test visualization saved to {output_path}")
