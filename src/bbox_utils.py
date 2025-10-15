"""
Bounding Box Validation Utilities
Ensures bboxes are well-formed before processing.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def ensure_valid_bbox(bbox: Dict[str, float], min_size: int = 1) -> Dict[str, float]:
    """
    Ensure bounding box has valid coordinates (x1 < x2, y1 < y2).

    Args:
        bbox: Bounding box dictionary with keys: x1, y1, x2, y2
        min_size: Minimum width/height (default: 1 pixel)

    Returns:
        Valid bounding box dictionary with corrected coordinates

    Note:
        - Swaps inverted coordinates (x1>x2, y1>y2)
        - Enforces minimum size
        - Recalculates area after normalization
    """
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

    # Swap if inverted
    if x1 > x2:
        x1, x2 = x2, x1
        logger.debug(f"Swapped inverted x coordinates: {bbox['x1']} > {bbox['x2']}")
    if y1 > y2:
        y1, y2 = y2, y1
        logger.debug(f"Swapped inverted y coordinates: {bbox['y1']} > {bbox['y2']}")

    # Ensure minimum size
    if x2 - x1 < min_size:
        x2 = x1 + min_size
        logger.debug(f"Enforced minimum width: {min_size}px")
    if y2 - y1 < min_size:
        y2 = y1 + min_size
        logger.debug(f"Enforced minimum height: {min_size}px")

    # Recalculate area
    width = x2 - x1
    height = y2 - y1
    area = width * height

    return {
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'width': width,
        'height': height,
        'area': area
    }


def validate_bbox_coords(x1: float, y1: float, x2: float, y2: float, min_size: int = 1) -> Tuple[float, float, float, float]:
    """
    Validate and normalize bbox coordinates (tuple form).

    Args:
        x1, y1, x2, y2: Bounding box coordinates
        min_size: Minimum width/height (default: 1 pixel)

    Returns:
        Tuple of (x1, y1, x2, y2) with normalized coordinates
    """
    # Swap if inverted
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Ensure minimum size
    if x2 - x1 < min_size:
        x2 = x1 + min_size
    if y2 - y1 < min_size:
        y2 = y1 + min_size

    return (x1, y1, x2, y2)


def is_valid_bbox(bbox: Dict[str, float], min_size: int = 1) -> bool:
    """
    Check if bounding box has valid coordinates.

    Args:
        bbox: Bounding box dictionary with keys: x1, y1, x2, y2
        min_size: Minimum width/height (default: 1 pixel)

    Returns:
        True if bbox is valid, False otherwise

    Note:
        This checks if a bbox is ALREADY valid (before normalization).
        Use ensure_valid_bbox() to normalize an invalid bbox.
        The definition of "valid" matches ensure_valid_bbox(): proper
        ordering (x1 < x2, y1 < y2) and minimum size.
    """
    try:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        # Check for proper ordering
        if x1 >= x2 or y1 >= y2:
            return False

        # Check for minimum size (consistent with ensure_valid_bbox)
        if x2 - x1 < min_size or y2 - y1 < min_size:
            return False

        # Check for negative coordinates (optional, depends on use case)
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return False

        return True
    except (KeyError, TypeError):
        return False
