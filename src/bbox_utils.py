"""
Bounding Box Validation Utilities
Ensures bboxes are well-formed before processing.
"""

import logging
from typing import Dict, Any, Tuple, Optional, Union, List

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


def validate_bbox(bbox: Union[Tuple, List], frame_shape: Optional[Tuple[int, int]] = None, min_size: int = 1) -> Optional[Tuple[int, int, int, int]]:
    """
    Validate and normalize bounding box.
    
    Args:
        bbox: (x1, y1, x2, y2) tuple/list
        frame_shape: Optional (height, width) to clip to frame bounds
        min_size: Minimum width/height (default 1)
    
    Returns:
        Normalized (x1, y1, x2, y2) tuple or None if invalid
    """
    try:
        x1, y1, x2, y2 = map(float, bbox)
        
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
        
        # Clip to frame bounds if provided
        if frame_shape is not None:
            height, width = frame_shape
            x1 = max(0, min(x1, width - min_size))
            y1 = max(0, min(y1, height - min_size))
            x2 = max(x1 + min_size, min(x2, width))
            y2 = max(y1 + min_size, min(y2, height))
        
        # Final validation
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid bbox after normalization: {bbox}")
            return None
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    except (ValueError, TypeError) as e:
        logger.error(f"Error validating bbox {bbox}: {e}")
        return None


def bbox_area(bbox: Union[Tuple, List, Dict[str, float]]) -> int:
    """
    Calculate bounding box area, returns 0 if invalid.
    
    Args:
        bbox: Bounding box as tuple (x1, y1, x2, y2) or dict with x1, y1, x2, y2 keys
    
    Returns:
        Area in pixels, 0 if invalid
    """
    try:
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        else:
            x1, y1, x2, y2 = bbox
        
        return max(0, int((x2 - x1) * (y2 - y1)))
    except (ValueError, TypeError, KeyError, IndexError):
        return 0


def bbox_iou(bbox1: Union[Tuple, List], bbox2: Union[Tuple, List]) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        bbox1: First bounding box as (x1, y1, x2, y2)
        bbox2: Second bounding box as (x1, y1, x2, y2)
    
    Returns:
        IoU value between 0.0 and 1.0, or 0.0 if invalid
    """
    try:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if there's an intersection
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Avoid division by zero
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    except (ValueError, TypeError, IndexError):
        return 0.0
