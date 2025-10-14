"""
Time-of-Day Detection Filter
Filters or adjusts detections based on species activity patterns.
"""

import logging
from datetime import datetime, time
from typing import List, Dict, Any, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivityPattern(Enum):
    """Species activity patterns."""
    DIURNAL = "diurnal"  # Active during day
    NOCTURNAL = "nocturnal"  # Active at night
    CREPUSCULAR = "crepuscular"  # Active at dawn/dusk
    CATHEMERAL = "cathemeral"  # Active any time


class TimeOfDay(Enum):
    """Time of day categories."""
    DAWN = "dawn"  # 5:00 AM - 8:00 AM
    DAY = "day"  # 8:00 AM - 5:00 PM
    DUSK = "dusk"  # 5:00 PM - 9:00 PM
    NIGHT = "night"  # 9:00 PM - 5:00 AM


class TimeOfDayFilter:
    """
    Filters detections based on species activity patterns.

    Example:
        - Birds are diurnal (active during day)
        - Bird detections at night are likely bugs/bats → reduce confidence or filter
    """

    # Default activity patterns for common classes
    DEFAULT_ACTIVITY_PATTERNS = {
        # Birds - diurnal (active during daylight)
        'bird': ActivityPattern.DIURNAL,
        'quail': ActivityPattern.DIURNAL,
        'roadrunner': ActivityPattern.DIURNAL,
        'hawk': ActivityPattern.DIURNAL,
        'raven': ActivityPattern.DIURNAL,
        'dove': ActivityPattern.DIURNAL,
        'owl': ActivityPattern.NOCTURNAL,  # Exception: owls are nocturnal

        # Mammals - mostly crepuscular or cathemeral
        'coyote': ActivityPattern.CREPUSCULAR,  # Active dawn/dusk/night
        'rabbit': ActivityPattern.CREPUSCULAR,
        'fox': ActivityPattern.CREPUSCULAR,
        'deer': ActivityPattern.CREPUSCULAR,
        'javelina': ActivityPattern.CREPUSCULAR,
        'bobcat': ActivityPattern.CREPUSCULAR,
        'cat': ActivityPattern.CATHEMERAL,  # Cats can be active anytime
        'dog': ActivityPattern.CATHEMERAL,

        # Reptiles - diurnal (need warmth)
        'lizard': ActivityPattern.DIURNAL,
        'iguana': ActivityPattern.DIURNAL,
        'tortoise': ActivityPattern.DIURNAL,
        'snake': ActivityPattern.CREPUSCULAR,  # Some snakes hunt at dusk

        # Humans - cathemeral
        'person': ActivityPattern.CATHEMERAL,

        # Nocturnal animals
        'bat': ActivityPattern.NOCTURNAL,
        'scorpion': ActivityPattern.NOCTURNAL,
    }

    # Time ranges for time-of-day categories (24-hour format)
    TIME_RANGES = {
        TimeOfDay.DAWN: (time(5, 0), time(8, 0)),  # 5:00 AM - 8:00 AM
        TimeOfDay.DAY: (time(8, 0), time(17, 0)),  # 8:00 AM - 5:00 PM
        TimeOfDay.DUSK: (time(17, 0), time(21, 0)),  # 5:00 PM - 9:00 PM
        TimeOfDay.NIGHT: (time(21, 0), time(5, 0)),  # 9:00 PM - 5:00 AM
    }

    def __init__(
        self,
        enabled: bool = True,
        confidence_penalty: float = 0.5,  # Multiply confidence by this for out-of-pattern detections
        hard_filter: bool = False,  # If True, completely remove out-of-pattern detections
        activity_patterns: Optional[Dict[str, ActivityPattern]] = None,
        timezone_offset: int = 0  # Hours to add/subtract from UTC (e.g., -7 for MST)
    ):
        """
        Initialize time-of-day filter.

        Args:
            enabled: Whether to enable filtering
            confidence_penalty: Multiplier for confidence (0.0-1.0) when species detected out of pattern
            hard_filter: If True, remove out-of-pattern detections entirely
            activity_patterns: Custom activity patterns (overrides defaults)
            timezone_offset: Timezone offset from UTC in hours
        """
        self.enabled = enabled
        self.confidence_penalty = confidence_penalty
        self.hard_filter = hard_filter
        self.timezone_offset = timezone_offset

        # Merge custom patterns with defaults
        self.activity_patterns = self.DEFAULT_ACTIVITY_PATTERNS.copy()
        if activity_patterns:
            self.activity_patterns.update(activity_patterns)

        # Statistics
        self.filtered_count = 0
        self.penalized_count = 0
        self.total_processed = 0

        logger.info(f"TimeOfDayFilter initialized (enabled={enabled}, penalty={confidence_penalty}, hard_filter={hard_filter})")

    def get_current_time_of_day(self, current_time: Optional[datetime] = None) -> TimeOfDay:
        """
        Determine current time of day category.

        Args:
            current_time: Optional datetime to check (defaults to now)

        Returns:
            TimeOfDay category
        """
        if current_time is None:
            current_time = datetime.now()

        # Apply timezone offset
        if self.timezone_offset != 0:
            from datetime import timedelta
            current_time = current_time + timedelta(hours=self.timezone_offset)

        current_hour_minute = current_time.time()

        # Check each time range
        for time_category, (start, end) in self.TIME_RANGES.items():
            if time_category == TimeOfDay.NIGHT:
                # Night wraps around midnight (21:00 - 5:00)
                if current_hour_minute >= start or current_hour_minute < end:
                    return time_category
            else:
                # Normal ranges
                if start <= current_hour_minute < end:
                    return time_category

        # Default to day if no match
        return TimeOfDay.DAY

    def is_activity_likely(
        self,
        species: str,
        time_of_day: TimeOfDay
    ) -> bool:
        """
        Check if a species is likely to be active at this time of day.

        Args:
            species: Species/class name
            time_of_day: Current time of day

        Returns:
            True if activity is likely, False otherwise
        """
        # Get activity pattern for this species
        pattern = self.activity_patterns.get(species.lower())
        if pattern is None:
            # Unknown species - assume cathemeral (active any time)
            return True

        # Check if activity matches time of day
        if pattern == ActivityPattern.CATHEMERAL:
            return True  # Active any time
        elif pattern == ActivityPattern.DIURNAL:
            return time_of_day in [TimeOfDay.DAWN, TimeOfDay.DAY, TimeOfDay.DUSK]
        elif pattern == ActivityPattern.NOCTURNAL:
            return time_of_day in [TimeOfDay.DUSK, TimeOfDay.NIGHT, TimeOfDay.DAWN]
        elif pattern == ActivityPattern.CREPUSCULAR:
            return time_of_day in [TimeOfDay.DAWN, TimeOfDay.DUSK, TimeOfDay.NIGHT]

        return True  # Default to allowing

    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        current_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter detections based on time-of-day activity patterns.

        Args:
            detections: List of detection dicts
            current_time: Optional datetime (defaults to now)

        Returns:
            Filtered list of detections
        """
        if not self.enabled or not detections:
            return detections

        self.total_processed += len(detections)
        time_of_day = self.get_current_time_of_day(current_time)

        filtered_detections = []
        for detection in detections:
            class_name = detection.get('class_name', '').lower()
            confidence = detection.get('confidence', 0.0)

            # Check if activity is likely for this species
            is_likely = self.is_activity_likely(class_name, time_of_day)

            if not is_likely:
                if self.hard_filter:
                    # Remove detection entirely
                    self.filtered_count += 1
                    logger.debug(f"Filtered {class_name} detection at {time_of_day.value} (confidence={confidence:.2f})")
                    continue  # Skip this detection
                else:
                    # Reduce confidence
                    original_confidence = confidence
                    detection['confidence'] = confidence * self.confidence_penalty
                    detection['time_of_day_penalty'] = True
                    detection['original_confidence'] = original_confidence
                    self.penalized_count += 1
                    logger.debug(f"Penalized {class_name} at {time_of_day.value}: {original_confidence:.2f} → {detection['confidence']:.2f}")

            filtered_detections.append(detection)

        return filtered_detections

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            'enabled': self.enabled,
            'total_processed': self.total_processed,
            'filtered_count': self.filtered_count,
            'penalized_count': self.penalized_count,
            'confidence_penalty': self.confidence_penalty,
            'hard_filter': self.hard_filter
        }


if __name__ == "__main__":
    # Test the time-of-day filter
    logger.info("Testing TimeOfDayFilter")

    # Create filter
    tod_filter = TimeOfDayFilter(
        enabled=True,
        confidence_penalty=0.5,
        hard_filter=False
    )

    # Test detections at different times
    test_cases = [
        # Bird during day (should pass)
        {
            'time': datetime(2025, 10, 14, 12, 0, 0),  # Noon
            'detections': [
                {'class_name': 'bird', 'confidence': 0.85, 'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}}
            ],
            'expected': 'bird should have full confidence'
        },
        # Bird at night (should be penalized)
        {
            'time': datetime(2025, 10, 14, 23, 0, 0),  # 11 PM
            'detections': [
                {'class_name': 'bird', 'confidence': 0.85, 'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}}
            ],
            'expected': 'bird confidence should be reduced (likely a bug or bat)'
        },
        # Coyote at dusk (should pass)
        {
            'time': datetime(2025, 10, 14, 19, 0, 0),  # 7 PM
            'detections': [
                {'class_name': 'coyote', 'confidence': 0.75, 'bbox': {'x1': 100, 'y1': 100, 'x2': 300, 'y2': 300}}
            ],
            'expected': 'coyote should have full confidence (crepuscular)'
        },
        # Lizard at night (should be penalized)
        {
            'time': datetime(2025, 10, 14, 22, 0, 0),  # 10 PM
            'detections': [
                {'class_name': 'lizard', 'confidence': 0.70, 'bbox': {'x1': 100, 'y1': 100, 'x2': 150, 'y2': 150}}
            ],
            'expected': 'lizard confidence should be reduced (diurnal, unlikely at night)'
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['expected']} ---")
        time_of_day = tod_filter.get_current_time_of_day(test_case['time'])
        print(f"Time: {test_case['time'].strftime('%Y-%m-%d %H:%M')} ({time_of_day.value})")

        filtered = tod_filter.filter_detections(test_case['detections'], test_case['time'])

        for detection in filtered:
            class_name = detection['class_name']
            original_conf = detection.get('original_confidence', detection['confidence'])
            current_conf = detection['confidence']
            penalized = detection.get('time_of_day_penalty', False)

            print(f"  {class_name}: confidence {original_conf:.2f} → {current_conf:.2f} (penalized={penalized})")

    # Print stats
    print("\n--- Filter Statistics ---")
    stats = tod_filter.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    logger.info("✅ TimeOfDayFilter test completed")
