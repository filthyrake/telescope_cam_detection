#!/usr/bin/env python3
"""
Example visualization of tracking output.
Shows how tracking data would appear in real system.
"""

import json

# Example: Three frames of detection with tracking

frame1 = {
    "type": "detections",
    "camera_id": "cam1",
    "camera_name": "Main Backyard View",
    "frame_id": 12345,
    "timestamp": 1729538627.0,
    "total_detections": 1,
    "detections": [
        {
            "class_name": "coyote",
            "confidence": 0.85,
            "bbox": {"x1": 500, "y1": 300, "x2": 700, "y2": 500}
        }
    ],
    "tracks": {
        "550e8400-e29b-41d4-a716-446655440000": {
            "track_id": "550e8400-e29b-41d4-a716-446655440000",
            "class_name": "coyote",
            "species": "Canis latrans",
            "camera_id": "cam1",
            "first_seen": 1729538627.0,
            "last_seen": 1729538627.0,
            "frames_detected": 1,
            "current_bbox": {"x1": 500, "y1": 300, "x2": 700, "y2": 500},
            "current_confidence": 0.85,
            "avg_confidence": 0.85,
            "distance_traveled": 0.0,
            "speed": 0.0,
            "dwell_time": 0.0,
            "status": "active",
            "trajectory": [[600, 400, 1729538627.0]]
        }
    },
    "total_active_tracks": 1
}

frame2 = {
    "type": "detections",
    "camera_id": "cam1",
    "camera_name": "Main Backyard View",
    "frame_id": 12346,
    "timestamp": 1729538627.5,
    "total_detections": 2,
    "detections": [
        {
            "class_name": "coyote",
            "confidence": 0.87,
            "bbox": {"x1": 520, "y1": 305, "x2": 720, "y2": 505}
        },
        {
            "class_name": "rabbit",
            "confidence": 0.80,
            "bbox": {"x1": 300, "y1": 400, "x2": 350, "y2": 450}
        }
    ],
    "tracks": {
        "550e8400-e29b-41d4-a716-446655440000": {
            "track_id": "550e8400-e29b-41d4-a716-446655440000",
            "class_name": "coyote",
            "species": "Canis latrans",
            "camera_id": "cam1",
            "first_seen": 1729538627.0,
            "last_seen": 1729538627.5,
            "frames_detected": 2,
            "current_bbox": {"x1": 520, "y1": 305, "x2": 720, "y2": 505},
            "current_confidence": 0.87,
            "avg_confidence": 0.86,
            "distance_traveled": 22.4,
            "speed": 44.8,
            "dwell_time": 0.5,
            "status": "active",
            "trajectory": [
                [600, 400, 1729538627.0],
                [620, 405, 1729538627.5]
            ]
        },
        "7f9c3d21-a4b5-4c8e-9f1d-2e6a8b7c4d3f": {
            "track_id": "7f9c3d21-a4b5-4c8e-9f1d-2e6a8b7c4d3f",
            "class_name": "rabbit",
            "species": None,
            "camera_id": "cam1",
            "first_seen": 1729538627.5,
            "last_seen": 1729538627.5,
            "frames_detected": 1,
            "current_bbox": {"x1": 300, "y1": 400, "x2": 350, "y2": 450},
            "current_confidence": 0.80,
            "avg_confidence": 0.80,
            "distance_traveled": 0.0,
            "speed": 0.0,
            "dwell_time": 0.0,
            "status": "active",
            "trajectory": [[325, 425, 1729538627.5]]
        }
    },
    "total_active_tracks": 2
}

frame3 = {
    "type": "detections",
    "camera_id": "cam1",
    "camera_name": "Main Backyard View",
    "frame_id": 12347,
    "timestamp": 1729538628.0,
    "total_detections": 2,
    "detections": [
        {
            "class_name": "coyote",
            "confidence": 0.88,
            "bbox": {"x1": 540, "y1": 310, "x2": 740, "y2": 510}
        },
        {
            "class_name": "rabbit",
            "confidence": 0.82,
            "bbox": {"x1": 280, "y1": 395, "x2": 330, "y2": 445}
        }
    ],
    "tracks": {
        "550e8400-e29b-41d4-a716-446655440000": {
            "track_id": "550e8400-e29b-41d4-a716-446655440000",
            "class_name": "coyote",
            "species": "Canis latrans",
            "camera_id": "cam1",
            "first_seen": 1729538627.0,
            "last_seen": 1729538628.0,
            "frames_detected": 3,
            "current_bbox": {"x1": 540, "y1": 310, "x2": 740, "y2": 510},
            "current_confidence": 0.88,
            "avg_confidence": 0.867,
            "distance_traveled": 44.7,
            "speed": 44.6,
            "dwell_time": 1.0,
            "status": "active",
            "trajectory": [
                [600, 400, 1729538627.0],
                [620, 405, 1729538627.5],
                [640, 410, 1729538628.0]
            ]
        },
        "7f9c3d21-a4b5-4c8e-9f1d-2e6a8b7c4d3f": {
            "track_id": "7f9c3d21-a4b5-4c8e-9f1d-2e6a8b7c4d3f",
            "class_name": "rabbit",
            "species": "Sylvilagus audubonii",
            "camera_id": "cam1",
            "first_seen": 1729538627.5,
            "last_seen": 1729538628.0,
            "frames_detected": 2,
            "current_bbox": {"x1": 280, "y1": 395, "x2": 330, "y2": 445},
            "current_confidence": 0.82,
            "avg_confidence": 0.81,
            "distance_traveled": 22.4,
            "speed": 44.8,
            "dwell_time": 0.5,
            "status": "active",
            "trajectory": [
                [325, 425, 1729538627.5],
                [305, 420, 1729538628.0]
            ]
        }
    },
    "total_active_tracks": 2
}

print("=" * 80)
print("EXAMPLE: Object Tracking in Action")
print("=" * 80)
print()

print("Frame 1 (t=0.0s):")
print("-" * 80)
print(f"  Detections: {frame1['total_detections']}")
print(f"  Active Tracks: {frame1['total_active_tracks']}")
print()
for track_id, track in frame1['tracks'].items():
    print(f"  Track {track_id[:8]}...")
    print(f"    Class: {track['class_name']}")
    print(f"    Status: {track['status']}")
    print(f"    Dwell Time: {track['dwell_time']:.1f}s")
    print(f"    Distance: {track['distance_traveled']:.1f}px")
    print()

print("\nFrame 2 (t=0.5s):")
print("-" * 80)
print(f"  Detections: {frame2['total_detections']}")
print(f"  Active Tracks: {frame2['total_active_tracks']}")
print()
for track_id, track in frame2['tracks'].items():
    print(f"  Track {track_id[:8]}...")
    print(f"    Class: {track['class_name']}")
    print(f"    Status: {track['status']}")
    print(f"    Frames: {track['frames_detected']}")
    print(f"    Dwell Time: {track['dwell_time']:.1f}s")
    print(f"    Distance: {track['distance_traveled']:.1f}px")
    print(f"    Speed: {track['speed']:.1f}px/s")
    print()

print("\nFrame 3 (t=1.0s):")
print("-" * 80)
print(f"  Detections: {frame3['total_detections']}")
print(f"  Active Tracks: {frame3['total_active_tracks']}")
print()
for track_id, track in frame3['tracks'].items():
    print(f"  Track {track_id[:8]}...")
    print(f"    Class: {track['class_name']}")
    if track['species']:
        print(f"    Species: {track['species']}")
    print(f"    Status: {track['status']}")
    print(f"    Frames: {track['frames_detected']}")
    print(f"    Dwell Time: {track['dwell_time']:.1f}s")
    print(f"    Distance: {track['distance_traveled']:.1f}px")
    print(f"    Speed: {track['speed']:.1f}px/s")
    print(f"    Avg Confidence: {track['avg_confidence']:.3f}")
    print()

print("=" * 80)
print("TRACKING INSIGHTS")
print("=" * 80)
print()
print("Coyote (Track 550e8400...):")
print("  - Moving east at ~45 px/s")
print("  - In view for 1.0 seconds")
print("  - High confidence (87%)")
print("  - Species identified: Canis latrans")
print()
print("Rabbit (Track 7f9c3d21...):")
print("  - Moving west at ~45 px/s")
print("  - Recently appeared (0.5s ago)")
print("  - Potentially fleeing from coyote")
print("  - Species identified: Desert Cottontail")
print()
print("=" * 80)

# Save example data
with open('/tmp/tracking_example.json', 'w') as f:
    json.dump({
        'frame1': frame1,
        'frame2': frame2,
        'frame3': frame3
    }, f, indent=2)

print(f"\nFull example data saved to: /tmp/tracking_example.json")
