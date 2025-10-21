#!/usr/bin/env python3
"""
Simple integration test for tracking functionality.
Tests that tracking works end-to-end with detection processor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from queue import Queue
from src.detection_processor import DetectionProcessor


def test_tracking_integration():
    """Test tracking integration with detection processor."""
    print("Testing tracking integration...")
    
    # Create queues
    input_queue = Queue(maxsize=10)
    output_queue = Queue(maxsize=10)
    
    # Initialize detection processor with tracking enabled
    tracking_config = {
        'algorithm': 'iou',
        'max_age': 30,
        'min_hits': 3,
        'iou_threshold': 0.3,
        'per_camera': True
    }
    
    processor = DetectionProcessor(
        input_queue=input_queue,
        output_queue=output_queue,
        detection_history_size=30,
        enable_tracking=True,
        tracking_config=tracking_config
    )
    
    # Start processor
    assert processor.start(), "Failed to start detection processor"
    print("✓ Detection processor started with tracking enabled")
    
    # Simulate detection frames
    timestamp = time.time()
    
    # Frame 1: Single coyote detection
    detection1 = {
        'frame_id': 1,
        'camera_id': 'cam1',
        'camera_name': 'Test Camera',
        'timestamp': timestamp,
        'inference_time': 0.015,
        'detections': [
            {
                'class_name': 'coyote',
                'confidence': 0.85,
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'camera_id': 'cam1'
            }
        ],
        'frame_shape': (720, 1280, 3)
    }
    
    input_queue.put(detection1)
    time.sleep(0.1)  # Allow processing
    
    # Get result
    result1 = output_queue.get(timeout=1.0)
    assert 'tracks' in result1, "Tracks not in result"
    assert result1['total_active_tracks'] == 1, f"Expected 1 track, got {result1['total_active_tracks']}"
    
    track_id = list(result1['tracks'].keys())[0]
    track1 = result1['tracks'][track_id]
    assert track1['class_name'] == 'coyote'
    assert track1['frames_detected'] == 1
    print(f"✓ Track created: {track_id} (coyote)")
    
    # Frame 2: Same coyote moved slightly
    detection2 = {
        'frame_id': 2,
        'camera_id': 'cam1',
        'camera_name': 'Test Camera',
        'timestamp': timestamp + 0.1,
        'inference_time': 0.015,
        'detections': [
            {
                'class_name': 'coyote',
                'confidence': 0.87,
                'bbox': {'x1': 110, 'y1': 100, 'x2': 210, 'y2': 200},
                'camera_id': 'cam1'
            }
        ],
        'frame_shape': (720, 1280, 3)
    }
    
    input_queue.put(detection2)
    time.sleep(0.1)
    
    result2 = output_queue.get(timeout=1.0)
    assert result2['total_active_tracks'] == 1, "Track should persist"
    assert track_id in result2['tracks'], "Track ID should be the same"
    
    track2 = result2['tracks'][track_id]
    assert track2['frames_detected'] == 2, f"Expected 2 frames, got {track2['frames_detected']}"
    assert track2['distance_traveled'] > 0, "Distance should be non-zero"
    print(f"✓ Track updated: {track_id} (2 frames, {track2['distance_traveled']:.1f}px traveled)")
    
    # Frame 3: New rabbit detection
    detection3 = {
        'frame_id': 3,
        'camera_id': 'cam1',
        'camera_name': 'Test Camera',
        'timestamp': timestamp + 0.2,
        'inference_time': 0.015,
        'detections': [
            {
                'class_name': 'coyote',
                'confidence': 0.88,
                'bbox': {'x1': 120, 'y1': 100, 'x2': 220, 'y2': 200},
                'camera_id': 'cam1'
            },
            {
                'class_name': 'rabbit',
                'confidence': 0.8,
                'bbox': {'x1': 400, 'y1': 300, 'x2': 450, 'y2': 350},
                'camera_id': 'cam1'
            }
        ],
        'frame_shape': (720, 1280, 3)
    }
    
    input_queue.put(detection3)
    time.sleep(0.1)
    
    result3 = output_queue.get(timeout=1.0)
    assert result3['total_active_tracks'] == 2, f"Expected 2 tracks, got {result3['total_active_tracks']}"
    
    # Find rabbit track
    rabbit_track = None
    for tid, track in result3['tracks'].items():
        if track['class_name'] == 'rabbit':
            rabbit_track = track
            break
    
    assert rabbit_track is not None, "Rabbit track should exist"
    print(f"✓ New track created for rabbit")
    
    # Check stats
    stats = processor.get_stats()
    assert stats['tracking_enabled'] == True
    assert 'tracking_stats' in stats
    tracking_stats = stats['tracking_stats']
    assert tracking_stats['total_active_tracks'] == 2
    print(f"✓ Tracking stats: {tracking_stats['total_active_tracks']} active tracks")
    
    # Stop processor
    processor.stop()
    print("✓ Detection processor stopped")
    
    print("\n✅ All tracking integration tests passed!")
    return True


if __name__ == '__main__':
    try:
        success = test_tracking_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
