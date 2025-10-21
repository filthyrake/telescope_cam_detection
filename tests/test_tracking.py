#!/usr/bin/env python3
"""
Tests for object tracking functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import time
from src.object_tracker import ObjectTracker, Track


class TestTrack(unittest.TestCase):
    """Tests for Track class."""

    def test_track_initialization(self):
        """Test track initialization from detection."""
        detection = {
            'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
            'class_name': 'coyote',
            'confidence': 0.85,
            'timestamp': time.time(),
            'camera_id': 'cam1'
        }
        
        track = Track(detection)
        
        self.assertIsNotNone(track.track_id)
        self.assertEqual(track.class_name, 'coyote')
        self.assertEqual(track.camera_id, 'cam1')
        self.assertEqual(track.frames_detected, 1)
        self.assertEqual(track.status, 'active')
        self.assertEqual(len(track.trajectory), 1)
        self.assertEqual(len(track.bbox_history), 1)

    def test_track_update(self):
        """Test track update with new detection."""
        detection1 = {
            'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
            'class_name': 'coyote',
            'confidence': 0.85,
            'timestamp': time.time(),
        }
        
        track = Track(detection1)
        initial_frames = track.frames_detected
        
        # Update with new detection (moved right by 10 pixels)
        detection2 = {
            'bbox': {'x1': 110, 'y1': 100, 'x2': 210, 'y2': 200},
            'class_name': 'coyote',
            'confidence': 0.87,
            'timestamp': time.time() + 0.1,
        }
        
        track.update(detection2)
        
        self.assertEqual(track.frames_detected, initial_frames + 1)
        self.assertEqual(track.current_bbox, detection2['bbox'])
        self.assertEqual(track.current_confidence, 0.87)
        self.assertEqual(len(track.trajectory), 2)
        self.assertGreater(track.distance_traveled, 0)

    def test_track_velocity(self):
        """Test velocity calculation."""
        timestamp = time.time()
        detection1 = {
            'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
            'class_name': 'bird',
            'confidence': 0.9,
            'timestamp': timestamp,
        }
        
        track = Track(detection1)
        
        # Move right by 50 pixels in 0.1 seconds
        detection2 = {
            'bbox': {'x1': 150, 'y1': 100, 'x2': 250, 'y2': 200},
            'class_name': 'bird',
            'confidence': 0.92,
            'timestamp': timestamp + 0.1,
        }
        
        track.update(detection2)
        
        vx, vy = track.get_velocity()
        speed = track.get_speed()
        
        # Should move at ~500 pixels/second horizontally
        self.assertAlmostEqual(vx, 500.0, delta=10.0)
        self.assertAlmostEqual(vy, 0.0, delta=10.0)
        self.assertAlmostEqual(speed, 500.0, delta=10.0)

    def test_track_dwell_time(self):
        """Test dwell time calculation."""
        timestamp = time.time()
        detection1 = {
            'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
            'class_name': 'rabbit',
            'confidence': 0.8,
            'timestamp': timestamp,
        }
        
        track = Track(detection1)
        
        # Update after 2 seconds
        detection2 = {
            'bbox': {'x1': 105, 'y1': 105, 'x2': 205, 'y2': 205},
            'class_name': 'rabbit',
            'confidence': 0.82,
            'timestamp': timestamp + 2.0,
        }
        
        track.update(detection2)
        
        dwell_time = track.get_dwell_time()
        self.assertAlmostEqual(dwell_time, 2.0, delta=0.1)

    def test_track_to_dict(self):
        """Test track serialization to dict."""
        detection = {
            'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
            'class_name': 'coyote',
            'confidence': 0.85,
            'timestamp': time.time(),
            'camera_id': 'cam1',
            'species': 'Canis latrans'
        }
        
        track = Track(detection)
        track_dict = track.to_dict()
        
        self.assertEqual(track_dict['track_id'], track.track_id)
        self.assertEqual(track_dict['class_name'], 'coyote')
        self.assertEqual(track_dict['species'], 'Canis latrans')
        self.assertEqual(track_dict['camera_id'], 'cam1')
        self.assertEqual(track_dict['status'], 'active')
        self.assertIn('trajectory', track_dict)
        self.assertIn('avg_confidence', track_dict)


class TestObjectTracker(unittest.TestCase):
    """Tests for ObjectTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ObjectTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        self.assertEqual(tracker.max_age, 30)
        self.assertEqual(tracker.min_hits, 3)
        self.assertEqual(tracker.iou_threshold, 0.3)
        self.assertEqual(tracker.total_tracks_created, 0)

    def test_tracker_create_new_track(self):
        """Test creating new track from detection."""
        tracker = ObjectTracker()
        
        detections = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'coyote',
                'confidence': 0.85,
                'timestamp': time.time(),
            }
        ]
        
        tracks = tracker.update(detections, camera_id='cam1')
        
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracker.total_tracks_created, 1)
        
        track = list(tracks.values())[0]
        self.assertEqual(track.class_name, 'coyote')
        self.assertEqual(track.frames_detected, 1)

    def test_tracker_update_existing_track(self):
        """Test updating existing track with matching detection."""
        tracker = ObjectTracker(iou_threshold=0.3)
        
        # Frame 1: Create track
        detections1 = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'bird',
                'confidence': 0.9,
                'timestamp': time.time(),
            }
        ]
        
        tracks = tracker.update(detections1, camera_id='cam1')
        track_id = list(tracks.keys())[0]
        
        # Frame 2: Update track (slightly moved)
        detections2 = [
            {
                'bbox': {'x1': 105, 'y1': 105, 'x2': 205, 'y2': 205},
                'class_name': 'bird',
                'confidence': 0.92,
                'timestamp': time.time() + 0.1,
            }
        ]
        
        tracks = tracker.update(detections2, camera_id='cam1')
        
        # Should still have 1 track (same ID)
        self.assertEqual(len(tracks), 1)
        self.assertIn(track_id, tracks)
        self.assertEqual(tracks[track_id].frames_detected, 2)

    def test_tracker_multiple_tracks(self):
        """Test tracking multiple objects simultaneously."""
        tracker = ObjectTracker()
        
        detections = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'coyote',
                'confidence': 0.85,
                'timestamp': time.time(),
            },
            {
                'bbox': {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400},
                'class_name': 'rabbit',
                'confidence': 0.8,
                'timestamp': time.time(),
            }
        ]
        
        tracks = tracker.update(detections, camera_id='cam1')
        
        self.assertEqual(len(tracks), 2)
        class_names = {track.class_name for track in tracks.values()}
        self.assertEqual(class_names, {'coyote', 'rabbit'})

    def test_tracker_age_out_tracks(self):
        """Test aging out tracks that are not detected."""
        tracker = ObjectTracker(max_age=2)
        
        # Frame 1: Create track
        detections1 = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'cat',
                'confidence': 0.9,
                'timestamp': time.time(),
            }
        ]
        
        tracker.update(detections1, camera_id='cam1')
        
        # Frame 2: No detections (age = 1)
        tracker.update([], camera_id='cam1')
        tracks = tracker.get_active_tracks('cam1')
        self.assertEqual(len(tracks), 1)
        
        # Frame 3: No detections (age = 2)
        tracker.update([], camera_id='cam1')
        tracks = tracker.get_active_tracks('cam1')
        self.assertEqual(len(tracks), 1)
        
        # Frame 4: No detections (age = 3, exceeds max_age=2)
        tracker.update([], camera_id='cam1')
        tracks = tracker.get_active_tracks('cam1')
        self.assertEqual(len(tracks), 0)

    def test_tracker_min_hits_threshold(self):
        """Test minimum hits threshold for valid tracks."""
        tracker = ObjectTracker(max_age=1, min_hits=3)
        
        # Create track with only 1 detection
        detections = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'dog',
                'confidence': 0.9,
                'timestamp': time.time(),
            }
        ]
        
        tracker.update(detections, camera_id='cam1')
        
        # Age out without reaching min_hits
        tracker.update([], camera_id='cam1')
        tracker.update([], camera_id='cam1')
        
        # Track should not be in completed tracks (didn't meet min_hits)
        self.assertEqual(len(tracker.completed_tracks), 0)

    def test_tracker_class_filtering(self):
        """Test that tracks only match detections of same class."""
        tracker = ObjectTracker(iou_threshold=0.5)
        
        # Frame 1: Bird at position A
        detections1 = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'bird',
                'confidence': 0.9,
                'timestamp': time.time(),
            }
        ]
        
        tracker.update(detections1, camera_id='cam1')
        
        # Frame 2: Cat at same position (high IoU but different class)
        detections2 = [
            {
                'bbox': {'x1': 105, 'y1': 105, 'x2': 205, 'y2': 205},
                'class_name': 'cat',
                'confidence': 0.85,
                'timestamp': time.time() + 0.1,
            }
        ]
        
        tracks = tracker.update(detections2, camera_id='cam1')
        
        # Should have 2 tracks (bird and cat, not matched)
        self.assertEqual(len(tracks), 2)
        class_names = {track.class_name for track in tracks.values()}
        self.assertEqual(class_names, {'bird', 'cat'})

    def test_tracker_per_camera(self):
        """Test per-camera tracking isolation."""
        tracker = ObjectTracker(per_camera=True)
        
        # Camera 1: Coyote
        detections_cam1 = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'coyote',
                'confidence': 0.85,
                'timestamp': time.time(),
            }
        ]
        
        # Camera 2: Coyote (different individual)
        detections_cam2 = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'coyote',
                'confidence': 0.85,
                'timestamp': time.time(),
            }
        ]
        
        tracker.update(detections_cam1, camera_id='cam1')
        tracker.update(detections_cam2, camera_id='cam2')
        
        # Should have separate tracks per camera
        tracks_cam1 = tracker.get_active_tracks('cam1')
        tracks_cam2 = tracker.get_active_tracks('cam2')
        
        self.assertEqual(len(tracks_cam1), 1)
        self.assertEqual(len(tracks_cam2), 1)
        
        # Track IDs should be different
        track_id_cam1 = list(tracks_cam1.keys())[0]
        track_id_cam2 = list(tracks_cam2.keys())[0]
        self.assertNotEqual(track_id_cam1, track_id_cam2)

    def test_tracker_stats(self):
        """Test tracker statistics."""
        tracker = ObjectTracker(max_age=1, min_hits=2)
        
        # Create and complete a track
        timestamp = time.time()
        for i in range(3):
            detections = [
                {
                    'bbox': {'x1': 100 + i*5, 'y1': 100, 'x2': 200 + i*5, 'y2': 200},
                    'class_name': 'rabbit',
                    'confidence': 0.8,
                    'timestamp': timestamp + i * 0.1,
                }
            ]
            tracker.update(detections, camera_id='cam1')
        
        # Age out the track
        tracker.update([], camera_id='cam1')
        tracker.update([], camera_id='cam1')
        
        stats = tracker.get_stats(camera_id='cam1')
        
        self.assertEqual(stats['total_active_tracks'], 0)
        self.assertEqual(stats['total_completed_tracks'], 1)
        self.assertEqual(stats['completed_by_class']['rabbit'], 1)
        self.assertGreater(stats['avg_dwell_time_seconds'], 0)

    def test_tracker_get_track(self):
        """Test retrieving specific track by ID."""
        tracker = ObjectTracker()
        
        detections = [
            {
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'class_name': 'coyote',
                'confidence': 0.85,
                'timestamp': time.time(),
            }
        ]
        
        tracks = tracker.update(detections, camera_id='cam1')
        track_id = list(tracks.keys())[0]
        
        # Get track by ID
        track = tracker.get_track(track_id)
        
        self.assertIsNotNone(track)
        self.assertEqual(track.track_id, track_id)
        self.assertEqual(track.class_name, 'coyote')


if __name__ == '__main__':
    unittest.main()
