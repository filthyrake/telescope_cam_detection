"""
Object Tracking Module
Implements tracking of detected objects across frames with unique IDs.
Supports IoU-based tracking with configurable parameters.
"""

import uuid
import time
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from src.bbox_utils import bbox_iou

logger = logging.getLogger(__name__)


class Track:
    """
    Represents a single tracked object across multiple frames.
    """

    def __init__(self, detection: Dict[str, Any], track_id: Optional[str] = None):
        """
        Initialize a new track from a detection.

        Args:
            detection: Detection dictionary with bbox, class_name, confidence
            track_id: Optional UUID string, generates new one if not provided
        """
        self.track_id = track_id if track_id else str(uuid.uuid4())
        self.class_name = detection['class_name']
        self.camera_id = detection.get('camera_id', 'default')
        
        # Species classification (if available)
        self.species = detection.get('species', None)
        
        # Lifecycle
        self.first_seen = detection.get('timestamp', time.time())
        self.last_seen = self.first_seen
        self.frames_detected = 1
        self.age = 0  # Frames since last detection
        
        # Current state
        self.current_bbox = detection['bbox']
        self.current_confidence = detection['confidence']
        
        # History (limit size to prevent memory issues)
        max_history = 100
        self.trajectory = deque(maxlen=max_history)  # [(x, y, timestamp), ...]
        self.bbox_history = deque(maxlen=max_history)  # [bbox, ...]
        self.confidence_history = deque(maxlen=max_history)  # [conf, ...]
        
        # Add initial point
        center_x, center_y = self._bbox_center(self.current_bbox)
        self.trajectory.append((center_x, center_y, self.first_seen))
        self.bbox_history.append(self.current_bbox.copy())
        self.confidence_history.append(self.current_confidence)
        
        # Metrics
        self.distance_traveled = 0.0
        self.status = "active"  # active, lost, completed

    def _bbox_center(self, bbox: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate center point of bounding box.

        Args:
            bbox: Bounding box dict with x1, y1, x2, y2

        Returns:
            Tuple of (center_x, center_y)
        """
        center_x = (bbox['x1'] + bbox['x2']) / 2.0
        center_y = (bbox['y1'] + bbox['y2']) / 2.0
        return center_x, center_y

    def update(self, detection: Dict[str, Any]):
        """
        Update track with new detection.

        Args:
            detection: Detection dictionary
        """
        self.last_seen = detection.get('timestamp', time.time())
        self.frames_detected += 1
        self.age = 0  # Reset age since we have a new detection
        
        # Update current state
        prev_bbox = self.current_bbox
        self.current_bbox = detection['bbox']
        self.current_confidence = detection['confidence']
        
        # Update species if provided and not already set
        if 'species' in detection and detection['species']:
            self.species = detection['species']
        
        # Add to history
        center_x, center_y = self._bbox_center(self.current_bbox)
        self.trajectory.append((center_x, center_y, self.last_seen))
        self.bbox_history.append(self.current_bbox.copy())
        self.confidence_history.append(self.current_confidence)
        
        # Calculate distance traveled
        prev_center_x, prev_center_y = self._bbox_center(prev_bbox)
        distance = math.sqrt(
            (center_x - prev_center_x) ** 2 + 
            (center_y - prev_center_y) ** 2
        )
        self.distance_traveled += distance

    def mark_lost(self):
        """Mark track as lost (not seen in current frame)."""
        self.age += 1
        if self.status == "active":
            self.status = "lost"

    def mark_completed(self):
        """Mark track as completed (exceeded max_age)."""
        self.status = "completed"

    def get_avg_confidence(self) -> float:
        """
        Calculate average confidence across all detections.

        Returns:
            Average confidence value
        """
        if not self.confidence_history:
            return self.current_confidence
        return sum(self.confidence_history) / len(self.confidence_history)

    def get_velocity(self) -> Tuple[float, float]:
        """
        Calculate velocity vector (pixels/second) from recent trajectory.

        Returns:
            Tuple of (vx, vy) velocity components
        """
        if len(self.trajectory) < 2:
            return (0.0, 0.0)
        
        # Use last 2 points for velocity calculation
        x2, y2, t2 = self.trajectory[-1]
        x1, y1, t1 = self.trajectory[-2]
        
        time_delta = t2 - t1
        if time_delta <= 0:
            return (0.0, 0.0)
        
        vx = (x2 - x1) / time_delta
        vy = (y2 - y1) / time_delta
        
        return (vx, vy)

    def get_speed(self) -> float:
        """
        Calculate speed (magnitude of velocity) in pixels/second.

        Returns:
            Speed value
        """
        vx, vy = self.get_velocity()
        return math.sqrt(vx ** 2 + vy ** 2)

    def get_dwell_time(self) -> float:
        """
        Calculate dwell time (time between first and last seen).

        Returns:
            Dwell time in seconds
        """
        return self.last_seen - self.first_seen

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert track to dictionary for API/WebSocket.

        Returns:
            Dictionary representation of track
        """
        return {
            'track_id': self.track_id,
            'class_name': self.class_name,
            'species': self.species,
            'camera_id': self.camera_id,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'frames_detected': self.frames_detected,
            'current_bbox': self.current_bbox,
            'current_confidence': self.current_confidence,
            'avg_confidence': self.get_avg_confidence(),
            'distance_traveled': self.distance_traveled,
            'speed': self.get_speed(),
            'dwell_time': self.get_dwell_time(),
            'status': self.status,
            'trajectory': list(self.trajectory)[-10:],  # Last 10 points only
        }


class ObjectTracker:
    """
    Tracks objects across frames using IoU matching.
    Implements simple online tracking with unique IDs.
    """

    def __init__(
        self,
        algorithm: str = "iou",
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_camera: bool = True
    ):
        """
        Initialize object tracker.

        Args:
            algorithm: Tracking algorithm (currently only "iou" supported)
            max_age: Delete track after N frames without detection
            min_hits: Require N detections before considering track valid
            iou_threshold: Minimum IoU for association
            per_camera: Track separately per camera (no cross-camera tracking)
        """
        self.algorithm = algorithm
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.per_camera = per_camera
        
        # Tracks by camera_id (if per_camera=True) or global (if False)
        self.tracks: Dict[str, Dict[str, Track]] = {}  # {camera_id: {track_id: Track}}
        self.completed_tracks: List[Track] = []  # Tracks that are done
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_completed = 0

    def _get_tracks_for_camera(self, camera_id: str) -> Dict[str, Track]:
        """
        Get tracks for a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Dictionary of track_id -> Track
        """
        if self.per_camera:
            if camera_id not in self.tracks:
                self.tracks[camera_id] = {}
            return self.tracks[camera_id]
        else:
            # Global tracking (single camera key)
            if 'global' not in self.tracks:
                self.tracks['global'] = {}
            return self.tracks['global']

    def update(
        self,
        detections: List[Dict[str, Any]],
        camera_id: str = "default"
    ) -> Dict[str, Track]:
        """
        Update tracks with new detections from a frame.

        Args:
            detections: List of detection dictionaries
            camera_id: Camera identifier for per-camera tracking

        Returns:
            Dictionary of active tracks (track_id -> Track)
        """
        camera_tracks = self._get_tracks_for_camera(camera_id)
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(
            detections, camera_tracks
        )
        
        # Update matched tracks
        for track_id, detection in matched:
            camera_tracks[track_id].update(detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_dets:
            # Ensure camera_id is in detection
            detection['camera_id'] = camera_id
            new_track = Track(detection)
            camera_tracks[new_track.track_id] = new_track
            self.total_tracks_created += 1
            logger.debug(f"Created new track: {new_track.track_id} for {detection['class_name']}")
        
        # Age out unmatched tracks
        tracks_to_remove = []
        for track_id in unmatched_tracks:
            track = camera_tracks[track_id]
            track.mark_lost()
            
            if track.age > self.max_age:
                track.mark_completed()
                # Only keep tracks that meet minimum hits threshold
                if track.frames_detected >= self.min_hits:
                    self.completed_tracks.append(track)
                    self.total_tracks_completed += 1
                    logger.debug(
                        f"Completed track: {track_id} "
                        f"({track.class_name}, {track.frames_detected} frames, "
                        f"{track.get_dwell_time():.1f}s dwell time)"
                    )
                tracks_to_remove.append(track_id)
        
        # Remove completed tracks
        for track_id in tracks_to_remove:
            del camera_tracks[track_id]
        
        return camera_tracks

    def _associate(
        self,
        detections: List[Dict[str, Any]],
        tracks: Dict[str, Track]
    ) -> Tuple[List[Tuple[str, Dict]], List[Dict], List[str]]:
        """
        Associate detections with existing tracks using IoU.

        Args:
            detections: List of detection dictionaries
            tracks: Dictionary of existing tracks

        Returns:
            Tuple of (matched, unmatched_detections, unmatched_tracks)
            - matched: List of (track_id, detection) tuples
            - unmatched_detections: List of detections without matches
            - unmatched_tracks: List of track_ids without matches
        """
        if not detections or not tracks:
            return [], list(detections), list(tracks.keys())
        
        # Build IoU matrix
        track_ids = list(tracks.keys())
        iou_matrix = []
        
        for detection in detections:
            det_bbox = detection['bbox']
            det_class = detection['class_name']
            row = []
            
            for track_id in track_ids:
                track = tracks[track_id]
                # Only match same class
                if track.class_name != det_class:
                    row.append(0.0)
                    continue
                
                # Calculate IoU
                track_bbox = track.current_bbox
                iou = self._calculate_iou(det_bbox, track_bbox)
                row.append(iou)
            
            iou_matrix.append(row)
        
        # Perform greedy matching (highest IoU first)
        matched = []
        unmatched_det_indices = set(range(len(detections)))
        unmatched_track_indices = set(range(len(track_ids)))
        
        # Sort all matches by IoU (descending)
        all_matches = []
        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                iou = iou_matrix[i][j]
                if iou >= self.iou_threshold:
                    all_matches.append((iou, i, j))
        
        all_matches.sort(reverse=True, key=lambda x: x[0])
        
        # Greedy assignment
        for iou, det_idx, track_idx in all_matches:
            if det_idx in unmatched_det_indices and track_idx in unmatched_track_indices:
                track_id = track_ids[track_idx]
                detection = detections[det_idx]
                matched.append((track_id, detection))
                unmatched_det_indices.discard(det_idx)
                unmatched_track_indices.discard(track_idx)
        
        # Gather unmatched
        unmatched_detections = [detections[i] for i in unmatched_det_indices]
        unmatched_track_ids = [track_ids[i] for i in unmatched_track_indices]
        
        return matched, unmatched_detections, unmatched_track_ids

    def _calculate_iou(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """
        Calculate IoU between two bounding boxes.

        Args:
            bbox1: First bbox dict with x1, y1, x2, y2
            bbox2: Second bbox dict with x1, y1, x2, y2

        Returns:
            IoU value between 0.0 and 1.0
        """
        # Convert to tuple format for bbox_iou function
        b1 = (bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2'])
        b2 = (bbox2['x1'], bbox2['y1'], bbox2['x2'], bbox2['y2'])
        return bbox_iou(b1, b2)

    def get_active_tracks(self, camera_id: Optional[str] = None) -> Dict[str, Track]:
        """
        Get all active tracks.

        Args:
            camera_id: Optional camera ID to filter tracks

        Returns:
            Dictionary of track_id -> Track
        """
        if camera_id:
            return self._get_tracks_for_camera(camera_id)
        else:
            # Return all tracks from all cameras
            all_tracks = {}
            for cam_tracks in self.tracks.values():
                all_tracks.update(cam_tracks)
            return all_tracks

    def get_track(self, track_id: str) -> Optional[Track]:
        """
        Get a specific track by ID.

        Args:
            track_id: Track identifier

        Returns:
            Track object or None if not found
        """
        # Search in active tracks
        for cam_tracks in self.tracks.values():
            if track_id in cam_tracks:
                return cam_tracks[track_id]
        
        # Search in completed tracks
        for track in self.completed_tracks:
            if track.track_id == track_id:
                return track
        
        return None

    def get_stats(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get tracking statistics.

        Args:
            camera_id: Optional camera ID to filter stats
            start_time: Optional start timestamp to filter completed tracks

        Returns:
            Dictionary of tracking statistics
        """
        active_tracks = self.get_active_tracks(camera_id)
        
        # Filter completed tracks by camera_id and time
        completed = self.completed_tracks
        if camera_id:
            completed = [t for t in completed if t.camera_id == camera_id]
        if start_time:
            completed = [t for t in completed if t.first_seen >= start_time]
        
        # Count by class
        active_by_class = {}
        for track in active_tracks.values():
            class_name = track.class_name
            active_by_class[class_name] = active_by_class.get(class_name, 0) + 1
        
        completed_by_class = {}
        for track in completed:
            class_name = track.class_name
            completed_by_class[class_name] = completed_by_class.get(class_name, 0) + 1
        
        # Calculate averages from completed tracks
        avg_dwell_time = 0.0
        longest_track = None
        
        if completed:
            dwell_times = [t.get_dwell_time() for t in completed]
            avg_dwell_time = sum(dwell_times) / len(dwell_times)
            
            # Find longest track
            longest_track = max(completed, key=lambda t: t.get_dwell_time())
        
        stats = {
            'total_active_tracks': len(active_tracks),
            'total_completed_tracks': len(completed),
            'total_unique_tracks': len(active_tracks) + len(completed),
            'active_by_class': active_by_class,
            'completed_by_class': completed_by_class,
            'avg_dwell_time_seconds': avg_dwell_time,
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_completed': self.total_tracks_completed,
        }
        
        if longest_track:
            stats['longest_track'] = {
                'track_id': longest_track.track_id,
                'class_name': longest_track.class_name,
                'duration_seconds': longest_track.get_dwell_time(),
                'distance_traveled': longest_track.distance_traveled,
                'frames_detected': longest_track.frames_detected,
            }
        
        return stats

    def reset(self, camera_id: Optional[str] = None):
        """
        Reset tracker state.

        Args:
            camera_id: Optional camera ID to reset only that camera's tracks
        """
        if camera_id:
            if camera_id in self.tracks:
                self.tracks[camera_id] = {}
        else:
            self.tracks = {}
            self.completed_tracks = []
            self.total_tracks_created = 0
            self.total_tracks_completed = 0
