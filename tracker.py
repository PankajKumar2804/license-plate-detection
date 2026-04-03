"""Advanced vehicle tracking system for license plate detection."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from loguru import logger

@dataclass
class VehicleTrack:
    """Vehicle tracking with trajectory history."""
    
    track_id: str
    first_detection: datetime
    last_detection: datetime
    plate_text: str
    confidence: float
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=100))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=100))
    locations: List[Tuple[float, float]] = field(default_factory=list)
    detected_frames: int = 0
    direction_vector: Optional[Tuple[float, float]] = None
    is_active: bool = True
    
    def update(self, bbox: Tuple, speed: float):
        """Update track with new detection."""
        self.bbox_history.append(bbox)
        self.speed_history.append(speed)
        self.last_detection = datetime.now()
        self.detected_frames += 1
        
        # Calculate direction
        if len(self.bbox_history) >= 2:
            prev = self.bbox_history[-2]
            curr = self.bbox_history[-1]
            self.direction_vector = (
                curr[0] - prev[0],
                curr[1] - prev[1]
            )
    
    def predict_next_position(self) -> Optional[Tuple[float, float]]:
        """Predict next vehicle position using Kalman filter."""
        if not self.direction_vector or len(self.speed_history) < 5:
            return None
        
        avg_speed = np.mean(list(self.speed_history)[-5:])
        dx, dy = self.direction_vector
        magnitude = np.sqrt(dx**2 + dy**2)
        
        if magnitude == 0:
            return None
        
        unit_vec = (dx/magnitude, dy/magnitude)
        next_pos = (
            self.bbox_history[-1][0] + unit_vec[0] * avg_speed,
            self.bbox_history[-1][1] + unit_vec[1] * avg_speed
        )
        return next_pos

class VehicleTracker:
    """Advanced multi-object tracking system."""
    
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        """Initialize tracker with Hungarian algorithm for association."""
        self.tracks: Dict[str, VehicleTrack] = {}
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_track_id = 0
        self.frame_count = 0
        
    def update(self, detections: List[Dict], frame_shape: Tuple[int, int]):
        """Update tracks with new detections using Hungarian algorithm."""
        self.frame_count += 1
        
        # Get cost matrix using IOU
        cost_matrix = self._compute_iou_matrix(detections)
        
        # Hungarian algorithm for matching
        matches = self._hungarian_matching(cost_matrix)
        
        # Update matched tracks
        for track_id, det_idx in matches:
            if track_id in self.tracks:
                det = detections[det_idx]
                speed = self._calculate_speed(
                    self.tracks[track_id].bbox_history[-1] if self.tracks[track_id].bbox_history else det['bbox'],
                    det['bbox']
                )
                self.tracks[track_id].update(det['bbox'], speed)
        
        # Create new tracks for unmatched detections
        unmatched_dets = set(range(len(detections))) - set(det_idx for _, det_idx in matches)
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = f"track_{self.next_track_id}"
            self.next_track_id += 1
            
            track = VehicleTrack(
                track_id=track_id,
                first_detection=datetime.now(),
                last_detection=datetime.now(),
                plate_text=det.get('text', 'UNKNOWN'),
                confidence=det.get('confidence', 0.0)
            )
            track.update(det['bbox'], 0.0)
            self.tracks[track_id] = track
        
        # Remove inactive tracks
        inactive = [
            tid for tid, track in self.tracks.items()
            if (datetime.now() - track.last_detection).seconds > self.max_age
        ]
        for tid in inactive:
            self.tracks[tid].is_active = False
    
    def _compute_iou_matrix(self, detections: List[Dict]) -> np.ndarray:
        """Compute Intersection over Union matrix."""
        active_tracks = [t for t in self.tracks.values() if t.is_active]
        matrix = np.zeros((len(active_tracks), len(detections)))
        
        for i, track in enumerate(active_tracks):
            if not track.bbox_history:
                continue
            track_bbox = track.bbox_history[-1]
            
            for j, det in enumerate(detections):
                det_bbox = det['bbox']
                iou = self._calculate_iou(track_bbox, det_bbox)
                matrix[i, j] = iou
        
        return 1 - matrix  # Convert to cost (lower is better)
    
    @staticmethod
    def _calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_min_x = max(x1_min, x2_min)
        inter_min_y = max(y1_min, y2_min)
        inter_max_x = min(x1_max, x2_max)
        inter_max_y = min(y1_max, y2_max)
        
        if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
            return 0.0
        
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def _calculate_speed(prev_bbox: Tuple, curr_bbox: Tuple) -> float:
        """Calculate speed from bbox movement."""
        prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
        curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
        
        distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
        return distance
    
    @staticmethod
    def _hungarian_matching(cost_matrix: np.ndarray) -> List[Tuple[str, int]]:
        """Hungarian algorithm for optimal track-detection matching."""
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return list(zip(row_ind, col_ind))
        except ImportError:
            logger.warning("scipy not available, using greedy matching")
            return []
    
    def get_active_tracks(self) -> Dict[str, VehicleTrack]:
        """Get all active tracks."""
        return {tid: t for tid, t in self.tracks.items() if t.is_active}
    
    def get_track_statistics(self) -> Dict:
        """Get tracking statistics."""
        active = self.get_active_tracks()
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len(active),
            'avg_detections': np.mean([t.detected_frames for t in active.values()]) if active else 0,
            'frame_count': self.frame_count,
            'tracks_data': {
                tid: {
                    'plate': t.plate_text,
                    'confidence': t.confidence,
                    'detections': t.detected_frames,
                    'duration': (t.last_detection - t.first_detection).total_seconds()
                }
                for tid, t in active.items()
            }
        }
