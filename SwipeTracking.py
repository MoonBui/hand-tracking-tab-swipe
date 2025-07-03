# swipe_tracker.py
import time
import math
import cv2
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

class SwipeTracker:
    """
    A standalone swipe gesture tracking module for finger movements.
    Can be used with any hand tracking system that provides finger positions.
    """
    
    def __init__(self, max_points: int = 10, min_distance: float = 50, 
                 min_velocity: float = 100, trail_timeout: float = 2.0):
        """
        Initialize the SwipeTracker.
        
        Args:
            max_points: Maximum number of points to store in trail
            min_distance: Minimum distance for a valid swipe
            min_velocity: Minimum velocity (pixels/second) for a valid swipe
            trail_timeout: Time in seconds before clearing inactive trails
        """
        self.finger_trails: Dict[int, deque] = {}
        self.finger_times: Dict[int, deque] = {}
        self.last_update_time: Dict[int, float] = {}
        
        self.max_points = max_points
        self.min_distance = min_distance
        self.min_velocity = min_velocity
        self.trail_timeout = trail_timeout
        
        # Gesture state tracking
        self.active_gestures: Dict[int, bool] = {}
        self.last_swipe_time: Dict[int, float] = {}
        self.swipe_cooldown = 2.0  # Minimum time between swipes for same finger
    
    def add_point(self, finger_id: int, point: Tuple[int, int]) -> None:
        """Add a new point to the finger's trail."""
        current_time = time.time()
        
        # Initialize finger data if not exists
        if finger_id not in self.finger_trails:
            self.finger_trails[finger_id] = deque(maxlen=self.max_points)
            self.finger_times[finger_id] = deque(maxlen=self.max_points)
            self.active_gestures[finger_id] = False
            self.last_swipe_time[finger_id] = 0
        
        self.finger_trails[finger_id].append(point)
        self.finger_times[finger_id].append(current_time)
        self.last_update_time[finger_id] = current_time
    
    def detect_swipe(self, finger_id: int) -> Optional[Dict[str, Any]]:
        """
        Detect if a swipe gesture has occurred for the given finger.
        
        Returns:
            Dict with swipe information if detected, None otherwise
        """
        if not self._is_valid_trail(finger_id):
            return None
        
        # Check cooldown
        current_time = time.time()
        if (current_time - self.last_swipe_time[finger_id]) < self.swipe_cooldown:
            return None
        
        trail = list(self.finger_trails[finger_id])
        times = list(self.finger_times[finger_id])
        
        # Analyze the trail for swipe characteristics
        swipe_data = self._analyze_trail(trail, times, finger_id)
        
        if swipe_data and self._validate_swipe(swipe_data):
            self.last_swipe_time[finger_id] = current_time
            return swipe_data
        
        return None
    
    def _is_valid_trail(self, finger_id: int) -> bool:
        """Check if finger has a valid trail for analysis."""
        return (finger_id in self.finger_trails and 
                len(self.finger_trails[finger_id]) >= 3)
    
    def _analyze_trail(self, trail: List[Tuple[int, int]], 
                      times: List[float], finger_id: int) -> Optional[Dict[str, Any]]:
        """Analyze the trail to extract swipe characteristics."""
        if len(trail) < 3 or len(times) < 3:
            return None
        
        start_point = trail[0]
        end_point = trail[-1]
        start_time = times[0]
        end_time = times[-1]
        
        # Calculate basic metrics
        distance = math.sqrt((end_point[0] - start_point[0])**2 + 
                           (end_point[1] - start_point[1])**2)
        time_diff = end_time - start_time
        
        if time_diff <= 0:
            return None
        
        velocity = distance / time_diff
        direction = self._get_swipe_direction(start_point, end_point)
        
        # Calculate smoothness (how straight the path is)
        smoothness = self._calculate_smoothness(trail)
        
        return {
            'finger_id': finger_id,
            'start': start_point,
            'end': end_point,
            'distance': distance,
            'velocity': velocity,
            'direction': direction,
            'smoothness': smoothness,
            'duration': time_diff,
            'timestamp': end_time
        }
    
    def _validate_swipe(self, swipe_data: Dict[str, Any]) -> bool:
        """Validate if the analyzed data represents a valid swipe."""
        return (swipe_data['distance'] > self.min_distance and
                swipe_data['velocity'] > self.min_velocity and
                swipe_data['smoothness'] > 0.3)  # Reasonably straight path
    
    def _get_swipe_direction(self, start: Tuple[int, int], 
                           end: Tuple[int, int]) -> str:
        """Calculate swipe direction with 8-directional support."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        angle = math.atan2(dy, dx) * 180 / math.pi
        
        # 8-directional classification
        if -22.5 <= angle <= 22.5:
            return "right"
        elif 22.5 < angle <= 67.5:
            return "down-right"
        elif 67.5 < angle <= 112.5:
            return "down"
        elif 112.5 < angle <= 157.5:
            return "down-left"
        elif 157.5 < angle or angle <= -157.5:
            return "left"
        elif -157.5 < angle <= -112.5:
            return "up-left"
        elif -112.5 < angle <= -67.5:
            return "up"
        else:  # -67.5 < angle <= -22.5
            return "up-right"
    
    def _calculate_smoothness(self, trail: List[Tuple[int, int]]) -> float:
        """Calculate how smooth/straight the trail is (0-1, higher is smoother)."""
        if len(trail) < 3:
            return 0.0
        
        start = trail[0]
        end = trail[-1]
        
        # Calculate expected path length (straight line)
        expected_distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        if expected_distance == 0:
            return 0.0
        
        # Calculate actual path length
        actual_distance = 0
        for i in range(1, len(trail)):
            actual_distance += math.sqrt((trail[i][0] - trail[i-1][0])**2 + 
                                       (trail[i][1] - trail[i-1][1])**2)
        
        # Smoothness is ratio of expected to actual distance
        return min(expected_distance / actual_distance, 1.0) if actual_distance > 0 else 0.0
    
    def clear_trail(self, finger_id: int) -> None:
        """Clear the trail for a specific finger."""
        if finger_id in self.finger_trails:
            self.finger_trails[finger_id].clear()
            self.finger_times[finger_id].clear()
    
    def clear_all_trails(self) -> None:
        """Clear all finger trails."""
        for finger_id in list(self.finger_trails.keys()):
            self.clear_trail(finger_id)
    
    def cleanup_inactive_trails(self) -> None:
        """Remove trails that haven't been updated recently."""
        current_time = time.time()
        inactive_fingers = []
        
        for finger_id, last_time in self.last_update_time.items():
            if current_time - last_time > self.trail_timeout:
                inactive_fingers.append(finger_id)
        
        for finger_id in inactive_fingers:
            self.clear_trail(finger_id)
    
    def draw_trail(self, img, finger_id: int, color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> None:
        """Draw the trail for a specific finger on the image."""
        if finger_id in self.finger_trails and len(self.finger_trails[finger_id]) > 1:
            trail = list(self.finger_trails[finger_id])
            
            # Draw trail with fading effect
            for i in range(1, len(trail)):
                # Calculate alpha based on position in trail (newer points brighter)
                alpha = i / len(trail)
                fade_color = tuple(int(c * alpha) for c in color)
                
                cv2.line(img, trail[i-1], trail[i], fade_color, thickness)
                
            # Draw start and end points
            if len(trail) >= 2:
                cv2.circle(img, trail[0], 5, (0, 0, 255), -1)  # Red start
                cv2.circle(img, trail[-1], 5, (255, 0, 0), -1)  # Blue end
    
    def draw_all_trails(self, img, colors: Optional[Dict[int, Tuple[int, int, int]]] = None) -> None:
        """Draw trails for all active fingers."""
        default_colors = {
            4: (255, 0, 0),    # Thumb - Red
            8: (0, 255, 0),    # Index - Green  
            12: (0, 0, 255),   # Middle - Blue
            16: (255, 255, 0), # Ring - Yellow
            20: (255, 0, 255)  # Pinky - Magenta
        }
        
        colors = colors or default_colors
        
        for finger_id in self.finger_trails:
            color = colors.get(finger_id, (128, 128, 128))  # Gray default
            self.draw_trail(img, finger_id, color)
    
    def get_trail_info(self, finger_id: int) -> Optional[Dict[str, Any]]:
        """Get current trail information for a finger."""
        if not self._is_valid_trail(finger_id):
            return None
        
        trail = list(self.finger_trails[finger_id])
        times = list(self.finger_times[finger_id])
        
        return {
            'finger_id': finger_id,
            'trail_length': len(trail),
            'current_position': trail[-1] if trail else None,
            'trail_duration': times[-1] - times[0] if len(times) >= 2 else 0,
            'is_active': time.time() - self.last_update_time.get(finger_id, 0) < 0.1
        }
    
    def get_all_active_fingers(self) -> List[int]:
        """Get list of all fingers with active trails."""
        current_time = time.time()
        return [finger_id for finger_id, last_time in self.last_update_time.items()
                if current_time - last_time < 0.5]