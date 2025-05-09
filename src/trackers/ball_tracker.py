import uuid  # For unique IDs
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from trackers.object_tracker import ObjectTracker, TrackerConfig


class BallTracker(ObjectTracker):
    """
    Specialized tracker for ping pong balls.
    Inherits from the base ObjectTracker class.
    """

    def __init__(
        self,
        initial_box,
        frame_shape,
        tracker_id=None,
        initial_confidence=1.0,
        source_region_id=None,
        position_stability_factor=0.02,
        min_confidence_frames=1,
        max_lost_frames=20,
        search_expansion_factor=2.0,
        directional_search_expansion=True,
        **kwargs
    ):
        # Create ball-specific tracker configuration
        ball_config = TrackerConfig(
            tracker_type="Ball",
            color=(0, 255, 255),  # Yellow for balls
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            search_expansion_factor=search_expansion_factor,
            position_stability_factor=position_stability_factor,
            velocity_smoothing=0.5,  # Lower smoothing for faster tracking
            velocity_threshold=0.5,  # Lower threshold to detect small movements
            use_x_distance_only=False,  # Use full distance for balls
            trajectory_len=10,  # Keep longer trajectory for balls
            directional_search_expansion=directional_search_expansion,
            source_region_id=source_region_id,
        )

        # Initialize parent class with ball configuration
        super().__init__(
            initial_box=initial_box,
            frame_shape=frame_shape,
            tracker_id=tracker_id,
            initial_confidence=initial_confidence,
            config=ball_config,
            **kwargs
        )
        
        # Ball-specific properties
        self.diameter = max(self.width, self.height)
        self.is_detected_this_frame = True
        self.last_detection_confidence = initial_confidence
        
        # Add size history tracking for better matching
        self.size_history = []
        self.max_size_history = 5
        self.size_history.append((self.width, self.height))
        
    def update(self, detection_box, detection_confidence=1.0, **kwargs):
        """
        Update the ball tracker with new detection information.
        
        Args:
            detection_box: New bounding box [x1, y1, x2, y2]
            detection_confidence: Confidence of the detection
        """
        # Mark as detected this frame
        self.is_detected_this_frame = True
        self.last_detection_confidence = detection_confidence
        
        # Update ball diameter (useful for physics calculations)
        width = detection_box[2] - detection_box[0]
        height = detection_box[3] - detection_box[1]
        self.diameter = max(width, height)
        
        # Call parent class update method
        super().update(detection_box, detection_confidence, **kwargs)
    
    def _extend_state(self, state):
        """
        Add ball-specific fields to tracker state.
        
        Args:
            state: Dictionary to extend with ball-specific fields
        """
        # Add ball-specific fields
        state["is_detected_this_frame"] = self.is_detected_this_frame
        state["diameter"] = self.diameter
            
        # Add size information
        avg_width, avg_height = self.get_average_size()
        state["avg_width"] = avg_width
        state["avg_height"] = avg_height
            
        # Add source region if available
        if self.config.source_region_id is not None:
            state["source_region_id"] = self.config.source_region_id
        
    def _custom_draw(self, frame, show_search_box):
        """
        Add ball-specific visualization elements.
        """
        # Draw ball trajectory if showing detailed debug info
        if show_search_box and len(self.trajectory_positions) > 1:
            # Draw trajectory path
            points = np.array(self.trajectory_positions, dtype=np.int32)
            cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, (0, 255, 255), 2)
            
            # Draw velocity vector
            if not np.all(self.velocity == 0):
                # Calculate endpoint based on velocity
                start_point = tuple(self.center.astype(int))
                # Scale velocity for visibility
                scale = 5.0
                end_point = tuple((self.center + self.velocity * scale).astype(int))
                # Draw arrow
                cv2.arrowedLine(frame, start_point, end_point, (0, 200, 255), 2)
        
    def get_center(self):
        """Get the center point of the ball"""
        return self.center.copy()
        
    def get_search_box(self):
        """
        Get the current search box for the ball.
        
        Returns:
            Tuple (x1, y1, x2, y2) of search box coordinates
        """
        # Predict to update search box based on lost frames and velocity
        self.predict()
        return self.search_box.astype(int)
        
    def get_average_size(self):
        """
        Get the average size of the ball from recent history.
        
        Returns:
            Tuple (avg_width, avg_height) of the ball's average dimensions
        """
        if not self.size_history:
            return (self.width, self.height)
            
        widths, heights = zip(*self.size_history)
        return (sum(widths) / len(widths), sum(heights) / len(heights))
        
    def _custom_update(self, **kwargs):
        """
        Add custom update logic for ball tracker.
        """
        # Update size history
        self.size_history.append((self.width, self.height))
        if len(self.size_history) > self.max_size_history:
            self.size_history.pop(0)
            
    def predict(self):
        """
        Predict the next position of the ball.
        """
        # Reset detection flag at beginning of frame
        self.is_detected_this_frame = False
        
        # Use the normal prediction method from the parent class
        return super().predict() 