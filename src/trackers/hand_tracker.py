import cv2
import numpy as np

from trackers.object_tracker import ObjectTracker, TrackerConfig


class HandTracker(ObjectTracker):
    """
    Specific implementation for tracking a hand.
    Defines a region around the hand where balls might be detected.
    """

    def __init__(
        self, 
        initial_box, 
        frame_shape, 
        initial_confidence, 
        ball_region_expansion=1.3,
        **kwargs
    ):
        config = TrackerConfig(
            tracker_type="Hand",
            color=(255, 0, 0),  # Blue in BGR format
            position_stability_factor=0.3,  # Moderate stability for hands which move
            use_x_distance_only=True,  # Use x-direction for hand distance measurements
            velocity_smoothing=0.8,  # Higher smoothing for hands
            velocity_threshold=2.0,  # Higher threshold to reduce jitter
        )
        super().__init__(
            initial_box,
            frame_shape,
            initial_confidence=initial_confidence,
            config=config,
            **kwargs
        )
        
        # Hand-specific properties
        self.ball_region_expansion = ball_region_expansion
        self.ball_region = None
        
        # Initialize ball region
        if self.ball_region_expansion is not None:
            self.ball_region = self._calculate_ball_region()
            
    def _calculate_ball_region(self):
        """
        Calculate a region around the hand where balls might be detected.
        """
        if self.ball_region_expansion is None:
            return None
            
        center = self.center.copy()
        width = self.width * self.ball_region_expansion
        height = self.height * self.ball_region_expansion
        
        # Create a symmetrical region
        x1 = center[0] - width * 0.5
        x2 = center[0] + width * 0.5
        
        # Expand more upward than downward for hand tracking
        y1 = center[1] - height * 0.7
        y2 = center[1] + height * 0.3
        
        # Create and clip the region
        region = np.array([x1, y1, x2, y2])
        return self._clip_to_frame(region)
        
    def _custom_update(self, **kwargs):
        """Update ball region when hand position changes"""
        if self.ball_region_expansion is not None:
            self.ball_region = self._calculate_ball_region()
    
    def _custom_predict(self):
        """Update ball region during prediction steps"""
        if self.ball_region_expansion is not None:
            self.ball_region = self._calculate_ball_region()
            
    def _extend_state(self, state):
        """Add ball region to state dictionary"""
        if self.ball_region is not None:
            state["ball_region"] = self.ball_region.astype(int).tolist()
            
    def _custom_draw(self, frame, show_search_box):
        """Draw ball region if showing search box"""
        if show_search_box and self.ball_region is not None:
            br = self.ball_region.astype(int)
            cv2.rectangle(
                frame,
                (br[0], br[1]),
                (br[2], br[3]),
                (0, 255, 255),  # Yellow for ball region
                1,
                cv2.LINE_DASH
            )
            cv2.putText(
                frame,
                "Ball Region",
                (br[0], br[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            ) 