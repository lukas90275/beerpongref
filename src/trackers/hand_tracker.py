import cv2
import numpy as np

from trackers.object_tracker import ObjectTracker


class HandTracker(ObjectTracker):
    """
    Specific implementation for tracking a hand.
    Uses parent class for matching score (IoU). Stores handedness.
    """

    def __init__(
        self, 
        initial_box, 
        frame_shape, 
        hand_type, 
        initial_confidence, 
        position_stability_factor=0.3,  # Moderate stability for hands which move
        use_x_distance_only=True,  # Use x-direction for hand distance measurements
        ball_region_expansion=1.3,  # Factor to expand hand box for ball detection region
        **kwargs
    ):
        super().__init__(
            initial_box,
            frame_shape,
            initial_confidence=initial_confidence,
            tracker_type="Hand",
            color=(255, 0, 0),
            position_stability_factor=position_stability_factor,
            velocity_smoothing=0.8,  # Higher smoothing for hands
            velocity_threshold=2.0,  # Higher threshold to reduce jitter
            use_x_distance_only=use_x_distance_only,
            **kwargs
        )
        self.hand_type = hand_type
        self.ball_region_expansion = ball_region_expansion
        self.ball_region = self._calculate_ball_region()
        
        # Track motion state for gesture detection
        self.is_moving = False
        self.motion_threshold = max(5.0, self.width * 0.15)  # Scale threshold by hand size

    def _calculate_ball_region(self):
        """
        Calculate a region around the hand where balls might be detected.
        Expands the hand box by the ball_region_expansion factor.
        """
        center = self.center.copy()
        width = self.width * self.ball_region_expansion
        height = self.height * self.ball_region_expansion
        
        # For hand regions, we're more interested in the area in front of and above the hand
        # Adjust the region based on hand type
        if self.hand_type == "left":
            # Expand more to the right for left hand
            x1 = center[0] - width * 0.3
            x2 = center[0] + width * 0.7
        else:
            # Expand more to the left for right hand
            x1 = center[0] - width * 0.7
            x2 = center[0] + width * 0.3
        
        # Expand more upward
        y1 = center[1] - height * 0.7
        y2 = center[1] + height * 0.3
        
        # Create and clip the region
        region = np.array([x1, y1, x2, y2])
        return self._clip_to_frame(region)

    def update(self, detection_box, detection_confidence=1.0, hand_type=None):
        """
        Update tracker, including handedness if provided.
        Also update motion state for the hand and ball detection region.
        """
        # Store previous position for motion calculation
        prev_position = self.center.copy() if hasattr(self, 'center') else None
        
        # Call parent update
        super().update(detection_box, detection_confidence)
        
        # Update handedness if provided
        if hand_type and hand_type != self.hand_type:
            self.hand_type = hand_type
            
        # Update motion state if we have previous position, using appropriate distance method
        if prev_position is not None:
            if self.use_x_distance_only:
                motion = abs(self.center[0] - prev_position[0])
            else:
                motion = np.linalg.norm(self.center - prev_position)
                
            # Scale motion threshold by hand size
            self.motion_threshold = max(5.0, self.width * 0.15)
            self.is_moving = motion > self.motion_threshold
        
        # Update ball detection region
        self.ball_region = self._calculate_ball_region()

    def get_state(self):
        """Add hand_type and ball_region to the state dictionary."""
        state = super().get_state()
        state["hand_type"] = self.hand_type
        state["is_moving"] = self.is_moving
        state["motion_threshold"] = self.motion_threshold
        state["ball_region"] = self.ball_region.astype(int).tolist()
        return state
        
    def draw(self, frame, show_search_box=False):
        """
        Draw the hand tracker's state onto the frame.
        Add hand-specific visualizations.
        """
        # Call parent draw method first
        frame = super().draw(frame, show_search_box)
        
        # Add hand type label
        b = self.box.astype(int)
        cv2.putText(
            frame,
            f"{self.hand_type}",
            (b[0], b[1] - 30),  # Position above tracker_type label
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if self.is_confident else (0, 0, 255),
            2,
        )
        
        # Draw ball detection region if showing search boxes
        if show_search_box:
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
        
        return frame
