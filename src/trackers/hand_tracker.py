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
        
        # Track motion state for gesture detection
        self.is_moving = False
        self.motion_threshold = max(5.0, self.width * 0.15)  # Scale threshold by hand size

    def update(self, detection_box, detection_confidence=1.0, hand_type=None):
        """
        Update tracker, including handedness if provided.
        Also update motion state for the hand.
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

    def get_state(self):
        """Add hand_type to the state dictionary."""
        state = super().get_state()
        state["hand_type"] = self.hand_type
        state["is_moving"] = self.is_moving
        state["motion_threshold"] = self.motion_threshold
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
        
        return frame
