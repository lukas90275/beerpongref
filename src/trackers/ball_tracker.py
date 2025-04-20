import cv2
import numpy as np

from trackers.object_tracker import ObjectTracker


class BallTracker(ObjectTracker):
    """
    Specific implementation for tracking a ping pong ball.
    Tracks which region the ball was initially detected in.
    """

    def __init__(
        self, 
        initial_box, 
        frame_shape, 
        initial_confidence,
        source_region_id=None,  # ID of the region where the ball was initially detected
        position_stability_factor=0.1,  # Light stability for balls which move quickly
        use_x_distance_only=False,  # Use full 2D distance for balls
        velocity_smoothing=0.5,  # Lower smoothing for faster response to ball movement
        **kwargs
    ):
        super().__init__(
            initial_box,
            frame_shape,
            initial_confidence=initial_confidence,
            tracker_type="Ball",
            color=(0, 255, 255),  # Yellow for balls
            position_stability_factor=position_stability_factor,
            velocity_smoothing=velocity_smoothing,
            velocity_threshold=0.5,  # Lower threshold to capture small movements
            use_x_distance_only=use_x_distance_only,
            **kwargs
        )
        self.source_region_id = source_region_id
        
        # Track diameter for size information
        self.diameter = max(self.width, self.height)  # Estimated diameter in pixels
        
        # Track position history for trajectory analysis
        self.trajectory_positions = []  # List of center positions
        self.trajectory_max_length = 10  # Number of positions to track
        
        # Store initial position
        if hasattr(self, 'center'):
            self.trajectory_positions.append(self.center.copy())

    def update(self, detection_box, detection_confidence=1.0):
        """
        Update tracker with new detection
        """
        # Store previous position for reference
        prev_position = self.center.copy() if hasattr(self, 'center') else None
        
        # Call parent update
        super().update(detection_box, detection_confidence)
        
        # Update diameter estimate
        self.diameter = max(self.width, self.height)
        
        # Update trajectory history
        self.trajectory_positions.append(self.center.copy())
        if len(self.trajectory_positions) > self.trajectory_max_length:
            self.trajectory_positions.pop(0)

    def get_state(self):
        """Add ball-specific information to the state dictionary."""
        state = super().get_state()
        state["source_region_id"] = self.source_region_id
        state["diameter"] = self.diameter
        return state
        
    def draw(self, frame, show_search_box=False):
        """
        Draw the ball tracker's state onto the frame with enhanced visibility.
        """
        b = self.box.astype(int)
        
        # Draw search box with higher visibility if requested
        if show_search_box:
            sb = self.search_box.astype(int)
            cv2.rectangle(
                frame,
                (sb[0], sb[1]),
                (sb[2], sb[3]),
                (0, 0, 255),  # Change to RED for high visibility
                2,  # Thicker line
                cv2.LINE_AA
            )
            # Add search box label with high visibility
            cv2.putText(
                frame,
                "Ball Search Area",
                (sb[0], sb[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),  # RED text
                2
            )
        
        # Draw brighter yellow outline
        cv2.rectangle(
            frame,
            (b[0], b[1]),
            (b[2], b[3]),
            (0, 255, 255),  # Bright yellow
            2  # Thicker line
        )
        
        # Draw ball label
        cv2.putText(
            frame,
            f"Ball {self.diameter:.0f}px",
            (b[0], b[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),  # Yellow
            2
        )
        
        # Draw trajectory if showing search box
        if show_search_box and len(self.trajectory_positions) >= 2:
            for i in range(1, len(self.trajectory_positions)):
                pt1 = self.trajectory_positions[i-1].astype(int)
                pt2 = self.trajectory_positions[i].astype(int)
                cv2.line(frame, tuple(pt1), tuple(pt2), (0, 200, 255), 2)
        
        return frame 