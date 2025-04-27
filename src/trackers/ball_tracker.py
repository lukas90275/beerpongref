import cv2
import numpy as np

from trackers.object_tracker import ObjectTracker, TrackerConfig


class BallTracker(ObjectTracker):
    """
    Specific implementation for tracking a ping pong ball.
    Uses object tracker with trajectory tracking and specialized visualization.
    """

    def __init__(
        self, 
        initial_box, 
        frame_shape, 
        initial_confidence,
        source_region_id=None,
        **kwargs
    ):
        config = TrackerConfig(
            tracker_type="Ball",
            color=(0, 255, 255),  # Yellow for balls
            position_stability_factor=0.1,  # Light stability for balls which move quickly
            use_x_distance_only=False,  # Use full 2D distance for balls
            velocity_smoothing=0.5,  # Lower smoothing for faster response to ball movement
            velocity_threshold=0.5,  # Lower threshold to capture small movements
            trajectory_len=10,  # Track trajectory for visualization
            source_region_id=source_region_id
        )
        super().__init__(
            initial_box,
            frame_shape,
            initial_confidence=initial_confidence,
            config=config,
            **kwargs
        ) 