import cv2
import numpy as np

from trackers.object_tracker import ObjectTracker, TrackerConfig


class CupTracker(ObjectTracker):
    """
    Specific implementation for tracking cups on the beer pong table.
    Cups are mostly stationary and should only move when hit or manipulated.
    """

    def __init__(
        self, 
        initial_box, 
        frame_shape, 
        search_region_bounds=None,
        **kwargs
    ):
        config = TrackerConfig(
            tracker_type="Cup",
            color=(0, 0, 255),  # Red in BGR format
            search_expansion_factor=1.2,  # Reduced because cups don't move much
            velocity_smoothing=0.95,  # Increased to reduce jitter
            velocity_threshold=3.0,  # Increased to ignore small movements
            position_stability_factor=0.95,  # High stability for stationary cups
            use_x_distance_only=True,  # Use only x-distance for cup matching (side view)
            search_region_bounds=search_region_bounds,
            detect_aspect_ratio_change=True,  # Track if cup is knocked over
            aspect_ratio_change_factor=1.5,
            aspect_ratio_history_len=5
        )
        super().__init__(
            initial_box,
            frame_shape,
            config=config,
            **kwargs
        ) 