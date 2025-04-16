
from trackers.object_tracker import ObjectTracker


class TableTracker(ObjectTracker):
    """
    Specific implementation for tracking a table.
    Uses parent class for matching score (IoU). Assumes only one table.
    Modified to use velocity-based prediction for more robust tracking.
    Tables are typically stationary in the world frame, so velocity is
    mostly due to camera motion which is compensated at the manager level.
    """

    def __init__(self, initial_box, frame_shape, **kwargs):
        super().__init__(
            initial_box,
            frame_shape,
            max_lost_frames=5,
            min_confidence_frames=3,
            tracker_type="Table",
            color=(0, 255, 0),
            search_expansion_factor=1.3,
            velocity_smoothing=0.9,
            velocity_threshold=2.0,
        )
