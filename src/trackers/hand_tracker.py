
from trackers.object_tracker import ObjectTracker


class HandTracker(ObjectTracker):
    """
    Specific implementation for tracking a hand.
    Uses parent class for matching score (IoU). Stores handedness.
    """

    def __init__(
        self, initial_box, frame_shape, hand_type, initial_confidence, **kwargs
    ):
        super().__init__(
            initial_box,
            frame_shape,
            initial_confidence=initial_confidence,
            max_lost_frames=8,
            min_confidence_frames=3,
            tracker_type="Hand",
            color=(255, 0, 0),
        )
        self.hand_type = hand_type

    def update(self, detection_box, detection_confidence=1.0, hand_type=None):
        """Update tracker, including handedness if provided."""
        super().update(detection_box, detection_confidence)
        if hand_type and hand_type != self.hand_type:
            self.hand_type = hand_type  # Update handedness if detection provides it

    def get_state(self):
        """Add hand_type to the state dictionary."""
        state = super().get_state()
        state["hand_type"] = self.hand_type
        return state
