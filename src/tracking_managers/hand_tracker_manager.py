from trackers.hand_tracker import HandTracker

from .tracker_manager import TrackerManager


class HandTrackerManager(TrackerManager):
    """
    Manages multiple HandTracker instances.
    Assigns new detections to existing trackers using IoU and the Hungarian algorithm.
    Creates new trackers for unmatched detections and removes lost trackers.
    """

    def __init__(
        self,
        iou_threshold=0.3,
        min_confidence_frames=3,
        max_lost_frames=8,
        detection_threshold=0.5,
    ):
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=False,  # Hands are multi-instance trackers
            motion_history_max_len=5,  # Default motion history for hands
        )

    def _extract_bbox_from_detection(self, detection):
        """Extract bounding box from hand detection"""
        return detection["bbox"]

    def _create_tracker_from_detection(self, detection, frame_shape):
        """Create a new HandTracker from a detection"""
        return HandTracker(
            initial_box=detection["bbox"],
            frame_shape=frame_shape,
            hand_type=detection["hand_type"],
            initial_confidence=detection["confidence"],
            min_confidence_frames=self.min_confidence_frames,
            max_lost_frames=self.max_lost_frames,
        )

    def _update_tracker_with_detection(self, tracker, detection):
        """Update a HandTracker with new detection"""
        tracker.update(
            detection["bbox"],
            detection_confidence=detection["confidence"],
            hand_type=detection["hand_type"],
        )

    def get_confident_hands(self):
        """Get the state of hand trackers that are currently confident"""
        return self.get_confident_trackers()
