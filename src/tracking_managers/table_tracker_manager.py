from trackers.table_tracker import TableTracker

from .tracker_manager import TrackerManager


class TableTrackerManager(TrackerManager):
    """
    Manages TableTracker instances.
    Unlike hand manager, typically only tracks a single table instance.
    Implements velocity-based prediction and expanding search areas.
    Tables are typically stationary in world coordinates, so their apparent motion
    is primarily due to camera movement which is compensated for.
    """

    def __init__(
        self,
        iou_threshold=0.2,
        min_confidence_frames=3,
        max_lost_frames=5,
        detection_threshold=0.7,
    ):
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=True,  # Tables are single-instance trackers
            motion_history_max_len=8,  # Longer motion history for tables (more stable)
            target_labels=["dining table", "table"],  # Labels to detect
        )

    def _extract_bbox_from_detection(self, detection):
        """Extract bounding box from table detection"""
        return detection["box"]

    def _create_tracker_from_detection(self, detection, frame_shape):
        """Create a new TableTracker from a detection"""
        return TableTracker(
            initial_box=detection["box"],
            frame_shape=frame_shape,
            initial_confidence=detection["confidence"],
            min_confidence_frames=self.min_confidence_frames,
            max_lost_frames=self.max_lost_frames,
        )

    def _update_tracker_with_detection(self, tracker, detection):
        """Update a TableTracker with new detection"""
        tracker.update(
            detection["box"],
            detection_confidence=detection["confidence"],
        )

    def get_table_bounds(self):
        """Get the bounding box of the tracked table if confident"""
        return self.get_bounds()

    def get_primary_tracker(self):
        """
        Get the primary table tracker.

        Returns:
            TableTracker or None: The primary tracker if one exists, None otherwise
        """
        if not self.trackers:
            return None

        # Since we're using track_single_instance=True, we should only have one tracker
        # But just to be safe, return the most confident one
        return (
            max(self.trackers, key=lambda tracker: tracker.last_confidence)
            if self.trackers
            else None
        )
