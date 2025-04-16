from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm


class TrackerManager(ABC):
    """
    Abstract base class for tracker managers.
    Manages multiple tracker instances of a specific type.
    Handles assignment of detections to trackers and lifecycle management.
    """

    def __init__(
        self,
        iou_threshold=0.3,
        min_confidence_frames=3,
        max_lost_frames=5,
        detection_threshold=0.7,
        track_single_instance=False,
        motion_history_max_len=5,
    ):
        self.trackers = []
        self.iou_threshold = iou_threshold
        self.min_confidence_frames = min_confidence_frames
        self.max_lost_frames = max_lost_frames
        self.detection_threshold = detection_threshold
        self._next_tracker_id = 0  # Simple incrementing ID
        self.last_frame_shape = None
        self.track_single_instance = (
            track_single_instance  # Whether to track only one instance (for tables)
        )

        # Camera motion estimation
        self.camera_motion = np.zeros(2, dtype=float)
        self.motion_history = []  # Store recent motion vectors for smoothing
        self.motion_history_max_len = (
            motion_history_max_len  # Maximum history length for motion estimation
        )

    def reset(self):
        """Reset all trackers - useful when needing to clear current tracking state"""
        self.trackers = []
        self.camera_motion = np.zeros(2, dtype=float)
        self.motion_history = []

    def get_filtered_detections(self, detections):
        """
        Filter detections based on confidence threshold.
        Only returns detections that meet the minimum confidence threshold.

        Args:
            detections: List of detection dictionaries with 'confidence' field

        Returns:
            List of detections meeting the confidence threshold
        """
        return [
            d
            for d in detections
            if self._get_detection_confidence(d) >= self.detection_threshold
        ]

    def update(self, detections, frame_shape):
        """
        Update trackers with new detections for the current frame.

        Args:
            detections: List of detection dictionaries
            frame_shape: Tuple (height, width) of the current frame

        Returns:
            List of active tracker states or single tracker state if track_single_instance=True
        """
        # Store frame shape for coordinate clipping
        self.last_frame_shape = frame_shape

        # Filter out low confidence detections
        high_confidence_detections = self.get_filtered_detections(detections)

        # If no high-confidence detections, mark all trackers as lost
        if not high_confidence_detections and self.trackers:
            for tracker in self.trackers:
                tracker.mark_lost()

            # Remove trackers that have been lost for too long
            self.remove_lost_trackers()

            # Return the state of remaining trackers or None
            if self.track_single_instance:
                return self.trackers[0].get_state() if self.trackers else None
            return [t.get_state() for t in self.trackers]

        # 1. Predict next state for existing trackers
        self.predict_all_trackers()

        # 2. Match detections to trackers
        matched_indices, unmatched_trackers, unmatched_detections = (
            self.match_detections_to_trackers(high_confidence_detections)
        )

        # 3. Update matched trackers
        for t_idx, d_idx in matched_indices:
            self._update_tracker_with_detection(
                self.trackers[t_idx], high_confidence_detections[d_idx]
            )

        # 4. Mark unmatched trackers as lost
        for t_idx in unmatched_trackers:
            self.trackers[t_idx].mark_lost()

        # 5. Handle unmatched detections differently based on tracking mode
        if self.track_single_instance:
            # For single instance tracking (like tables)
            # Only create a new tracker if none exists or all are lost
            if not self.trackers or all(t.is_lost for t in self.trackers):
                best_detection = self._find_best_detection(high_confidence_detections)
                if best_detection:
                    new_tracker = self._create_tracker_from_detection(
                        best_detection, frame_shape
                    )
                    self.trackers.append(new_tracker)
        else:
            # For multi-instance tracking (like hands)
            # Create new trackers for each unmatched detection
            for d_idx in unmatched_detections:
                new_tracker = self._create_tracker_from_detection(
                    high_confidence_detections[d_idx], frame_shape
                )
                self.trackers.append(new_tracker)

        # 6. Remove lost trackers
        self.remove_lost_trackers()

        # 7. Return states differently based on tracking mode
        if self.track_single_instance:
            # For single instance tracking (like tables)
            if self.trackers and not self.trackers[0].is_lost or self.trackers:
                return self.trackers[0].get_state()
            return None
        # For multi-instance tracking (like hands)
        return [t.get_state() for t in self.trackers]

    def predict_all_trackers(self):
        """Predict next state for all existing trackers"""
        # First estimate camera motion from previous frame if we have enough trackers
        self._estimate_camera_motion()

        # Now predict with camera motion compensation
        for tracker in self.trackers:
            # Apply camera motion compensation - subtract camera motion from object velocity
            tracker.velocity = tracker.velocity - self.camera_motion
            tracker.predict()

    def _estimate_camera_motion(self):
        """
        Estimate global camera motion by analyzing velocities of all tracked objects.
        Assumes that majority of the scene is stationary, so average motion
        vector represents camera movement.
        """
        # Skip if we don't have enough trackers (need at least 2 for stable estimation)
        if len(self.trackers) < 2:
            self.camera_motion = np.zeros(2, dtype=float)
            return

        # Get motion vectors from confident trackers only
        velocities = []
        for tracker in self.trackers:
            if tracker.is_confident and tracker.lost_frames == 0:
                velocities.append(tracker.velocity)

        if not velocities:
            self.camera_motion = np.zeros(2, dtype=float)
            return

        # Convert to numpy array for easier processing
        velocities = np.array(velocities)

        # Calculate median of all velocities (more robust than mean)
        # This represents the common motion shared by objects (likely camera motion)
        median_motion = np.median(velocities, axis=0)

        # Add to motion history
        self.motion_history.append(median_motion)
        if len(self.motion_history) > self.motion_history_max_len:
            self.motion_history.pop(0)  # Remove oldest

        # Use average of recent motion history for smoothing
        if self.motion_history:
            self.camera_motion = np.mean(self.motion_history, axis=0)
        else:
            self.camera_motion = np.zeros(2, dtype=float)

    def match_detections_to_trackers(self, detections):
        """
        Match detections to trackers using IoU and Hungarian algorithm.

        Args:
            detections: List of detection dictionaries

        Returns:
            matched_indices: List of (tracker_idx, detection_idx) tuples
            unmatched_trackers: List of unmatched tracker indices
            unmatched_detections: List of unmatched detection indices
        """
        if not self.trackers or not detections:
            # Handle cases with no trackers or no detections
            matched_indices = []
            unmatched_detections = list(range(len(detections)))
            unmatched_trackers = list(range(len(self.trackers)))
            return matched_indices, unmatched_trackers, unmatched_detections

        # Calculate cost matrix (negative IoU, higher IoU = lower cost)
        cost_matrix = np.full((len(self.trackers), len(detections)), np.inf)

        for t_idx, tracker in enumerate(self.trackers):
            for d_idx, det in enumerate(detections):
                # Extract bbox from detection (implementation may vary)
                det_bbox = self._extract_bbox_from_detection(det)

                # Calculate IoU between tracker box and detection
                iou = tracker.calculate_iou(tracker.box, det_bbox)
                if iou >= self.iou_threshold:
                    cost_matrix[t_idx, d_idx] = 1.0 - iou  # Cost is 1 - IoU

        # Check if cost matrix is all infinite (no matches exceed threshold)
        if np.all(np.isinf(cost_matrix)):
            # No valid assignments can be made - all trackers will be unmatched
            matched_indices = []
            unmatched_trackers = list(range(len(self.trackers)))
            unmatched_detections = list(range(len(detections)))
            return matched_indices, unmatched_trackers, unmatched_detections

        # Some valid matches exist - run Hungarian algorithm
        # Replace np.inf with a large finite number for the algorithm
        finite_cost_matrix = np.where(np.isinf(cost_matrix), 1e10, cost_matrix)

        # Use Hungarian algorithm to find optimal assignments
        row_ind, col_ind = linear_sum_assignment(finite_cost_matrix)

        # Filter out matches with cost > threshold (i.e., IoU < threshold)
        matched_indices = []
        unmatched_trackers = list(range(len(self.trackers)))
        unmatched_detections = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (
                1.0 - self.iou_threshold
            ):  # Check if cost is low enough
                matched_indices.append((r, c))
                unmatched_trackers.remove(r)
                unmatched_detections.remove(c)

        return matched_indices, unmatched_trackers, unmatched_detections

    def _find_best_detection(self, detections):
        """Find the detection with highest confidence above threshold"""
        if not detections:
            return None

        best_detection = None
        highest_confidence = self.detection_threshold  # Minimum threshold

        for detection in detections:
            detection_confidence = self._get_detection_confidence(detection)
            if detection_confidence > highest_confidence:
                highest_confidence = detection_confidence
                best_detection = detection

        return best_detection

    def _get_detection_confidence(self, detection):
        """
        Extract confidence from a detection.
        Default implementation assumes 'confidence' key in detection dict.
        Override if confidence is stored differently.
        """
        return detection.get("confidence", 0.0)

    @abstractmethod
    def _extract_bbox_from_detection(self, detection):
        """
        Extract bounding box from detection dictionary.
        Implementation depends on detection format.

        Args:
            detection: Detection dictionary

        Returns:
            bbox: List or array [x1, y1, x2, y2]
        """
        pass

    @abstractmethod
    def _create_tracker_from_detection(self, detection, frame_shape):
        """
        Create a new tracker from a detection.
        Implementation depends on tracker type.

        Args:
            detection: Detection dictionary
            frame_shape: Frame dimensions

        Returns:
            tracker: New tracker instance
        """
        pass

    @abstractmethod
    def _update_tracker_with_detection(self, tracker, detection):
        """
        Update a tracker with a detection.
        Implementation depends on tracker type and detection format.

        Args:
            tracker: Tracker instance
            detection: Detection dictionary
        """
        pass

    def remove_lost_trackers(self):
        """Remove trackers that are marked as lost"""
        self.trackers = [t for t in self.trackers if not t.is_lost]

    def draw_trackers(self, frame):
        """Draw all active trackers onto the frame"""
        for tracker in self.trackers:
            frame = tracker.draw(frame)
        return frame

    def get_confident_trackers(self):
        """Get the state of trackers that are currently confident"""
        return [t.get_state() for t in self.trackers if t.is_confident]

    def get_bounds(self):
        """Get the bounding box of the tracked object if confident"""
        if self.trackers and self.trackers[0].is_confident:
            return self.trackers[0].box.astype(int).tolist()

        # If we have any tracker at all, return its bounds even if not confident
        if self.trackers:
            return self.trackers[0].box.astype(int).tolist()

        return None
