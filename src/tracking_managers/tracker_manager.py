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
        target_labels=None,  # Labels this tracker looks for
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
        self.target_labels = target_labels or []  # Labels this tracker looks for

        # Camera motion estimation
        self.camera_motion = np.zeros(2, dtype=float)
        self.motion_history = []  # Store recent motion vectors for smoothing
        self.motion_history_max_len = (
            motion_history_max_len  # Maximum history length for motion estimation
        )
        # For cup tracking where relative position is important, we want to prioritize
        # position (distance) over IoU since cups look similar
        self.cost_weight_iou = 0.3  # Weight for IoU in cost function (reduced for cups)
        self.cost_weight_distance = 0.7  # Weight for distance (increased for cups)

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

    def _calculate_centroid(self, box):
        """Helper to calculate the center of a bounding box."""
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def match_detections_to_trackers(self, detections):
        """
        Match detections to trackers using a combined cost of IoU and centroid distance,
        solved with the Hungarian algorithm. Ensures one detection matches at most one tracker.

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

        # Calculate frame diagonal for distance normalization
        frame_height, frame_width = self.last_frame_shape[:2] if self.last_frame_shape else (1, 1)
        frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        # Avoid division by zero if frame size is degenerate
        frame_diagonal = max(frame_diagonal, 1.0)

        # Calculate cost matrix using combined cost
        cost_matrix = np.full((len(self.trackers), len(detections)), np.inf)

        # Track historical assignment distances to use for resolving conflicts
        historical_distances = np.zeros((len(self.trackers), len(detections)))

        for t_idx, tracker in enumerate(self.trackers):
            # Tracker's predicted center (predict() was called before this)
            tracker_predicted_center = tracker.center
            tracker_predicted_box = tracker.box # The box after prediction

            for d_idx, det in enumerate(detections):
                # Extract bbox and calculate center from detection
                det_bbox = self._extract_bbox_from_detection(det)
                det_center = self._calculate_centroid(det_bbox)

                # Calculate IoU between tracker's *predicted* box and detection box
                iou = tracker.calculate_iou(tracker_predicted_box, det_bbox)

                # Calculate Euclidean distance between predicted center and detection center
                distance = np.linalg.norm(tracker_predicted_center - det_center)
                normalized_distance = distance / frame_diagonal
                
                # Store the raw distance for conflict resolution
                historical_distances[t_idx, d_idx] = distance

                # For objects that are very close to each other, IoU could be confusing
                # So we'll rely more on the distance for these cases
                
                # Only allow matches where the normalized distance is below a reasonable threshold
                # This prevents distant matches even when IoU might indicate an overlap
                max_distance_threshold = 0.2  # 20% of the frame diagonal as a max distance
                
                if normalized_distance <= max_distance_threshold:
                    # If IoU is adequate, use combined cost
                    if iou >= self.iou_threshold:
                        # Combined cost: lower is better
                        cost = (self.cost_weight_iou * (1.0 - iou) +
                                self.cost_weight_distance * normalized_distance)
                    else:
                        # If IoU is below threshold but distance is small enough,
                        # still consider the match but penalize it
                        cost = (self.cost_weight_iou * 1.0 + 
                                self.cost_weight_distance * normalized_distance)
                    
                    cost_matrix[t_idx, d_idx] = cost

        # Check if cost matrix is all infinite (no matches meet our criteria)
        if np.all(np.isinf(cost_matrix)):
            # No valid assignments can be made - all trackers will be unmatched
            matched_indices = []
            unmatched_trackers = list(range(len(self.trackers)))
            unmatched_detections = list(range(len(detections)))
            return matched_indices, unmatched_trackers, unmatched_detections

        # Some valid matches might exist - run Hungarian algorithm
        # Replace np.inf with a large finite number for the algorithm
        finite_cost_matrix = np.where(np.isinf(cost_matrix), 1e10, cost_matrix)

        # Use Hungarian algorithm to find optimal assignments
        row_ind, col_ind = linear_sum_assignment(finite_cost_matrix)

        # Filter out assignments with excessive cost
        # 1.0 is a reasonable maximum as it represents the worst case scenario
        # for the combined iou (0) and normalized distance (up to our max threshold)
        cost_threshold = 0.85  # Slightly reduced from 1.0 to be more selective
        
        # Process matches in order of cost (best matches first)
        # This ensures that when there are multiple trackers competing for the same detection,
        # the one with the lowest cost (best match) gets it
        potential_matches = [(r, c, cost_matrix[r, c]) for r, c in zip(row_ind, col_ind)
                             if cost_matrix[r, c] <= cost_threshold]
        
        # Sort by cost (lowest first)
        potential_matches.sort(key=lambda x: x[2])
        
        # Track which detections and trackers have been assigned
        assigned_detections = set()
        assigned_trackers = set()
        matched_indices = []
        
        # Resolve conflicts: assign detections to trackers in order of match quality
        for r, c, cost in potential_matches:
            # Skip if this tracker or detection is already assigned
            if r in assigned_trackers or c in assigned_detections:
                # If this tracker is unassigned but the detection is already taken,
                # we have a conflict. Check if this tracker has a better historical claim
                if r not in assigned_trackers and c in assigned_detections:
                    # Find which tracker currently has this detection
                    for existing_r, existing_c in matched_indices:
                        if existing_c == c:
                            # Compare historical distances to see who has better claim
                            if historical_distances[r, c] < historical_distances[existing_r, c] * 0.8:
                                # This tracker has a much better claim - swap assignments
                                matched_indices.remove((existing_r, existing_c))
                                assigned_trackers.remove(existing_r)
                                matched_indices.append((r, c))
                                assigned_trackers.add(r)
                            break
                continue
                
            # This is a valid assignment
            matched_indices.append((r, c))
            assigned_detections.add(c)
            assigned_trackers.add(r)
        
        # Generate unmatched trackers and detections based on assignments
        unmatched_trackers = [i for i in range(len(self.trackers)) if i not in assigned_trackers]
        unmatched_detections = [i for i in range(len(detections)) if i not in assigned_detections]

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

    def draw_trackers(self, frame, show_search_box=False):
        """Draw all active trackers onto the frame"""
        for tracker in self.trackers:
            frame = tracker.draw(frame, show_search_box=show_search_box)
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

    def process_detr_results(self, results, model, frame_shape):
        """
        Process DETR detection results to extract relevant objects based on target_labels.
        
        Args:
            results: DETR post-processed results
            model: DETR model (for id2label mapping)
            frame_shape: Tuple (height, width) of the current frame
            
        Returns:
            tracker_state: Current state of all trackers after processing
        """
        # Extract detections that match our target labels
        raw_detections = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            box_coords = [round(i) for i in box.tolist()]
            label_name = model.config.id2label[label.item()]
            
            if label_name in self.target_labels and score_val >= self.detection_threshold:
                raw_detections.append({"box": box_coords, "confidence": score_val, "label": label_name})
        
        # Update trackers with these detections
        return self.update(raw_detections, frame_shape)
