import uuid  # For unique IDs
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
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
        tracker_overlap_threshold=0.7,  # IoU threshold to consider trackers overlapping
        min_tracker_separation=0.3,  # Minimum normalized distance between trackers (as fraction of frame diagonal)
        cost_weight_iou=0.3,  # Weight for IoU in cost function
        cost_weight_distance=0.7,  # Weight for distance in cost function
        position_stability_factor=0.0,  # Factor to stabilize object positions (0.0 = off)
        max_position_history=5,  # Maximum number of historical positions to track per object
    ):
        self.trackers = []
        self.iou_threshold = iou_threshold
        self.min_confidence_frames = min_confidence_frames
        self.max_lost_frames = max_lost_frames
        self.detection_threshold = detection_threshold
        self._next_tracker_id = 0  # Simple incrementing ID
        self.last_frame_shape = None
        self.track_single_instance = track_single_instance
        self.target_labels = target_labels or []  # Labels this tracker looks for
        self.tracker_overlap_threshold = tracker_overlap_threshold
        self.min_tracker_separation = min_tracker_separation
        self.cost_weight_iou = cost_weight_iou
        self.cost_weight_distance = cost_weight_distance
        self.position_stability_factor = position_stability_factor
        self.max_position_history = max_position_history

        # Camera motion estimation
        self.camera_motion = np.zeros(2, dtype=float)
        self.motion_history = []  # Store recent motion vectors for smoothing
        self.motion_history_max_len = motion_history_max_len

        # Track tracker statistics over time
        self.tracker_stats = {}  # Tracker ID -> stats dict
        
        # Track object position history for stable identity and merging
        self.position_history = {}  # Region/category -> [(x, y, id), ...]

    def reset(self):
        """Reset all trackers - useful when needing to clear current tracking state"""
        self.trackers = []
        self.camera_motion = np.zeros(2, dtype=float)
        self.motion_history = []
        self.tracker_stats = {}
        self.position_history = {}

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

        # Pre-process detections (allow subclasses to do additional filtering)
        detections = self._pre_process_detections(detections)

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
            
            # Update tracker stats
            tracker_id = str(self.trackers[t_idx].id)
            if tracker_id not in self.tracker_stats:
                self.tracker_stats[tracker_id] = {"matches": 0, "misses": 0}
            self.tracker_stats[tracker_id]["matches"] += 1

        # 4. Mark unmatched trackers as lost
        for t_idx in unmatched_trackers:
            self.trackers[t_idx].mark_lost()
            
            # Update tracker stats
            tracker_id = str(self.trackers[t_idx].id)
            if tracker_id not in self.tracker_stats:
                self.tracker_stats[tracker_id] = {"matches": 0, "misses": 0}
            self.tracker_stats[tracker_id]["misses"] += 1

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
                    
                    # Initialize stats for new tracker
                    self.tracker_stats[str(new_tracker.id)] = {"matches": 1, "misses": 0}
        else:
            # For multi-instance tracking (like hands, cups)
            # Create new trackers for each unmatched detection that isn't too close to existing ones
            for d_idx in unmatched_detections:
                detection = high_confidence_detections[d_idx]
                if self._should_create_new_tracker(detection):
                    new_tracker = self._create_tracker_from_detection(
                        detection, frame_shape
                    )
                    self.trackers.append(new_tracker)
                    
                    # Initialize stats for new tracker
                    self.tracker_stats[str(new_tracker.id)] = {"matches": 1, "misses": 0}

        # 6. Check for and merge overlapping trackers
        self.check_and_merge_overlapping_trackers()

        # 7. Post process trackers (allow subclasses to do additional work)
        self._post_process_trackers()

        # 8. Remove lost trackers
        self.remove_lost_trackers()

        # 9. Return states differently based on tracking mode
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

    def _should_create_new_tracker(self, detection):
        """
        Determine if a new tracker should be created for this detection.
        Prevents creating trackers too close to existing ones.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            bool: True if a new tracker should be created
        """
        # Allow subclasses to implement custom pre-filtering
        if not self._pre_create_tracker_filter(detection):
            return False
            
        # If no existing trackers, always create new one
        if not self.trackers:
            return True
            
        # Get detection box and center
        det_box = self._extract_bbox_from_detection(detection)
        det_center = self._calculate_centroid(det_box)
        
        # Calculate frame diagonal for distance normalization
        frame_height, frame_width = self.last_frame_shape[:2] if self.last_frame_shape else (1, 1)
        frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        
        # Check distance and overlap with all existing trackers
        for tracker in self.trackers:
            # Only check against confident or recently confident trackers
            if tracker.is_confident or tracker.lost_frames <= 2:
                # Check IoU overlap
                iou = tracker.calculate_iou(tracker.box, det_box)
                if iou > self.tracker_overlap_threshold:
                    return False  # Too much overlap with existing tracker
                
                # Check distance
                distance = np.linalg.norm(tracker.center - det_center)
                normalized_distance = distance / frame_diagonal
                
                if normalized_distance < self.min_tracker_separation:
                    return False  # Too close to existing tracker
        
        # Allow subclasses to implement custom post-filtering
        return self._post_create_tracker_filter(detection)

    def check_and_merge_overlapping_trackers(self):
        """
        Check for overlapping trackers and merge them if necessary.
        Prioritizes more confident trackers when merging.
        """
        if len(self.trackers) <= 1:
            return  # Nothing to merge
            
        # Sort trackers by confidence (confident ones first, then by confidence frames)
        sorted_trackers = sorted(
            enumerate(self.trackers),
            key=lambda x: (x[1].is_confident, x[1].confidence_frames),
            reverse=True
        )
        
        # Track which trackers should be removed after merging
        to_remove = set()
        
        # Compare each tracker with the ones after it in the sorted list
        for i, (idx_a, tracker_a) in enumerate(sorted_trackers[:-1]):
            if idx_a in to_remove:
                continue  # Skip if this tracker is already marked for removal
                
            for idx_b, tracker_b in sorted_trackers[i+1:]:
                if idx_b in to_remove:
                    continue  # Skip if this tracker is already marked for removal
                    
                # Calculate IoU between trackers
                iou = tracker_a.calculate_iou(tracker_a.box, tracker_b.box)
                
                # If significant overlap, merge the trackers (keep tracker_a, remove tracker_b)
                if iou > self.tracker_overlap_threshold:
                    # Additional check: If trackers are of different types, don't merge
                    if hasattr(tracker_a, 'tracker_type') and hasattr(tracker_b, 'tracker_type'):
                        if tracker_a.tracker_type != tracker_b.tracker_type:
                            continue

                    # Additional check: Let subclasses decide if these should merge
                    if not self._should_merge_trackers(tracker_a, tracker_b):
                        continue
                    
                    # We might consider a more sophisticated merging strategy here,
                    # but for now we'll just keep the more confident tracker
                    to_remove.add(idx_b)
                    
                    # Merge tracker stats
                    tracker_b_id = str(tracker_b.id)
                    tracker_a_id = str(tracker_a.id)
                    if tracker_b_id in self.tracker_stats and tracker_a_id in self.tracker_stats:
                        self.tracker_stats[tracker_a_id]["matches"] += self.tracker_stats[tracker_b_id]["matches"]
                        # Don't transfer misses - that would penalize the surviving tracker
        
        # Remove merged trackers
        if to_remove:
            self.trackers = [t for i, t in enumerate(self.trackers) if i not in to_remove]

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
        
        # Track IoU values for resolving conflicts
        iou_values = np.zeros((len(self.trackers), len(detections)))

        # Group trackers by confidence level for assignment priority
        confident_trackers = []
        uncertain_trackers = []
        
        for t_idx, tracker in enumerate(self.trackers):
            if tracker.is_confident:
                confident_trackers.append(t_idx)
            else:
                uncertain_trackers.append(t_idx)

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
                iou_values[t_idx, d_idx] = iou

                # Calculate Euclidean distance between predicted center and detection center
                distance = np.linalg.norm(tracker_predicted_center - det_center)
                normalized_distance = distance / frame_diagonal
                
                # Store the raw distance for conflict resolution
                historical_distances[t_idx, d_idx] = distance

                # Let subclasses implement custom matching logic/preferences
                custom_cost = self._calculate_custom_matching_cost(tracker, det, iou, normalized_distance)
                if custom_cost is not None:
                    cost_matrix[t_idx, d_idx] = custom_cost
                    continue
                
                # Only allow matches where the normalized distance is below a reasonable threshold
                # This prevents distant matches even when IoU might indicate an overlap
                max_distance_threshold = 0.2  # 20% of the frame diagonal as a max distance
                
                if normalized_distance <= max_distance_threshold:
                    # Calculate confidence weight - prioritize more confident trackers
                    confidence_weight = 1.0
                    if tracker.is_confident:
                        confidence_weight = 0.7  # Lower cost for confident trackers (better chance of matching)
                    
                    # Check tracker history - prioritize trackers with good track records
                    tracker_id = str(tracker.id)
                    history_weight = 1.0
                    if tracker_id in self.tracker_stats:
                        matches = self.tracker_stats[tracker_id]["matches"]
                        misses = self.tracker_stats[tracker_id]["misses"]
                        if matches + misses > 0:  # Avoid division by zero
                            match_ratio = matches / (matches + misses)
                            history_weight = max(0.5, 1.0 - match_ratio)  # Lower is better, minimum 0.5
                    
                    # If IoU is adequate, use combined cost
                    if iou >= self.iou_threshold:
                        # Combined cost: lower is better
                        cost = (self.cost_weight_iou * (1.0 - iou) +
                                self.cost_weight_distance * normalized_distance) * confidence_weight * history_weight
                    else:
                        # If IoU is below threshold but distance is small enough,
                        # still consider the match but penalize it
                        cost = (self.cost_weight_iou * 1.0 + 
                                self.cost_weight_distance * normalized_distance) * confidence_weight * history_weight
                        
                        # Further penalize if IoU is very low
                        if iou < 0.1:
                            cost *= 1.5
                    
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
        potential_matches = [(r, c, cost_matrix[r, c], iou_values[r, c], historical_distances[r, c]) 
                           for r, c in zip(row_ind, col_ind)
                           if cost_matrix[r, c] <= cost_threshold]
        
        # Sort by cost (lowest first)
        potential_matches.sort(key=lambda x: x[2])
        
        # Track which detections and trackers have been assigned
        assigned_detections = set()
        assigned_trackers = set()
        matched_indices = []
        
        # First assign confident trackers, then uncertain ones
        # This two-pass approach helps maintain identity of confident trackers
        
        # First pass: assign confident trackers
        for r, c, cost, iou, distance in potential_matches:
            # Skip if this tracker is not confident or already assigned
            if r not in confident_trackers or r in assigned_trackers or c in assigned_detections:
                continue
                
            # Allow subclass to veto this match if needed
            if not self._validate_match(self.trackers[r], detections[c], iou, distance):
                continue
                
            # This is a valid assignment for a confident tracker
            matched_indices.append((r, c))
            assigned_detections.add(c)
            assigned_trackers.add(r)
        
        # Second pass: assign remaining trackers
        for r, c, cost, iou, distance in potential_matches:
            # Skip if already assigned
            if r in assigned_trackers or c in assigned_detections:
                continue

            # Allow subclass to veto this match if needed
            if not self._validate_match(self.trackers[r], detections[c], iou, distance):
                continue
                
            # Check for conflicts with much better historical claims
            conflict = False
            for other_r in range(len(self.trackers)):
                if other_r != r and other_r not in assigned_trackers:
                    # Check if other tracker has much better IoU or distance
                    if (iou_values[other_r, c] > iou * 1.5 or 
                        (historical_distances[other_r, c] < distance * 0.6 and iou_values[other_r, c] > iou * 0.7)):
                        conflict = True
                        break
            
            if not conflict:
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

    def _pre_process_detections(self, detections):
        """
        Pre-process detections before matching. Override in subclasses for specific filtering.
        Default implementation returns detections unchanged.
        """
        return detections

    def _post_process_trackers(self):
        """
        Post-process trackers after update. Override in subclasses for specific operations.
        Default implementation does nothing.
        """
        pass

    def _pre_create_tracker_filter(self, detection):
        """
        Additional filtering before creating a new tracker.
        Override in subclasses for specific filtering.
        Default implementation allows all detections.
        """
        return True

    def _post_create_tracker_filter(self, detection):
        """
        Final filtering after basic checks before creating a new tracker.
        Override in subclasses for specific filtering.
        Default implementation allows all detections.
        """
        return True

    def _should_merge_trackers(self, tracker_a, tracker_b):
        """
        Determine if two overlapping trackers should be merged.
        Override in subclasses for specialized logic.
        Default implementation allows all merges.
        """
        return True

    def _calculate_custom_matching_cost(self, tracker, detection, iou, normalized_distance):
        """
        Calculate custom matching cost between tracker and detection.
        Override in subclasses for specialized cost functions.
        Return None to use default cost calculation.
        """
        return None

    def _validate_match(self, tracker, detection, iou, distance):
        """
        Validate a potential match between tracker and detection.
        Override in subclasses to add additional validation.
        Default implementation allows all matches.
        """
        return True

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

    def process_detr_results(self, results, model, frame_shape, **kwargs):
        """
        Process DETR detection results to extract relevant objects based on target_labels.
        
        Args:
            results: DETR post-processed results
            model: DETR model (for id2label mapping)
            frame_shape: Tuple (height, width) of the current frame
            **kwargs: Additional arguments for subclasses
            
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
                raw_detections.append({
                    "box": box_coords, 
                    "confidence": score_val, 
                    "label": label_name
                })
        
        # Update trackers with these detections
        return self.update(raw_detections, frame_shape)
