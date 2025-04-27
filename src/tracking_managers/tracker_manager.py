import uuid  # For unique IDs
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm


class TrackerManager(ABC):
    """
    Abstract base class for tracker managers.
    Manages multiple tracker instances of a specific type.
    Handles assignment of detections to trackers and lifecycle management.
    Provides integrated DETR processing by default.
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
        size_filter_func=None,  # Optional function to filter detections by size
        custom_cost_func=None,  # Optional function to override cost calculation
        validate_match_func=None,  # Optional function to validate matches
        # Size filtering parameters
        size_min_width=None,
        size_max_width=None,
        size_min_height=None,
        size_max_height=None,
        size_min_aspect_ratio=None,
        size_max_aspect_ratio=None,
        size_calibration_func=None,  # Optional function to calibrate expected size
        # Match validation parameters
        match_iou_leniency=1.0,  # Factor to multiply iou_threshold for leniency (< 1 is more lenient)
        match_distance_multiplier=1.0,  # Factor to multiply object dimensions for max acceptable distance
        # Position-based ID parameters
        position_id_distance_threshold=20,  # Distance threshold for reusing position-based IDs
        position_id_prefer_x_distance=False,  # Whether to prefer x-distance for ID assignment (side view)
        position_id_name_format="{region}_{object}_{index}",  # Format string for position-based IDs
        # Search box parameters
        search_box_expansion_factor=1.5,  # Factor to expand object size for search box
        # Region drawing parameters
        draw_region_color=(100, 255, 255),  # Default color for region boxes (BGR)
        draw_region_label_prefix="Region",  # Prefix for region labels
        # Lifecycle hooks
        region_calibration=None,  # Optional function for region calibration
        on_tracker_update=None,  # Optional function called after tracker update
        on_tracker_removed=None,  # Optional function called when tracker is removed
        # New parameters for region fallback
        require_regions=False,  # Whether to require detection regions 
        high_confidence_fallback=False,  # Allow high-confidence detections when no regions
        high_confidence_threshold=None,  # Threshold for high-confidence fallback
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
        
        # Region requirements and fallback
        self.require_regions = require_regions
        self.high_confidence_fallback = high_confidence_fallback
        self.high_confidence_threshold = high_confidence_threshold or detection_threshold

        # Camera motion estimation
        self.camera_motion = np.zeros(2, dtype=float)
        self.motion_history = []  # Store recent motion vectors for smoothing
        self.motion_history_max_len = motion_history_max_len

        # Track tracker statistics over time
        self.tracker_stats = {}  # Tracker ID -> stats dict
        
        # Track object position history for stable identity and merging
        self.position_history = {}  # Region/category -> [(x, y, id), ...]
        
        # Region management
        self.regions = {}  # region_id -> box
        
        # Tracker factory - subclasses should register their factory method
        self.tracker_factory = None
        
        # Hook functions
        self.size_filter_func = size_filter_func or self._default_size_filter
        self.custom_cost_func = custom_cost_func
        self.validate_match_func = validate_match_func or self._default_validate_match
        
        # Size filtering parameters
        self.size_min_width = size_min_width
        self.size_max_width = size_max_width
        self.size_min_height = size_min_height
        self.size_max_height = size_max_height
        self.size_min_aspect_ratio = size_min_aspect_ratio
        self.size_max_aspect_ratio = size_max_aspect_ratio
        self.size_calibration_func = size_calibration_func
        
        # Match validation parameters
        self.match_iou_leniency = match_iou_leniency
        self.match_distance_multiplier = match_distance_multiplier

        # Position-based ID parameters
        self.position_id_distance_threshold = position_id_distance_threshold
        self.position_id_prefer_x_distance = position_id_prefer_x_distance
        self.position_id_name_format = position_id_name_format

        # Search box parameters
        self.search_box_expansion_factor = search_box_expansion_factor
        
        # Region drawing parameters
        self.draw_region_color = draw_region_color
        self.draw_region_label_prefix = draw_region_label_prefix

        # Lifecycle hooks
        self.region_calibration = region_calibration
        self.on_tracker_update = on_tracker_update
        self.on_tracker_removed = on_tracker_removed

    def register_tracker_factory(self, factory_method):
        """Register a method that creates appropriate tracker instances"""
        self.tracker_factory = factory_method

    def reset(self):
        """Reset all trackers - useful when needing to clear current tracking state"""
        self.trackers = []
        self.camera_motion = np.zeros(2, dtype=float)
        self.motion_history = []
        self.tracker_stats = {}
        self.position_history = {}
        self.regions = {}

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

    def update(self, detections, frame_shape, **kwargs):
        """
        Update trackers with new detections for the current frame.

        Args:
            detections: List of detection dictionaries
            frame_shape: Tuple (height, width) of the current frame
            **kwargs: Additional arguments that may be passed to region_calibration

        Returns:
            List of active tracker states or single tracker state if track_single_instance=True
        """
        # Store frame shape for coordinate clipping
        self.last_frame_shape = frame_shape

        # Run region calibration if provided
        if self.region_calibration:
            self.region_calibration(**kwargs)

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
            detection = high_confidence_detections[d_idx]
            tracker = self.trackers[t_idx]
            
            self._update_tracker_with_detection(tracker, detection)
            
            # Update tracker stats
            tracker_id = str(tracker.id)
            if tracker_id not in self.tracker_stats:
                self.tracker_stats[tracker_id] = {"matches": 0, "misses": 0}
            self.tracker_stats[tracker_id]["matches"] += 1
            
            # Call the update hook if provided
            if self.on_tracker_update:
                self.on_tracker_update(tracker, detection)

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
                    
                    # Call the update hook if provided
                    if self.on_tracker_update:
                        self.on_tracker_update(new_tracker, best_detection)
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
                    
                    # Call the update hook if provided
                    if self.on_tracker_update:
                        self.on_tracker_update(new_tracker, detection)

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
        # New region fallback logic
        if self.require_regions and not self.regions:
            # If high confidence fallback is enabled and the detection is confident enough
            if self.high_confidence_fallback:
                confidence = self._get_detection_confidence(detection)
                if confidence >= self.high_confidence_threshold:
                    return True
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
        
        return True

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
                # Using the new hook if provided
                if self.custom_cost_func:
                    custom_cost = self.custom_cost_func(tracker, det, iou, normalized_distance)
                    if custom_cost is not None:
                        cost_matrix[t_idx, d_idx] = custom_cost
                        continue
                else:
                    # Fallback to the traditional method
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
            # Use the new validation hook if provided
            if self.validate_match_func:
                if not self.validate_match_func(self.trackers[r], detections[c], iou, distance):
                    continue
            elif not self._validate_match(self.trackers[r], detections[c], iou, distance):
                # Fallback to traditional method
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
            if self.validate_match_func:
                if not self.validate_match_func(self.trackers[r], detections[c], iou, distance):
                    continue
            elif not self._validate_match(self.trackers[r], detections[c], iou, distance):
                # Fallback to traditional method
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
        Pre-process detections before matching.
        Filters by labels, regions, and size.
        """
        # First filter by labels if target_labels is specified
        if self.target_labels:
            detections = [d for d in detections if d.get("label") in self.target_labels]
        
        # Next filter by regions if any are defined
        if self.regions:
            filtered_detections = []
            for detection in detections:
                box = self._extract_bbox_from_detection(detection)
                if box is None:
                    continue
                
                # Calculate centroid
                center = self._calculate_centroid(box)
                
                # Check if centroid is in any region
                region_id = self._get_region_for_point(center[0], center[1])
                if region_id:
                    # Tag detection with region ID
                    detection_copy = detection.copy()
                    detection_copy["region_id"] = region_id
                    filtered_detections.append(detection_copy)
            detections = filtered_detections
        
        # Finally apply size filter if provided
        if self.size_filter_func:
            detections = [d for d in detections if self.size_filter_func(d)]
            
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

    def _extract_bbox_from_detection(self, detection):
        """
        Extract bounding box from detection dictionary.
        Default implementation assumes 'box' field in a standard DETR detection.

        Args:
            detection: Detection dictionary

        Returns:
            bbox: List or array [x1, y1, x2, y2]
        """
        # Most detections use "box" field, especially DETR outputs
        if "box" in detection:
            return detection["box"]
        # Fallbacks for other formats
        elif "bbox" in detection:
            return detection["bbox"]
        elif "bounding_box" in detection:
            return detection["bounding_box"]
        return None

    def _create_tracker_from_detection(self, detection, frame_shape):
        """
        Create a new tracker from a detection.
        Base implementation that subclasses will override with specific tracker creation.

        Args:
            detection: Detection dictionary
            frame_shape: Frame dimensions

        Returns:
            tracker: New tracker instance
        """
        if self.tracker_factory is None:
            raise NotImplementedError("Tracker factory method must be registered or this method must be overridden")
        
        return self.tracker_factory(detection, frame_shape)

    def _update_tracker_with_detection(self, tracker, detection):
        """
        Update a tracker with a detection.
        Default implementation extracts the box and confidence and calls tracker.update().

        Args:
            tracker: Tracker instance
            detection: Detection dictionary
        """
        box = self._extract_bbox_from_detection(detection)
        confidence = self._get_detection_confidence(detection)
        tracker.update(box, detection_confidence=confidence)

    def remove_lost_trackers(self):
        """Remove trackers that are marked as lost"""
        trackers_to_remove = [t for t in self.trackers if t.is_lost]
        
        # Call the removal hook for each tracker being removed
        if self.on_tracker_removed:
            for tracker in trackers_to_remove:
                self.on_tracker_removed(str(tracker.id))
        
        # Now actually remove the trackers
        self.trackers = [t for t in self.trackers if not t.is_lost]

    def draw_trackers(self, frame, show_search_box=False):
        """
        Draw all active trackers onto the frame, optionally with search boxes.
        First updates search boxes if needed.
        
        Args:
            frame: Image to draw on
            show_search_box: Whether to show expanded search boxes
            
        Returns:
            frame: Frame with trackers drawn
        """
        # Update search boxes if showing them
        if show_search_box:
            self.update_search_boxes()
            
        # Draw each tracker using its own draw method
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
        
    # Region management
    def add_region(self, box):
        """
        Add a detection region.
        
        Args:
            box: List or array [x1, y1, x2, y2]
            
        Returns:
            region_id: Unique identifier for the region
        """
        region_id = str(uuid.uuid4())
        self.regions[region_id] = np.array(box)
        return region_id
        
    def set_regions(self, boxes):
        """
        Set multiple detection regions at once, clearing any existing regions.
        
        Args:
            boxes: List of [x1, y1, x2, y2] boxes
        
        Returns:
            region_ids: List of region IDs generated
        """
        self.regions = {}
        region_ids = []
        for box in boxes:
            region_id = self.add_region(box)
            region_ids.append(region_id)
        return region_ids
        
    def clear_regions(self):
        """Clear all detection regions."""
        self.regions = {}
        
    def _get_region_for_point(self, x, y):
        """
        Determine which region a point belongs to.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            region_id: ID of the region containing the point, or None
        """
        for region_id, box in self.regions.items():
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                return region_id
        return None
        
    def process_detr_results(self, results, model, frame_shape, frame=None, **kwargs):
        """
        Process DETR detection results to extract relevant objects based on target_labels.
        
        Args:
            results: DETR post-processed results
            model: DETR model (for id2label mapping)
            frame_shape: Tuple (height, width) of the current frame
            frame: Optional frame data, used by some tracker managers like HandTrackerManager
            **kwargs: Additional arguments passed to update
            
        Returns:
            tracker_state: Current state of all trackers after processing
        """
        # Extract detections that match our target labels
        raw_detections = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            box_coords = [round(i) for i in box.tolist()]
            label_name = model.config.id2label[label.item()]
            
            raw_detections.append({
                "box": box_coords, 
                "confidence": score_val, 
                "label": label_name
            })
        
        # Update trackers with these detections and pass along any kwargs
        return self.update(raw_detections, frame_shape, **kwargs)

    def _default_size_filter(self, detection):
        """
        Default size filter implementation that uses the configured size parameters.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            bool: True if detection passes size filter
        """
        # If no size parameters are set, accept all
        if not any([self.size_min_width, self.size_max_width, 
                   self.size_min_height, self.size_max_height,
                   self.size_min_aspect_ratio, self.size_max_aspect_ratio]):
            return True
            
        # Extract box
        box = self._extract_bbox_from_detection(detection)
        if box is None:
            return False
            
        # Calculate width and height
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Calculate aspect ratio
        aspect_ratio = width / max(height, 0.001)  # Avoid division by zero
        
        # Check width constraints
        if self.size_min_width is not None and width < self.size_min_width:
            return False
        if self.size_max_width is not None and width > self.size_max_width:
            return False
            
        # Check height constraints
        if self.size_min_height is not None and height < self.size_min_height:
            return False
        if self.size_max_height is not None and height > self.size_max_height:
            return False
            
        # Check aspect ratio constraints
        if self.size_min_aspect_ratio is not None and aspect_ratio < self.size_min_aspect_ratio:
            return False
        if self.size_max_aspect_ratio is not None and aspect_ratio > self.size_max_aspect_ratio:
            return False
            
        return True
        
    def _default_validate_match(self, tracker, detection, iou, distance):
        """
        Default match validation based on configured parameters.
        
        Args:
            tracker: Tracker instance
            detection: Detection dictionary
            iou: IoU between tracker and detection
            distance: Distance between tracker and detection
            
        Returns:
            bool: Whether the match is valid
        """
        # Check if IoU is above the threshold with leniency applied
        adjusted_threshold = self.iou_threshold * self.match_iou_leniency
        if iou > adjusted_threshold:
            return True
            
        # Check if distance is below maximum acceptable distance
        det_box = self._extract_bbox_from_detection(detection)
        det_center = self._calculate_centroid(det_box)
        
        # Calculate maximum acceptable distance based on object dimensions
        tracker_size = max(tracker.width, tracker.height)
        max_acceptable_distance = tracker_size * self.match_distance_multiplier
        
        # Calculate distance between centers
        distance_to_center = np.linalg.norm(tracker.center - det_center)
        
        # Accept match if distance is acceptable
        return distance_to_center < max_acceptable_distance

    def get_region_for_point(self, x, y):
        """
        Determine which named region a point belongs to.
        Base implementation uses generic region IDs from self.regions.
        Override in subclasses for named regions.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            region_id: ID of the region containing the point, or None
        """
        return self._get_region_for_point(x, y)

    def get_stable_id_for_position(self, region_id, center_x, center_y, object_type="object"):
        """
        Get a stable ID for an object at the given position.
        Tries to maintain consistent IDs across frames by finding the nearest historical position.
        
        Args:
            region_id: ID of the region this object belongs to
            center_x, center_y: Center coordinates of the object
            object_type: Type of object (for ID generation)
            
        Returns:
            stable_id: A stable ID for this object, or None if not possible
        """
        if not region_id or region_id not in self.position_history:
            self.position_history[region_id] = []
            
        # Find closest historical position (if any) to reuse ID
        min_dist = float('inf')
        closest_idx = -1
        positions = self.position_history[region_id]
        
        for i, (pos_x, pos_y, existing_id) in enumerate(positions):
            # Calculate distance (either x-only or Euclidean)
            if self.position_id_prefer_x_distance:
                dist = abs(center_x - pos_x)
            else:
                dist = np.sqrt((center_x - pos_x)**2 + (center_y - pos_y)**2)
            
            # If very close to existing position, reuse that ID
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # If close enough to existing position, reuse ID
        if closest_idx >= 0 and min_dist < self.position_id_distance_threshold:
            object_id = positions[closest_idx][2]
            # Update the position
            positions[closest_idx] = (center_x, center_y, object_id)
            return object_id
        
        # Generate new ID for this object
        object_id = self.position_id_name_format.format(
            region=region_id,
            object=object_type,
            index=len(positions) + 1
        )
        positions.append((center_x, center_y, object_id))
        return object_id

    def update_search_boxes(self):
        """
        Update search boxes for all trackers based on current positions and dimensions.
        The search box is an expanded version of the bounding box used for searching 
        in the next frame and visualization.
        """
        for tracker in self.trackers:
            # Calculate expanded search box based on object size
            width = tracker.box[2] - tracker.box[0]
            height = tracker.box[3] - tracker.box[1]
            
            # Use configured expansion factor
            search_width = width * self.search_box_expansion_factor
            search_height = height * self.search_box_expansion_factor
            
            # Center the search box around the current center
            if hasattr(tracker, '_update_box_from_center'):
                tracker.search_box = tracker._update_box_from_center(
                    tracker.center, search_width, search_height
                )
            else:
                # Fallback for trackers without that method
                center_x, center_y = tracker.center
                half_width = search_width / 2
                half_height = search_height / 2
                tracker.search_box = np.array([
                    center_x - half_width,
                    center_y - half_height,
                    center_x + half_width,
                    center_y + half_height
                ])
            
            # Clip to frame boundaries if method exists
            if hasattr(tracker, '_clip_to_frame'):
                tracker.search_box = tracker._clip_to_frame(tracker.search_box)
            elif self.last_frame_shape:
                # Basic clipping fallback
                h, w = self.last_frame_shape[:2]
                tracker.search_box[0] = max(0, tracker.search_box[0])
                tracker.search_box[1] = max(0, tracker.search_box[1])
                tracker.search_box[2] = min(w, tracker.search_box[2])
                tracker.search_box[3] = min(h, tracker.search_box[3])

    def draw_regions(self, frame, color=None, label_prefix=None, show_expected_size=False):
        """
        Draw all detection regions onto the frame.
        
        Args:
            frame: Image to draw on
            color: Override color for region boxes (BGR)
            label_prefix: Override prefix for region labels
            show_expected_size: Whether to show expected size information
            
        Returns:
            frame: Frame with regions drawn
        """
        # Use provided values or defaults
        color = color or self.draw_region_color
        label_prefix = label_prefix or self.draw_region_label_prefix
        
        for region_id, region_box in self.regions.items():
            rb = region_box.astype(int)
            cv2.rectangle(
                frame,
                (rb[0], rb[1]),
                (rb[2], rb[3]),
                color,
                1,
                cv2.LINE_AA
            )
            
            # Show shortened region ID for brevity
            label = f"{label_prefix} {region_id[-4:]}" if len(region_id) > 4 else f"{label_prefix} {region_id}"
            cv2.putText(
                frame,
                label,
                (rb[0], rb[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        
        # Draw expected size information if requested and available
        if show_expected_size:
            # Check for expected size attributes
            size_info = []
            
            # For cups - check both attributes and make sure they're not None
            if (hasattr(self, 'expected_cup_width_pixels') and 
                hasattr(self, 'expected_cup_height_pixels') and
                self.expected_cup_width_pixels is not None and 
                self.expected_cup_height_pixels is not None):
                size_info.append(f"Expected Size (px): {self.expected_cup_width_pixels:.1f}W x {self.expected_cup_height_pixels:.1f}H")
            
            # For balls - check attribute and make sure it's not None
            elif hasattr(self, 'expected_ball_diameter_px') and self.expected_ball_diameter_px is not None:
                size_info.append(f"Expected Diameter (px): {self.expected_ball_diameter_px:.1f}")
                
            # Draw each piece of size info
            for i, text in enumerate(size_info):
                cv2.putText(
                    frame,
                    text,
                    (10, frame.shape[0] - 10 - (20 * i)),  # Position at bottom-left with offset
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        
        return frame

    def release(self):
        """Release any resources used by the tracker manager. Default implementation does nothing."""
        pass

    def region_calibration(self, **kwargs):
        """
        Calibrate regions based on provided data.
        Default implementation does nothing.
        Should be overridden by subclasses that need region calibration.
        """
        pass
