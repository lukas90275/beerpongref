import uuid
from typing import Dict, List, Tuple

import cv2
import numpy as np

from trackers.ball_tracker import BallTracker
from tracking_managers.tracker_manager import TrackerManager


class BallTrackerManager(TrackerManager):
    """
    Specialized tracker manager for ping pong balls.
    Manages ball trackers and detects balls within specified regions.
    """

    def __init__(
        self,
        iou_threshold=0.4,
        min_confidence_frames=3,
        max_lost_frames=15,  # Balls can get occluded briefly
        detection_threshold=0.3,
        position_stability_factor=0.1,  # Low stability to allow accurate tracking of fast movements
        size_filtering=True,  # Flag to enable size filtering specific to ping pong balls
    ):
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=False,  # Can have multiple balls
            position_stability_factor=position_stability_factor,
            target_labels=["sports ball", "ball", "baseball", "tennis ball", "sphere"],
            cost_weight_iou=0.3,
            cost_weight_distance=0.7,
        )
        # Dictionary to store detection regions: region_id -> region box
        self.detection_regions = {}
        
        # Track which balls were detected in which regions
        self.ball_source_regions = {}  # tracker_id -> region_id
        
        # Size filtering for ping pong balls (1.57 inches diameter)
        self.size_filtering = size_filtering
        # Expected ball diameter in pixels (will be calibrated based on frame size)
        self.expected_ball_diameter_px = None 
        # Acceptable deviation from expected size (percentage)
        self.size_tolerance = 0.8  # Allow 80% deviation from expected size - much more lenient

    def add_detection_region(self, region_box, region_id=None):
        """
        Add a region where balls should be detected.
        
        Args:
            region_box: List or array [x1, y1, x2, y2]
            region_id: Unique identifier for this region (generated if not provided)
            
        Returns:
            region_id: The ID of the added region
        """
        # Generate a region ID if not provided
        if region_id is None:
            region_id = str(uuid.uuid4())
            
        # Store the region
        self.detection_regions[region_id] = np.array(region_box)
        
        return region_id

    def update_detection_region(self, region_id, region_box):
        """
        Update an existing detection region.
        
        Args:
            region_id: ID of the region to update
            region_box: New region box [x1, y1, x2, y2]
        """
        if region_id in self.detection_regions:
            self.detection_regions[region_id] = np.array(region_box)

    def remove_detection_region(self, region_id):
        """
        Remove a detection region.
        
        Args:
            region_id: ID of the region to remove
        """
        if region_id in self.detection_regions:
            del self.detection_regions[region_id]

    def _extract_bbox_from_detection(self, detection):
        """
        Extract bounding box from detection dictionary.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            bbox: List or array [x1, y1, x2, y2]
        """
        # Handle different detection formats
        if "box" in detection:
            return detection["box"]
        elif "bbox" in detection:
            return detection["bbox"]
        elif "bounding_box" in detection:
            return detection["bounding_box"]
        return None

    def _create_tracker_from_detection(self, detection, frame_shape):
        """
        Create a new tracker from a detection.
        Also stores which region the ball was detected in.
        
        Args:
            detection: Detection dictionary
            frame_shape: Frame dimensions
            
        Returns:
            tracker: New BallTracker instance
        """
        box = self._extract_bbox_from_detection(detection)
        confidence = self._get_detection_confidence(detection)
        
        # Determine which region this ball was detected in
        source_region_id = detection.get("source_region_id", None)
        
        # Create new tracker
        tracker = BallTracker(
            initial_box=box,
            frame_shape=frame_shape,
            initial_confidence=confidence,
            source_region_id=source_region_id,
            position_stability_factor=self.position_stability_factor,
            min_confidence_frames=self.min_confidence_frames,
            max_lost_frames=self.max_lost_frames,
            search_expansion_factor=2.5  # Use larger search boxes for balls for better visualization
        )
        
        # Force balls to be confident immediately for better tracking continuity
        # This ensures ball trackers are marked as confident without waiting
        if self.min_confidence_frames == 1:
            tracker.is_confident = True
            tracker.confidence_frames = 1
        
        # Store the source region
        if source_region_id:
            self.ball_source_regions[str(tracker.id)] = source_region_id
            
        return tracker

    def _update_tracker_with_detection(self, tracker, detection):
        """
        Update a tracker with a detection.
        
        Args:
            tracker: BallTracker instance
            detection: Detection dictionary
        """
        box = self._extract_bbox_from_detection(detection)
        confidence = self._get_detection_confidence(detection)
        tracker.update(box, confidence)

    def _pre_process_detections(self, detections):
        """
        Pre-process detections:
        1. Filter by label (sports ball)
        2. Apply size filtering for ping pong balls if enabled
        3. ONLY accept detections that are in hand regions
        4. Add region_id to detections that are in regions
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            filtered_detections: List of processed detections
        """
        # Process detections first with parent method (label filtering)
        detections = super()._pre_process_detections(detections)
        
        # If we have no detection regions (no hands), return empty list
        if not self.detection_regions:
            return []
        
        # Apply size filtering if enabled
        if self.size_filtering and detections:
            filtered_by_size = []
            
            # If we haven't calibrated the expected ball size yet, do it now
            if self.expected_ball_diameter_px is None and self.last_frame_shape is not None:
                # Ping pong ball is ~1.57 inches diameter (from background-info)
                # Estimate pixels based on frame width
                frame_width = self.last_frame_shape[1]
                # Assuming the frame shows about 10 feet width (table is 8ft + some margin)
                # 1.57 inches / (10 feet * 12 inches/foot) * frame_width
                self.expected_ball_diameter_px = int((1.57 / (10 * 12)) * frame_width)
                # Ensure a minimum size
                self.expected_ball_diameter_px = max(self.expected_ball_diameter_px, 8)  # Lower minimum size
                print(f"Calibrated ping pong ball diameter: ~{self.expected_ball_diameter_px} pixels")
            
            # Apply size filtering
            for detection in detections:
                box = self._extract_bbox_from_detection(detection)
                if box is None:
                    continue
                    
                # Calculate width and height
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                # For a ball, width and height should be similar
                aspect_ratio = max(width, height) / (min(width, height) + 0.001)  # Avoid division by zero
                
                # Check if the size is within the expected range for a ping pong ball
                if self.expected_ball_diameter_px:
                    min_size = self.expected_ball_diameter_px * (1 - self.size_tolerance)
                    max_size = self.expected_ball_diameter_px * (1 + self.size_tolerance)
                    avg_size = (width + height) / 2
                    
                    # Accept if size is in range and aspect ratio is close to 1 (circle)
                    if min_size <= avg_size <= max_size and aspect_ratio <= 2.0:  # More lenient aspect ratio
                        filtered_by_size.append(detection)
                    elif width < min_size * 4 and height < min_size * 4:  # More lenient small object acceptance
                        # Also accept very small objects that could be balls in the distance
                        # But decrease their confidence to prioritize better matches
                        detection_copy = detection.copy()
                        original_confidence = self._get_detection_confidence(detection)
                        detection_copy["confidence"] = original_confidence * 0.9  # Less confidence penalty
                        filtered_by_size.append(detection_copy)
                else:
                    # If we haven't calibrated yet, accept all
                    filtered_by_size.append(detection)
            
            detections = filtered_by_size
        
        # For each detection, check if it's within a detection region
        processed_detections = []
        
        for detection in detections:
            box = self._extract_bbox_from_detection(detection)
            if box is None:
                continue
                
            # Convert to numpy array for easier processing
            box = np.array(box)
            
            # Get the center of the detection
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            
            # Check if the ball is in any detection region
            for region_id, region_box in self.detection_regions.items():
                if (region_box[0] <= center[0] <= region_box[2] and 
                    region_box[1] <= center[1] <= region_box[3]):
                    # Ball is in this region - add region_id to detection
                    detection_copy = detection.copy()
                    detection_copy["source_region_id"] = region_id
                    processed_detections.append(detection_copy)
                    break
        
        # Only return detections that were found in hand regions
        return processed_detections

    def _same_detection(self, detection1, detection2):
        """Helper to check if two detections refer to the same object"""
        box1 = self._extract_bbox_from_detection(detection1)
        box2 = self._extract_bbox_from_detection(detection2)
        
        if box1 is None or box2 is None:
            return False
            
        # Calculate IoU
        iou = BallTracker.calculate_iou(box1, box2)
        return iou > 0.7  # High threshold means they're the same object

    def update_with_hand_trackers(self, hand_trackers, detections, frame_shape):
        """
        Update ball tracking using hand trackers for detection regions.
        
        Args:
            hand_trackers: List of hand tracker states
            detections: List of detection dictionaries
            frame_shape: Frame dimensions
            
        Returns:
            List of active ball tracker states
        """
        # Clear previous detection regions
        self.detection_regions = {}
        
        # Add new detection regions from hand trackers
        for hand in hand_trackers:
            if "ball_region" in hand:
                region_id = f"hand_{hand['id']}"
                self.add_detection_region(hand["ball_region"], region_id)
        
        # Update trackers with detections
        tracked_balls = self.update(detections, frame_shape)
        
        # Extra step: Force all trackers to be confident if min_confidence_frames is 1
        # This ensures immediate confidence for all tracked balls
        if self.min_confidence_frames == 1:
            for tracker in self.trackers:
                tracker.is_confident = True
                tracker.confidence_frames = max(tracker.confidence_frames, 1)
            
            # Also update the returned state to reflect this change
            if tracked_balls:
                for ball_state in tracked_balls:
                    ball_state["is_confident"] = True
                    ball_state["confidence_frames"] = max(ball_state["confidence_frames"], 1)
        
        return tracked_balls

    def draw_regions(self, frame):
        """
        Draw all detection regions onto the frame.
        
        Args:
            frame: Image to draw on
            
        Returns:
            frame: Frame with regions drawn
        """
        for region_id, region_box in self.detection_regions.items():
            rb = region_box.astype(int)
            cv2.rectangle(
                frame,
                (rb[0], rb[1]),
                (rb[2], rb[3]),
                (100, 255, 255),  # Light yellow for regions
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"Ball Region {region_id[-4:]}",  # Show last 4 chars of ID for brevity
                (rb[0], rb[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 255, 255),
                1,
            )
        
        return frame 

    def _update_search_boxes(self):
        """
        Update search boxes for all trackers to ensure they're visible.
        Call this right before drawing to ensure search boxes are properly sized.
        """
        for tracker in self.trackers:
            # Calculate expanded search box based on ball size
            width = tracker.box[2] - tracker.box[0]
            height = tracker.box[3] - tracker.box[1]
            
            # Use a larger expansion factor for visualization
            search_width = width * 2.5
            search_height = height * 2.5
            
            # Center the search box around the current center
            tracker.search_box = tracker._update_box_from_center(
                tracker.center, search_width, search_height
            )
            
            # Clip to frame boundaries
            tracker.search_box = tracker._clip_to_frame(tracker.search_box)

    def draw_trackers(self, frame, show_search_box=False):
        """
        Draw all trackers with enhanced visibility for balls.
        
        Args:
            frame: Frame to draw on
            show_search_box: Whether to show search boxes
            
        Returns:
            Frame with trackers drawn
        """
        # Update search boxes for better visualization if showing them
        if show_search_box:
            self._update_search_boxes()
            
        # Draw each tracker using its own draw method
        for tracker in self.trackers:
            frame = tracker.draw(frame, show_search_box)
        
        return frame 

    def _validate_match(self, tracker, detection, iou, distance):
        """
        Additional validation for matches between trackers and detections.
        For balls, be more lenient with matching to ensure tracking continuity.
        
        Args:
            tracker: BallTracker instance
            detection: Detection dictionary
            iou: IoU between tracker and detection
            distance: Distance between tracker and detection
            
        Returns:
            bool: Whether the match is valid
        """
        # For balls, we want to be more lenient with matching
        # Accept matches even with lower IoU as long as they're in the general area
        if iou > self.iou_threshold * 0.7:  # Lower threshold for balls
            return True
            
        # Check if detection is in the tracker's search box
        det_box = self._extract_bbox_from_detection(detection)
        det_center = self._calculate_centroid(det_box)
        
        # More lenient distance-based matching for balls
        distance_to_center = np.linalg.norm(tracker.center - det_center)
        max_acceptable_distance = max(tracker.width, tracker.height) * 2
        
        # If detection is close enough to the tracker, consider it a match
        return distance_to_center < max_acceptable_distance 

    def _calculate_custom_matching_cost(self, tracker, detection, iou, normalized_distance):
        """
        Calculate a custom matching cost for balls.
        Prioritize matching existing ball trackers with detections to improve continuity.
        
        Args:
            tracker: BallTracker instance
            detection: Detection dictionary
            iou: IoU between tracker and detection
            normalized_distance: Normalized distance between tracker and detection
            
        Returns:
            float: Custom matching cost (lower values mean better matches)
        """
        # For balls, we want to prioritize continuing existing tracks
        # Use a lower cost to encourage matching with existing trackers
        
        # Start with a base cost that combines IoU and distance
        if iou > 0.1:  # Even a small overlap is meaningful for balls
            # Prioritize IoU more for balls since they can move quickly
            cost = (1.0 - iou) * 0.8 + normalized_distance * 0.2
            
            # Apply additional scaling to favor existing trackers more
            if tracker.confidence_frames > 0:
                # The more frames we've been tracking, the more we want to keep tracking
                confidence_scale = max(0.5, 1.0 - (tracker.confidence_frames * 0.1))
                cost *= confidence_scale
                
            return cost
            
        # If very low IoU but close distance, still consider it but with higher cost
        if normalized_distance < 0.15:  # Ball is relatively close
            return 0.8 + normalized_distance
            
        # Default to None which will fall back to the base implementation
        return None 