import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Import our Kalman tracker - adjusted relative import
from trackers.kalman_ball_tracker import KalmanBallTracker
from tracking_managers.tracker_manager import TrackerManager


class BallTrackerManager(TrackerManager):
    """
    Specialized tracker manager for ping pong balls.
    Uses Hough Circle Transform for detection and Kalman filtering for tracking.
    """

    # Real-world dimensions (constants)
    REAL_TABLE_WIDTH_FEET = 8.0  # Standard beer pong table length
    REAL_BALL_DIAMETER_INCHES = 1.57  # Standard ping pong ball diameter

    def __init__(
        self,
        iou_threshold=0.1,  # Lower threshold to be more lenient for matching
        min_confidence_frames=1,  # Detect balls immediately
        max_lost_frames=20,  # Keep tracking much longer
        detection_threshold=0.01,  # Accept very low confidence detections
        # Circle detection parameters - deliberately relaxed
        hough_dp=1.2,  # Resolution accumulator ratio
        hough_min_dist=20,  # Minimum distance between circles
        hough_param1=50,  # Higher threshold for Canny edge detector
        hough_param2=15,  # Lower accumulator threshold for less strict detection
        hough_min_radius=5,  # Minimum radius in pixels
        hough_max_radius=30,  # Maximum radius in pixels
        # White ball detection parameters
        white_threshold=180,  # Min value to consider a pixel "white" (0-255)
        min_white_percent=0.1,  # Minimum percentage of white pixels in a circle to be considered
        # Kalman filter parameters
        process_noise=1e-5,  # Process noise - lower for smoother trajectory
        measurement_noise=0.1,  # Measurement noise - higher means less trust in measurements
        motion_blur_velocity_threshold=15.0, # Velocity threshold for motion blur
        motion_blur_noise_multiplier=5.0,    # Noise multiplier for motion blur
        motion_blur_search_expansion_factor=2.5, # Search expansion for motion blur
        # Debug options
        debug_visualization=False,
        **kwargs
    ):
        # Initialize with ball-specific parameters
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=False,  # Can have multiple balls
            cost_weight_iou=0.2,  # Reduced to favor distance more
            cost_weight_distance=0.8,  # Increased to favor distance more
            match_iou_leniency=0.9,  # More lenient for fast-moving balls
            match_distance_multiplier=3.0,  # Allow matches at greater distances
            search_box_expansion_factor=2.0,
            directional_search_expansion=True,
            region_calibration=self._add_hand_regions,
            draw_region_color=(100, 255, 255),
            draw_region_label_prefix="Ball Region",
            require_regions=False,  # Don't enforce region detection for now
            high_confidence_fallback=True,  # Enable high confidence fallback outside regions
            high_confidence_threshold=0.1,
            **kwargs
        )
        
        # Store Hough Circle parameters
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.hough_min_radius = hough_min_radius
        self.hough_max_radius = hough_max_radius
        
        # Store white color parameters
        self.white_threshold = white_threshold
        self.min_white_percent = min_white_percent
        
        # Store Kalman filter parameters
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.motion_blur_velocity_threshold = motion_blur_velocity_threshold
        self.motion_blur_noise_multiplier = motion_blur_noise_multiplier
        self.motion_blur_search_expansion_factor = motion_blur_search_expansion_factor
        
        # Ball size calibration (will be set dynamically)
        self.expected_ball_diameter_px = None
        self.expected_ball_area_px = None
        
        # Debug settings
        self.debug_visualization = debug_visualization
        
        # Configure logger
        self.logger = logging.getLogger("BallTracker")
        
        # Register the Kalman ball tracker factory
        self.register_tracker_factory(self._create_kalman_ball_tracker)
        
        # Store active hand regions
        self.active_hand_regions = []
        
        self.logger.info(f"Initialized Kalman Filter ball tracker")

    def _create_tracker_from_detection(self, detection: Dict[str, Any], frame_shape: Tuple[int, int]) -> KalmanBallTracker:
        """Create a new tracker from detection - implements abstract method"""
        return self._create_kalman_ball_tracker(detection, frame_shape)

    def _create_kalman_ball_tracker(self, detection: Dict[str, Any], frame_shape: Tuple[int, int]) -> KalmanBallTracker:
        """Create a new Kalman filter ball tracker from a detection."""
        box = self._extract_bbox_from_detection(detection)
        confidence = self._get_detection_confidence(detection)
        source_region_id = detection.get("region_id", None)
        
        # Create new tracker
        tracker = KalmanBallTracker(
            initial_box=box,
            frame_shape=frame_shape,
            initial_confidence=confidence,
            source_region_id=source_region_id,
            max_lost_frames=self.max_lost_frames,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            motion_blur_velocity_threshold=self.motion_blur_velocity_threshold,
            motion_blur_noise_multiplier=self.motion_blur_noise_multiplier,
            motion_blur_search_expansion_factor=self.motion_blur_search_expansion_factor
        )
            
        return tracker

    def _add_hand_regions(self, hand_trackers=None, **_):
        """
        Set up ball detection regions based on hand tracker positions.
        Generate stable region IDs based on hand tracker IDs to prevent changing IDs.
        
        Args:
            hand_trackers: List of hand tracker states
        """
        # Store old region mappings before clearing
        old_regions = {}
        for region_id, box in self.regions.items():
            old_regions[region_id] = box.copy()
            
        # Clear previous detection regions
        self.clear_regions()
        self.active_hand_regions = []
        
        # If no hand trackers provided, do nothing
        if not hand_trackers:
            return
            
        # Add new detection regions from hand trackers
        added_regions = 0
        for hand in hand_trackers:
            # Check if hand has ball_region
            if "ball_region" in hand:
                # Use hand ID to create stable region ID
                if "id" in hand:
                    # Use the hand ID to create a stable region ID
                    hand_id = hand["id"]
                    region_id = f"hand_{hand_id}_ball_region"
                else:
                    # Fallback to using a hash of the region coordinates for stability
                    region_box = hand["ball_region"]
                    box_str = f"{region_box[0]}_{region_box[1]}_{region_box[2]}_{region_box[3]}"
                    region_id = f"region_{hash(box_str) % 10000}"
                
                # Add the region with the stable ID
                self.regions[region_id] = np.array(hand["ball_region"])
                
                # Store in active hand regions
                self.active_hand_regions.append({
                    "region_id": region_id,
                    "box": hand["ball_region"]
                })
                added_regions += 1
                
        if added_regions > 0:
            self.logger.info(f"Added {added_regions} ball regions from hand trackers")
        
        # Calibrate ball size based on frame dimensions if needed
        if self.last_frame_shape and self.expected_ball_diameter_px is None:
            # Get table width in pixels from frame width
            frame_width = self.last_frame_shape[1]
            
            # Calculate pixels per foot using actual table width
            pixels_per_foot = frame_width / self.REAL_TABLE_WIDTH_FEET
            
            # Convert ball diameter from inches to pixels
            self.expected_ball_diameter_px = int((self.REAL_BALL_DIAMETER_INCHES / 12.0) * pixels_per_foot)
            
            # Ensure a minimum size
            self.expected_ball_diameter_px = max(self.expected_ball_diameter_px, 4)  # Minimum 4 pixels
            
            # Update Hough circle parameters based on expected size
            # Leave a wide margin to ensure we don't miss the ball
            self.hough_min_radius = max(3, int(self.expected_ball_diameter_px * 0.3))
            self.hough_max_radius = int(self.expected_ball_diameter_px * 0.7) 
            
            # Calculate expected area
            self.expected_ball_area_px = np.pi * ((self.expected_ball_diameter_px / 2) ** 2)
            
            self.logger.info(f"Calibrated ball diameter: ~{self.expected_ball_diameter_px} pixels")
            self.logger.info(f"Adjusted Hough circles params: min_radius={self.hough_min_radius}, max_radius={self.hough_max_radius}")

    def detect_circles(self, frame):
        """
        Detect circles in the frame using HoughCircles.
        
        Args:
            frame: BGR image
            
        Returns:
            List of detected circles as detections with confidence
        """
        if frame is None or frame.size == 0:
            return []
            
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optional: Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.hough_min_radius,
            maxRadius=self.hough_max_radius
        )
        
        # Check if any circles were found
        if circles is None:
            return []
            
        # Convert to detections format
        detections = []
        
        # Create a binary mask for white pixels
        _, white_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        for circle in circles[0]:
            x, y, radius = circle
            
            # Convert circle to bounding box
            x1 = int(x - radius)
            y1 = int(y - radius)
            x2 = int(x + radius) 
            y2 = int(y + radius)
            
            # Ensure box is within frame
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w-1, x2)
            y2 = min(h-1, y2)
            
            # Skip if box is invalid
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Calculate percentage of white pixels in the circle
            # Create a circular mask
            circle_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            cv2.circle(circle_mask, 
                      (int((x2-x1)/2), int((y2-y1)/2)), 
                      int(radius), 
                      255, 
                      -1)
            
            # Extract the region from white mask
            white_roi = white_mask[y1:y2, x1:x2]
            
            # Calculate percentage of white pixels in the circle
            white_pixels = cv2.countNonZero(cv2.bitwise_and(white_roi, circle_mask))
            circle_area = cv2.countNonZero(circle_mask)
            white_percentage = white_pixels / max(1, circle_area)
            
            # Set confidence based on white percentage
            confidence = white_percentage
            
            # For MVP, all circles get high confidence - we'll filter later
            # This ensures we don't miss the ball
            if confidence >= self.min_white_percent:
                detection = {
                    "box": [x1, y1, x2, y2],
                    "confidence": max(0.5, confidence),  # Minimum 0.5 confidence
                    "label": "ball",
                    "radius": radius,
                    "center": (x, y)
                }
                
                # Check if this circle is within any region
                center_point = (int(x), int(y))
                region_id = self._get_region_for_point(center_point[0], center_point[1])
                if region_id:
                    detection["region_id"] = region_id
                    
                detections.append(detection)
        
        return detections

    def process_detr_results(self, results, model, frame_shape, frame=None, hand_trackers=None, **kwargs):
        """
        Process detection with Hough Circles instead of DETR.
        
        Args:
            results: DETR results (not used)
            model: DETR model (not used)
            frame_shape: Frame dimensions
            frame: Original frame (required)
            hand_trackers: Optional list of hand tracker states
            **kwargs: Additional arguments
            
        Returns:
            List of active ball tracker states
        """
        # If hand_trackers are provided, update regions
        if hand_trackers:
            self._add_hand_regions(hand_trackers=hand_trackers)
        
        if frame is None:
            self.logger.warning("Frame is required for circle detection")
            return [tracker.get_state() for tracker in self.trackers]
            
        # Detect circles in the frame
        circle_detections = self.detect_circles(frame)
        
        # Update trackers with circle detections
        return self.update(circle_detections, frame_shape, frame=frame)

    def draw_trackers(self, frame, show_search_box=False, show_blob_detections=False, **kwargs):
        """
        Draw all ball trackers with enhanced visibility.
        
        Args:
            frame: Frame to draw on
            show_search_box: Whether to show search boxes
            show_blob_detections: Whether to show detected circles
            
        Returns:
            Frame with trackers drawn
        """
        # Update trackers' search boxes before drawing to ensure proper expansion
        for tracker in self.trackers:
            # Force prediction to update search box with velocity and lost frames
            tracker.predict()
            
        # Draw each tracker using its own draw method
        for tracker in self.trackers:
            # Draw the Kalman tracker visualization
            frame = tracker.draw(frame, show_search_box)
                
        return frame
        
    def draw_detections(self, frame, detections):
        """
        Draw circle detections on the frame for debugging.
        
        Args:
            frame: Frame to draw on
            detections: List of detected circles
            
        Returns:
            Frame with detections drawn
        """
        for det in detections:
            if "center" in det and "radius" in det:
                center = (int(det["center"][0]), int(det["center"][1]))
                radius = int(det["radius"])
                confidence = det.get("confidence", 0.0)
                
                # Color based on confidence: green for high confidence, yellow for medium, red for low
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.3:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw circle
                cv2.circle(frame, center, radius, color, 2)
                
                # Draw center point
                cv2.circle(frame, center, 2, color, -1)
                
                # Add confidence text
                cv2.putText(
                    frame,
                    f"{confidence:.2f}",
                    (center[0], center[1] - radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
                
        return frame 