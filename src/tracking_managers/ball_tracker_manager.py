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
    Optimized for handling motion blur and high-speed ball trajectories.
    """

    # Real-world dimensions (constants)
    REAL_TABLE_WIDTH_FEET = 8.0  # Standard beer pong table length
    REAL_BALL_DIAMETER_INCHES = 1.57  # Standard ping pong ball diameter

    def __init__(
        self,
        iou_threshold=0.05,  # Even lower threshold for fast-moving balls
        min_confidence_frames=1,  # Detect balls immediately
        max_lost_frames=30,  # Keep tracking much longer for better trajectory prediction
        detection_threshold=0.01,  # Accept very low confidence detections
        # Circle detection parameters - deliberately relaxed
        hough_dp=1.2,  # Resolution accumulator ratio
        hough_min_dist=20,  # Minimum distance between circles
        hough_param1=50,  # Higher threshold for Canny edge detector
        hough_param2=12,  # Even lower accumulator threshold for less strict detection (was 15)
        hough_min_radius=5,  # Minimum radius in pixels
        hough_max_radius=35,  # Maximum radius in pixels (increased to handle motion blur)
        # White ball detection parameters
        white_threshold=165,  # Significantly lowered to capture very blurry balls (was 175)
        min_white_percent=0.03,  # Extremely low threshold to detect heavily blurred balls (was 0.05)
        # Blob detection for elongated balls (motion blur)
        use_blob_detection=True,  # Enable blob detection for elongated balls
        max_elongation_ratio=5.0,  # Increased max ratio to catch extreme motion blur (was 3.5)
        # Kalman filter parameters
        process_noise=1e-4,  # Increased for better handling of rapid movements
        measurement_noise=0.05,  # Lower to trust detections more
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
            match_iou_leniency=0.8,  # More lenient for fast-moving balls (was 0.9)
            match_distance_multiplier=4.0,  # Allow matches at even greater distances (was 3.0)
            search_box_expansion_factor=2.5,  # Larger search boxes (was 2.0)
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
        
        # Blob detection parameters
        self.use_blob_detection = use_blob_detection
        self.max_elongation_ratio = max_elongation_ratio
        
        # Create blob detector for elongated balls
        if self.use_blob_detection:
            # Setup SimpleBlobDetector parameters for elongated white blobs
            params = cv2.SimpleBlobDetector_Params()
            
            # Change thresholds - we want white blobs
            params.minThreshold = white_threshold
            params.maxThreshold = 255
            params.thresholdStep = 10
            
            # Filter by color (255 = white)
            params.filterByColor = True
            params.blobColor = 255
            
            # Filter by area
            params.filterByArea = True
            params.minArea = 30  # Lower minimum area (was 50)
            params.maxArea = 3000  # Higher maximum area (was 2000)
            
            # Filter by circularity - allow much lower values for extremely elongated blobs
            params.filterByCircularity = True
            params.minCircularity = 0.05  # Extremely low (was 0.1)
            
            # Filter by convexity
            params.filterByConvexity = True
            params.minConvexity = 0.5  # Lower for less perfect shapes (was 0.7)
            
            # Filter by inertia (elongation) - allow elongated blobs
            params.filterByInertia = True
            params.minInertiaRatio = 0.01  # Extremely low to catch very elongated blobs (was 0.05)
            
            # Create the detector
            self.blob_detector = cv2.SimpleBlobDetector_create(params)
        else:
            self.blob_detector = None
        
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
        
        self.logger.info(f"Initialized Motion-Optimized Kalman Filter ball tracker with blob detection")

    def _create_tracker_from_detection(self, detection: Dict[str, Any], frame_shape: Tuple[int, int]) -> KalmanBallTracker:
        """Create a new tracker from detection - implements abstract method"""
        return self._create_kalman_ball_tracker(detection, frame_shape)

    def _create_kalman_ball_tracker(self, detection: Dict[str, Any], frame_shape: Tuple[int, int]) -> KalmanBallTracker:
        """Create a new Kalman filter ball tracker from a detection."""
        box = self._extract_bbox_from_detection(detection)
        confidence = self._get_detection_confidence(detection)
        source_region_id = detection.get("region_id", None)
        
        # Check if there's motion direction information from blob detection
        initial_velocity = None
        if "motion_direction" in detection:
            # Get motion direction (unit vector)
            motion_dir = np.array(detection["motion_direction"])
            
            # Estimate speed based on elongation ratio
            estimated_speed = 0
            if "elongation_ratio" in detection:
                # Higher elongation = higher speed
                elongation_ratio = detection["elongation_ratio"]
                estimated_speed = (elongation_ratio - 1.0) * 30.0  # Scale factor
                
            # Create initial velocity vector
            initial_velocity = motion_dir * estimated_speed
        
        # Create new tracker
        tracker = KalmanBallTracker(
            initial_box=box,
            frame_shape=frame_shape,
            initial_confidence=confidence,
            source_region_id=source_region_id,
            max_lost_frames=self.max_lost_frames,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            initial_velocity=initial_velocity  # Pass the estimated velocity
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

    def _calculate_custom_matching_cost(self, tracker, detection, iou, normalized_distance):
        """
        Calculate custom matching cost between tracker and detection.
        Optimized for fast-moving balls with motion blur.
        """
        # Get detection box
        det_box = self._extract_bbox_from_detection(detection)
        det_center = self._calculate_centroid(det_box)
        
        # Get ball speed and use it to adjust matching criteria
        if hasattr(tracker, 'speed') and tracker.speed > 10.0:
            # For fast balls, rely more on distance and direction of motion
            
            # Check if detection is in the general direction of motion
            motion_aligned = False
            if hasattr(tracker, 'velocity'):
                # Get normalized velocity direction
                velocity = tracker.velocity
                speed = np.linalg.norm(velocity)
                if speed > 0:
                    vel_dir = velocity / speed
                    
                    # Vector from tracker center to detection
                    to_det = det_center - tracker.center
                    dist = np.linalg.norm(to_det)
                    if dist > 0:
                        to_det_dir = to_det / dist
                        
                        # Dot product to check alignment (1=same direction, -1=opposite)
                        alignment = np.dot(vel_dir, to_det_dir)
                        motion_aligned = alignment > 0.3  # Detection is somewhat in direction of motion
                        
            # Adjust cost based on motion alignment and speed
            if motion_aligned:
                # Detection is in the direction of motion - lower cost
                direction_bonus = 0.3
                # Distance is more important than IoU for fast balls
                custom_cost = 0.1 * (1.0 - iou) + 0.9 * normalized_distance
                # Apply direction bonus
                custom_cost = max(0.01, custom_cost - direction_bonus)
                return custom_cost
            
        # For slower balls or non-aligned detections, use default cost calculation
        return None  # Use default cost calculation

    def _validate_match(self, tracker, detection, iou, distance):
        """
        Validate a match between tracker and detection.
        More lenient for fast-moving balls.
        """
        # Get detection info
        det_box = self._extract_bbox_from_detection(detection)
        det_center = self._calculate_centroid(det_box)
        
        # For fast balls, use more lenient validation
        if hasattr(tracker, 'speed') and tracker.speed > 10.0:
            # Scale validation based on speed
            speed_factor = min(1.0 + tracker.speed / 50.0, 3.0)
            
            # Get adjusted thresholds
            iou_threshold = self.iou_threshold * self.match_iou_leniency * 0.5  # Even more lenient
            
            # Calculate object size and use it for distance tolerance
            tracker_size = max(tracker.width, tracker.height) 
            max_distance = tracker_size * self.match_distance_multiplier * speed_factor
            
            # Check if either criterion is met
            if iou > iou_threshold or distance < max_distance:
                return True
                
            return False
            
        # For slower balls, use the default validation
        return super()._validate_match(tracker, detection, iou, distance)

    def detect_circles(self, frame):
        """
        Detect circles in the frame using HoughCircles and blob detection for fast-moving,
        elongated balls caused by motion blur.
        
        Args:
            frame: BGR image
            
        Returns:
            List of detected circles as detections with confidence
        """
        if frame is None or frame.size == 0:
            return []
            
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur to reduce noise but preserve edges
        gray = cv2.GaussianBlur(gray, (5, 5), 0.8)  # Reduced blur sigma
        
        # Create a binary mask for white pixels (for both methods)
        _, white_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        # Get detections from circle detection
        circle_detections = self._detect_circles_hough(gray, white_mask, frame.shape[:2])
        
        # Get detections from blob detection for elongated shapes (if enabled)
        blob_detections = []
        if self.use_blob_detection:
            blob_detections = self._detect_elongated_blobs(white_mask, frame.shape[:2])
            
        # Combine detections
        all_detections = circle_detections + blob_detections
        
        # If we have detections from both methods that are very close, keep the better one
        if circle_detections and blob_detections:
            all_detections = self._merge_nearby_detections(all_detections)
            
        return all_detections
        
    def _detect_circles_hough(self, gray, white_mask, frame_shape):
        """
        Detect circles using Hough Circle Transform.
        
        Args:
            gray: Grayscale image
            white_mask: Binary mask of white pixels
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            List of circle detections
        """
        detections = []
        
        # Try different parameter sets to capture both sharp and blurry balls
        all_circles = []
        
        # First attempt - standard parameters for clear balls
        circles1 = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.hough_min_radius,
            maxRadius=self.hough_max_radius
        )
        
        if circles1 is not None:
            all_circles.extend(circles1[0])
        
        # Second attempt - more lenient parameters for blurry/less-defined balls
        circles2 = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1 * 0.8,  # Lower edge sensitivity
            param2=self.hough_param2 * 0.8,  # Lower voting threshold
            minRadius=self.hough_min_radius,
            maxRadius=int(self.hough_max_radius * 1.2)  # Larger max radius for blurred balls
        )
        
        if circles2 is not None:
            all_circles.extend(circles2[0])
            
        # If no circles found, try with even more relaxed parameters
        if not all_circles:
            circles3 = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=self.hough_dp,
                minDist=self.hough_min_dist,
                param1=self.hough_param1 * 0.7,
                param2=self.hough_param2 * 0.6,  # Much lower threshold
                minRadius=self.hough_min_radius,
                maxRadius=int(self.hough_max_radius * 1.5)
            )
            
            if circles3 is not None:
                all_circles.extend(circles3[0])
        
        # If still no circles, return empty list
        if not all_circles:
            return []
            
        # Process all detected circles
        for circle in all_circles:
            x, y, radius = circle
            
            # Convert circle to bounding box
            x1 = int(x - radius)
            y1 = int(y - radius)
            x2 = int(x + radius) 
            y2 = int(y + radius)
            
            # Ensure box is within frame
            h, w = frame_shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w-1, x2)
            y2 = min(h-1, y2)
            
            # Skip if box is invalid
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Calculate percentage of white pixels in the circle
            circle_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            cv2.circle(circle_mask, 
                      (int((x2-x1)/2), int((y2-y1)/2)), 
                      int(radius), 
                      255, 
                      -1)
            
            # Extract the region from white mask
            white_roi = white_mask[y1:y2, x1:x2]
            
            # Check if the ROI and mask shapes match
            if white_roi.shape != circle_mask.shape:
                continue
                
            # Calculate percentage of white pixels in the circle
            white_pixels = cv2.countNonZero(cv2.bitwise_and(white_roi, circle_mask))
            circle_area = cv2.countNonZero(circle_mask)
            white_percentage = white_pixels / max(1, circle_area)
            
            # Set confidence based on white percentage
            confidence = white_percentage
            
            # For MVP, all circles that meet minimum threshold get high confidence
            if confidence >= self.min_white_percent:
                detection = {
                    "box": [x1, y1, x2, y2],
                    "confidence": max(0.5, confidence),  # Minimum 0.5 confidence
                    "label": "ball",
                    "radius": radius,
                    "center": (x, y),
                    "detection_type": "circle"
                }
                
                # Check if this circle is within any region
                center_point = (int(x), int(y))
                region_id = self._get_region_for_point(center_point[0], center_point[1])
                if region_id:
                    detection["region_id"] = region_id
                    
                detections.append(detection)
                
        return detections
        
    def _detect_elongated_blobs(self, white_mask, frame_shape):
        """
        Detect elongated white blobs that could be motion-blurred balls.
        
        Args:
            white_mask: Binary mask of white pixels
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            List of blob detections
        """
        h, w = frame_shape
        detections = []
        
        # First try to detect keypoints using the blob detector
        keypoints = self.blob_detector.detect(white_mask)
        
        # Process each keypoint
        for kp in keypoints:
            # Get keypoint info
            x, y = kp.pt
            size = kp.size  # Diameter of the blob
            
            # Skip if too close to frame edges (these are usually false positives)
            edge_margin = 3  # Reduced from 5 to catch balls closer to edges
            if (x < edge_margin or x > w - edge_margin or 
                y < edge_margin or y > h - edge_margin):
                continue
                
            # Create an initial bounding box from keypoint
            radius = size / 2
            x1 = int(x - radius * 1.5)  # Larger box to ensure we capture the whole blur (was 1.2)
            y1 = int(y - radius * 1.5)
            x2 = int(x + radius * 1.5)
            y2 = int(y + radius * 1.5)
            
            # Ensure box is within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w-1, x2)
            y2 = min(h-1, y2)
            
            # Skip if box is invalid
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Extract the white mask region
            roi = white_mask[y1:y2, x1:x2]
            
            # Find contours to get precise shape
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Skip if no contours found
            if not contours:
                continue
                
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Skip if contour is too small
            area = cv2.contourArea(largest_contour)
            if area < 20:  # Even lower minimum area threshold (was 30)
                continue
                
            # Find the rotated bounding rectangle of the contour
            rect = cv2.minAreaRect(largest_contour)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            
            # Get width and height of the rotated box
            (center_x, center_y), (width, height), angle = rect
            
            # Ensure width is the longer dimension
            if width < height:
                width, height = height, width
                
            # Check elongation ratio
            elongation_ratio = width / max(height, 1)
            
            # Accept even slightly elongated shapes (1.1+)
            if elongation_ratio < 1.1 or elongation_ratio > self.max_elongation_ratio:
                continue
                
            # Calculate the axis-aligned bounding box for the blob
            min_x = np.min(box_points[:, 0])
            min_y = np.min(box_points[:, 1])
            max_x = np.max(box_points[:, 0])
            max_y = np.max(box_points[:, 1])
            
            # Adjust coordinates to frame coordinates
            min_x += x1
            min_y += y1
            max_x += x1
            max_y += y1
            
            # Ensure within frame
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(w-1, max_x)
            max_y = min(h-1, max_y)
            
            # Skip if invalid
            if min_x >= max_x or min_y >= max_y:
                continue
                
            # Calculate confidence based on blob properties
            # Higher confidence for more elongated blobs within reasonable limits
            confidence_factor = min(elongation_ratio / 2.0, 1.0)
            confidence = 0.6 * confidence_factor  # Base confidence
            
            # Check whiteness percentage - be more lenient for extremely elongated shapes
            white_count = cv2.countNonZero(white_mask[min_y:max_y, min_x:max_x])
            total_area = (max_x - min_x) * (max_y - min_y)
            white_percent = white_count / max(total_area, 1)
            
            # Reduced whiteness requirement for extremely elongated blobs
            min_white_percent_required = self.min_white_percent
            if elongation_ratio > 3.0:
                # Even lower threshold for very elongated shapes that might be dimmer
                min_white_percent_required *= 0.7
                
            # Adjust confidence based on whiteness
            if white_percent >= min_white_percent_required:
                # Create detection
                detection = {
                    "box": [min_x, min_y, max_x, max_y],
                    "confidence": max(0.5, confidence),
                    "label": "ball",
                    "center": (int(x), int(y)),
                    "elongation_ratio": elongation_ratio,
                    "angle": angle,
                    "detection_type": "blob"
                }
                
                # Add motion direction info (perpendicular to elongation)
                motion_angle = angle + 90 if width > height else angle
                motion_angle_rad = np.radians(motion_angle)
                motion_dir = np.array([np.cos(motion_angle_rad), np.sin(motion_angle_rad)])
                detection["motion_direction"] = motion_dir.tolist()
                
                # Check if this blob is within any region
                region_id = self._get_region_for_point(int(x), int(y))
                if region_id:
                    detection["region_id"] = region_id
                    
                detections.append(detection)
                
        # Additional approach: Find contours directly for more challenging cases
        # This catches extremely elongated blobs that SimpleBlobDetector might miss
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area - allow a wider range of sizes
            area = cv2.contourArea(contour)
            if area < 25 or area > 1500:  # More inclusive area range (was 50-1000)
                continue
                
            # Find bounding rectangle
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            # Ensure width is the longer dimension
            if width < height:
                width, height = height, width
                angle += 90
                
            # Accept blobs with elongation ratio 1.5+ (was 2.0+)
            elongation_ratio = width / max(height, 1)
            if elongation_ratio < 1.5 or elongation_ratio > self.max_elongation_ratio:
                continue
                
            # Get the center coordinates
            center_x, center_y = int(center_x), int(center_y)
            
            # Calculate axis-aligned bounding box
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            min_x = max(0, np.min(box_points[:, 0]))
            min_y = max(0, np.min(box_points[:, 1]))
            max_x = min(w-1, np.max(box_points[:, 0]))
            max_y = min(h-1, np.max(box_points[:, 1]))
            
            # Skip if invalid
            if min_x >= max_x or min_y >= max_y:
                continue
            
            # Higher base confidence for more extreme elongation
            confidence = min(0.5 + (elongation_ratio - 1.5) * 0.1, 0.8)
            
            # Create detection
            detection = {
                "box": [min_x, min_y, max_x, max_y],
                "confidence": confidence,
                "label": "ball",
                "center": (center_x, center_y),
                "elongation_ratio": elongation_ratio,
                "angle": angle,
                "detection_type": "elongated_contour"
            }
            
            # Add motion direction info (perpendicular to elongation)
            motion_angle_rad = np.radians(angle)
            motion_dir = np.array([np.cos(motion_angle_rad), np.sin(motion_angle_rad)])
            detection["motion_direction"] = motion_dir.tolist()
            
            # Check if this blob is within any region
            region_id = self._get_region_for_point(center_x, center_y)
            if region_id:
                detection["region_id"] = region_id
                
            detections.append(detection)
        
        # Try another approach - dilate the mask and look for elongated shapes
        # This can help recover extremely faint motion blurs
        dilated_mask = cv2.dilate(white_mask, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Skip small or huge contours
            area = cv2.contourArea(contour)
            if area < 50 or area > 2000:
                continue
                
            # Skip contours that are already in the white mask (already detected)
            if np.any([cv2.pointPolygonTest(contour, point, False) >= 0 for detection in detections
                     for point in [detection.get("center", (0,0))]]):
                continue
                
            # Find bounding rectangle
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            # Ensure width is the longer dimension
            if width < height:
                width, height = height, width
                angle += 90
                
            # Only consider significantly elongated shapes
            elongation_ratio = width / max(height, 1)
            if elongation_ratio < 2.0 or elongation_ratio > self.max_elongation_ratio:
                continue
                
            # Calculate axis-aligned bounding box
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            min_x = max(0, np.min(box_points[:, 0]))
            min_y = max(0, np.min(box_points[:, 1]))
            max_x = min(w-1, np.max(box_points[:, 0]))
            max_y = min(h-1, np.max(box_points[:, 1]))
            
            # Skip if invalid
            if min_x >= max_x or min_y >= max_y:
                continue
                
            # Check if this is a plausible ball - must have some white pixels
            roi = white_mask[min_y:max_y, min_x:max_x]
            white_pixels = cv2.countNonZero(roi)
            total_pixels = (max_x - min_x) * (max_y - min_y)
            white_ratio = white_pixels / max(total_pixels, 1)
            
            # Much lower threshold for dilated objects
            if white_ratio < self.min_white_percent * 0.5:
                continue
                
            # Lower confidence for detections from dilation
            confidence = 0.4 * min(elongation_ratio / 4.0, 1.0)
            
            # Create detection
            detection = {
                "box": [min_x, min_y, max_x, max_y],
                "confidence": confidence,
                "label": "ball",
                "center": (int(center_x), int(center_y)),
                "elongation_ratio": elongation_ratio,
                "angle": angle,
                "detection_type": "dilated_blob"
            }
            
            # Add motion direction
            motion_angle_rad = np.radians(angle)
            motion_dir = np.array([np.cos(motion_angle_rad), np.sin(motion_angle_rad)])
            detection["motion_direction"] = motion_dir.tolist()
            
            # Check if this blob is within any region
            region_id = self._get_region_for_point(int(center_x), int(center_y))
            if region_id:
                detection["region_id"] = region_id
                
            detections.append(detection)
            
        return detections
        
    def _merge_nearby_detections(self, detections, distance_threshold=30):
        """
        Merge nearby detections, keeping the highest confidence one.
        
        Args:
            detections: List of detection dictionaries
            distance_threshold: Maximum distance to consider detections as duplicates
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
            
        # Sort detections by confidence
        sorted_detections = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
        
        # List to track which detections to keep
        keep = [True] * len(sorted_detections)
        
        # Check each detection against others
        for i in range(len(sorted_detections)):
            if not keep[i]:
                continue
                
            det_i = sorted_detections[i]
            center_i = det_i.get("center", None)
            
            if center_i is None:
                # Get center from box if center not provided
                box_i = det_i.get("box", [0, 0, 0, 0])
                center_i = ((box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2)
                
            # Check against remaining detections
            for j in range(i+1, len(sorted_detections)):
                if not keep[j]:
                    continue
                    
                det_j = sorted_detections[j]
                center_j = det_j.get("center", None)
                
                if center_j is None:
                    # Get center from box if center not provided
                    box_j = det_j.get("box", [0, 0, 0, 0])
                    center_j = ((box_j[0] + box_j[2]) / 2, (box_j[1] + box_j[3]) / 2)
                    
                # Calculate distance between centers
                distance = np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
                
                # If close enough, mark the lower confidence one for removal
                if distance < distance_threshold:
                    keep[j] = False
                    
                    # Special case: if j is a blob detection (elongated) and i is circle
                    # copy motion direction to the higher confidence detection
                    if (det_i.get("detection_type", "") == "circle" and 
                        det_j.get("detection_type", "") in ["blob", "elongated_contour"] and
                        "motion_direction" in det_j):
                        det_i["motion_direction"] = det_j["motion_direction"]
                        det_i["is_elongated"] = True
                        det_i["elongation_ratio"] = det_j.get("elongation_ratio", 2.0)
        
        # Return only the detections marked to keep
        return [det for i, det in enumerate(sorted_detections) if keep[i]]

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
        Draw circle and blob detections on the frame for debugging.
        
        Args:
            frame: Frame to draw on
            detections: List of detected circles
            
        Returns:
            Frame with detections drawn
        """
        for det in detections:
            det_type = det.get("detection_type", "unknown")
            
            # Common detection properties
            confidence = det.get("confidence", 0.0)
            box = det.get("box", [0, 0, 0, 0])
            
            # Different colors based on detection type
            if det_type == "circle":
                center = det["center"]
                radius = det["radius"]
                
                # Color based on confidence: green for high confidence, yellow for medium, red for low
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.3:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw circle
                cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), color, 2)
                
                # Draw center point
                cv2.circle(frame, (int(center[0]), int(center[1])), 2, color, -1)
                
            elif det_type in ["blob", "elongated_contour"]:
                # For blob detections
                if "angle" in det and "elongation_ratio" in det:
                    # Draw elongated detection with its orientation
                    center = det.get("center", ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2))
                    angle = det["angle"]
                    elongation = det["elongation_ratio"]
                    
                    # Color based on elongation: more orange for more elongated
                    elong_factor = min((elongation - 1.0) / 2.0, 1.0)
                    blue = 0
                    green = int(165 * (1 - elong_factor))
                    red = 255
                    color = (blue, green, red)
                    
                    # Draw rotated rectangle
                    rect = ((center[0], center[1]), 
                           (int(elongation * 20), 20),  # Width, height
                           angle)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int0(box_points)
                    cv2.drawContours(frame, [box_points], 0, color, 2)
                    
                    # Draw motion direction
                    if "motion_direction" in det:
                        motion_dir = np.array(det["motion_direction"])
                        start_pt = (int(center[0]), int(center[1]))
                        end_pt = (int(center[0] + motion_dir[0] * 30), 
                                 int(center[1] + motion_dir[1] * 30))
                        cv2.arrowedLine(frame, start_pt, end_pt, (255, 0, 255), 2)
                else:
                    # Draw regular box for other blobs
                    color = (255, 128, 0)  # Orange
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            else:
                # Unknown detection type - use default drawing
                color = (0, 128, 255)  # Orange
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Add confidence text
            cv2.putText(
                frame,
                f"{confidence:.2f}",
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
                
        return frame 