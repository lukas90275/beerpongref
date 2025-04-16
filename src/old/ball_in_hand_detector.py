from collections import deque

import cv2
import numpy as np


class BallInHandDetector:
    def __init__(
        self,
        consecutive_frames_threshold=3,
        max_tracking_history=10,
        table_size_inches=(96, 48),
    ):
        """
        Initialize a detector to find ping pong balls within hand regions

        Args:
            consecutive_frames_threshold: Number of consecutive frames a ball must be
                                          detected to be considered valid
            max_tracking_history: Maximum number of frames to keep in tracking history
            table_size_inches: Size of the beer pong table in inches (length, width)
        """
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.max_tracking_history = max_tracking_history

        # Ball size in inches (standard ping pong ball is 1.57 inches in diameter)
        self.ball_diameter_inches = 1.57
        self.table_size_inches = table_size_inches

        # Pixel size will be calculated dynamically based on detected table
        self.pixels_per_inch = None
        self.estimated_ball_radius_pixels = None

        # Dictionary to track ball detections across frames
        # Key: ball_id (based on position), Value: deque of detection counts
        self.ball_tracking = {}
        self.next_ball_id = 0

        # Store the original image for debugging
        self.debug_original = None

    def calibrate_from_table(self, table_bounds):
        """
        Calculate pixels per inch based on detected table bounds

        Args:
            table_bounds: Dictionary with x1, y1, x2, y2 coordinates of table
        """
        if table_bounds:
            table_width_pixels = table_bounds["x2"] - table_bounds["x1"]

            # Calculate pixels per inch based on table width
            self.pixels_per_inch = table_width_pixels / self.table_size_inches[0]

            # Estimate ball radius in pixels
            self.estimated_ball_radius_pixels = (
                self.ball_diameter_inches * self.pixels_per_inch
            ) / 2

            print(f"Table calibration: {self.pixels_per_inch:.2f} pixels/inch")
            print(
                f"Estimated ball radius: {self.estimated_ball_radius_pixels:.2f} pixels"
            )

    def detect_balls_in_hands(self, frame, hand_detections, table_bounds=None):
        """
        Detect ping pong balls in hand regions

        Args:
            frame: The input frame
            hand_detections: List of hand detection dictionaries with bounding boxes
            table_bounds: Optional table bounds for size calibration

        Returns:
            annotated_frame: Frame with detected balls highlighted
            ball_detections: List of dictionaries with ball information
        """
        # Store original frame for reference
        self.debug_original = frame.copy()

        # Calibrate using table size if provided and not already calibrated
        if table_bounds and not self.pixels_per_inch:
            self.calibrate_from_table(table_bounds)

        annotated_frame = frame.copy()
        ball_detections = []

        # Set of currently detected ball_ids in this frame
        current_ball_ids = set()

        # Process each hand region
        for hand in hand_detections:
            bbox = hand["bbox"]

            # Extract the hand region with expanded padding to account for motion blur
            # Use larger padding for hands that might be moving
            padding_x = 60  # Increased horizontal padding for motion blur
            padding_y = 60  # Increased vertical padding

            x1 = max(0, bbox["x1"] - padding_x)
            y1 = max(0, bbox["y1"] - padding_y)
            x2 = min(frame.shape[1], bbox["x2"] + padding_x)
            y2 = min(frame.shape[0], bbox["y2"] + padding_y)

            hand_region = frame[y1:y2, x1:x2]

            if hand_region.size == 0:  # Skip if region is empty
                continue

            # Debug visualization of hand region being analyzed
            hand_region_vis = hand_region.copy()

            # Detect balls in this hand region
            balls_in_hand = self._detect_balls(hand_region, hand_region_vis)

            # Optional: Uncomment to show hand region analysis for debugging
            # cv2.imshow(f"Hand Region {hand['id']}", hand_region_vis)

            # Adjust ball coordinates to the full frame
            for ball in balls_in_hand:
                # Adjust coordinates
                ball["center"] = (ball["center"][0] + x1, ball["center"][1] + y1)

                # Generate a unique ID or match to existing ball
                ball_id = self._get_ball_id(ball["center"], current_ball_ids)
                ball["id"] = ball_id
                current_ball_ids.add(ball_id)

                # Initialize tracking if this is a new ball
                if ball_id not in self.ball_tracking:
                    self.ball_tracking[ball_id] = deque(
                        [0] * self.max_tracking_history,
                        maxlen=self.max_tracking_history,
                    )

                # Update tracking (1 means detected in this frame)
                self.ball_tracking[ball_id].appendleft(1)

                # Count consecutive detections
                consecutive_detections = 0
                for detection in self.ball_tracking[ball_id]:
                    if detection == 1:
                        consecutive_detections += 1
                    else:
                        break

                ball["consecutive_detections"] = consecutive_detections
                ball["is_consistent"] = (
                    consecutive_detections >= self.consecutive_frames_threshold
                )

                # Add to result list
                ball_detections.append(ball)

                # Draw the ball on the frame
                self._draw_ball(annotated_frame, ball)

        # Update tracking for balls not detected in this frame
        for ball_id in list(self.ball_tracking.keys()):
            if ball_id not in current_ball_ids:
                self.ball_tracking[ball_id].appendleft(0)

                # Remove balls that haven't been seen for a while
                if sum(self.ball_tracking[ball_id]) == 0:
                    del self.ball_tracking[ball_id]

        return annotated_frame, ball_detections

    def _detect_balls(self, hand_region, debug_frame=None):
        """
        Detect potential ping pong balls in a hand region using simple image processing

        Args:
            hand_region: Region of the frame to analyze
            debug_frame: Optional frame for visualization

        Returns:
            balls: List of detected balls
        """
        # Try multiple approaches to handle different lighting conditions and motion blur
        detected_balls = []

        # Approach 1: HSV white detection
        detected_balls.extend(self._detect_white_balls_hsv(hand_region, debug_frame))

        # Approach 2: Enhanced method for blurred balls using HoughCircles
        detected_balls.extend(
            self._detect_blurred_white_balls(hand_region, debug_frame)
        )

        # Remove duplicates - merge balls that are too close to each other
        merged_balls = self._merge_close_balls(detected_balls)

        return merged_balls

    def _detect_white_balls_hsv(self, hand_region, debug_frame=None):
        """
        Detect white ping pong balls using HSV color filtering
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)

        # Define range for white color in HSV
        # White has low saturation and high value
        lower_white = np.array(
            [0, 0, 200]
        )  # Increased value threshold to focus on brighter whites
        upper_white = np.array([180, 40, 255])  # Decreased saturation tolerance

        # Create mask for white regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Apply morphological operations to reduce noise and handle motion blur
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(opening, kernel, iterations=1)

        # Visualize mask in debug frame
        if debug_frame is not None:
            debug_frame_mask = cv2.bitwise_and(hand_region, hand_region, mask=dilated)
            cv2.addWeighted(debug_frame, 0.7, debug_frame_mask, 0.3, 0, debug_frame)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        balls = []

        # Define size constraints based on calibration or defaults
        min_ball_area = 50  # Default minimum area
        max_ball_area = 800  # Default maximum area

        if self.estimated_ball_radius_pixels:
            # If we have calibrated size, use it for constraints
            ball_area = np.pi * (self.estimated_ball_radius_pixels**2)

            # Account for motion blur by allowing a range around the estimated size
            min_ball_area = max(
                30, int(ball_area * 0.3)
            )  # Decreased to catch smaller blurred balls
            max_ball_area = int(
                ball_area * 3.0
            )  # Increased to catch elongated blurred balls

        # Process each contour
        for contour in contours:
            # Filter by area - ping pong balls are small but not tiny
            area = cv2.contourArea(contour)
            if area < min_ball_area or area > max_ball_area:
                continue

            # Calculate circularity - but be more lenient due to motion blur
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            )

            # Lower threshold for circularity to account for motion blur
            circularity_threshold = 0.25  # Even more lenient

            # For debugging
            if debug_frame is not None and area > min_ball_area:
                # Draw all potential candidates with circularity values
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        debug_frame,
                        f"C:{circularity:.2f}",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            # Ping pong balls might not be perfectly circular due to motion blur
            if circularity < circularity_threshold:
                continue

            # Get the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Check if radius is reasonable for a ping pong ball
            if self.estimated_ball_radius_pixels:
                # Allow for more variation due to camera angle and motion blur
                if (
                    radius < self.estimated_ball_radius_pixels * 0.3
                    or radius > self.estimated_ball_radius_pixels * 3.0
                ):
                    continue

            # Calculate average brightness of the region inside the circle
            mask = np.zeros(hand_region.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_brightness = cv2.mean(
                cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY), mask=mask
            )[0]

            # Ensure it's bright enough to be a white ball
            if mean_brightness < 120:  # Adjust threshold based on lighting
                continue

            # Add to balls list
            balls.append(
                {
                    "center": center,
                    "radius": radius,
                    "circularity": circularity,
                    "area": area,
                    "brightness": mean_brightness,
                    "method": "hsv",
                }
            )

            # Draw on debug frame if provided
            if debug_frame is not None:
                cv2.circle(debug_frame, center, radius, (255, 0, 0), 1)
                cv2.putText(
                    debug_frame,
                    f"B:{mean_brightness:.0f}",
                    (center[0], center[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

        return balls

    def _detect_blurred_white_balls(self, hand_region, debug_frame=None):
        """
        Alternative method to detect white ping pong balls when heavily blurred
        """
        balls = []

        # Convert to grayscale
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Detect circles using Hough transform
        estimated_radius = 15  # Default if not calibrated
        if self.estimated_ball_radius_pixels:
            estimated_radius = max(10, int(self.estimated_ball_radius_pixels))

        # Parameters for HoughCircles
        min_radius = int(estimated_radius * 0.5)
        max_radius = int(estimated_radius * 2.5)

        # HoughCircles parameters may need tuning for your specific videos
        circles = cv2.HoughCircles(
            filtered,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=estimated_radius * 2,
            param1=80,  # Higher edge threshold
            param2=20,  # Lower accumulator threshold for motion blurred circles
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)

            for x, y, r in circles:
                # Verify it's a white ball by checking brightness
                mask = np.zeros(hand_region.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                mean_brightness = cv2.mean(gray, mask=mask)[0]

                # Only consider bright circles
                if mean_brightness > 150:  # Higher threshold for Hough detection
                    # Check for whiteness (low color saturation)
                    hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
                    mean_saturation = cv2.mean(hsv[:, :, 1], mask=mask)[0]

                    if mean_saturation < 40:  # Low saturation indicates white
                        balls.append(
                            {
                                "center": (x, y),
                                "radius": r,
                                "circularity": 0.8,  # Assumed value for Hough circles
                                "area": np.pi * r * r,
                                "brightness": mean_brightness,
                                "method": "hough",
                            }
                        )

                        # Draw on debug frame
                        if debug_frame is not None:
                            cv2.circle(debug_frame, (x, y), r, (0, 255, 255), 1)
                            cv2.putText(
                                debug_frame,
                                f"H:{mean_brightness:.0f}",
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1,
                            )

        return balls

    def _merge_close_balls(self, balls, distance_threshold=30):
        """
        Merge balls that are too close to each other (likely duplicates)
        """
        if len(balls) <= 1:
            return balls

        merged_balls = []
        used_indices = set()

        for i in range(len(balls)):
            if i in used_indices:
                continue

            current_ball = balls[i]
            merged_center_x = current_ball["center"][0]
            merged_center_y = current_ball["center"][1]
            merged_radius = current_ball["radius"]
            merged_count = 1

            # Find all balls close to this one
            for j in range(i + 1, len(balls)):
                if j in used_indices:
                    continue

                # Calculate distance between centers
                distance = np.sqrt(
                    (current_ball["center"][0] - balls[j]["center"][0]) ** 2
                    + (current_ball["center"][1] - balls[j]["center"][1]) ** 2
                )

                # If balls are close, merge them
                if distance < distance_threshold:
                    merged_center_x += balls[j]["center"][0]
                    merged_center_y += balls[j]["center"][1]
                    merged_radius += balls[j]["radius"]
                    merged_count += 1
                    used_indices.add(j)

            # If we have a merged ball, average the values
            if merged_count > 1:
                used_indices.add(i)
                merged_ball = current_ball.copy()
                merged_ball["center"] = (
                    int(merged_center_x / merged_count),
                    int(merged_center_y / merged_count),
                )
                merged_ball["radius"] = int(merged_radius / merged_count)
                merged_ball["merged_count"] = merged_count
                merged_balls.append(merged_ball)
            elif i not in used_indices:  # Just add the original ball
                merged_balls.append(current_ball)

        return merged_balls

    def _get_ball_id(self, center, current_ball_ids):
        """
        Get the ID for a ball based on its position.
        Either match to an existing ball or create a new ID.
        """
        # Check if this ball is close to any existing tracked ball
        for ball_id, history in self.ball_tracking.items():
            if ball_id in current_ball_ids:
                continue  # Skip balls already matched in this frame

            # Get the last known position of this ball
            last_positions = getattr(self, f"ball_{ball_id}_positions", [])

            if last_positions:
                last_center = last_positions[-1]
                distance = np.sqrt(
                    (center[0] - last_center[0]) ** 2
                    + (center[1] - last_center[1]) ** 2
                )

                # If the distance is small, it's likely the same ball
                # Use larger threshold to account for motion blur
                if distance < 80:  # Increased from 70 to account for fast movement
                    # Update position history
                    if len(last_positions) >= 10:
                        last_positions.pop(0)
                    last_positions.append(center)
                    setattr(self, f"ball_{ball_id}_positions", last_positions)
                    return ball_id

        # If no matching ball found, create a new ID
        new_id = self.next_ball_id
        self.next_ball_id += 1

        # Initialize position history
        setattr(self, f"ball_{new_id}_positions", [center])

        return new_id

    def _draw_ball(self, frame, ball):
        """Draw a detected ball on the frame"""
        center = ball["center"]
        radius = ball["radius"]
        is_consistent = ball["is_consistent"]

        # Get detection method for color coding
        method = ball.get("method", "unknown")
        merged_count = ball.get("merged_count", 1)

        # Color based on consistency and detection method
        if is_consistent:
            color = (0, 255, 0)  # Green for consistent
        elif method == "hough":
            color = (0, 255, 255)  # Yellow for Hough detections
        elif method == "hsv":
            color = (255, 0, 255)  # Magenta for HSV detections
        else:
            color = (0, 165, 255)  # Orange for other methods

        thickness = 2 if is_consistent else 1

        # Draw the ball
        cv2.circle(frame, center, radius, color, thickness)

        # Add label for consistent balls or merged detections
        if is_consistent or merged_count > 1:
            label = f"Ball {ball['id']}"
            if merged_count > 1:
                label += f" (M{merged_count})"

            cv2.putText(
                frame,
                label,
                (center[0] - radius, center[1] - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            # Draw additional marker for emphasis
            cv2.circle(frame, center, radius + 5, color, 1)
