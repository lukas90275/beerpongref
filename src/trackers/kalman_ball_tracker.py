import uuid
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class KalmanBallTracker:
    """
    Ball tracker using Kalman filter for accurate prediction of ping pong ball trajectory.
    
    The Kalman filter uses a constant acceleration model with gravity to predict
    the ball's trajectory even during occlusions and erratic camera movements.
    """

    def __init__(
        self,
        initial_box,
        frame_shape,
        tracker_id=None,
        initial_confidence=1.0,
        source_region_id=None,
        max_lost_frames=30,  # Increased to be more tolerant of motion blur
        process_noise=1e-4,  # Increased 10x to allow more motion flexibility
        measurement_noise=0.05,  # Reduced to trust detections more
        initial_velocity=None,  # Initial velocity for fast-moving elongated balls
    ):
        # Basic identification
        self.id = tracker_id if tracker_id is not None else str(uuid.uuid4())
        self.frame_shape = frame_shape
        self.source_region_id = source_region_id
        
        # Status tracking
        self.lost_frames = 0
        self.max_lost_frames = max_lost_frames
        self.confidence_frames = 1
        self.is_confident = True
        self.is_lost = False
        self.last_confidence = initial_confidence
        self.is_detected_this_frame = True
        
        # Extract initial position from bounding box
        self.box = np.array(initial_box, dtype=float)
        self.center = self._calculate_center(self.box)
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]
        self.diameter = max(self.width, self.height)
        
        # Motion tracking
        self.velocity = np.zeros(2, dtype=float) if initial_velocity is None else np.array(initial_velocity)
        self.speed = np.linalg.norm(self.velocity)  # Speed magnitude
        self.prev_detected = True
        self.is_elongated = False  # Flag for elongated ball detection
        self.elongation_ratio = 1.0  # Default is circular
        
        # Trajectory history for visualization
        self.trajectory_positions = [self.center.copy()]
        self.trajectory_len = 10
        
        # Search box for detection
        self.search_box = self.box.copy()
        
        # Size history tracking for better matching
        self.size_history = [(self.width, self.height)]
        self.max_size_history = 5
        
        # Initialize Kalman filter with constant acceleration model (x, y, vx, vy, ax, ay)
        # State: [x, y, vx, vy, ax, ay]
        # Measurement: [x, y]
        self.kalman = cv2.KalmanFilter(6, 2)
        
        # State transition matrix (physics model)
        # [ 1, 0, dt, 0, 0.5*dt^2, 0       ]
        # [ 0, 1, 0,  dt, 0,        0.5*dt^2]
        # [ 0, 0, 1,  0,  dt,       0       ]
        # [ 0, 0, 0,  1,  0,        dt      ]
        # [ 0, 0, 0,  0,  1,        0       ]
        # [ 0, 0, 0,  0,  0,        1       ]
        dt = 1.0  # Time step
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Measurement matrix (only position is directly measured)
        # [1, 0, 0, 0, 0, 0]
        # [0, 1, 0, 0, 0, 0]
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        # Process noise covariance (how much we trust the model)
        # Higher values = more responsive to measurements for velocity components
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        # Much more process noise for velocity components to handle fast motion better
        self.kalman.processNoiseCov[2,2] = process_noise * 100  # vx
        self.kalman.processNoiseCov[3,3] = process_noise * 100  # vy
        # More process noise for acceleration components
        self.kalman.processNoiseCov[4,4] = process_noise * 50   # ax
        self.kalman.processNoiseCov[5,5] = process_noise * 50   # ay
        
        # Measurement noise covariance (how much we trust measurements)
        # Lower values = more trust in measurements
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initialize state with position, zero velocity and gravity acceleration
        self.kalman.statePre = np.zeros((6, 1), np.float32)
        self.kalman.statePre[0] = self.center[0]
        self.kalman.statePre[1] = self.center[1]
        
        # Initialize with provided velocity if available
        if initial_velocity is not None:
            self.kalman.statePre[2] = initial_velocity[0]  # vx
            self.kalman.statePre[3] = initial_velocity[1]  # vy
        
        # Initialize with gravity for y-axis acceleration (positive since y increases downwards in image)
        self.kalman.statePre[5] = 9.8  # Gravity in pixels/frame^2 (will be scaled automatically by the filter)
        
        # Initial state covariance
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)
        
        # For fast-moving balls with initial velocity, increase uncertainty in position
        if initial_velocity is not None and np.linalg.norm(initial_velocity) > 10:
            self.kalman.errorCovPost[0, 0] *= 2.0  # x position
            self.kalman.errorCovPost[1, 1] *= 2.0  # y position
    
    def _calculate_center(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        
    def _update_box_from_center(self, center, width, height):
        x1 = center[0] - width / 2
        y1 = center[1] - height / 2
        x2 = center[0] + width / 2
        y2 = center[1] + height / 2
        return np.array([x1, y1, x2, y2])
        
    def _clip_to_frame(self, box):
        """Clip box coordinates to stay within frame boundaries."""
        h, w = self.frame_shape[:2]
        box[0] = np.clip(box[0], 0, w - 1)
        box[1] = np.clip(box[1], 0, h - 1)
        box[2] = np.clip(box[2], 0, w - 1)
        box[3] = np.clip(box[3], 0, h - 1)
        # Ensure x1 < x2 and y1 < y2
        if box[0] >= box[2]:
            box[2] = box[0] + 1
        if box[1] >= box[3]:
            box[3] = box[1] + 1
        box[2] = np.clip(box[2], 0, w - 1)  # Clip again if adjusted
        box[3] = np.clip(box[3], 0, h - 1)
        return box
        
    def predict(self):
        """
        Predict the ball's next position using Kalman filter.
        Uses a physics-based model including gravity.
        """
        # Store previous detection state
        self.prev_detected = self.is_detected_this_frame
        
        # Reset detection flag at beginning of frame
        self.is_detected_this_frame = False
        
        # Increment lost frames counter
        if not self.prev_detected:
            self.lost_frames += 1
            
            # Mark as lost if too many frames without detection
            if self.lost_frames >= self.max_lost_frames:
                self.is_lost = True
                self.is_confident = False
        
        # Predict next state
        predicted_state = self.kalman.predict()
        
        # Extract position from state
        predicted_center = np.array([predicted_state[0, 0], predicted_state[1, 0]])
        
        # Extract velocity for search box expansion
        self.velocity = np.array([predicted_state[2, 0], predicted_state[3, 0]])
        
        # Calculate speed (magnitude of velocity)
        self.speed = np.linalg.norm(self.velocity)
        
        # Update box position using predicted center
        self.box = self._update_box_from_center(predicted_center, self.width, self.height)
        self.center = predicted_center
        
        # Expand search box based on velocity and lost frames
        # The longer we've lost the ball, the larger the search area
        # Base expansion factor scales with lost frames
        expansion_factor = 1.5 + (self.lost_frames * 0.25)  # More aggressive growth
        expansion_factor = min(expansion_factor, 4.0)  # Cap at 4x size (up from 3x)
        
        # Also scale expansion with speed for fast-moving balls
        speed_factor = 1.0
        if self.speed > 10.0:  # For high-speed balls
            speed_factor = 1.0 + min(self.speed / 50.0, 2.0)  # Cap at 3x expansion
            
        # Apply combined scaling
        expansion_factor *= speed_factor
        
        # Get velocity direction for directional expansion
        vel_dir = self.velocity / (self.speed + 1e-6)  # Normalized direction vector
        
        # Determine search box shape based on motion
        if self.speed > 5.0:  # Only apply directional expansion for significant motion
            # Higher expansion in velocity direction to handle motion blur
            motion_scale = 2.0 + min(self.speed / 20.0, 3.0)  # More expansion at higher speeds
            search_forward = self.diameter * expansion_factor * motion_scale
            search_side = self.diameter * expansion_factor * 0.8  # Less expansion perpendicular to motion
            
            # For diagonal motion, create rotated search box
            if abs(vel_dir[0]) > 0.3 and abs(vel_dir[1]) > 0.3:
                # Calculate major and minor axis lengths
                major_axis = self.diameter * expansion_factor * motion_scale
                minor_axis = self.diameter * expansion_factor * 0.8
                
                # Extended box in velocity direction
                forward_pt = self.center + vel_dir * major_axis
                backward_pt = self.center - vel_dir * (major_axis * 0.5)  # Less expansion backward
                
                # Create perpendicular vector for minor axis
                perp_vec = np.array([-vel_dir[1], vel_dir[0]])
                
                # Calculate four corners of the search box
                c1 = forward_pt + perp_vec * minor_axis * 0.5
                c2 = forward_pt - perp_vec * minor_axis * 0.5
                c3 = backward_pt - perp_vec * minor_axis * 0.5
                c4 = backward_pt + perp_vec * minor_axis * 0.5
                
                # Find bounding box of these points
                min_x = min(c1[0], c2[0], c3[0], c4[0])
                min_y = min(c1[1], c2[1], c3[1], c4[1])
                max_x = max(c1[0], c2[0], c3[0], c4[0])
                max_y = max(c1[1], c2[1], c3[1], c4[1])
                
                self.search_box = np.array([min_x, min_y, max_x, max_y])
            else:
                # More expansion in primary direction of motion
                if abs(vel_dir[0]) > abs(vel_dir[1]):  # Moving horizontally
                    search_x_expansion = search_forward if vel_dir[0] > 0 else search_forward * 0.7
                    search_x_backward = search_forward * 0.7 if vel_dir[0] > 0 else search_forward
                    search_y_expansion = search_side
                    
                    self.search_box = np.array([
                        self.center[0] - search_x_backward,
                        self.center[1] - search_y_expansion,
                        self.center[0] + search_x_expansion,
                        self.center[1] + search_y_expansion
                    ])
                else:  # Moving vertically
                    search_y_expansion = search_forward if vel_dir[1] > 0 else search_forward * 0.7
                    search_y_backward = search_forward * 0.7 if vel_dir[1] > 0 else search_forward
                    search_x_expansion = search_side
                    
                    self.search_box = np.array([
                        self.center[0] - search_x_expansion,
                        self.center[1] - search_y_backward,
                        self.center[0] + search_x_expansion,
                        self.center[1] + search_y_expansion
                    ])
        else:
            # Standard expansion for slow or stationary balls
            search_expansion = self.diameter * expansion_factor
            self.search_box = np.array([
                self.center[0] - search_expansion,
                self.center[1] - search_expansion,
                self.center[0] + search_expansion,
                self.center[1] + search_expansion
            ])
        
        # Clip boxes to frame boundaries
        self.box = self._clip_to_frame(self.box)
        self.search_box = self._clip_to_frame(self.search_box)
        
        return self.box.astype(int), self.search_box.astype(int)
        
    def update(self, detection_box, detection_confidence=1.0, **kwargs):
        """
        Update the Kalman filter with a new detection.
        
        Args:
            detection_box: New bounding box [x1, y1, x2, y2]
            detection_confidence: Confidence of the detection
            **kwargs: Additional arguments including possible motion direction
        """
        # Mark as detected this frame
        self.is_detected_this_frame = True
        self.last_confidence = detection_confidence
        self.lost_frames = 0
        self.is_lost = False
        self.confidence_frames += 1
        
        # Extract the center point from the detection box
        detected_center = self._calculate_center(detection_box)
        
        # Get detection width and height
        detected_width = detection_box[2] - detection_box[0]
        detected_height = detection_box[3] - detection_box[1]
        
        # Calculate elongation ratio
        width_to_height = detected_width / max(detected_height, 1.0)
        height_to_width = detected_height / max(detected_width, 1.0)
        current_elongation = max(width_to_height, height_to_width)
        
        # Update elongation state
        self.is_elongated = current_elongation > 1.5
        self.elongation_ratio = current_elongation
        
        # Check if motion direction provided (from blob detection)
        motion_direction = kwargs.get("motion_direction", None)
        
        # For elongated detections, adapt the box and handle potential motion blur
        if self.is_elongated:
            # Get the longer dimension
            is_horizontal = detected_width > detected_height
            
            if motion_direction is None and self.speed > 5.0:
                # Estimate motion direction from current velocity
                motion_direction = self.velocity / max(self.speed, 1e-6)
            
            # Adjust size to compensate for motion blur
            if is_horizontal:
                # For horizontal elongation, reduce width to approach original aspect ratio
                adjusted_width = detected_width
                if self.speed > 10.0:
                    # Scale down width based on elongation ratio (more elongated = more reduction)
                    reduction_factor = min(1.0, 1.5 / current_elongation)
                    adjusted_width = detected_width * reduction_factor
                
                # Keep height as detected
                self.width = adjusted_width
                self.height = detected_height
            else:
                # For vertical elongation, reduce height
                adjusted_height = detected_height
                if self.speed > 10.0:
                    reduction_factor = min(1.0, 1.5 / current_elongation)
                    adjusted_height = detected_height * reduction_factor
                
                # Keep width as detected
                self.width = detected_width
                self.height = adjusted_height
                
            # Update diameter
            self.diameter = max(self.width, self.height)
        else:
            # Normal non-elongated update
            self.width = detected_width
            self.height = detected_height
            self.diameter = max(self.width, self.height)
        
        # Update size history
        self.size_history.append((self.width, self.height))
        if len(self.size_history) > self.max_size_history:
            self.size_history.pop(0)
        
        # Create measurement matrix with detected position
        measurement = np.array([[detected_center[0]], [detected_center[1]]], dtype=np.float32)
        
        # Update Kalman filter with measurement
        corrected_state = self.kalman.correct(measurement)
        
        # Update bounding box with corrected position
        corrected_center = np.array([corrected_state[0, 0], corrected_state[1, 0]])
        self.box = self._update_box_from_center(corrected_center, self.width, self.height)
        self.center = corrected_center
        
        # Update velocity from filter state
        self.velocity = np.array([corrected_state[2, 0], corrected_state[3, 0]])
        self.speed = np.linalg.norm(self.velocity)
        
        # Update trajectory history
        self.trajectory_positions.append(self.center.copy())
        if len(self.trajectory_positions) > self.trajectory_len:
            self.trajectory_positions.pop(0)
            
        # Ensure box stays within frame
        self.box = self._clip_to_frame(self.box)
            
    def get_state(self):
        """
        Return the current state for external use.
        """
        velocity = np.array([self.kalman.statePost[2, 0], self.kalman.statePost[3, 0]])
        acceleration = np.array([self.kalman.statePost[4, 0], self.kalman.statePost[5, 0]])
        
        state = {
            "id": self.id,
            "box": self.box.astype(int).tolist(),
            "search_box": self.search_box.astype(int).tolist(),
            "center": self.center.astype(int).tolist(),
            "velocity": velocity.tolist(),
            "acceleration": acceleration.tolist(),
            "speed": float(self.speed),  # Add speed magnitude
            "lost_frames": self.lost_frames,
            "confidence_frames": self.confidence_frames,
            "is_confident": self.is_confident,
            "is_lost": self.is_lost,
            "last_confidence": self.last_confidence,
            "tracker_type": "KalmanBall",
            "width": self.width,
            "height": self.height,
            "diameter": self.diameter,
            "is_detected_this_frame": self.is_detected_this_frame,
            "is_elongated": self.is_elongated,
            "elongation_ratio": self.elongation_ratio
        }
        
        # Add source region if available
        if self.source_region_id is not None:
            state["source_region_id"] = self.source_region_id
            
        return state
    
    def get_search_box(self):
        """Get the current search box for the ball."""
        return self.search_box.astype(int)
    
    def get_center(self):
        """Get the center point of the ball"""
        return self.center.copy()
        
    def get_average_size(self):
        """
        Get the average size of the ball from recent history.
        
        Returns:
            Tuple (avg_width, avg_height) of the ball's average dimensions
        """
        if not self.size_history:
            return (self.width, self.height)
            
        widths, heights = zip(*self.size_history)
        return (sum(widths) / len(widths), sum(heights) / len(heights))
    
    def mark_lost(self):
        """Mark the ball as lost for this frame."""
        self.lost_frames += 1
        
        # Don't reset confidence too quickly for fast-moving balls
        if self.speed < 5.0 or self.lost_frames > 5:
            self.confidence_frames = max(0, self.confidence_frames - 1)
            if self.confidence_frames < 2:
                self.is_confident = False
        
        if self.lost_frames >= self.max_lost_frames:
            self.is_lost = True
            self.is_confident = False
    
    def calculate_iou(self, boxA, boxB):
        """Calculate Intersection over Union between two boxes."""
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union
        iou = (
            interArea / float(boxAArea + boxBArea - interArea)
            if (boxAArea + boxBArea - interArea) > 0
            else 0
        )

        return iou
    
    def match_score(self, detection_box):
        """
        Calculate a match score for a detection.
        Uses search box for lost trackers, regular box otherwise.
        
        For fast-moving objects, we're more lenient with matching.
        """
        if self.lost_frames > 0:
            # Check if detection center is within the search box
            detection_center = self._calculate_center(detection_box)
            
            # Check search box bounds
            if not (
                self.search_box[0] <= detection_center[0] <= self.search_box[2]
                and self.search_box[1] <= detection_center[1] <= self.search_box[3]
            ):
                return 0.0
                    
            # If in search box, calculate base match score from IoU
            base_score = self.calculate_iou(self.box, detection_box)
            
            # For fast-moving balls, we're more lenient - use distance-based score
            if self.speed > 10.0:
                # Calculate distance between predicted center and detection
                dist = np.linalg.norm(self.center - detection_center)
                
                # Scale by the ball diameter to get a normalized distance
                scaled_dist = dist / max(self.diameter, 1.0)
                
                # Convert to a score (1.0 is perfect match, 0.0 is bad)
                # We're more lenient with faster balls
                dist_score = max(0.0, 1.0 - scaled_dist / (1.0 + self.speed / 50.0))
                
                # Use higher of IoU or distance score
                return max(base_score, dist_score * 0.8)  # Cap distance score at 0.8
            
            return base_score
            
        # Normal tracking - use IoU with current box
        return self.calculate_iou(self.box, detection_box)
    
    def draw(self, frame, show_search_box=False):
        """
        Draw the tracker's visualization on the frame.
        """
        # Draw search box if requested
        if show_search_box:
            sb = self.search_box.astype(int)
            cv2.rectangle(
                frame, (sb[0], sb[1]), (sb[2], sb[3]), (0, 255, 255), 2, cv2.LINE_AA
            )
            # Add search box label
            cv2.putText(
                frame,
                "Kalman Ball Search",
                (sb[0], sb[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Draw current box
        b = self.box.astype(int)
        
        # Choose color based on ball state
        if self.lost_frames == 0:
            if self.is_elongated:
                # Orange for elongated/fast balls
                color = (0, 165, 255)
            elif self.speed > 20.0:  
                # Yellow-orange for fast balls
                color = (0, 200, 255)
            else:
                # Yellow for normal balls
                color = (0, 255, 255)
        else:
            # Red for lost balls
            color = (0, 0, 255)
            
        thickness = 2
        
        # Draw special visualization for elongated balls
        if self.is_elongated and self.elongation_ratio > 1.5:
            # Draw a rotated rectangle to show orientation based on velocity
            if np.linalg.norm(self.velocity) > 1.0:
                # Calculate angle from velocity vector
                angle_rad = np.arctan2(self.velocity[1], self.velocity[0])
                angle_deg = np.degrees(angle_rad)
                
                # Create a rotated rectangle
                center = tuple(self.center.astype(int))
                width = int(self.width)
                height = int(self.height)
                
                # Ensure width is the longer dimension
                if width < height:
                    width, height = height, width
                    angle_deg += 90
                
                # Create the rotated rectangle
                rect = ((center[0], center[1]), (width, height), angle_deg)
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)
                
                # Draw rotated rectangle
                cv2.drawContours(frame, [box_points], 0, color, thickness)
                
                # Add elongation info to label
                label = f"Ball {self.speed:.1f} ({self.elongation_ratio:.1f}x)"
            else:
                # Fallback to regular rectangle if velocity is too small
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, thickness)
                label = f"Ball {self.elongation_ratio:.1f}x"
        else:
            # Regular rectangle for normal balls
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, thickness)
            
            # Add speed to label
            label = f"Ball {self.speed:.1f}" if self.speed > 1.0 else "Ball"
            
        # Draw label
        cv2.putText(
            frame,
            label,
            (b[0], b[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        
        # Draw trajectory if showing detailed debug info
        if show_search_box and len(self.trajectory_positions) > 1:
            # Draw trajectory path
            points = np.array(self.trajectory_positions, dtype=np.int32)
            cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, (0, 255, 255), 2)
            
            # Draw velocity vector
            if hasattr(self, 'velocity') and not np.all(self.velocity == 0):
                # Calculate endpoint based on velocity
                start_point = tuple(self.center.astype(int))
                # Scale velocity for visibility
                scale = 5.0
                end_point = tuple((self.center + self.velocity * scale).astype(int))
                # Draw arrow
                cv2.arrowedLine(frame, start_point, end_point, (0, 200, 255), 2)
                
            # Draw acceleration vector if available
            if hasattr(self.kalman, 'statePost') and self.kalman.statePost is not None:
                acceleration = np.array([self.kalman.statePost[4, 0], self.kalman.statePost[5, 0]])
                if not np.all(acceleration == 0):
                    # Calculate endpoint based on acceleration
                    start_point = tuple(self.center.astype(int))
                    # Scale acceleration for visibility
                    scale = 10.0
                    end_point = tuple((self.center + acceleration * scale).astype(int))
                    # Draw arrow (different color for acceleration)
                    cv2.arrowedLine(frame, start_point, end_point, (255, 0, 255), 2)

        return frame 