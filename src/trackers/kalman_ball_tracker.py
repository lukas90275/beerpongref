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
        max_lost_frames=20,
        process_noise=1e-5,  # Process noise - lower for smoother trajectory
        measurement_noise=0.1,  # Measurement noise - higher means less trust in measurements
        motion_blur_velocity_threshold=15.0, # Velocity above which we suspect motion blur
        motion_blur_noise_multiplier=5.0,    # Factor to increase measurement noise by
        motion_blur_search_expansion_factor=2.5 # Factor to expand search box for blurred objects
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
        # Lower values = smoother trajectory, higher values = more responsive to measurements
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        # More process noise for acceleration components
        self.kalman.processNoiseCov[4,4] = process_noise * 10
        self.kalman.processNoiseCov[5,5] = process_noise * 10
        
        # Measurement noise covariance (how much we trust measurements)
        # Higher values = less trust in measurements, more trust in model predictions
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Store original measurement noise for adaptive adjustments
        self._original_measurement_noise = measurement_noise
        self._motion_blur_velocity_threshold = motion_blur_velocity_threshold
        self._motion_blur_noise_multiplier = motion_blur_noise_multiplier
        self._motion_blur_search_expansion_factor = motion_blur_search_expansion_factor
        
        # Initialize state with position, zero velocity and gravity acceleration
        self.kalman.statePre = np.zeros((6, 1), np.float32)
        self.kalman.statePre[0] = self.center[0]
        self.kalman.statePre[1] = self.center[1]
        # Initialize with gravity for y-axis acceleration (positive since y increases downwards in image)
        self.kalman.statePre[5] = 9.8  # Gravity in pixels/frame^2 (will be scaled automatically by the filter)
        
        # Initial state covariance
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)
        
        # Save initial velocity for directional search expansion
        self.velocity = np.zeros(2, dtype=float)

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
        # Reset detection flag at beginning of frame
        self.is_detected_this_frame = False
        
        # Increment lost frames counter
        if not self.is_detected_this_frame:
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
        
        # Update box position using predicted center
        self.box = self._update_box_from_center(predicted_center, self.width, self.height)
        self.center = predicted_center
        
        # Enhanced search box calculation for motion blur
        vel_mag = np.linalg.norm(self.velocity)
        vel_dir = self.velocity / (vel_mag + 1e-6)  # Normalized direction vector

        search_expansion_x = 0
        search_expansion_y = 0

        if self.lost_frames == 0 and vel_mag > self._motion_blur_velocity_threshold:
            # Case 1: Motion blur suspected (not lost, high velocity)
            # Aggressive and directional expansion
            base_expansion = self.diameter * self._motion_blur_search_expansion_factor
            
            # Expand more in the dominant direction of velocity
            # These multipliers control how much more the search box extends along the velocity vector
            # vs. perpendicular to it.
            dominant_axis_multiplier = 2.0  # e.g., expand 2x along the main motion axis
            minor_axis_multiplier = 1.0     # e.g., expand 1x perpendicular to main motion

            if abs(vel_dir[0]) > abs(vel_dir[1]):  # Primarily horizontal motion
                search_expansion_x = base_expansion * dominant_axis_multiplier
                search_expansion_y = base_expansion * minor_axis_multiplier
            else:  # Primarily vertical motion (or equal)
                search_expansion_x = base_expansion * minor_axis_multiplier
                search_expansion_y = base_expansion * dominant_axis_multiplier
        
        elif self.lost_frames > 0:
            # Case 2: Ball is lost
            # Expansion based on lost duration, capped, can be slightly directional
            lost_expansion_factor = 1.5 + (self.lost_frames * 0.2) # Slightly more aggressive
            lost_expansion_factor = min(lost_expansion_factor, 4.0) # Increased cap
            
            current_diameter = max(self.width, self.height, 1) # Use current diameter or a minimum of 1
            base_lost_expansion = current_diameter * lost_expansion_factor

            if vel_mag > 1.0: # If there's some residual velocity, make it somewhat directional
                directional_lost_multiplier = 1.2
                search_expansion_x = base_lost_expansion * (directional_lost_multiplier if abs(vel_dir[0]) > 0.7 else 1.0)
                search_expansion_y = base_lost_expansion * (directional_lost_multiplier if abs(vel_dir[1]) > 0.7 else 1.0)
            else: # Uniform expansion if no significant velocity
                search_expansion_x = base_lost_expansion
                search_expansion_y = base_lost_expansion
        else:
            # Case 3: Standard expansion (not lost, low velocity)
            standard_expansion_factor = 1.5 
            current_diameter = max(self.width, self.height, 1) # Use current diameter or a minimum of 1
            search_expansion_x = current_diameter * standard_expansion_factor
            search_expansion_y = current_diameter * standard_expansion_factor

        # Create the search box using calculated x and y expansions
        self.search_box = np.array([
            self.center[0] - search_expansion_x,
            self.center[1] - search_expansion_y,
            self.center[0] + search_expansion_x,
            self.center[1] + search_expansion_y
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
        """
        # Mark as detected this frame
        self.is_detected_this_frame = True
        self.last_confidence = detection_confidence
        self.lost_frames = 0
        self.is_lost = False
        self.confidence_frames += 1
        
        # Extract the center point from the detection box
        detected_center = self._calculate_center(detection_box)
        
        # Update width, height, and diameter
        self.width = detection_box[2] - detection_box[0]
        self.height = detection_box[3] - detection_box[1]
        self.diameter = max(self.width, self.height)
        
        # Update size history
        self.size_history.append((self.width, self.height))
        if len(self.size_history) > self.max_size_history:
            self.size_history.pop(0)
        
        # Create measurement matrix with detected position
        measurement = np.array([[detected_center[0]], [detected_center[1]]], dtype=np.float32)
        
        # Adapt measurement noise based on predicted velocity (to handle motion blur)
        predicted_velocity_magnitude = np.linalg.norm(self.kalman.statePre[2:4])
        
        original_measurement_noise_cov = self.kalman.measurementNoiseCov.copy() # Store current
        
        if predicted_velocity_magnitude > self._motion_blur_velocity_threshold:
            # If moving fast, increase measurement noise (trust prediction more)
            adapted_measurement_noise = self._original_measurement_noise * self._motion_blur_noise_multiplier
            self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * adapted_measurement_noise
        else:
            # Otherwise, use original measurement noise
            self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self._original_measurement_noise
            
        # Update Kalman filter with measurement
        corrected_state = self.kalman.correct(measurement)
        
        # Restore measurement noise covariance if it was changed
        # This ensures the next frame's prediction isn't unduly affected if this frame had blur
        if not np.array_equal(self.kalman.measurementNoiseCov, original_measurement_noise_cov):
            self.kalman.measurementNoiseCov = original_measurement_noise_cov
            
        # Update bounding box with corrected position
        corrected_center = np.array([corrected_state[0, 0], corrected_state[1, 0]])
        self.box = self._update_box_from_center(corrected_center, self.width, self.height)
        self.center = corrected_center
        
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
        self.confidence_frames = 0
        self.is_confident = False
        
        if self.lost_frames >= self.max_lost_frames:
            self.is_lost = True
    
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
                    
            # If in search box, use IoU with predicted box
            return self.calculate_iou(self.box, detection_box)
            
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
        # Yellow for balls, red if lost
        color = (0, 255, 255) if self.lost_frames == 0 else (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, thickness)

        # Draw label
        cv2.putText(
            frame,
            f"Kalman Ball",
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