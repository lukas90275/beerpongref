import uuid  # For unique IDs
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class TrackerConfig:
    """Configuration object for ObjectTracker to reduce parameter explosion"""
    
    def __init__(
        self,
        tracker_type="Object",
        color=(255, 0, 0),
        min_confidence_frames=3,
        max_lost_frames=5,
        search_expansion_factor=1.5,
        velocity_smoothing=0.7,
        velocity_threshold=1.0,
        position_stability_factor=0.0,
        use_x_distance_only=True,
        trajectory_len=0,
        detect_aspect_ratio_change=False,
        aspect_ratio_change_factor=1.5,
        aspect_ratio_history_len=5,
        search_region_bounds=None,
        source_region_id=None,
        directional_search_expansion=False,  # Whether to expand search box more in velocity direction
    ):
        # Basic tracker identification
        self.tracker_type = tracker_type
        self.color = color
        
        # Core tracking parameters
        self.min_confidence_frames = min_confidence_frames
        self.max_lost_frames = max_lost_frames
        self.search_expansion_factor = search_expansion_factor
        
        # Motion and stability parameters
        self.velocity_smoothing = velocity_smoothing
        self.velocity_threshold = velocity_threshold
        self.position_stability_factor = position_stability_factor
        self.use_x_distance_only = use_x_distance_only
        self.directional_search_expansion = directional_search_expansion
        
        # Feature flags and specialized parameters
        self.trajectory_len = trajectory_len
        self.detect_aspect_ratio_change = detect_aspect_ratio_change
        self.aspect_ratio_change_factor = aspect_ratio_change_factor 
        self.aspect_ratio_history_len = aspect_ratio_history_len
        self.search_region_bounds = search_region_bounds
        self.source_region_id = source_region_id


class ObjectTracker(ABC):
    """
    Unified tracking class for all object types.
    Manages state like box, velocity, confidence, and lost frames.
    Supports specialized features through configuration and overridable methods.
    """

    def __init__(
        self,
        initial_box,
        frame_shape,
        tracker_id=None,
        initial_confidence=1.0,
        config=None,
        **kwargs
    ):
        # Allow passing either a TrackerConfig object or keyword arguments
        self.config = config if config is not None else TrackerConfig(**kwargs)
        
        self.id = tracker_id if tracker_id is not None else uuid.uuid4()
        self.box = np.array(initial_box, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.raw_velocity = np.zeros(2, dtype=float)
        self.center = self._calculate_center(self.box)
        self.prev_center = self.center.copy()
        self.lost_frames = 0
        self.confidence_frames = 1  # Starts at 1
        self.is_confident = False
        self.is_lost = False
        self.frame_shape = frame_shape
        self.search_box = self.box.copy()  # Initial search box is just the box
        self.last_confidence = initial_confidence
        
        # Calculate the object's width and height
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]

        # Initialize position history for stability and predictions
        self.position_history = []
        self.max_position_history = 5
        self.position_history.append(self.center.copy())
        
        # Initialize trajectory tracking if enabled
        self.trajectory_positions = []
        if self.config.trajectory_len > 0:
            self.trajectory_positions.append(self.center.copy())
            
        # Initialize aspect ratio tracking if enabled
        self.is_standing = True
        if self.config.detect_aspect_ratio_change:
            self.original_aspect_ratio = self.width / self.height if self.height > 0 else 1.0
            self.aspect_ratio_history = [self.original_aspect_ratio]

        if self.confidence_frames >= self.config.min_confidence_frames:
            self.is_confident = True

    def _calculate_center(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def _update_box_from_center(self, center, width, height):
        x1 = center[0] - width / 2
        y1 = center[1] - height / 2
        x2 = center[0] + width / 2
        y2 = center[1] + height / 2
        return np.array([x1, y1, x2, y2])
        
    def calculate_distance(self, point_a, point_b):
        """
        Calculate distance between two points, using only x-direction if configured.
        Can be scaled by object size if needed.
        
        Args:
            point_a: First point as (x, y) or [x, y]
            point_b: Second point as (x, y) or [x, y]
            
        Returns:
            Distance (float): Euclidean or x-only distance
        """
        if self.config.use_x_distance_only:
            # Use only x-direction distance
            return abs(point_a[0] - point_b[0])
        else:
            # Use full Euclidean distance
            return np.linalg.norm(np.array(point_a) - np.array(point_b))
    
    def calculate_scaled_distance(self, point_a, point_b):
        """
        Calculate distance scaled by object size (width).
        
        Args:
            point_a: First point as (x, y) or [x, y]
            point_b: Second point as (x, y) or [x, y]
            
        Returns:
            Scaled distance (float): Distance scaled by object width
        """
        raw_distance = self.calculate_distance(point_a, point_b)
        
        # If width is very small, avoid division by zero
        if self.width < 1.0:
            return raw_distance
            
        # Scale the distance by the object's width
        return raw_distance / self.width

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
        
    def _clip_to_bounds(self, box, bounds):
        """Clip box coordinates to stay within specified bounds."""
        if bounds is None:
            return box
        box[0] = np.clip(box[0], bounds[0], bounds[2])
        box[1] = np.clip(box[1], bounds[1], bounds[3])
        box[2] = np.clip(box[2], bounds[0], bounds[2])
        box[3] = np.clip(box[3], bounds[1], bounds[3])
        # Ensure x1 < x2 and y1 < y2 after clipping
        if box[0] >= box[2]: box[2] = box[0] + 1
        if box[1] >= box[3]: box[3] = box[1] + 1
        # Clip again to ensure adjustment didn't exceed bounds
        box[2] = np.clip(box[2], bounds[0], bounds[2]) 
        box[3] = np.clip(box[3], bounds[1], bounds[3])
        return box

    def predict(self):
        """
        Predict the bounding box and search area for the next frame based on velocity.
        The search box is expanded based on lost frames.
        Returns the predicted box (current self.box) and the search box.
        """
        # Scale velocity by lost frames if we're searching for a lost object
        velocity_scale = 1.0
        if self.lost_frames > 0:
            velocity_scale = min(self.lost_frames, 3)  # Scale up to 3x for lost objects

        # Predict next center based on velocity
        predicted_center = self.center + (self.velocity * velocity_scale)

        # Update predicted box position (this becomes the tracker's current 'box')
        width = self.box[2] - self.box[0]
        height = self.box[3] - self.box[1]
        self.box = self._update_box_from_center(predicted_center, width, height)
        self.center = predicted_center  # Update center as well
        
        # Update width and height properties
        self.width = width
        self.height = height
        
        # Apply position stability if enabled
        if self.config.position_stability_factor > 0 and len(self.position_history) >= 2:
            # Check if the object is moving significantly
            # Use configured distance calculation method
            recent_motion = 0
            if self.config.use_x_distance_only:
                recent_motion = abs(self.position_history[-1][0] - self.position_history[0][0])
            else:
                recent_motion = np.linalg.norm(self.position_history[-1] - self.position_history[0])
            
            # Define threshold based on object size
            motion_threshold = min(5.0, self.width * 0.2)
            
            # If object is relatively stationary, apply stability correction
            if recent_motion < motion_threshold:
                # Average position from history for stability
                stable_center = np.mean(self.position_history, axis=0)
                
                # Blend current predicted center with stable center
                stabilized_center = (
                    self.center * (1 - self.config.position_stability_factor) + 
                    stable_center * self.config.position_stability_factor
                )
                
                # Update box position based on stabilized center
                self.box = self._update_box_from_center(stabilized_center, width, height)
                
                # Update center
                self.center = stabilized_center

        # Define base expansion factor
        if self.lost_frames > 0:
            # Expand search area if object was lost
            base_expansion = self.config.search_expansion_factor * self.lost_frames
        else:
            base_expansion = self.config.search_expansion_factor
            
        # Use directional search expansion if enabled and x-velocity is significant
        if self.config.directional_search_expansion and abs(self.velocity[0]) > self.config.velocity_threshold:
            # Only consider x-direction velocity for directional expansion
            direction_x = np.sign(self.velocity[0])
            
            # Calculate expansions - double in direction of motion, normal in other direction
            if direction_x > 0:  # Moving right
                left_expansion = width * base_expansion
                right_expansion = width * base_expansion * 2
            else:  # Moving left
                left_expansion = width * base_expansion * 2
                right_expansion = width * base_expansion
            
            # Normal expansion in y-direction
            top_expansion = height * base_expansion
            bottom_expansion = height * base_expansion
        else:
            # Standard expansion in all directions
            left_expansion = width * base_expansion
            right_expansion = width * base_expansion
            top_expansion = height * base_expansion
            bottom_expansion = height * base_expansion
        
        # Create search box with directional expansion
        self.search_box = np.array([
            self.center[0] - left_expansion,
            self.center[1] - top_expansion,
            self.center[0] + right_expansion,
            self.center[1] + bottom_expansion
        ])

        # Clip boxes to frame boundaries
        self.box = self._clip_to_frame(self.box)
        self.search_box = self._clip_to_frame(self.search_box)
        
        # Clip search box to region bounds if specified (for cup tracking)
        if self.config.search_region_bounds is not None:
            self.search_box = self._clip_to_bounds(
                self.search_box, self.config.search_region_bounds
            )
            
        # Hook for subclasses to customize prediction
        self._custom_predict()

        return self.box.astype(int), self.search_box.astype(int)
        
    def _custom_predict(self):
        """Hook for subclasses to add custom prediction logic without overriding the full predict method"""
        pass

    def update(self, detection_box, detection_confidence=1.0, **kwargs):
        """
        Update the tracker state with a new assigned detection.
        """
        detection_box = np.array(detection_box, dtype=float)
        new_center = self._calculate_center(detection_box)

        # Store previous center
        self.prev_center = self.center.copy()

        # Calculate raw velocity - use the appropriate distance method
        if self.config.use_x_distance_only:
            # Only track x velocity
            self.raw_velocity[0] = new_center[0] - self.center[0]
            # Still calculate y velocity but with less influence
            self.raw_velocity[1] = (new_center[1] - self.center[1]) * 0.5
        else:
            self.raw_velocity = new_center - self.center

        # Apply smoothing to reduce jitter
        self.velocity = self.velocity * self.config.velocity_smoothing + self.raw_velocity * (
            1 - self.config.velocity_smoothing
        )

        # Apply threshold to velocity to reduce small jitter
        mask = np.abs(self.velocity) < self.config.velocity_threshold
        self.velocity[mask] = 0.0

        # Update state
        self.box = detection_box
        self.center = new_center
        self.lost_frames = 0
        self.is_lost = False
        self.confidence_frames += 1
        self.last_confidence = detection_confidence
        
        # Update width and height properties
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]
        
        if self.confidence_frames >= self.config.min_confidence_frames:
            self.is_confident = True
            
        # Update position history
        self.position_history.append(self.center.copy())
        if len(self.position_history) > self.max_position_history:
            self.position_history.pop(0)
            
        # Update trajectory if enabled
        if self.config.trajectory_len > 0:
            self.trajectory_positions.append(self.center.copy())
            if len(self.trajectory_positions) > self.config.trajectory_len:
                self.trajectory_positions.pop(0)
                
        # Update aspect ratio history if enabled
        if self.config.detect_aspect_ratio_change and self.height > 0:
            current_aspect = self.width / self.height
            self.aspect_ratio_history.append(current_aspect)
            
            if len(self.aspect_ratio_history) > self.config.aspect_ratio_history_len:
                self.aspect_ratio_history.pop(0)
                
            # Analyze aspect ratio changes to detect if object has changed state (e.g., cup knocked over)
            if len(self.aspect_ratio_history) >= 3:
                avg_recent_aspect = np.mean(self.aspect_ratio_history[-3:])
                
                # If aspect ratio has changed significantly, state might have changed
                if self.is_standing and avg_recent_aspect > self.original_aspect_ratio * self.config.aspect_ratio_change_factor:
                    self.is_standing = False
                # If aspect ratio is back to normal, state might have reverted
                elif not self.is_standing and abs(avg_recent_aspect - self.original_aspect_ratio) < 0.2:
                    self.is_standing = True
            
        # Call hook for subclass-specific updates
        self._custom_update(**kwargs)

    def _custom_update(self, **kwargs):
        """Hook for subclasses to add custom update logic without overriding the full update method"""
        pass

    def mark_lost(self):
        """
        Mark the object as lost for this frame.
        """
        self.lost_frames += 1
        self.confidence_frames = 0  # Reset confidence streak
        self.is_confident = False

        # Dampen velocity less aggressively to allow better momentum-based prediction
        # Higher value = more retention of velocity when object is lost
        self.velocity *= 0.9  # Changed from 0.8 to 0.9

        if self.lost_frames >= self.config.max_lost_frames:
            self.is_lost = True

    def match_score(self, detection_box):
        """
        Calculate a score indicating how well the detection matches this tracker's current state.
        When the object is lost, use the search_box instead of the box for matching to allow
        for velocity-based prediction and expanding search area.
        """
        if self.lost_frames > 0:
            # Check if detection is within the search box
            detection_center = self._calculate_center(detection_box)
            
            # Use appropriate distance check based on configuration
            if self.config.use_x_distance_only:
                # Only check x-bounds for search box
                if not (self.search_box[0] <= detection_center[0] <= self.search_box[2]):
                    return 0.0
                # For y, be more lenient but still check
                if not (self.search_box[1] - self.height <= detection_center[1] <= self.search_box[3] + self.height):
                    return 0.0
            else:
                # Check full bounds
                if not (
                    self.search_box[0] <= detection_center[0] <= self.search_box[2]
                    and self.search_box[1] <= detection_center[1] <= self.search_box[3]
                ):
                    return 0.0
                    
            # If detection is in search box, use IoU with predicted box
            return self.calculate_iou(self.box, detection_box)
            
        # Normal tracking - use IoU with current box
        return self.calculate_iou(self.box, detection_box)

    def get_state(self):
        """
        Return the current state of the tracker.
        Includes all base fields and any enabled specialized fields.
        """
        state = {
            "id": self.id,
            "box": self.box.astype(int).tolist(),
            "search_box": self.search_box.astype(int).tolist(),
            "center": self.center.astype(int).tolist(),
            "velocity": self.velocity.tolist(),
            "lost_frames": self.lost_frames,
            "confidence_frames": self.confidence_frames,
            "is_confident": self.is_confident,
            "is_lost": self.is_lost,
            "last_confidence": self.last_confidence,
            "tracker_type": self.config.tracker_type,
            "width": self.width,
            "height": self.height,
            "diameter": max(self.width, self.height),  # Always include diameter
        }
        
        # Add specialized fields based on enabled features
        if self.config.source_region_id is not None:
            state["source_region_id"] = self.config.source_region_id
            
        if self.config.detect_aspect_ratio_change:
            state["is_standing"] = self.is_standing
            if len(self.aspect_ratio_history) > 0:
                state["aspect_ratio"] = self.aspect_ratio_history[-1]
            
        # Allow subclasses to add their own state
        self._extend_state(state)
            
        return state
        
    def _extend_state(self, state):
        """Hook for subclasses to extend the state without overriding the full get_state method"""
        pass

    def draw(self, frame, show_search_box=False):
        """
        Draw the tracker's state onto the frame.
        Handles drawing all enabled specialized visualizations.
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
                f"{self.config.tracker_type} Search",
                (sb[0], sb[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Draw current box
        b = self.box.astype(int)
        # Use tracker's own color if confident, red if lost
        color = self.config.color if self.lost_frames == 0 else (0, 0, 255)
        thickness = 2  # Consistent thickness
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, thickness)

        # Draw simple label (just the type)
        cv2.putText(
            frame,
            f"{self.config.tracker_type}",
            (b[0], b[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        
        # Draw trajectory if enabled and showing search box
        if show_search_box and self.config.trajectory_len > 0 and len(self.trajectory_positions) >= 2:
            for i in range(1, len(self.trajectory_positions)):
                pt1 = self.trajectory_positions[i-1].astype(int)
                pt2 = self.trajectory_positions[i].astype(int)
                cv2.line(frame, tuple(pt1), tuple(pt2), (0, 200, 255), 2)
                
        # Add custom drawing from subclasses
        self._custom_draw(frame, show_search_box)

        return frame
        
    def _custom_draw(self, frame, show_search_box):
        """Hook for subclasses to add custom drawing without overriding the full draw method"""
        pass

    # Helper function for IoU calculation
    @staticmethod
    def calculate_iou(boxA, boxB):
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

        # Return the intersection over union value
        return iou
