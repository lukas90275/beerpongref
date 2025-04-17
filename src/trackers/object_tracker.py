import uuid  # For unique IDs
from abc import ABC

import cv2
import numpy as np


class ObjectTracker(ABC):
    """
    Abstract base class for tracking objects across frames.
    Manages state like box, velocity, confidence, and lost frames.
    """

    def __init__(
        self,
        initial_box,
        frame_shape,
        tracker_id=None,  # Allow assigning specific ID or generate one
        min_confidence_frames=3,
        max_lost_frames=5,
        search_expansion_factor=1.5,
        initial_confidence=1.0,  # Add initial confidence if available
        tracker_type="Object",  # Added tracker type for labeling
        color=(255, 0, 0),  # Default color for visualization
        velocity_smoothing=0.7,  # Velocity smoothing factor (higher = more smoothing)
        velocity_threshold=1.0,  # Minimum threshold to consider velocity non-zero
        position_stability_factor=0.0,  # Factor to stabilize stationary objects (0.0 = off, 1.0 = max stability)
        use_x_distance_only=True,  # Whether to use only x-direction for distance calculations
    ):
        self.id = tracker_id if tracker_id is not None else uuid.uuid4()
        self.box = np.array(initial_box, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.raw_velocity = np.zeros(
            2, dtype=float
        )  # Store raw velocity before smoothing
        self.center = self._calculate_center(self.box)
        self.prev_center = (
            self.center.copy()
        )  # Store previous center for velocity calculation
        self.lost_frames = 0
        self.confidence_frames = 1  # Starts at 1
        self.is_confident = False
        self.is_lost = False
        self.frame_shape = frame_shape
        self.min_confidence_frames = min_confidence_frames
        self.max_lost_frames = max_lost_frames
        self.search_expansion_factor = search_expansion_factor
        self.search_box = self.box.copy()  # Initial search box is just the box
        self.last_confidence = initial_confidence  # Store confidence of last detection
        self.tracker_type = tracker_type  # Type of tracker (e.g., "Hand", "Table")
        self.color = color  # Color for visualization
        self.velocity_smoothing = velocity_smoothing  # Higher = more smoothing
        self.velocity_threshold = (
            velocity_threshold  # Minimum threshold to consider velocity non-zero
        )
        self.position_stability_factor = position_stability_factor  # How much to stabilize position
        self.use_x_distance_only = use_x_distance_only  # Whether to use only x-direction for distance
        
        # Track position history for stability and smoother predictions
        self.position_history = []
        self.max_position_history = 5
        self.position_history.append(self.center.copy())

        # Calculate the object's width and height (used for scaled distance calculations)
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]

        if self.confidence_frames >= self.min_confidence_frames:
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
        if self.use_x_distance_only:
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
        if self.position_stability_factor > 0 and len(self.position_history) >= 2:
            # Check if the object is moving significantly
            # Use configured distance calculation method
            recent_motion = 0
            if self.use_x_distance_only:
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
                    self.center * (1 - self.position_stability_factor) + 
                    stable_center * self.position_stability_factor
                )
                
                # Update box position based on stabilized center
                self.box = self._update_box_from_center(stabilized_center, width, height)
                
                # Update center
                self.center = stabilized_center

        # Define search box based on the *predicted* box
        if self.lost_frames > 0:
            # Expand search area if object was lost
            search_width = width * (self.search_expansion_factor**self.lost_frames)
            search_height = height * (self.search_expansion_factor**self.lost_frames)
        else:
            search_width = width * self.search_expansion_factor
            search_height = height * self.search_expansion_factor

        # Center the search box around the predicted center
        self.search_box = self._update_box_from_center(
            self.center, search_width, search_height
        )

        # Clip boxes to frame boundaries
        self.box = self._clip_to_frame(self.box)
        self.search_box = self._clip_to_frame(self.search_box)

        return self.box.astype(int), self.search_box.astype(int)

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

    def update(self, detection_box, detection_confidence=1.0):
        """
        Update the tracker state with a new assigned detection.
        """
        detection_box = np.array(detection_box, dtype=float)
        new_center = self._calculate_center(detection_box)

        # Store previous center
        self.prev_center = self.center.copy()

        # Calculate raw velocity - use the appropriate distance method
        if self.use_x_distance_only:
            # Only track x velocity
            self.raw_velocity[0] = new_center[0] - self.center[0]
            # Still calculate y velocity but with less influence
            self.raw_velocity[1] = (new_center[1] - self.center[1]) * 0.5
        else:
            self.raw_velocity = new_center - self.center

        # Apply smoothing to reduce jitter
        self.velocity = self.velocity * self.velocity_smoothing + self.raw_velocity * (
            1 - self.velocity_smoothing
        )

        # Apply threshold to velocity to reduce small jitter
        mask = np.abs(self.velocity) < self.velocity_threshold
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
        
        if self.confidence_frames >= self.min_confidence_frames:
            self.is_confident = True
            
        # Update position history
        self.position_history.append(self.center.copy())
        if len(self.position_history) > self.max_position_history:
            self.position_history.pop(0)

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

        if self.lost_frames >= self.max_lost_frames:
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
            if self.use_x_distance_only:
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
        Subclasses might add more info.
        """
        return {
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
            "tracker_type": self.tracker_type,
            "width": self.width,
            "height": self.height,
        }

    def draw(self, frame, show_search_box=False):
        """
        Draw the tracker's state onto the frame.
        Only shows search box with label if show_search_box is True.
        """
        # Draw search box (dashed yellow) if requested
        if show_search_box:
            sb = self.search_box.astype(int)
            cv2.rectangle(
                frame, (sb[0], sb[1]), (sb[2], sb[3]), (0, 255, 255), 2, cv2.LINE_AA
            )
            # Add search box label
            cv2.putText(
                frame,
                f"{self.tracker_type} Search",
                (sb[0], sb[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Draw current box
        b = self.box.astype(int)
        # Use tracker's own color if confident, red if lost
        color = self.color if self.lost_frames == 0 else (0, 0, 255)
        thickness = 2  # Consistent thickness
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, thickness)

        # Draw simple label (just the type)
        cv2.putText(
            frame,
            f"{self.tracker_type}",
            (b[0], b[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        return frame

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
