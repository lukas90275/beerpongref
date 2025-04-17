import cv2  # Import cv2 for drawing
import numpy as np

from trackers.object_tracker import ObjectTracker


class CupTracker(ObjectTracker):
    """
    Specific implementation for tracking cups on the beer pong table.
    Cups are mostly stationary and should only move when hit or manipulated.
    Search area is constrained to a predefined region (left/right table end).
    """

    def __init__(
        self, 
        initial_box, 
        frame_shape, 
        search_region_bounds=None,
        position_stability_factor=0.95,  # Higher value = more stability for stationary cups
        use_x_distance_only=True,  # Use only x-distance for cup matching (side view)
        **kwargs
    ):
        """
        Initialize CupTracker.
        
        Args:
            initial_box: Initial bounding box [x1, y1, x2, y2].
            frame_shape: Shape of the video frame (height, width).
            search_region_bounds: Optional bounds [x1, y1, x2, y2] to constrain the search box.
            position_stability_factor: Factor to stabilize stationary cup positions (0.0-1.0)
            use_x_distance_only: Whether to use only x-direction for distance calculations
            **kwargs: Additional arguments for ObjectTracker.
        """
        # Pass parameters through kwargs
        kwargs["position_stability_factor"] = position_stability_factor
        kwargs["use_x_distance_only"] = use_x_distance_only
        
        super().__init__(
            initial_box,
            frame_shape,
            tracker_type="Cup",
            color=(0, 0, 255),  # Red in BGR format
            search_expansion_factor=1.2,  # Reduced because cups don't move much
            velocity_smoothing=0.95,  # Increased to reduce jitter (higher = more smoothing)
            velocity_threshold=3.0,  # Increased to ignore small movements
            **kwargs
        )
        self.search_region_bounds = np.array(search_region_bounds, dtype=float) if search_region_bounds else None
        
        # Track historical positions for stability
        self.position_history = []
        self.max_history_len = 5
        self.position_history.append(self.center.copy())
        
        # Track cup state (standing, knocked over, etc.)
        self.is_standing = True
        self.original_aspect_ratio = (self.box[2] - self.box[0]) / (self.box[3] - self.box[1])
        self.aspect_ratio_history = [self.original_aspect_ratio]
        self.max_aspect_ratio_history = 5
        
        # Cup diameter is approximately equal to its width
        self.cup_diameter = self.width

    def _clip_box(self, box, bounds):
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
        Predict the bounding box and search area, constraining search box to region bounds.
        For cups, prediction is heavily biased toward stability when not in motion.
        """
        # Call parent predict first (calculates predicted box and search box, clips to frame)
        predicted_box, search_box = super().predict()
        
        # Apply position stability for cups (they rarely move unless hit)
        # Note: Basic stability is now handled by the parent class
        
        # Update cup diameter based on current width
        self.cup_diameter = self.width
        
        # Now, further clip the search_box to the specific region bounds if they exist
        if self.search_region_bounds is not None:
            self.search_box = self._clip_box(self.search_box.copy(), self.search_region_bounds)
            search_box = self.search_box.copy()  # Update return value

        # Return the predicted box and the final search box (frame and region clipped)
        return predicted_box.astype(int), search_box.astype(int)

    def update(self, detection_box, detection_confidence=1.0):
        """
        Update the tracker state with a new assigned detection.
        Overrides parent to add cup-specific position stability and state tracking.
        """
        # Store previous box for aspect ratio analysis
        prev_box = self.box.copy()
        
        # Call parent update
        super().update(detection_box, detection_confidence)
        
        # Update cup diameter based on new width
        self.cup_diameter = self.width
        
        # Cup-specific position history update already handled by parent class
        
        # Calculate and track aspect ratio changes (to detect knocked over cups)
        current_width = self.width
        current_height = self.height
        
        if current_height > 0:  # Avoid division by zero
            current_aspect = current_width / current_height
            self.aspect_ratio_history.append(current_aspect)
            
            if len(self.aspect_ratio_history) > self.max_aspect_ratio_history:
                self.aspect_ratio_history.pop(0)
                
            # Analyze aspect ratio changes to detect if cup has been knocked over
            if len(self.aspect_ratio_history) >= 3:
                avg_recent_aspect = np.mean(self.aspect_ratio_history[-3:])
                
                # If aspect ratio has changed significantly, cup might be knocked over
                if self.is_standing and avg_recent_aspect > self.original_aspect_ratio * 1.5:
                    self.is_standing = False
                # If aspect ratio is back to normal, cup might be standing again
                elif not self.is_standing and abs(avg_recent_aspect - self.original_aspect_ratio) < 0.2:
                    self.is_standing = True

    def get_state(self):
        """
        Return the current state of the cup tracker.
        Extends parent to add cup-specific state information.
        """
        state = super().get_state()
        # Add cup-specific state information
        state["is_standing"] = self.is_standing
        state["cup_diameter"] = self.cup_diameter
        if len(self.aspect_ratio_history) > 0:
            state["aspect_ratio"] = self.aspect_ratio_history[-1]
        return state

    def draw(self, frame, show_search_box=False):
        """
        Draw the tracker's state onto the frame.
        Overrides parent to add cup-specific visualizations.
        """
        # Draw standard tracker elements (box, search box, label)
        frame = super().draw(frame, show_search_box=show_search_box)
        
        # Add cup state indicator
        b = self.box.astype(int)
        # Choose color based on cup state
        state_color = (0, 255, 0) if self.is_standing else (0, 0, 255)  # Green if standing, red if knocked over
        
        return frame 