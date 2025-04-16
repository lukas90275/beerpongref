import cv2  # Import cv2 for drawing
import numpy as np

from trackers.object_tracker import ObjectTracker


class CupTracker(ObjectTracker):
    """
    Specific implementation for tracking cups on the beer pong table.
    Cups are mostly stationary and should only move when hit or manipulated.
    Search area is constrained to a predefined region (left/right table end).
    """

    def __init__(self, initial_box, frame_shape, search_region_bounds=None, **kwargs):
        """
        Initialize CupTracker.
        
        Args:
            initial_box: Initial bounding box [x1, y1, x2, y2].
            frame_shape: Shape of the video frame (height, width).
            search_region_bounds: Optional bounds [x1, y1, x2, y2] to constrain the search box.
            **kwargs: Additional arguments for ObjectTracker.
        """
        super().__init__(
            initial_box,
            frame_shape,
            max_lost_frames=15,  # Increased to avoid losing cups too quickly
            min_confidence_frames=5,  # Increased to require more consistent detections
            tracker_type="Cup",
            color=(0, 0, 255),  # Red in BGR format
            search_expansion_factor=1.2,  # Reduced because cups don't move much
            velocity_smoothing=0.95,  # Increased to reduce jitter (higher = more smoothing)
            velocity_threshold=3.0,  # Increased to ignore small movements
            **kwargs
        )
        self.search_region_bounds = np.array(search_region_bounds, dtype=float) if search_region_bounds else None

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
        """
        # Call parent predict first (calculates predicted box and search box, clips to frame)
        predicted_box, search_box = super().predict()
        
        # Now, further clip the search_box to the specific region bounds if they exist
        if self.search_region_bounds is not None:
            self.search_box = self._clip_box(self.search_box.copy(), self.search_region_bounds)
            search_box = self.search_box.copy()  # Update return value

        # Return the predicted box and the final search box (frame and region clipped)
        return predicted_box.astype(int), search_box.astype(int)

    def draw(self, frame, show_search_box=False):
        """
        Draw the tracker's state onto the frame.
        Overrides parent to potentially draw region bounds if desired (optional).
        """
        # Draw standard tracker elements (box, search box, label)
        frame = super().draw(frame, show_search_box=show_search_box)
        
        # Optionally draw the specific search region bounds for this tracker (e.g., in grey)
        # if self.search_region_bounds is not None:
        #     b = self.search_region_bounds.astype(int)
        #     cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (128, 128, 128), 1) # Grey dashed line? 
            
        return frame 