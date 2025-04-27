import cv2
import numpy as np

from trackers.ball_tracker import BallTracker
from tracking_managers.tracker_manager import TrackerManager


class BallTrackerManager(TrackerManager):
    """
    Specialized tracker manager for ping pong balls.
    Manages ball trackers and detects balls within specified regions.
    Uses size filtering and specialized search expansion for visualization.
    """

    # Real-world dimensions (constants)
    REAL_TABLE_WIDTH_FEET = 8.0  # Standard beer pong table length
    REAL_BALL_DIAMETER_INCHES = 1.57  # Standard ping pong ball diameter

    def __init__(
        self,
        iou_threshold=0.2,  # Reduced from 0.4 to allow more lenient matching
        min_confidence_frames=1,  # Reduced from 3 to detect balls immediately
        max_lost_frames=30,  # Increased from 15 to keep tracking longer
        detection_threshold=0.1,  # Reduced from 0.3 to detect lower confidence balls
        position_stability_factor=0.05,  # Reduced from 0.1 for even faster movement tracking
        size_filtering=True,  # Keep size filtering but we'll make it more lenient
        **kwargs
    ):
        # Initialize with ball-specific parameters
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=False,  # Can have multiple balls
            position_stability_factor=position_stability_factor,
            target_labels=["sports ball", "ball", "baseball", "tennis ball", "sphere"],
            cost_weight_iou=0.2,  # Reduced from 0.3 to favor distance more
            cost_weight_distance=0.8,  # Increased from 0.7 to favor distance more
            # Use more lenient match validation for fast-moving balls
            match_iou_leniency=0.9,  # More lenient (was 0.7)
            match_distance_multiplier=3.0,  # Allow matches at even greater distances (was 2.0)
            # Larger search boxes for visualization
            search_box_expansion_factor=3.0,  # Increased from 2.5
            # Custom region calibration for handling hand trackers
            region_calibration=self._add_hand_regions,
            # Region drawing parameters
            draw_region_color=(100, 255, 255),
            draw_region_label_prefix="Ball Region",
            # New parameters for region requirements
            require_regions=False,  # Changed from True to allow detection anywhere
            high_confidence_fallback=True,
            high_confidence_threshold=0.3,  # Reduced from 0.7
            **kwargs
        )
        # Initial values - size calibration will be done dynamically
        self.expected_ball_diameter_px = None
        
        # Register the ball tracker factory
        self.register_tracker_factory(self._create_ball_tracker)

    def _create_tracker_from_detection(self, detection, frame_shape):
        """Create a new tracker from detection - implements abstract method"""
        return self._create_ball_tracker(detection, frame_shape)

    def _create_ball_tracker(self, detection, frame_shape):
        """Create a new ball tracker from a detection."""
        box = self._extract_bbox_from_detection(detection)
        confidence = self._get_detection_confidence(detection)
        
        # Determine which region this ball was detected in
        source_region_id = detection.get("region_id", None)
        
        # Create new tracker
        tracker = BallTracker(
            initial_box=box,
            frame_shape=frame_shape,
            initial_confidence=confidence,
            source_region_id=source_region_id,
            position_stability_factor=self.position_stability_factor,
            min_confidence_frames=self.min_confidence_frames,
            max_lost_frames=self.max_lost_frames,
            search_expansion_factor=self.search_box_expansion_factor
        )
        
        # Force balls to be confident immediately for better tracking continuity
        if self.min_confidence_frames == 1:
            tracker.is_confident = True
            tracker.confidence_frames = 1
            
        return tracker

    def _add_hand_regions(self, hand_trackers=None, **_):
        """
        Set up ball detection regions based on hand tracker positions.
        
        Args:
            hand_trackers: List of hand tracker states
        """
        # Clear previous detection regions
        self.clear_regions()
        
        # If no hand trackers provided, do nothing
        if not hand_trackers:
            return
            
        # Add new detection regions from hand trackers
        added_regions = 0
        for hand in hand_trackers:
            if "ball_region" in hand:
                self.add_region(hand["ball_region"])
                added_regions += 1
                
        if added_regions > 0:
            print(f"Added {added_regions} ball regions from hand trackers")
        
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
            
            # Calculate size limits with much higher tolerance (150% instead of 80%)
            tolerance = 1.5
            min_size = self.expected_ball_diameter_px * (1 - tolerance)
            max_size = self.expected_ball_diameter_px * (1 + tolerance)
            
            # Set sizing parameters with very relaxed constraints
            self.size_min_width = max(min_size, 3)  # Ensure we don't go too small
            self.size_max_width = max_size
            self.size_min_height = max(min_size, 3)  # Ensure we don't go too small
            self.size_max_height = max_size
            self.size_min_aspect_ratio = 0.3  # Allow more oval shapes (was 0.5)
            self.size_max_aspect_ratio = 3.0  # Allow more oval shapes (was 2.0)
            
            print(f"Calibrated ping pong ball diameter: ~{self.expected_ball_diameter_px} pixels")

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
        # Use the new update method with hand_trackers passed as a keyword argument
        return self.update(detections, frame_shape, hand_trackers=hand_trackers)
        
    def draw_regions(self, frame, **kwargs):
        """
        Draw all ball detection regions onto the frame.
        
        Args:
            frame: Image to draw on
            **kwargs: Additional arguments passed to base class method
            
        Returns:
            frame: Frame with regions drawn
        """
        # Use the base class implementation with ball-specific defaults
        return super().draw_regions(
            frame,
            color=(100, 255, 255),  # Light yellow for ball regions
            label_prefix="Ball Region",
            **kwargs
        )

    def draw_trackers(self, frame, show_search_box=False):
        """
        Draw all trackers with enhanced visibility for balls.
        
        Args:
            frame: Frame to draw on
            show_search_box: Whether to show search boxes
            
        Returns:
            Frame with trackers drawn
        """
        # Update trackers' search boxes before drawing to ensure proper expansion based on velocity and lost frames
        for tracker in self.trackers:
            # Force prediction to update search box with velocity and lost frames
            tracker.predict()
            
        # Draw each tracker using its own draw method
        for tracker in self.trackers:
            frame = tracker.draw(frame, show_search_box)
        
        return frame
        
    # Legacy method for backward compatibility
    def process_detr_results(self, results, model, frame_shape, frame=None, hand_trackers=None, **kwargs):
        """
        Process DETR results and update with hand tracker regions if provided.
        Legacy method for backward compatibility.
        """
        # Forward to base implementation with hand_trackers passed along
        return super().process_detr_results(results, model, frame_shape, frame=frame, 
                                           hand_trackers=hand_trackers, **kwargs) 