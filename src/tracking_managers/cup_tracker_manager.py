import cv2
import numpy as np  # Import numpy for calculations

from trackers.cup_tracker import CupTracker
from tracking_managers.tracker_manager import TrackerManager


class CupTrackerManager(TrackerManager):
    """
    Manager for tracking multiple cups on a beer pong table.
    Uses table bounds to focus detection on the ends of the table.
    Filters detections based on expected size derived from table dimensions.
    Optimized for tracking stationary cups that only move when hit.
    """

    # Real-world dimensions (constants)
    REAL_TABLE_WIDTH_FEET = 8.0
    REAL_CUP_HEIGHT_INCHES = 4.85
    REAL_CUP_WIDTH_INCHES = 3.84

    def __init__(
        self,
        iou_threshold=0.6,  # Increased to require better overlap for matches
        min_confidence_frames=5,  # Increased to require more consistent detections
        max_lost_frames=15,  # Increased to avoid losing cups too quickly
        detection_threshold=0.65,  # Use this threshold for filtering
        size_tolerance=0.5,  # Allow 50% deviation in size
    ):
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=False,  # Track multiple cups
            motion_history_max_len=5,  # Increased for smoother motion estimation
            target_labels=["cup", "wine glass", "bottle"],  # Labels for cup detection
        )
        
        # These will be set when calibrating with the table
        self.left_region = None
        self.right_region = None
        self.pixels_per_foot = None
        self.expected_cup_height_pixels = None
        self.expected_cup_width_pixels = None
        self.size_tolerance = size_tolerance
        
    def _estimate_pixels_per_foot(self, table_width_pixels):
        """Estimate pixels per foot based on assumed table width."""
        if table_width_pixels > 0:
            return table_width_pixels / self.REAL_TABLE_WIDTH_FEET
        return None
        
    def calibrate_regions(self, table_bounds):
        """
        Calibrate the left and right cup regions based on the table bounds.
        Also estimates expected cup size based on table width.
        
        Args:
            table_bounds: Dictionary with x1, y1, x2, y2 coordinates of the table
        """
        if not table_bounds:
            self.pixels_per_foot = None
            self.expected_cup_height_pixels = None
            self.expected_cup_width_pixels = None
            return
            
        table_width_pixels = table_bounds["x2"] - table_bounds["x1"]
        table_height_pixels = table_bounds["y2"] - table_bounds["y1"]
        
        # Estimate scale
        self.pixels_per_foot = self._estimate_pixels_per_foot(table_width_pixels)
        
        if self.pixels_per_foot:
            self.expected_cup_height_pixels = (
                self.REAL_CUP_HEIGHT_INCHES / 12.0
            ) * self.pixels_per_foot
            self.expected_cup_width_pixels = (
                self.REAL_CUP_WIDTH_INCHES / 12.0
            ) * self.pixels_per_foot
        else:
            self.expected_cup_height_pixels = None
            self.expected_cup_width_pixels = None
            
        # Define regions as the outer 30% of each side of the table
        region_width = int(table_width_pixels * 0.3)
        
        # Calculate the height region to look slightly above and below the table top
        region_start_y = int(table_bounds["y1"] - table_height_pixels * 0.5)
        region_end_y = int(table_bounds["y1"] + table_height_pixels * 0.85)
        
        # Left region
        self.left_region = {
            "x1": table_bounds["x1"],
            "x2": table_bounds["x1"] + region_width,
            "y1": max(0, region_start_y),  # Ensure we don't go outside frame
            "y2": region_end_y,
        }
        
        # Right region
        self.right_region = {
            "x1": table_bounds["x2"] - region_width,
            "x2": table_bounds["x2"],
            "y1": max(0, region_start_y),  # Ensure we don't go outside frame
            "y2": region_end_y,
        }
        
    def _extract_bbox_from_detection(self, detection):
        """
        Extract bounding box from detection dictionary.
        
        Args:
            detection: Detection dictionary with 'box' field
            
        Returns:
            List [x1, y1, x2, y2]
        """
        return detection["box"]
        
    def _create_tracker_from_detection(self, detection, frame_shape):
        """
        Create a new cup tracker from a detection.
        Determines if the cup is in the left or right region and passes appropriate bounds.
        
        Args:
            detection: Detection dictionary with 'box' and 'confidence' fields
            frame_shape: Frame dimensions
            
        Returns:
            CupTracker instance or None if region cannot be determined
        """
        box = self._extract_bbox_from_detection(detection)
        confidence = detection.get("confidence", 0.0)
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        region_bounds = None
        if self.left_region and (self.left_region["x1"] <= center_x <= self.left_region["x2"] and self.left_region["y1"] <= center_y <= self.left_region["y2"]):
            region_bounds = [
                self.left_region["x1"],
                self.left_region["y1"],
                self.left_region["x2"],
                self.left_region["y2"],
            ]
        elif self.right_region and (self.right_region["x1"] <= center_x <= self.right_region["x2"] and self.right_region["y1"] <= center_y <= self.right_region["y2"]):
            region_bounds = [
                self.right_region["x1"],
                self.right_region["y1"],
                self.right_region["x2"],
                self.right_region["y2"],
            ]
        # else: # If neither region is defined or cup is outside, don't create tracker?
            # For now, let's allow creation without bounds if regions aren't set
            # Or should we prevent creation if it's outside known regions? Let's assume for now
            # that the filtering in process_detr_results handles this.
            
        return CupTracker(
            initial_box=box,
            frame_shape=frame_shape,
            initial_confidence=confidence,
            search_region_bounds=region_bounds # Pass the determined bounds
        )
        
    def _update_tracker_with_detection(self, tracker, detection):
        """
        Update a cup tracker with a new detection.
        
        Args:
            tracker: CupTracker instance
            detection: Detection dictionary with 'box' and 'confidence' fields
        """
        box = self._extract_bbox_from_detection(detection)
        confidence = detection.get("confidence", 0.0)
        
        tracker.update(box, detection_confidence=confidence)
        
    def _is_detection_in_regions(self, detection):
        """Check if a detection center falls within calibrated regions."""
        if not (self.left_region and self.right_region):
            return True # Cannot filter by region if not calibrated
            
        box = self._extract_bbox_from_detection(detection)
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        in_left = (
            self.left_region["x1"] <= center_x <= self.left_region["x2"]
            and self.left_region["y1"] <= center_y <= self.left_region["y2"]
        )
        
        in_right = (
            self.right_region["x1"] <= center_x <= self.right_region["x2"]
            and self.right_region["y1"] <= center_y <= self.right_region["y2"]
        )
        
        return in_left or in_right

    def _is_detection_correct_size(self, detection):
        """Check if a detection's bounding box size matches expected cup size."""
        if not self.expected_cup_width_pixels or not self.expected_cup_height_pixels:
            return True # Cannot filter by size if not calibrated
            
        box = self._extract_bbox_from_detection(detection)
        det_width = box[2] - box[0]
        det_height = box[3] - box[1]
        
        width_ok = (
            abs(det_width - self.expected_cup_width_pixels)
            <= self.expected_cup_width_pixels * self.size_tolerance
        )
        height_ok = (
            abs(det_height - self.expected_cup_height_pixels)
            <= self.expected_cup_height_pixels * self.size_tolerance
        )
        
        # Check aspect ratio as well (optional but helpful)
        expected_aspect_ratio = self.expected_cup_width_pixels / self.expected_cup_height_pixels
        detected_aspect_ratio = det_width / det_height if det_height > 0 else 0
        aspect_ratio_ok = abs(detected_aspect_ratio - expected_aspect_ratio) <= 0.5 # Allow 0.5 deviation in aspect ratio
        
        return width_ok and height_ok and aspect_ratio_ok

    def process_detr_results(self, results, model, frame_shape, table_bounds=None):
        """
        Process DETR detection results: filter by label, confidence, region, and size.
        Then update trackers.
        
        Args:
            results: DETR post-processed results
            model: DETR model (for id2label mapping)
            frame_shape: Tuple (height, width) of the current frame
            table_bounds: Optional table bounds for region and size calibration
            
        Returns:
            tracker_state: Current state of all cup trackers after processing
        """
        # 1. Calibrate regions and expected size if table_bounds are provided
        if table_bounds is not None:
            self.calibrate_regions(table_bounds)
            
        # 2. Initial filter by label and confidence
        raw_detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            box_coords = [round(i) for i in box.tolist()]
            label_name = model.config.id2label[label.item()]
            
            if label_name in self.target_labels and score_val >= self.detection_threshold:
                raw_detections.append({"box": box_coords, "confidence": score_val, "label": label_name})

        # 3. Filter by region and size
        filtered_detections = []
        for det in raw_detections:
            if self._is_detection_in_regions(det) and self._is_detection_correct_size(det):
                filtered_detections.append(det)
        
        # 4. Update trackers with the final filtered detections
        return self.update(filtered_detections, frame_shape) # Pass only filtered detections

    def update(self, detections, frame_shape, table_bounds=None):
        """
        Update cup trackers with new detections.
        
        Args:
            detections: List of cup detections
            frame_shape: Tuple (height, width) of the frame
            table_bounds: Optional table bounds for region calibration
            
        Returns:
            List of active cup tracker states
        """
        # If table bounds are provided, calibrate regions
        if table_bounds is not None:
            self.calibrate_regions(table_bounds)
            
        # Update trackers with filtered detections
        return super().update(detections, frame_shape)
        
    def draw_regions(self, frame):
        """
        Draw cup search regions and display expected size info on the frame.
        """
        if self.left_region:
            # Draw left region (yellow)
            left = self.left_region
            cv2.rectangle(
                frame,
                (left["x1"], left["y1"]),
                (left["x2"], left["y2"]),
                (0, 255, 255),  # Yellow
                2,
            )
            
            # Add label
            cv2.putText(
                frame,
                "Left Cup Region",
                (left["x1"], left["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
            
        if self.right_region:
            # Draw right region (yellow)
            right = self.right_region
            cv2.rectangle(
                frame,
                (right["x1"], right["y1"]),
                (right["x2"], right["y2"]),
                (0, 255, 255),  # Yellow
                2,
            )
            
            # Add label
            cv2.putText(
                frame,
                "Right Cup Region",
                (right["x1"], right["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
            
        # Display expected size info if calculated
        if self.expected_cup_width_pixels and self.expected_cup_height_pixels:
            size_text = f"Exp Cup Size (px): {self.expected_cup_width_pixels:.1f}W x {self.expected_cup_height_pixels:.1f}H"
            cv2.putText(frame, size_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        return frame 