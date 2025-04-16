import cv2
import mediapipe as mp

from trackers.hand_tracker import HandTracker

from .tracker_manager import TrackerManager


class HandTrackerManager(TrackerManager):
    """
    Manages multiple HandTracker instances.
    Assigns new detections to existing trackers using IoU and the Hungarian algorithm.
    Creates new trackers for unmatched detections and removes lost trackers.
    """

    def __init__(
        self,
        iou_threshold=0.3,
        min_confidence_frames=3,
        max_lost_frames=8,
        detection_threshold=0.5,
        min_tracking_confidence=0.5
    ):
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=False,  # Hands are multi-instance trackers
            motion_history_max_len=5,  # Default motion history for hands
        )
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Hands with better parameters for beer pong context
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Use tracking for better performance in videos
            max_num_hands=10,  # Increase to detect more hands for multi-player games
            min_detection_confidence=detection_threshold,  # Use the same threshold
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,  # Use higher model complexity (0, 1, or 2)
        )

    def _extract_bbox_from_detection(self, detection):
        """Extract bounding box from hand detection"""
        return detection["bbox"]

    def _create_tracker_from_detection(self, detection, frame_shape):
        """Create a new HandTracker from a detection"""
        return HandTracker(
            initial_box=detection["bbox"],
            frame_shape=frame_shape,
            hand_type=detection["hand_type"],
            initial_confidence=detection["confidence"],
            min_confidence_frames=self.min_confidence_frames,
            max_lost_frames=self.max_lost_frames,
        )

    def _update_tracker_with_detection(self, tracker, detection):
        """Update a HandTracker with new detection"""
        tracker.update(
            detection["bbox"],
            detection_confidence=detection["confidence"],
            hand_type=detection["hand_type"],
        )

    def get_confident_hands(self):
        """Get the state of hand trackers that are currently confident"""
        return self.get_confident_trackers()
        
    def detect_hands_raw(self, frame):
        """
        Detect hands in the frame and return raw data (bounding boxes, type, confidence).

        Args:
            frame: Input frame in BGR format (from OpenCV)

        Returns:
            hand_data: List of detected hand dictionaries:
                       [{'bbox': [x1, y1, x2, y2], 'hand_type': 'Left'/'Right', 'confidence': float}]
        """
        if frame is None or frame.size == 0:
            return []

        # Preprocessing (resize, etc.) - Consider keeping if beneficial
        height, width = frame.shape[:2]
        max_dimension = 1280
        working_frame = frame
        scale = 1.0
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            working_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        rgb_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hand_data = []

        if results.multi_hand_landmarks:
            frame_height, frame_width = frame.shape[:2]  # Use original frame dims for bbox

            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                hand_type = handedness.classification[0].label
                confidence = handedness.classification[0].score

                # Calculate bounding box based on original frame dimensions
                x_coords = [
                    landmark.x * frame_width for landmark in hand_landmarks.landmark
                ]
                y_coords = [
                    landmark.y * frame_height for landmark in hand_landmarks.landmark
                ]

                # Skip if coordinates are invalid (though unlikely with landmarks)
                if not x_coords or not y_coords:
                    continue

                padding = 15  # Slightly smaller padding maybe
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                bbox_coords = [
                    max(0, int(x_min - padding)),
                    max(0, int(y_min - padding)),
                    min(frame_width, int(x_max + padding)),
                    min(frame_height, int(y_max + padding)),
                ]

                # Basic aspect ratio check (optional, tracker might handle bad shapes)
                box_w = bbox_coords[2] - bbox_coords[0]
                box_h = bbox_coords[3] - bbox_coords[1]
                aspect_ratio = box_w / box_h if box_h > 0 else 0

                # Relaxed aspect ratio or remove if tracker is robust
                if 0.2 <= aspect_ratio <= 3.0:  # Broader range
                    hand_data.append(
                        {
                            "bbox": bbox_coords,  # [x1, y1, x2, y2]
                            "hand_type": hand_type,
                            "confidence": confidence,
                        }
                    )

        return hand_data

    def process_hand_detector_results(self, frame, frame_shape):
        """
        Process hand detections and update trackers.
        
        Args:
            frame: Current video frame
            frame_shape: Tuple (height, width) of the current frame
            
        Returns:
            tracker_state: Current state of all hand trackers after processing
        """
        # Get hand detections directly
        raw_hand_detections = self.detect_hands_raw(frame)
        
        # Update trackers with these detections
        return self.update(raw_hand_detections, frame_shape)
        
    def process_detr_results(self, results, model, frame_shape, frame=None, **kwargs):
        """
        Override to use integrated hand detection instead of DETR results.
        
        Args:
            results: Not used for hand detection
            model: Not used for hand detection
            frame_shape: Tuple (height, width) of the current frame
            frame: Current video frame
            
        Returns:
            tracker_state: Current state of all hand trackers after processing
        """
        if frame is None:
            raise ValueError("HandTrackerManager requires frame for processing")
            
        return self.process_hand_detector_results(frame, frame_shape)
        
    def release(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
