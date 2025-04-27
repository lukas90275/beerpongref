import cv2
import mediapipe as mp

from trackers.hand_tracker import HandTracker
from tracking_managers.tracker_manager import TrackerManager


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
            tracker_overlap_threshold=0.7,  # Standard overlap threshold for hands
            min_tracker_separation=0.05,  # Hands can be closer than other objects
            cost_weight_iou=0.5,  # Equal weight for IoU and distance for hands
            cost_weight_distance=0.5,
            position_stability_factor=0.3,  # Moderate stability for hands (which move)
        )
        
        # Hand-specific parameters
        self.min_tracking_confidence = min_tracking_confidence
        
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
        
        # Store hand types separately instead of in tracker objects
        self.hand_types = {}  # {tracker_id: hand_type}
        
        # Initialize position history for hands
        self.position_history["left_hands"] = []
        self.position_history["right_hands"] = []

    def _extract_bbox_from_detection(self, detection):
        """Extract bounding box from hand detection"""
        return detection["bbox"]

    def _create_tracker_from_detection(self, detection, frame_shape):
        """Create a new HandTracker from a detection"""
        box = detection["bbox"]
        confidence = detection.get("confidence", 0.0)
        hand_type = detection.get("hand_type", "Unknown")
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # Generate a stable ID based on hand type and position
        hand_id = self._get_hand_id(hand_type, center_x, center_y)
        
        # Create tracker without hand_type
        tracker = HandTracker(
            initial_box=box,
            frame_shape=frame_shape,
            initial_confidence=confidence,
            tracker_id=hand_id,
            position_stability_factor=self.position_stability_factor,
            use_x_distance_only=True,  # Use only x-distance for side view
        )
        
        # Store hand type separately
        self.hand_types[str(tracker.id)] = hand_type
        
        return tracker

    def _update_tracker_with_detection(self, tracker, detection):
        """Update a HandTracker with new detection"""
        box = detection["bbox"]
        confidence = detection.get("confidence", 0.0)
        hand_type = detection.get("hand_type", None)
        
        # Update tracker with basic information
        tracker.update(
            box,
            detection_confidence=confidence,
        )
        
        # Update hand type in our dictionary if provided
        if hand_type:
            self.hand_types[str(tracker.id)] = hand_type
        
        # Get hand type from our dictionary
        tracker_hand_type = self.hand_types.get(str(tracker.id), "Unknown")
        
        # Update position history for this hand
        hand_id = str(tracker.id)
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # Update position in history
        history_key = f"{tracker_hand_type.lower()}_hands"
        if history_key in self.position_history:
            history = self.position_history[history_key]
            # Look for existing entry with this ID
            found = False
            for i, (px, py, pid) in enumerate(history):
                if pid == hand_id:
                    history[i] = (center_x, center_y, hand_id)
                    found = True
                    break
                    
            # Add new entry if not found
            if not found:
                history.append((center_x, center_y, hand_id))

    def _get_hand_id(self, hand_type, center_x, center_y):
        """
        Generate a stable ID for a hand based on its type and position.
        Tries to maintain consistent IDs across frames.
        """
        if not hand_type:
            return None
            
        # Get the appropriate history list
        history_key = f"{hand_type.lower()}_hands"
        if history_key not in self.position_history:
            self.position_history[history_key] = []
            
        history = self.position_history[history_key]
        
        # Look for closest historical hand position to reuse ID
        if history:
            min_dist = float('inf')
            closest_idx = -1
            
            for i, (px, py, pid) in enumerate(history):
                # Use x-distance for hands in side view
                dist = abs(center_x - px)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    
            # Hand size is approximately 100-150 pixels wide at standard resolutions
            # Base distance threshold on reasonable hand width
            hand_size_estimate = 100  # Pixels
            distance_threshold = hand_size_estimate * 0.8
                    
            # If close enough to existing position, reuse ID
            if closest_idx >= 0 and min_dist < distance_threshold:
                hand_id = history[closest_idx][2]
                # Update position
                history[closest_idx] = (center_x, center_y, hand_id)
                return hand_id
                
        # Generate new ID
        hand_id = f"{hand_type.lower()}_hand_{len(history) + 1}"
        history.append((center_x, center_y, hand_id))
        return hand_id

    def _calculate_custom_matching_cost(self, tracker, detection, iou, normalized_distance):
        """
        Custom cost calculation for hands that considers hand type.
        Prioritizes matching hands of the same type.
        """
        # If hand types don't match, increase cost significantly
        if 'hand_type' in detection:
            tracker_hand_type = self.hand_types.get(str(tracker.id))
            if tracker_hand_type != detection['hand_type']:
                return 10.0  # Very high cost to discourage matching different hand types
        
        # For hands of the same type, use default cost
        return None  # Default to base implementation

    def _validate_match(self, tracker, detection, iou, distance):
        """
        Additional validation for hand matches.
        """
        # For hands, we can be stricter about matching only the same hand type
        if 'hand_type' in detection:
            tracker_hand_type = self.hand_types.get(str(tracker.id))
            if tracker_hand_type != detection['hand_type']:
                return False  # Don't match hands of different types
                
        return True

    def get_confident_hands(self):
        """Get the state of hand trackers that are currently confident"""
        hand_states = []
        confident_trackers = self.get_confident_trackers()
        
        for tracker in confident_trackers:
            # Get base state from tracker
            state = tracker.get_state()
            
            # Add hand type from our dictionary
            state["hand_type"] = self.hand_types.get(str(tracker.id), "Unknown")
            
            hand_states.append(state)
            
        return hand_states
        
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
        tracker_states = self.update(raw_hand_detections, frame_shape)
        
        # Add hand types to each tracker state
        for state in tracker_states:
            state["hand_type"] = self.hand_types.get(str(state["id"]), "Unknown")
            
        return tracker_states
        
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

    # Override to clean up hand types for removed trackers
    def remove_tracker(self, tracker_id):
        """Remove a tracker by ID"""
        # Clean up hand type entry
        if str(tracker_id) in self.hand_types:
            del self.hand_types[str(tracker_id)]
            
        # Call parent implementation
        super().remove_tracker(tracker_id)
