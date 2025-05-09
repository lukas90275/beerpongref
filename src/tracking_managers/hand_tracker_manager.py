import cv2
import mediapipe as mp

from trackers.hand_tracker import HandTracker
from tracking_managers.tracker_manager import TrackerManager


class HandTrackerManager(TrackerManager):
    """
    Manages multiple HandTracker instances.
    Uses MediaPipe for hand detection instead of DETR.
    """

    def __init__(
        self,
        iou_threshold=0.3,
        min_confidence_frames=3,
        max_lost_frames=8,
        detection_threshold=0.1,
        min_tracking_confidence=0.1,
        **kwargs
    ):
        # Hand-specific parameters
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=10,
            min_detection_confidence=detection_threshold,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )
        
        # Store hand types separately
        self.hand_types = {}  # {tracker_id: hand_type}
        
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=False,
            motion_history_max_len=5,
            tracker_overlap_threshold=0.7,
            min_tracker_separation=0.05,
            cost_weight_iou=0.5,
            cost_weight_distance=0.5,
            position_stability_factor=0.3,
            # Position-based ID parameters
            position_id_distance_threshold=80,
            position_id_prefer_x_distance=True,
            position_id_name_format="{region}_hand_{index}",
            # Pass lifecycle hooks
            validate_match_func=self._validate_match,
            on_tracker_update=self._sync_hand_type,
            on_tracker_removed=self._cleanup_hand_type,
            # Region drawing parameters
            draw_region_color=(0, 0, 255),
            draw_region_label_prefix="Hand Region",
            **kwargs
        )
        
        # Initialize position history for hands
        self.position_history["left"] = []
        self.position_history["right"] = []
        
        # Register the hand tracker factory
        self.register_tracker_factory(self._create_hand_tracker)

    def _create_tracker_from_detection(self, detection, frame_shape):
        """Create a new tracker from detection - implements abstract method"""
        return self._create_hand_tracker(detection, frame_shape)

    def _create_hand_tracker(self, detection, frame_shape):
        """Create a new HandTracker from a detection"""
        box = detection["bbox"]
        confidence = detection.get("confidence", 0.0)
        hand_type = detection.get("hand_type", "Unknown").lower()
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # Generate a stable ID based on hand type and position
        hand_id = self.get_stable_id_for_position(hand_type, center_x, center_y, "hand")
        
        # Create tracker
        tracker = HandTracker(
            initial_box=box,
            frame_shape=frame_shape,
            initial_confidence=confidence,
            tracker_id=hand_id,
            position_stability_factor=self.position_stability_factor,
            use_x_distance_only=True,
        )
        
        # Store hand type separately
        self.hand_types[str(tracker.id)] = hand_type
        
        return tracker

    def _sync_hand_type(self, tracker, detection):
        """
        Update hand type for a tracker based on detection.
        Called by the base class when a tracker is updated.
        """
        if "hand_type" in detection:
            self.hand_types[str(tracker.id)] = detection["hand_type"].lower()

    def _cleanup_hand_type(self, tracker_id):
        """
        Remove hand type entry when a tracker is removed.
        Called by the base class when a tracker is removed.
        """
        if tracker_id in self.hand_types:
            del self.hand_types[tracker_id]

    def _validate_match(self, tracker, detection, iou, distance):
        """Don't match hands of different types"""
        if "hand_type" in detection and str(tracker.id) in self.hand_types:
            if self.hand_types[str(tracker.id)] != detection["hand_type"].lower():
                return False
        return True

    def detect_hands_raw(self, frame):
        """
        Detect hands in the frame using MediaPipe.
        Returns detections in a format compatible with our tracker.
        """
        if frame is None or frame.size == 0:
            return []

        hand_data = []
        
        try:
            # Resize large frames for better performance
            height, width = frame.shape[:2]
            max_dimension = 1280
            working_frame = frame
            scale = 1.0
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                working_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if results.multi_hand_landmarks and results.multi_handedness:
                frame_height, frame_width = frame.shape[:2]

                for idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    if not (handedness.classification and len(handedness.classification) > 0):
                        continue
                        
                    hand_type = handedness.classification[0].label
                    confidence = handedness.classification[0].score

                    # Calculate bounding box
                    x_coords = [
                        landmark.x * frame_width for landmark in hand_landmarks.landmark
                    ]
                    y_coords = [
                        landmark.y * frame_height for landmark in hand_landmarks.landmark
                    ]

                    if not x_coords or not y_coords:
                        continue

                    padding = 15
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    bbox_coords = [
                        max(0, int(x_min - padding)),
                        max(0, int(y_min - padding)),
                        min(frame_width, int(x_max + padding)),
                        min(frame_height, int(y_max + padding)),
                    ]

                    # Basic aspect ratio check
                    box_w = bbox_coords[2] - bbox_coords[0]
                    box_h = bbox_coords[3] - bbox_coords[1]
                    aspect_ratio = box_w / max(box_h, 0.001)

                    if 0.2 <= aspect_ratio <= 3.0:  # Broader range for various hand poses
                        hand_data.append(
                            {
                                "bbox": bbox_coords,
                                "hand_type": hand_type,
                                "confidence": confidence,
                            }
                        )
        except Exception as e:
            print(f"Error in hand detection: {e}")
            # Return empty list on error
            return []

        return hand_data

    def process_detr_results(self, results, model, frame_shape, frame=None, **kwargs):
        """Use MediaPipe for hand detection instead of DETR."""
        try:
            if frame is None:
                # Return empty list instead of raising an error
                print("HandTrackerManager requires frame for processing, but none provided")
                return []
                
            # Get hand detections using MediaPipe
            raw_hand_detections = self.detect_hands_raw(frame)
            
            # Update trackers with the standard update method
            tracker_states = self.update(raw_hand_detections, frame_shape, **kwargs)
            
            # Ensure that all hand trackers have their ball_region property set and added to state
            for tracker in self.trackers:
                # Force prediction to ensure updated positions
                tracker.predict()
                
                # If the tracker doesn't have a ball_region, calculate it
                if not hasattr(tracker, "ball_region") or tracker.ball_region is None:
                    if hasattr(tracker, "_calculate_ball_region"):
                        tracker.ball_region = tracker._calculate_ball_region()
            
            # Get updated states with ball regions
            tracker_states = [tracker.get_state() for tracker in self.trackers]
            
            return tracker_states
        except Exception as e:
            print(f"Error in hand detection processing: {e}")
            return []
    
    def process_hand_detector_results(self, frame, frame_shape):
        """
        Process MediaPipe hand detections and update trackers.
        Legacy method for backward compatibility.
        """
        # Get hand detections
        raw_hand_detections = self.detect_hands_raw(frame)
        
        # Update trackers
        tracker_states = self.update(raw_hand_detections, frame_shape)
        
        # Add hand types to each tracker state
        for state in tracker_states:
            state["hand_type"] = self.hand_types.get(str(state["id"]), "unknown")
            
        return tracker_states
    
    def get_confident_hands(self):
        """
        Get hand tracker states with hand type information.
        Legacy method for backward compatibility.
        """
        hand_states = []
        confident_trackers = self.get_confident_trackers()
        
        for tracker in confident_trackers:
            # Get base state
            state = tracker.get_state()
            
            # Add hand type
            state["hand_type"] = self.hand_types.get(str(tracker.id), "unknown")
            
            hand_states.append(state)
            
        return hand_states
        
    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
