import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the hand detector with MediaPipe Hands
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize MediaPipe Hands with better parameters for beer pong context
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Use tracking for better performance in videos
            max_num_hands=10,  # Increase to detect more hands for multi-player games
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,  # Use higher model complexity (0, 1, or 2)
        )

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
            frame_height, frame_width = frame.shape[
                :2
            ]  # Use original frame dims for bbox

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
                            # Optionally add landmarks if needed later
                            # "landmarks": [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        }
                    )

        return hand_data

    def release(self):
        """Release resources"""
        self.hands.close()
