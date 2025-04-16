import cv2
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from hand_detector import HandDetector
from tracking_managers.hand_tracker_manager import HandTrackerManager
from tracking_managers.table_tracker_manager import TableTrackerManager

# Initialize the DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Initialize the cup detector
# cup_detector = CupDetector(processor, model)

# Initialize the raw hand detector (just gets data)
hand_detector = HandDetector(min_detection_confidence=0.3)

# Initialize the hand tracker manager
hand_tracker_manager = HandTrackerManager(iou_threshold=0.2, max_lost_frames=15)

# Initialize the table tracker manager
table_tracker_manager = TableTrackerManager(iou_threshold=0.2, max_lost_frames=20)

# Initialize the ball in hand detector
# ball_in_hand_detector = BallInHandDetector(consecutive_frames_threshold=3)

# Initialize the ball tracker manager
# ball_tracker_manager = BallTrackerManager()

# Frame counter for timestamping
frame_counter = 0


def overlay_transparent(background, overlay, alpha):
    """
    Add a transparent overlay on the background image
    """
    output = background.copy()
    cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0, output)
    return output


def analyze_frame(frame):
    """
    Analyze a frame using DETR to detect objects of interest in beer pong.

    Args:
        frame: numpy array of the image in BGR format (from cv2)

    Returns:
        annotated_frame: frame with bounding boxes and labels
        detections: dictionary containing detection information
    """
    global frame_counter, table_tracker_manager, hand_tracker_manager
    frame_counter += 1

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare image for the model
    inputs = processor(images=rgb_frame, return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process predictions
    target_sizes = torch.tensor([frame.shape[:2]])

    # Get all results with a low base threshold
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.1,  # Low base threshold to catch all potential objects
    )[0]

    # Updated detections dictionary
    detections = {
        "cups": [],
        "table": [],
        "table_tracker": None,
        "hands_tracked": [],
        "balls_in_hand": [],
    }

    # Create a copy for drawing
    annotated_frame = frame.copy()

    table_bounds = None
    raw_table_detections = []  # Store raw DETR table detections this frame

    # --- Table Detection and Tracking ---

    # Run DETR detection for tables
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        score_val = score.item()
        box_coords = [round(i) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]

        if (
            label_name in ["dining table", "table"] and score_val >= 0.2
        ):  # Lower threshold for potential matches
            raw_table_detections.append({"box": box_coords, "confidence": score_val})
            # Also add to the detections dict for debugging
            detections["table"].append({"box": box_coords, "confidence": score_val})

    # Update table tracker with detections
    table_tracker_state = table_tracker_manager.update(
        raw_table_detections, frame.shape[:2]
    )

    # Draw the table tracker on the frame
    annotated_frame = table_tracker_manager.draw_trackers(annotated_frame)

    # Get table bounds if the tracker is confident
    table_bounds = table_tracker_manager.get_table_bounds()

    # Update detections dictionary
    if table_tracker_state:
        detections["table_tracker"] = table_tracker_state

    # --- End Table Tracking ---

    # Second pass: detect cups using specialized detector
    # Ensure table_bounds are valid (list or tuple of 4 ints)
    valid_table_bounds = (
        isinstance(table_bounds, (list, tuple)) and len(table_bounds) == 4
    )

    # Comment out cup detection
    """
    if valid_table_bounds:
        cup_results = cup_detector.detect_cups(
            frame,
            {
                "x1": table_bounds[0],
                "y1": table_bounds[1],
                "x2": table_bounds[2],
                "y2": table_bounds[3],
            },
        )
        detections["cups"] = cup_results["cups"]

        # Overlay the cup detection visualization
        alpha = 0.7
        cup_detection_overlay = cv2.addWeighted(
            annotated_frame, 1 - alpha, cup_results["debug_frame"], alpha, 0
        )
        annotated_frame = cup_detection_overlay
    else:
        # Handle case where table bounds are not available (e.g., tracker lost or not initialized)
        # Optionally, run cup detection on the whole frame or skip
        pass  # Skipping cup detection if no table bounds
    """

    # Third pass: detect hands using MediaPipe
    raw_hand_detections = hand_detector.detect_hands_raw(frame)
    active_hand_states = hand_tracker_manager.update(
        raw_hand_detections, frame.shape[:2]
    )
    detections["hands_tracked"] = active_hand_states
    annotated_frame = hand_tracker_manager.draw_trackers(annotated_frame)

    # Comment out ball-in-hand detection
    """
    # Fourth pass: detect balls in hands
    # Get confident hands from the manager
    confident_hands = hand_tracker_manager.get_confident_hands()

    # Modify input to ball_in_hand_detector to use the format it expects
    # (It previously expected a list of dicts with 'id', 'bbox', etc.)
    # We adapt the tracked hand states to this format.
    formatted_confident_hands = []
    for hand_state in confident_hands:
        formatted_confident_hands.append(
            {
                "id": hand_state["id"],  # Use tracker ID
                "hand_type": hand_state["hand_type"],
                "confidence": hand_state["last_confidence"],
                "bbox": {  # Convert list [x1,y1,x2,y2] to dict
                    "x1": hand_state["box"][0],
                    "y1": hand_state["box"][1],
                    "x2": hand_state["box"][2],
                    "y2": hand_state["box"][3],
                },
            }
        )

    # Run ball-in-hand detection only if we have confident hands and a valid table
    if formatted_confident_hands and valid_table_bounds:
        ball_in_hand_frame, ball_in_hand_data = (
            ball_in_hand_detector.detect_balls_in_hands(
                annotated_frame,  # Use the frame with hand trackers drawn
                formatted_confident_hands,  # Use confident tracked hands
                {
                    "x1": table_bounds[0],
                    "y1": table_bounds[1],
                    "x2": table_bounds[2],
                    "y2": table_bounds[3],
                },
            )
        )
        detections["balls_in_hand"] = ball_in_hand_data
        annotated_frame = ball_in_hand_frame  # Update frame with ball-in-hand drawings

        # Track balls that are consistently detected in hands
        annotated_frame = ball_tracker_manager.update_trackers(
            ball_in_hand_data, annotated_frame, timestamp=frame_counter
        )
    """

    return annotated_frame, detections
