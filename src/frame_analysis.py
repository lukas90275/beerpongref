import cv2
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from tracking_managers.cup_tracker_manager import CupTrackerManager
from tracking_managers.hand_tracker_manager import HandTrackerManager
from tracking_managers.table_tracker_manager import TableTrackerManager

# Initialize the DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Store managers in a list for easier iteration
tracker_managers = {
    "table": TableTrackerManager(iou_threshold=0.6, max_lost_frames=20, detection_threshold=0.5),
    "cup": CupTrackerManager(
        iou_threshold=0.6,
        min_confidence_frames=5,
        max_lost_frames=20,
        detection_threshold=0.4
    ),
    "hand": HandTrackerManager(iou_threshold=0.2, max_lost_frames=15),
}

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


def analyze_frame(frame, table_viz=True, hand_viz=True, cup_viz=True, cup_search_viz=False):
    """
    Analyze a frame using DETR to detect objects of interest in beer pong.

    Args:
        frame: numpy array of the image in BGR format (from cv2)
        table_viz: whether to visualize table detection (default: True)
        hand_viz: whether to visualize hand detection (default: True)
        cup_viz: whether to visualize cup detection (default: True)
        cup_search_viz: whether to visualize cup search boxes (default: False)

    Returns:
        annotated_frame: frame with bounding boxes and labels
        detections: dictionary containing detection information
    """
    global frame_counter, tracker_managers
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

    # Detections dictionary to store results from each manager
    detections = {
        "table_tracker": None,
        "hands_tracked": [],
        "cups_tracked": [],
        "balls_in_hand": [], # Placeholder for future ball detection
    }

    # Create a copy for drawing
    annotated_frame = frame.copy()

    # --- Process Trackers ---
    table_bounds = None
    table_bounds_dict = None

    # 1. Process Table First (dependency for cups)
    table_manager = tracker_managers["table"]
    detections["table_tracker"] = table_manager.process_detr_results(
        results, model, frame.shape[:2]
    )
    table_bounds = table_manager.get_table_bounds()
    if table_bounds:
        table_bounds_dict = {
            "x1": table_bounds[0],
            "y1": table_bounds[1],
            "x2": table_bounds[2],
            "y2": table_bounds[3],
        }

    # 2. Process Hands
    hand_manager = tracker_managers["hand"]
    detections["hands_tracked"] = hand_manager.process_hand_detector_results(
        frame, frame.shape[:2]
    )

    # 3. Process Cups (only if table bounds exist)
    cup_manager = tracker_managers["cup"]
    if table_bounds_dict:
        detections["cups_tracked"] = cup_manager.process_detr_results(
            results, model, frame.shape[:2], table_bounds_dict
        )

    # --- Draw Trackers ---
    # Draw Table only if a table has been found and table visualization is enabled
    if table_bounds_dict and table_viz:
        annotated_frame = table_manager.draw_trackers(annotated_frame)
    
    # Draw Hands if hand visualization is enabled
    if hand_viz:
        annotated_frame = hand_manager.draw_trackers(annotated_frame)
    
    # Draw Cups (including regions) if cup visualization is enabled
    if table_bounds_dict and cup_viz:
        if cup_search_viz:
            annotated_frame = cup_manager.draw_regions(annotated_frame)
        annotated_frame = cup_manager.draw_trackers(annotated_frame, show_search_box=cup_search_viz)

    # Comment out ball-in-hand detection
    """
    # Detection for ball-in-hand would go here in the future
    """

    return annotated_frame, detections
