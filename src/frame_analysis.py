import cv2
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from tracking_managers.ball_tracker_manager import BallTrackerManager
from tracking_managers.cup_tracker_manager import CupTrackerManager
from tracking_managers.hand_tracker_manager import HandTrackerManager
from tracking_managers.table_tracker_manager import TableTrackerManager

# Initialize the DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Store managers in a list for easier iteration
tracker_managers = {
    "table": TableTrackerManager(),
    "cup": CupTrackerManager(),
    "hand": HandTrackerManager(),
    "ball": BallTrackerManager(),
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


def analyze_frame(frame, table_viz=True, hand_viz=True, cup_viz=True, ball_viz=True, 
                 cup_search_viz=False, table_search_viz=False, hand_search_viz=False, ball_search_viz=True):
    """
    Analyze a frame using DETR to detect objects of interest in beer pong.

    Args:
        frame: numpy array of the image in BGR format (from cv2)
        table_viz: whether to visualize table detection (default: True)
        hand_viz: whether to visualize hand detection (default: True)
        cup_viz: whether to visualize cup detection (default: True)
        ball_viz: whether to visualize ball detection (default: True)
        cup_search_viz: whether to visualize cup search boxes (default: False)
        table_search_viz: whether to visualize table search boxes (default: False)
        hand_search_viz: whether to visualize hand search boxes (default: False)
        ball_search_viz: whether to visualize ball search/detection regions (default: True)

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
        "balls_tracked": [],
    }

    # Create a copy for drawing
    annotated_frame = frame.copy()

    # --- Process Trackers in Dependency Order ---
    
    # 1. Process Table First (dependency for cups)
    detections["table_tracker"] = tracker_managers["table"].process_detr_results(
        results, model, frame.shape[:2]
    )
    
    # Get the table bounds for cup detection
    table_bounds_dict = None
    table_tracker = tracker_managers["table"].get_primary_tracker()
    if table_tracker and hasattr(table_tracker, 'box') and table_tracker.box is not None:
        bounds = table_tracker.box
        table_bounds_dict = {
            "x1": bounds[0],
            "y1": bounds[1],
            "x2": bounds[2],
            "y2": bounds[3],
        }

    # 2. Process Hands
    detections["hands_tracked"] = tracker_managers["hand"].process_detr_results(
        None, None, frame.shape[:2], frame=frame
    )

    # 3. Process Cups (using table bounds if available)
    detections["cups_tracked"] = tracker_managers["cup"].process_detr_results(
        results, model, frame.shape[:2], table_bounds=table_bounds_dict
    )
        
    # 4. Process Balls (using hand trackers for regions)  
    detections["balls_tracked"] = tracker_managers["ball"].process_detr_results(
        results, model, frame.shape[:2], frame=frame, hand_trackers=detections["hands_tracked"]
    )

    # --- Draw Trackers ---
    # Draw Table only if table visualization is enabled
    if table_viz and table_tracker:
        annotated_frame = tracker_managers["table"].draw_trackers(annotated_frame, show_search_box=table_search_viz)
    
    # Draw Hands if hand visualization is enabled
    if hand_viz:
        # Draw hand trackers first
        annotated_frame = tracker_managers["hand"].draw_trackers(annotated_frame, show_search_box=hand_search_viz)
        
        # If ball search visualization is enabled, ensure we draw those regions 
        # right after drawing hands to ensure they're shown even if no balls are detected
        if ball_search_viz:
            # Force recalculating regions from current hand trackers
            tracker_managers["ball"]._add_hand_regions(hand_trackers=detections["hands_tracked"])
            # Draw all ball search regions
            annotated_frame = tracker_managers["ball"].draw_regions(annotated_frame, show_expected_size=True)
    
    # Draw Cups (including regions) if cup visualization is enabled
    if cup_viz:
        if cup_search_viz:
            annotated_frame = tracker_managers["cup"].draw_regions(annotated_frame, show_expected_size=True)
        annotated_frame = tracker_managers["cup"].draw_trackers(annotated_frame, show_search_box=cup_search_viz)
        
    # Draw Balls if ball visualization is enabled
    if ball_viz:
        # We already drew ball regions above when hand_viz and ball_search_viz are both true
        # So only draw them here if hand_viz is false but ball_search_viz is true
        if ball_search_viz and not hand_viz:
            annotated_frame = tracker_managers["ball"].draw_regions(annotated_frame, show_expected_size=True)
            
        # If ball_search_viz is enabled, detect and show circles for debugging
        if ball_search_viz and frame is not None:
            # Get raw circle detections
            circle_detections = tracker_managers["ball"].detect_circles(frame)
            # Draw them
            if circle_detections:
                annotated_frame = tracker_managers["ball"].draw_detections(annotated_frame, circle_detections)
            
        # Option to show blob detections along with search boxes
        show_blob_detections = ball_search_viz  # Show blob detections when search box is shown
        annotated_frame = tracker_managers["ball"].draw_trackers(
            annotated_frame, 
            show_search_box=ball_search_viz,
            show_blob_detections=show_blob_detections
        )

    return annotated_frame, detections
