import cv2
import torch


class CupDetector:
    def __init__(self, processor, model):
        """
        Initialize the cup detector with the DETR model and processor.

        Args:
            processor: DetrImageProcessor instance
            model: DetrForObjectDetection instance
        """
        self.processor = processor
        self.model = model
        self.cup_threshold = 0.7

        # These will be set when calibrating
        self.left_region = None
        self.right_region = None

    def calibrate_regions(self, frame, table_bounds):
        """
        Calibrate the left and right cup regions based on the table bounds.
        Assumes the table is properly detected and cups are on either end.

        Args:
            frame: Full frame
            table_bounds: Dictionary with x1, y1, x2, y2 coordinates of the table
        """
        table_width = table_bounds["x2"] - table_bounds["x1"]
        table_height = table_bounds["y2"] - table_bounds["y1"]

        # Define regions as the outer 20% of each side of the table
        region_width = int(table_width * 0.3)

        # Calculate the height region to look slightly above and below the table top
        region_start_y = int(table_bounds["y1"] - table_height * 0.5)
        region_end_y = int(table_bounds["y1"] + table_height * 0.75)

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

    def detect_cups_in_region(self, frame, region):
        """
        Detect cups in a specific region of the frame.

        Args:
            frame: Full frame
            region: Dictionary with x1, y1, x2, y2 coordinates

        Returns:
            List of detected cups with adjusted coordinates
        """
        # Crop the region
        cropped = frame[region["y1"] : region["y2"], region["x1"] : region["x2"]]

        # Convert BGR to RGB
        rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Prepare image for the model
        inputs = self.processor(images=rgb_cropped, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process predictions
        target_sizes = torch.tensor([cropped.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.cup_threshold
        )[0]

        cups = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_name = self.model.config.id2label[label.item()]
            if label_name in ["cup", "wine glass", "bottle"]:
                # Adjust coordinates back to original frame
                box = [round(i) for i in box.tolist()]
                adjusted_box = [
                    box[0] + region["x1"],  # x1
                    box[1] + region["y1"],  # y1
                    box[2] + region["x1"],  # x2
                    box[3] + region["y1"],  # y2
                ]
                cups.append({"box": adjusted_box, "confidence": score.item()})

        return cups

    def detect_cups(self, frame, table_bounds=None):
        """
        Detect cups in both regions of the frame.

        Args:
            frame: Full frame
            table_bounds: Optional table bounds for calibration

        Returns:
            Dictionary with detected cups and debug frame
        """
        # If table bounds provided, calibrate regions
        if table_bounds is not None:
            self.calibrate_regions(frame, table_bounds)

        # If regions aren't calibrated, we can't detect
        if self.left_region is None or self.right_region is None:
            return {"cups": [], "debug_frame": frame}

        # Create debug frame
        debug_frame = frame.copy()

        # Draw regions on debug frame
        cv2.rectangle(
            debug_frame,
            (self.left_region["x1"], self.left_region["y1"]),
            (self.left_region["x2"], self.left_region["y2"]),
            (0, 255, 255),
            2,
        )  # Yellow for regions
        cv2.rectangle(
            debug_frame,
            (self.right_region["x1"], self.right_region["y1"]),
            (self.right_region["x2"], self.right_region["y2"]),
            (0, 255, 255),
            2,
        )

        # Detect cups in each region
        left_cups = self.detect_cups_in_region(frame, self.left_region)
        right_cups = self.detect_cups_in_region(frame, self.right_region)

        # Combine detections
        all_cups = left_cups + right_cups

        # Draw detections on debug frame
        for cup in all_cups:
            box = cup["box"]
            cv2.rectangle(
                debug_frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2
            )  # Blue for cups

            # Add confidence label
            label_text = f"Cup: {cup['confidence']:.2f}"
            cv2.putText(
                debug_frame,
                label_text,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        return {"cups": all_cups, "debug_frame": debug_frame}
