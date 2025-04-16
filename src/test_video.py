import argparse
import csv
import os
import sys

import cv2

import frame_analysis  # Import the module itself
from video_process.video_preprocessor import VideoPreprocessor
from video_process.video_processor import VideoProcessor


def display_frame(frame, table_viz=True, hand_viz=True, cup_viz=True, cup_search_viz=False):
    """Process and display frame with object detection"""
    # Analyze the frame using the updated analysis function
    annotated_frame, detections = frame_analysis.analyze_frame(frame, table_viz, hand_viz, cup_viz, cup_search_viz)

    # Print detection information
    print("\nDetections:")
    print("-----------")

    # Table Status (from tracker or initial detection)
    if detections.get("table_tracker"):
        tracker_state = detections["table_tracker"]
        status = (
            "Confident"
            if tracker_state["is_confident"]
            else f"Tracking (Lost: {tracker_state['lost_frames']})"
        )
        print(f"Table Status: {status}, Box: {tracker_state['box']}")
    elif detections.get("table"):
        print(f"Table Status: Initial Detection, Box: {detections['table'][0]['box']}")
    else:
        print("Table Status: Not Detected")

    # Cup tracking information
    cups_tracked = detections.get("cups_tracked", [])
    raw_cups = detections.get("cups", [])
    confident_cups = sum(1 for cup in cups_tracked if cup["is_confident"])
    
    print(f"Cups found: {len(raw_cups)}")
    print(f"Cups tracked: {len(cups_tracked)} (Confident: {confident_cups})")

    # Print info about tracked hands
    hands_tracked_count = len(detections.get("hands_tracked", []))
    confident_hands_count = sum(
        1 for hand in detections.get("hands_tracked", []) if hand["is_confident"]
    )
    print(f"Hands tracked: {hands_tracked_count} (Confident: {confident_hands_count})")

    # Comment out ball tracking output
    """
    # Print information about balls in hands
    consistent_balls = [
        ball for ball in detections["balls_in_hand"] if ball["is_consistent"]
    ]
    print(
        f"Balls in hands: {len(detections['balls_in_hand'])} (Consistent: {len(consistent_balls)})"
    )

    if consistent_balls:
        print("\nConsistent Balls in Hands:")
        print("-------------------------")
        for ball in consistent_balls:
            method = ball.get("method", "unknown")
            merged_count = ball.get("merged_count", 1)
            merged_info = f", merged×{merged_count}" if merged_count > 1 else ""
            print(
                f"Ball {ball['id']} ({method}{merged_info}): {ball['consecutive_detections']} consecutive frames"
            )
    """

    return annotated_frame


def read_video_info(csv_path, line_number):
    """
    Read video information from the specified line in the CSV file.
    Line numbers are 1-indexed (first data row is line 1).
    """
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if line_number < 1 or line_number > len(rows):
                raise ValueError(f"Line number must be between 1 and {len(rows)}")

            row = rows[line_number - 1]
            return {
                "video_path": row["video_path"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
            }
    except FileNotFoundError:
        raise ValueError(f"CSV file not found: {csv_path}")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid CSV format or data in line {line_number}: {str(e)}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Process and display video frames between timestamps from CSV"
    )
    parser.add_argument(
        "csv_path", type=str, help="Path to the CSV file containing video information"
    )
    parser.add_argument(
        "line_number", type=int, help="Line number in CSV to process (1-indexed)"
    )
    parser.add_argument(
        "--frame-by-frame",
        action="store_true",
        help="Enable frame-by-frame mode (press Enter to advance, 'q' to quit)",
    )
    parser.add_argument(
        "--table-viz",
        action="store_true",
        default=True,
        help="Enable table visualization",
    )
    parser.add_argument(
        "--hand-viz",
        action="store_true",
        default=True,
        help="Enable hand visualization",
    )
    parser.add_argument(
        "--cup-viz",
        action="store_true",
        default=True,
        help="Enable cup visualization",
    )
    parser.add_argument(
        "--cup-search-viz",
        action="store_true",
        default=False,
        help="Show cup search boxes (hidden by default)",
    )

    args = parser.parse_args()

    try:
        # Read video information from CSV
        video_info = read_video_info(args.csv_path, args.line_number)

        # Verify video path exists
        if not os.path.exists(video_info["video_path"]):
            raise ValueError(f"Video file not found: {video_info['video_path']}")

        # Preprocess the video to focus on the table
        preprocessor = VideoPreprocessor(video_info["video_path"])

        preprocessor.detect_stable_bounds()
        processor = VideoProcessor(video_info["video_path"])

        # Print video information
        print("\nVideo Information:")
        print("-----------------")
        for key, value in processor.video_info.items():
            print(f"{key}: {value}")

        if preprocessor.crop_bounds:
            print("\nTable Detection Bounds:")
            print("---------------------")
            for key, value in preprocessor.crop_bounds.items():
                print(f"{key}: {value}")

        print(
            f"\nProcessing video segment from {video_info['start_time']}s to {video_info['end_time']}s"
        )

        if args.frame_by_frame:
            print("Frame-by-frame mode: Press Enter to advance, 'q' to quit\n")
        else:
            print("Continuous mode: Press 'q' to quit\n")

        # Process and display frames
        for frame_number, frame in processor.process_video_segment(
            start_time=video_info["start_time"],
            end_time=video_info["end_time"],
            skip_frames=0,
        ):
            print(f"\nProcessing frame {frame_number}")

            # If using original video, crop to table bounds
            if preprocessor.crop_bounds:
                cropped_frame = frame[
                    preprocessor.crop_bounds["y1"] : preprocessor.crop_bounds["y2"],
                    preprocessor.crop_bounds["x1"] : preprocessor.crop_bounds["x2"],
                ]
            else:
                cropped_frame = frame

            # Process frame after cropping
            processed_frame = display_frame(
                cropped_frame,
                table_viz=args.table_viz,
                hand_viz=args.hand_viz,
                cup_viz=args.cup_viz,
                cup_search_viz=args.cup_search_viz
            )

            cv2.imshow("Frame", processed_frame)

            if args.frame_by_frame:
                # Wait for Enter key (13) or 'q' to quit
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        return
                    if key == 13:  # Enter key
                        break
            else:
                # In continuous mode, just check for 'q' with a small delay
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
