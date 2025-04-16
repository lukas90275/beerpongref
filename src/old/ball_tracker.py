from collections import deque

import cv2


class BallTracker:
    def __init__(
        self,
        ball_id,
        initial_position,
        initial_radius,
        max_history=10,
        search_scale=1.5,
    ):
        """
        Initialize a tracker for a ball once it's consistently found in a hand.

        Args:
            ball_id: Unique identifier for the ball
            initial_position: (x, y) tuple of initial ball position
            initial_radius: Radius of the ball in pixels
            max_history: Maximum number of positions to track
            search_scale: Scale factor for search area relative to ball size
        """
        self.ball_id = ball_id
        self.active = True
        self.positions = deque(maxlen=max_history)
        self.velocities = deque(maxlen=max_history - 1)
        self.timestamps = deque(maxlen=max_history)
        self.radii = deque(maxlen=max_history)
        self.search_scale = search_scale
        self.max_search_scale = search_scale * 2.0  # For extrapolation
        self.frames_since_last_detection = 0
        self.max_missed_frames = 30  # Deactivate after this many missed frames

        # Initialize with the first position
        self.positions.append(initial_position)
        self.radii.append(initial_radius)
        self.timestamps.append(0)  # Initial timestamp

        # Initial velocity is zero
        self.current_velocity = (0, 0)

    def update(self, new_position=None, new_radius=None, timestamp=None):
        """
        Update the tracker with a new ball position

        Args:
            new_position: (x, y) tuple of new ball position, None if not detected
            new_radius: Radius of the ball in pixels, None if not detected
            timestamp: Frame timestamp (or frame number)

        Returns:
            bool: True if the tracker is still active
        """
        if timestamp is None:
            timestamp = len(self.timestamps)

        if new_position is not None:
            # Ball was detected
            self.positions.append(new_position)
            self.radii.append(new_radius)
            self.timestamps.append(timestamp)
            self.frames_since_last_detection = 0

            # Calculate velocity if we have at least two positions
            if len(self.positions) >= 2:
                time_diff = self.timestamps[-1] - self.timestamps[-2]
                if time_diff > 0:
                    dx = self.positions[-1][0] - self.positions[-2][0]
                    dy = self.positions[-1][1] - self.positions[-2][1]
                    self.current_velocity = (dx / time_diff, dy / time_diff)
                    self.velocities.append(self.current_velocity)
        else:
            # Ball wasn't detected
            self.frames_since_last_detection += 1

            # Extrapolate position based on velocity
            if len(self.positions) >= 2:
                last_pos = self.positions[-1]
                extrapolated_x = int(last_pos[0] + self.current_velocity[0])
                extrapolated_y = int(last_pos[1] + self.current_velocity[1])
                extrapolated_pos = (extrapolated_x, extrapolated_y)

                self.positions.append(extrapolated_pos)
                self.radii.append(self.radii[-1])  # Keep the same radius
                self.timestamps.append(timestamp)

            # Deactivate tracker if we haven't seen the ball for too long
            if self.frames_since_last_detection > self.max_missed_frames:
                self.active = False

        return self.active

    def get_search_area(self):
        """
        Get the search area for the next frame based on current position and velocity

        Returns:
            tuple: (x1, y1, x2, y2) coordinates of search box
        """
        if not self.positions:
            return None

        # Get the most recent position and radius
        center = self.positions[-1]
        radius = self.radii[-1]

        # Scale search area based on missed frames
        scale = min(
            self.search_scale + (self.frames_since_last_detection * 0.1),
            self.max_search_scale,
        )

        # Calculate search box size
        search_width = int(radius * 2 * scale)
        search_height = int(radius * 2 * scale)

        # Calculate search box coordinates
        x1 = max(0, int(center[0] - search_width // 2))
        y1 = max(0, int(center[1] - search_height // 2))
        x2 = int(center[0] + search_width // 2)
        y2 = int(center[1] + search_height // 2)

        return (x1, y1, x2, y2)

    def draw_on_frame(self, frame):
        """
        Draw the tracker information on the frame

        Args:
            frame: The frame to draw on

        Returns:
            frame: The annotated frame
        """
        if not self.active or not self.positions:
            return frame

        center = self.positions[-1]
        radius = self.radii[-1] if self.radii else 10

        # Draw search area
        search_area = self.get_search_area()
        if search_area:
            x1, y1, x2, y2 = search_area

            # Use different color for extrapolated position
            if self.frames_since_last_detection > 0:
                # Orange for extrapolated position
                color = (0, 165, 255)
                label = f"Ball {self.ball_id} (predicted)"
            else:
                # Green for actual detection
                color = (0, 255, 0)
                label = f"Ball {self.ball_id} (tracked)"

            # Draw the search area
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Draw the ball position
            cv2.circle(frame, center, radius, color, 2)

            # Add label
            cv2.putText(
                frame,
                label,
                (center[0] - radius, center[1] - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            # Draw velocity vector if significant
            if len(self.positions) >= 2:
                velocity_scale = 3  # Scale factor for displaying velocity
                end_x = int(center[0] + self.current_velocity[0] * velocity_scale)
                end_y = int(center[1] + self.current_velocity[1] * velocity_scale)
                cv2.arrowedLine(frame, center, (end_x, end_y), color, 2)

        return frame


class BallTrackerManager:
    def __init__(self):
        """
        Manages multiple ball trackers
        """
        self.trackers = {}  # ball_id -> BallTracker

    def update_trackers(self, ball_detections, frame, timestamp=None):
        """
        Update all trackers with new ball detections

        Args:
            ball_detections: List of detected balls
            frame: Current video frame
            timestamp: Current timestamp or frame number

        Returns:
            frame: Annotated frame
        """
        # Track which ball_ids we've seen in this frame
        seen_ball_ids = set()

        # Process detected balls
        for ball in ball_detections:
            ball_id = ball["id"]
            position = ball["center"]
            radius = ball["radius"]
            is_consistent = ball.get("is_consistent", False)

            seen_ball_ids.add(ball_id)

            if is_consistent:
                if ball_id not in self.trackers:
                    # Create new tracker for consistently detected ball
                    self.trackers[ball_id] = BallTracker(
                        ball_id=ball_id,
                        initial_position=position,
                        initial_radius=radius,
                        timestamp=timestamp,
                    )
                else:
                    # Update existing tracker
                    self.trackers[ball_id].update(position, radius, timestamp)

        # Update trackers for balls not detected in this frame
        for ball_id, tracker in list(self.trackers.items()):
            if ball_id not in seen_ball_ids:
                # Ball not detected in this frame, update with None
                if not tracker.update(None, None, timestamp):
                    # Tracker is no longer active, remove it
                    del self.trackers[ball_id]

        # Draw all trackers on the frame
        for tracker in self.trackers.values():
            frame = tracker.draw_on_frame(frame)

        return frame
