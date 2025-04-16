from abc import ABC


class TrackerManager(ABC):
    # ... existing code ...

    def match_detections_to_trackers(self, detections):
        # ... existing code ...

        for t_idx, tracker in enumerate(self.trackers):
            for d_idx, det in enumerate(detections):
                # Extract bbox from detection (implementation may vary)
                det_bbox = self._extract_bbox_from_detection(det)

                # Calculate IoU between tracker box and detection
                # Call as static method from tracker class
                from ..trackers.object_tracker import ObjectTracker

                iou = ObjectTracker.calculate_iou(tracker.box, det_bbox)
                if iou >= self.iou_threshold:
                    cost_matrix[t_idx, d_idx] = 1.0 - iou  # Cost is 1 - IoU

        # ... rest of the method ...
