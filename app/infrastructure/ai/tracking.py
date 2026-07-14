import numpy as np
from typing import List, Tuple, Dict
from app.infrastructure.ai.detection import DetectionResult

class Track:
    """Represents an active or lost trajectory track of a detected face."""
    
    def __init__(self, bbox: np.ndarray, score: float, track_id: int, keypoints: np.ndarray = None):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.keypoints = keypoints  # [5, 2] landmarks
        self.state = "tracked"  # "tracked" or "lost"
        self.age = 1
        self.time_since_update = 0
        self.hits = 1

    def predict(self) -> None:
        """Projects the track's state into the current frame.
        In a simplified constant-velocity/overlap model, the prediction is the 
        previous bounding box. Velocity can be added if needed, but for 20-30 FPS face 
        tracking, last frame location is highly stable.
        """
        self.time_since_update += 1
        self.age += 1

    def update(self, bbox: np.ndarray, score: float, keypoints: np.ndarray = None) -> None:
        """Updates the track with a new detection match."""
        # Simple smoothing (moving average) for bounding box to reduce jitter
        alpha = 0.7
        self.bbox = alpha * bbox + (1 - alpha) * self.bbox
        self.score = score
        self.keypoints = keypoints
        self.state = "tracked"
        self.time_since_update = 0
        self.hits += 1


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


class ByteTracker:
    """Cython-free, lightweight implementation of ByteTrack multi-object tracker."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: float = 100.0
    ):
        self.track_thresh = track_thresh    # Threshold separating high vs low confidence
        self.match_thresh = match_thresh    # Maximum IoU distance allowed (1 - IoU threshold)
        self.track_buffer = track_buffer    # Maximum frames to keep a lost track
        self.min_box_area = min_box_area

        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.next_track_id = 1

    def update(self, detections: List[DetectionResult]) -> List[Track]:
        """Updates the tracker with detections from the current frame.

        Args:
            detections: List of DetectionResult from the face detector.

        Returns:
            List of active (tracked) tracks in the current frame.
        """
        # Step 1: Filter and split detections into high and low confidence
        high_detections: List[DetectionResult] = []
        low_detections: List[DetectionResult] = []

        for det in detections:
            # Filter extremely small boxes
            area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
            if area < self.min_box_area:
                continue

            if det.score >= self.track_thresh:
                high_detections.append(det)
            else:
                low_detections.append(det)

        # Step 2: Predict current state of tracked and lost tracks
        all_tracks = self.tracked_tracks + self.lost_tracks
        for track in all_tracks:
            track.predict()

        # Step 3: First association - match high confidence detections with active tracks
        matched_tracks, unmatched_tracks, unmatched_detections_high = self._associate(
            all_tracks, high_detections, 1.0 - self.match_thresh
        )

        # Step 4: Second association - match remaining tracks with low confidence detections
        # (This is the key trick in ByteTrack: recovering partially occluded faces)
        matched_tracks_second, unmatched_tracks_second, _ = self._associate(
            unmatched_tracks, low_detections, 1.0 - 0.45  # Slightly relaxed IoU threshold for low score
        )

        # Combine all matches
        all_matches = matched_tracks + matched_tracks_second
        
        # Determine remaining unmatched tracks (both active and lost)
        # If a tracked track remains unmatched, move it to lost.
        # If a lost track remains unmatched, keep it in lost (it will expire after track_buffer frames).
        new_tracked_tracks: List[Track] = []
        new_lost_tracks: List[Track] = []

        for track in all_matches:
            new_tracked_tracks.append(track)

        # For remaining unmatched tracks:
        # If they were tracked, they now become lost.
        # If they were already lost, we check if they exceed the buffer limit.
        remaining_unmatched = unmatched_tracks_second
        for track in remaining_unmatched:
            if track.state == "tracked":
                track.state = "lost"
                new_lost_tracks.append(track)
            elif track.state == "lost":
                if track.time_since_update <= self.track_buffer:
                    new_lost_tracks.append(track)

        # Step 5: Start new tracks from unmatched high confidence detections
        for det in unmatched_detections_high:
            new_track = Track(det.bbox, det.score, self.next_track_id, det.keypoints)
            self.next_track_id += 1
            new_tracked_tracks.append(new_track)

        self.tracked_tracks = new_tracked_tracks
        self.lost_tracks = new_lost_tracks

        return [t for t in self.tracked_tracks if t.state == "tracked"]

    def _associate(
        self,
        tracks: List[Track],
        detections: List[DetectionResult],
        min_iou: float
    ) -> Tuple[List[Track], List[Track], List[DetectionResult]]:
        """Associates tracks and detections based on IoU overlap using a greedy algorithm.
        
        Returns:
            Tuple of: (matched_tracks, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = calculate_iou(track.bbox, det.bbox)

        # Greedy matching: sort pairs by IoU score in descending order
        matched_track_indices = set()
        matched_det_indices = set()
        matched_tracks: List[Track] = []

        # Find all candidate pairs and sort
        pairs = []
        for t in range(len(tracks)):
            for d in range(len(detections)):
                if iou_matrix[t, d] >= min_iou:
                    pairs.append((iou_matrix[t, d], t, d))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for score, t, d in pairs:
            if t not in matched_track_indices and d not in matched_det_indices:
                matched_track_indices.add(t)
                matched_det_indices.add(d)
                # Update the matched track
                tracks[t].update(detections[d].bbox, detections[d].score, detections[d].keypoints)
                matched_tracks.append(tracks[t])

        unmatched_tracks = [tracks[i] for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_detections = [detections[i] for i in range(len(detections)) if i not in matched_det_indices]

        return matched_tracks, unmatched_tracks, unmatched_detections
