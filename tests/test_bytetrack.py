import numpy as np
from app.infrastructure.ai.tracking import ByteTracker
from app.infrastructure.ai.detection import DetectionResult

def test_bytetrack_association() -> None:
    """Verifies that the ByteTracker correctly tracks objects and persists track IDs."""
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=5)
    
    # Frame 1: 1 initial face detection
    detections_f1 = [
        DetectionResult(bbox=np.array([100.0, 100.0, 200.0, 200.0]), score=0.9, keypoints=np.zeros((5, 2)))
    ]
    
    tracks_f1 = tracker.update(detections_f1)
    assert len(tracks_f1) == 1
    initial_id = tracks_f1[0].track_id
    assert tracks_f1[0].state == "tracked"
    
    # Frame 2: Same face slightly shifted (IoU overlap high)
    detections_f2 = [
        DetectionResult(bbox=np.array([102.0, 103.0, 202.0, 203.0]), score=0.88, keypoints=np.zeros((5, 2)))
    ]
    
    tracks_f2 = tracker.update(detections_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == initial_id  # Bounding box ID must persist
    assert tracks_f2[0].state == "tracked"
    
    # Frame 3: Bounding box disappears (occluded / out of frame)
    tracks_f3 = tracker.update([])
    assert len(tracks_f3) == 0  # No active tracks visible
    assert len(tracker.lost_tracks) == 1  # Track should be in "lost" buffer
    
    # Frame 4: Bounding box reappears at similar coordinates before expiring
    detections_f4 = [
        DetectionResult(bbox=np.array([105.0, 105.0, 205.0, 205.0]), score=0.85, keypoints=np.zeros((5, 2)))
    ]
    tracks_f4 = tracker.update(detections_f4)
    assert len(tracks_f4) == 1
    assert tracks_f4[0].track_id == initial_id  # Track recovered!
