import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from loguru import logger

from app.infrastructure.ai.detection import FaceDetector
from app.infrastructure.ai.embedding import FaceEmbedder
from app.infrastructure.ai.tracking import ByteTracker, Track
from app.infrastructure.ai.alignment import align_face
from app.utils.cache import FaceEmbeddingCache
from app.services.monitor_service import MonitorService
from app.core.configs import get_settings

class AIService:
    """Orchestrates face detection, tracking, embedding, matching, and voting."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        tracker: ByteTracker
    ):
        self.detector = detector
        self.embedder = embedder
        self.tracker = tracker
        self.cache = FaceEmbeddingCache()
        self.monitor = MonitorService()
        
        # Load parameters from configuration
        settings = get_settings()
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.min_votes = settings.TEMPORAL_VOTING_MIN_VOTES
        self.buffer_size = settings.TEMPORAL_VOTING_BUFFER_SIZE

        # Track vote history: track_id -> List of (student_uuid, name, student_code)
        self.track_votes: Dict[int, List[Tuple[Optional[str], str, Optional[str]]]] = {}
        
        # Track identities that have already been finalized to avoid toggling
        self.finalized_identities: Dict[int, Tuple[str, str, str]] = {}

    def process_frame(self, image: np.ndarray) -> List[Track]:
        """Processes a single BGR frame, tracking and identifying faces.

        Args:
            image: BGR NumPy array from the camera.

        Returns:
            List of active tracks updated with identity metadata.
        """
        start_time = time.time()

        # 1. Run face detection
        t0 = time.time()
        detections = self.detector.detect(image)
        self.monitor.log_detection_latency(time.time() - t0)

        # 2. Update multi-object tracker
        active_tracks = self.tracker.update(detections)

        current_active_ids = set()

        # 3. Process each tracked face
        for track in active_tracks:
            current_active_ids.add(track.track_id)
            
            # Default metadata placeholders
            track.identified_name = "Unknown"
            track.identified_uuid = None
            track.identified_student_id = None
            track.is_recognized = False

            # Check if this track's identity was already locked
            if track.track_id in self.finalized_identities:
                s_uuid, s_name, s_code = self.finalized_identities[track.track_id]
                track.identified_uuid = s_uuid
                track.identified_name = s_name
                track.identified_student_id = s_code
                track.is_recognized = True
                continue

            # We only extract embeddings when a face is active (matched in current frame)
            # and has keypoints available.
            if track.time_since_update == 0 and track.keypoints is not None:
                try:
                    # 3.1. Align the face
                    aligned = align_face(image, track.keypoints)

                    # 3.2. Generate face embedding
                    t0 = time.time()
                    embedding = self.embedder.compute_embedding(aligned)
                    self.monitor.log_embedding_latency(time.time() - t0)

                    # 3.3. Query vector cache for similarity
                    t0 = time.time()
                    match = self.cache.search(embedding, threshold=self.similarity_threshold)
                    self.monitor.log_lookup_latency(time.time() - t0)

                    # Determine frame match identity
                    if match:
                        student_uuid, name, student_code, score = match
                        # Convert UUID to string for easy queue transfers
                        identity = (str(student_uuid), name, student_code)
                    else:
                        identity = (None, "Unknown", None)

                except Exception as e:
                    logger.error(f"Error processing track {track.track_id} features: {str(e)}")
                    identity = (None, "Unknown", None)

                # Update vote buffer
                if track.track_id not in self.track_votes:
                    self.track_votes[track.track_id] = []
                
                buffer = self.track_votes[track.track_id]
                buffer.append(identity)
                if len(buffer) > self.buffer_size:
                    buffer.pop(0)

                # Perform Temporal Voting
                # Count frequencies of each unique identity tuple in the sliding buffer
                counts: Dict[Tuple[Optional[str], str, Optional[str]], int] = {}
                for item in buffer:
                    counts[item] = counts.get(item, 0) + 1

                # Find the mode (identity with the highest votes)
                winner = max(counts, key=counts.get)
                winner_votes = counts[winner]

                winner_uuid, winner_name, winner_code = winner

                # Check if the winner meets the voting threshold and is not "Unknown"
                if winner_uuid is not None and winner_votes >= self.min_votes:
                    # Lock identity to avoid recognition flipping under occlusion
                    self.finalized_identities[track.track_id] = (winner_uuid, winner_name, winner_code)
                    
                    track.identified_uuid = winner_uuid
                    track.identified_name = winner_name
                    track.identified_student_id = winner_code
                    track.is_recognized = True
                    logger.debug(f"Track {track.track_id} locked to student: '{winner_name}' ({winner_votes}/{len(buffer)} votes).")
                else:
                    # Fallback or temporary votes
                    track.identified_uuid = winner_uuid
                    track.identified_name = winner_name
                    track.identified_student_id = winner_code
                    track.is_recognized = (winner_uuid is not None)

        # 4. Prune voting buffers for inactive/removed tracks to conserve memory
        all_vote_ids = list(self.track_votes.keys())
        for tid in all_vote_ids:
            if tid not in current_active_ids:
                # Keep lost tracks voting buffers for a short time, or prune if completely out of tracker
                tracker_has_id = any(t.track_id == tid for t in self.tracker.tracked_tracks + self.tracker.lost_tracks)
                if not tracker_has_id:
                    self.track_votes.pop(tid, None)
                    self.finalized_identities.pop(tid, None)

        # Log frame process time & FPS
        frame_time = time.time() - start_time
        if frame_time > 0:
            self.monitor.log_fps(1.0 / frame_time)

        return active_tracks

    def recognize_snapshot(self, image: np.ndarray) -> List[dict]:
        """Performs immediate face recognition on a static image (no tracking/voting).
        
        Args:
            image: BGR NumPy array of the snapshot.
            
        Returns:
            List of dictionaries containing bounding boxes, names, and match details.
        """
        t0 = time.time()
        detections = self.detector.detect(image)
        self.monitor.log_detection_latency(time.time() - t0)

        results = []
        for det in detections:
            try:
                # Align face
                aligned = align_face(image, det.keypoints)
                
                # Extract embedding
                t_emb = time.time()
                embedding = self.embedder.compute_embedding(aligned)
                self.monitor.log_embedding_latency(time.time() - t_emb)
                
                # Search local vector cache
                t_match = time.time()
                match = self.cache.search(embedding, threshold=self.similarity_threshold)
                self.monitor.log_lookup_latency(time.time() - t_match)
                
                if match:
                    student_uuid, name, student_code, score = match
                else:
                    student_uuid, name, student_code, score = None, "Unknown", None, 0.0
                    
                results.append({
                    "bbox": det.bbox,
                    "student_uuid": str(student_uuid) if student_uuid else None,
                    "name": name,
                    "student_code": student_code,
                    "score": score,
                    "is_recognized": student_uuid is not None
                })
            except Exception as e:
                logger.error(f"Failed to analyze face in static snapshot: {str(e)}")
                
        return results

