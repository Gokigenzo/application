import numpy as np
from typing import List, Optional
import uuid
from loguru import logger

from app.domain.entities import Student, FaceEmbedding
from app.domain.repositories import StudentRepository, FaceEmbeddingRepository
from app.infrastructure.ai.detection import FaceDetector
from app.infrastructure.ai.embedding import FaceEmbedder
from app.infrastructure.ai.alignment import align_face
from app.core.exceptions import DatabaseException, AIException
from app.utils.cache import FaceEmbeddingCache

class StudentService:
    """Coordinates student enrollment operations."""

    def __init__(
        self,
        student_repo: StudentRepository,
        embedding_repo: FaceEmbeddingRepository,
        detector: FaceDetector,
        embedder: FaceEmbedder
    ):
        self.student_repo = student_repo
        self.embedding_repo = embedding_repo
        self.detector = detector
        self.embedder = embedder
        self.cache = FaceEmbeddingCache()

    def register_student(
        self,
        student_id: str,
        name: str,
        email: Optional[str],
        face_images: List[np.ndarray]
    ) -> Student:
        """Registers a new student and saves their face embeddings.

        Args:
            student_id: Unique school registration ID.
            name: Full name of the student.
            email: Optional email address.
            face_images: List of BGR NumPy arrays containing enrollment faces.

        Returns:
            The registered Student domain entity.
        """
        # Validate that student code is unique
        existing = self.student_repo.get_by_student_id(student_id)
        if existing:
            raise DatabaseException(
                f"Student with ID '{student_id}' is already registered.", 
                "STUDENT_ALREADY_EXISTS"
            )

        embeddings_to_save = []
        for idx, img in enumerate(face_images):
            # Run detection on enrollment image
            detections = self.detector.detect(img)
            if not detections:
                raise AIException(
                    f"No face detected in enrollment image {idx+1}.", 
                    "NO_FACE_DETECTED"
                )
            if len(detections) > 1:
                raise AIException(
                    f"Multiple faces detected in enrollment image {idx+1}. Image must contain exactly one face.", 
                    "MULTIPLE_FACES_DETECTED"
                )

            # Align face based on eye, nose, and mouth positions
            det = detections[0]
            aligned = align_face(img, det.keypoints)
            
            # Extract normalized embedding vector
            embedding = self.embedder.compute_embedding(aligned)
            embeddings_to_save.append(embedding)

        if not embeddings_to_save:
            raise AIException(
                "No face embeddings could be extracted.", 
                "NO_EMBEDDINGS_EXTRACTED"
            )

        # Create Student domain model
        student = Student(student_id=student_id, name=name, email=email)
        self.student_repo.add(student)

        # Create face embeddings records
        for emb in embeddings_to_save:
            face_emb = FaceEmbedding(student_id=student.id, embedding=emb)
            self.embedding_repo.add(face_emb)
            
            # Instantly mirror vector embedding to local cache to keep it hot
            self.cache.add(student.id, student.name, student.student_id, emb)

        logger.info(f"Registered student: '{name}' (Code: {student_id}) with {len(embeddings_to_save)} embeddings.")
        return student

    def list_students(self) -> List[Student]:
        """Lists all registered students."""
        return self.student_repo.list_all()

    def get_student_by_student_id(self, student_id: str) -> Optional[Student]:
        """Looks up student metadata by registration code."""
        return self.student_repo.get_by_student_id(student_id)

    def delete_student(self, id_: uuid.UUID) -> bool:
        """Deletes a student record and drops their vector cache signatures."""
        success = self.student_repo.delete(id_)
        if success:
            self.cache.remove(id_)
            logger.info(f"Deleted student UUID '{id_}' and removed their signatures from vector cache.")
        return success
