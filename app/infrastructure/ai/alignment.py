import cv2
import numpy as np

# Standard template landmarks for a 112x112 face crop (InsightFace/MTCNN standard)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)

def align_face(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Aligns and crops a face image to 112x112 dimensions using affine transform.

    Args:
        image: Original BGR frame.
        keypoints: NumPy array of shape (5, 2) containing 5 facial keypoints.

    Returns:
        Aligned and cropped 112x112 BGR face image.
    """
    # Estimate similarity transformation matrix (rotation + translation + scale)
    M, _ = cv2.estimateAffinePartial2D(keypoints, REFERENCE_LANDMARKS, method=cv2.LMEDS)
    
    if M is None:
        # Fallback to simple bounding box crop & resize if transformation matrix fails
        x_min, y_min = np.min(keypoints, axis=0).astype(int)
        x_max, y_max = np.max(keypoints, axis=0).astype(int)
        
        # Add margin
        h, w = image.shape[:2]
        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return cv2.resize(image, (112, 112))
        return cv2.resize(cropped, (112, 112))

    # Apply the warp affine transformation
    aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0)
    return aligned
