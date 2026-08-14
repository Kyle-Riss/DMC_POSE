"""
Motion-based pre-filter to reduce redundant YOLO pose runs.
"""
import cv2
import numpy as np

class MotionDetector:
    """Simple frame-diff based motion detection.
    
    Useful for pre-filtering: only run YOLO pose if motion detected
    in bed ROI region.
    """
    
    def __init__(self, threshold: int = 30, blur_kernel: int = 5, min_motion_area: int = 100):
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.min_motion_area = min_motion_area
    
    def detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray, roi: tuple = None) -> bool:
        """Detect motion between two frames (optionally within ROI).
        
        Args:
            prev_frame: Previous BGR frame
            curr_frame: Current BGR frame
            roi: Optional (x1, y1, x2, y2) tuple for region of interest
        
        Returns:
            True if motion detected above threshold
        """
        if prev_frame is None or curr_frame is None:
            return False
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI if provided
        if roi is not None and len(roi) == 4:
            x1, y1, x2, y2 = roi
            prev_gray = prev_gray[y1:y2, x1:x2]
            curr_gray = curr_gray[y1:y2, x1:x2]
        
        # Blur to reduce noise
        prev_gray = cv2.GaussianBlur(prev_gray, (self.blur_kernel, self.blur_kernel), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Count motion pixels
        motion_area = cv2.countNonZero(thresh)
        
        return motion_area >= self.min_motion_area
