# Copied from https://github.com/roboflow/supervision/blob/develop/supervision/annotators/core.py
# significantly modified 
import numpy as np
import cv2
import math
from typing import Tuple

def clip_boxes(xyxy: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """
    Clips bounding boxes coordinates to fit within the frame resolution.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each
            row corresponds to a bounding box in
        the format `(x_min, y_min, x_max, y_max)`.
        resolution_wh (Tuple[int, int]): A tuple of the form `(width, height)`
            representing the resolution of the frame.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box with coordinates clipped to fit
            within the frame resolution.
    """
    result = np.copy(xyxy)
    width, height = resolution_wh
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    return result

class BlurAnnotator():
    """
    A class for blurring regions in an image using provided detections.
    """

    def __init__(self, blur_strength=30):
        """
        Initialize the blur annotator with customizable blur strength.
        
        Args:
            blur_strength (int): Not used for median_blur
        """
        self.blur_strength = blur_strength

    def annotate(
        self,
        scene: np.ndarray,
        detections,
    ) -> np.ndarray:
        """
        Annotates the given scene by blurring regions based on the provided detections.

        Args:
            scene (np.ndarray): The image where blurring will be applied.
            detections (Detections): Object detections to annotate.

        Returns:
            The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...F
            >>> detections = sv.Detections(...)

            >>> blur_annotator = sv.BlurAnnotator()
            >>> annotated_frame = circle_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```
        """
        image_height, image_width = scene.shape[:2]
        clipped_xyxy = clip_boxes(
            xyxy=detections.xyxy, resolution_wh=(image_width, image_height)
        ).astype(int)

        for x1, y1, x2, y2 in clipped_xyxy: # this loop is modified from the original
            
            roi = scene[y1:y2, x1:x2]
            kernel_size = max(1, math.floor(min(y2 -y1, x2 - x1) / 2))
            

            # Reduce the blur factor until it reaches 0
            while kernel_size > 0:
                kernel_size_n = kernel_size + 1
                if (kernel_size_n > 15):
                    kernel_size_n = 15

                # Make odd if even
                if kernel_size_n % 2 == 0:
                    kernel_size_n -= 1

                    
                roi = cv2.medianBlur(roi, kernel_size_n)            
                scene[y1:y2, x1:x2] = roi

                kernel_size = max(0, kernel_size - kernel_size_n - 1)

        return scene
