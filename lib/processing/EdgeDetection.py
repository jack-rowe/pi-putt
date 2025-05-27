# lib/processing/EdgeDetection.py
import cv2
import numpy as np


class EdgeDetection:
    def __init__(self, low_threshold=50, high_threshold=150, kernel_size=3):
        """
        Initialize edge detection processor

        Args:
            low_threshold (int): Lower threshold for the hysteresis procedure
            high_threshold (int): Higher threshold for the hysteresis procedure
            kernel_size (int): Sobel kernel size
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size

    def process(self, frame):
        """
        Apply Canny edge detection on the frame

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Processed image with detected edges
        """
        # Convert to grayscale if the image is colored
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.low_threshold,
            self.high_threshold,
            apertureSize=self.kernel_size,
        )

        # Convert back to 3 channels if input was colored
        if len(frame.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edges
