# lib/processing/Mask.py
import cv2
import numpy as np


class Mask:
    def __init__(self, lower_bound=None, upper_bound=None):
        """
        Initialize color mask processor

        Args:
            lower_bound (array): Lower bound for color filtering in HSV [H, S, V]
            upper_bound (array): Upper bound for color filtering in HSV [H, S, V]
        """
        # Default to golf ball white color range if none provided
        if lower_bound is None:
            self.lower_bound = np.array([0, 0, 200])  # White-ish color in HSV
        else:
            self.lower_bound = np.array(lower_bound)

        if upper_bound is None:
            self.upper_bound = np.array([180, 30, 255])  # White-ish color in HSV
        else:
            self.upper_bound = np.array(upper_bound)

    def process(self, frame):
        """
        Apply color masking to the frame

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Processed image with mask applied
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask using the specified color range
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)

        # Apply the mask to the original image
        result = cv2.bitwise_and(frame, frame, mask=mask)

        return result
