# lib/processing/Brightness.py
import cv2
import numpy as np


class Brightness:
    def __init__(self, alpha=1.0, beta=0):
        """
        Initialize brightness adjustment processor

        Args:
            alpha (float): Contrast control (1.0 means no change)
            beta (int): Brightness control (0 means no change)
        """
        self.alpha = alpha
        self.beta = beta

    def process(self, frame):
        """
        Adjust brightness and contrast of the frame

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Processed image
        """
        return cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)
