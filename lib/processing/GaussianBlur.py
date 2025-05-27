# lib/processing/GaussianBlur.py
import cv2
import numpy as np


class GaussianBlur:
    def __init__(self, kernel_size=5, sigma_x=0, sigma_y=0):
        """
        Initialize Gaussian Blur processor

        Args:
            kernel_size (int): Size of the Gaussian kernel. Must be odd and positive.
                              Common values: 3, 5, 7, 9, 11
            sigma_x (float): Gaussian kernel standard deviation in X direction.
                           If 0, calculated from kernel size
            sigma_y (float): Gaussian kernel standard deviation in Y direction.
                           If 0, uses same value as sigma_x
        """
        # Ensure kernel size is odd and positive
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")

        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        # Create kernel size tuple (width, height)
        self.ksize = (kernel_size, kernel_size)

    def process(self, frame):
        """
        Apply Gaussian blur to the input frame

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Blurred image
        """
        if frame is None:
            return frame

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, self.ksize, self.sigma_x, sigmaY=self.sigma_y)

        return blurred

    def set_kernel_size(self, kernel_size):
        """
        Update the kernel size

        Args:
            kernel_size (int): New kernel size (must be odd and positive)
        """
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")

        self.kernel_size = kernel_size
        self.ksize = (kernel_size, kernel_size)

    def set_sigma(self, sigma_x, sigma_y=None):
        """
        Update the sigma values

        Args:
            sigma_x (float): Standard deviation in X direction
            sigma_y (float): Standard deviation in Y direction (optional)
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y if sigma_y is not None else sigma_x

    def __str__(self):
        """
        String representation of the processor
        """
        return f"GaussianBlur(kernel_size={self.kernel_size}, sigma_x={self.sigma_x}, sigma_y={self.sigma_y})"
