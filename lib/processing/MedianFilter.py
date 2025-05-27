# lib/processing/MedianFilter.py
import cv2
import numpy as np


class MedianFilter:
    def __init__(self, kernel_size=5, iterations=1):
        """
        Initialize Median Filter processor

        Args:
            kernel_size (int): Size of the median filter kernel. Must be odd and >= 3.
                              Common values: 3, 5, 7, 9
                              Larger values remove more noise but blur more
            iterations (int): Number of times to apply the median filter.
                            Multiple iterations can remove stubborn noise
        """
        # Validate kernel size
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd and >= 3")

        self.kernel_size = kernel_size
        self.iterations = max(1, iterations)

    def process(self, frame):
        """
        Apply median filtering to remove salt-and-pepper noise

        Args:
            frame (numpy.ndarray): Input image (grayscale or color)

        Returns:
            numpy.ndarray: Filtered image with reduced salt-and-pepper noise
        """
        if frame is None:
            return frame

        # Start with the input frame
        result = frame.copy()

        # Apply median filter for the specified number of iterations
        for _ in range(self.iterations):
            result = cv2.medianBlur(result, self.kernel_size)

        return result

    def set_kernel_size(self, kernel_size):
        """
        Update the kernel size

        Args:
            kernel_size (int): New kernel size (must be odd and >= 3)
        """
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd and >= 3")

        self.kernel_size = kernel_size

    def set_iterations(self, iterations):
        """
        Update the number of iterations

        Args:
            iterations (int): Number of iterations to perform (minimum 1)
        """
        self.iterations = max(1, iterations)

    def get_noise_level_recommendation(self, frame):
        """
        Analyze the image and suggest appropriate kernel size based on noise level
        This is a helper method to automatically tune the filter

        Args:
            frame (numpy.ndarray): Input image to analyze

        Returns:
            int: Recommended kernel size
        """
        if frame is None:
            return self.kernel_size

        # Convert to grayscale if color image
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Calculate noise level using Laplacian variance
        # Higher variance indicates more edges/details, lower variance indicates more noise
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Recommend kernel size based on noise characteristics
        if laplacian_var < 100:  # Very noisy image
            return 7
        elif laplacian_var < 500:  # Moderately noisy
            return 5
        else:  # Clean image or high detail
            return 3

    def adaptive_filter(self, frame, auto_tune=True):
        """
        Apply median filter with automatic parameter tuning

        Args:
            frame (numpy.ndarray): Input image
            auto_tune (bool): Whether to automatically adjust kernel size

        Returns:
            numpy.ndarray: Filtered image
        """
        if frame is None:
            return frame

        if auto_tune:
            # Temporarily store original kernel size
            original_size = self.kernel_size

            # Get recommended size
            recommended_size = self.get_noise_level_recommendation(frame)
            self.kernel_size = recommended_size

            # Apply filter
            result = self.process(frame)

            # Restore original kernel size
            self.kernel_size = original_size

            return result
        else:
            return self.process(frame)

    def __str__(self):
        """
        String representation of the processor
        """
        return f"MedianFilter(kernel_size={self.kernel_size}, iterations={self.iterations})"
