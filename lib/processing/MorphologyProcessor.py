# lib/processing/MorphologyProcessor.py
import cv2
import numpy as np


class MorphologyProcessor:
    def __init__(
        self, operation="opening", kernel_size=5, kernel_shape="ellipse", iterations=1
    ):
        """
        Initialize Morphological processor

        Args:
            operation (str): Morphological operation to perform
                           Options: 'opening', 'closing', 'erosion', 'dilation', 'gradient', 'tophat', 'blackhat'
            kernel_size (int): Size of the morphological kernel
            kernel_shape (str): Shape of the kernel
                              Options: 'rect', 'ellipse', 'cross'
            iterations (int): Number of times to apply the operation
        """
        # Validate operation
        valid_operations = [
            "opening",
            "closing",
            "erosion",
            "dilation",
            "gradient",
            "tophat",
            "blackhat",
        ]
        if operation.lower() not in valid_operations:
            raise ValueError(
                f"Invalid operation '{operation}'. Must be one of: {valid_operations}"
            )

        # Validate kernel shape
        valid_shapes = ["rect", "ellipse", "cross"]
        if kernel_shape.lower() not in valid_shapes:
            raise ValueError(
                f"Invalid kernel shape '{kernel_shape}'. Must be one of: {valid_shapes}"
            )

        self.operation = operation.lower()
        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape.lower()
        self.iterations = iterations

        # Create the morphological kernel
        self.kernel = self._create_kernel()

        # Map operation names to OpenCV constants
        self.operation_map = {
            "opening": cv2.MORPH_OPEN,
            "closing": cv2.MORPH_CLOSE,
            "erosion": cv2.MORPH_ERODE,
            "dilation": cv2.MORPH_DILATE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }

    def _create_kernel(self):
        """
        Create the morphological kernel based on shape and size

        Returns:
            numpy.ndarray: Morphological kernel
        """
        if self.kernel_shape == "rect":
            return cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.kernel_size, self.kernel_size)
            )
        elif self.kernel_shape == "ellipse":
            return cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
            )
        elif self.kernel_shape == "cross":
            return cv2.getStructuringElement(
                cv2.MORPH_CROSS, (self.kernel_size, self.kernel_size)
            )

    def process(self, frame):
        """
        Apply morphological operation to the input frame

        Args:
            frame (numpy.ndarray): Input image (can be grayscale or color)

        Returns:
            numpy.ndarray: Processed image after morphological operation
        """
        if frame is None:
            return frame

        # Convert to grayscale if the image is colored and we're doing morphological operations
        # Some operations work better on binary/grayscale images
        if len(frame.shape) == 3:
            # For color images, we can either:
            # 1. Apply to each channel separately
            # 2. Convert to grayscale first
            # For golf ball tracking, we'll apply to each channel
            processed = np.zeros_like(frame)
            for i in range(frame.shape[2]):
                processed[:, :, i] = cv2.morphologyEx(
                    frame[:, :, i],
                    self.operation_map[self.operation],
                    self.kernel,
                    iterations=self.iterations,
                )
            return processed
        else:
            # Grayscale image
            return cv2.morphologyEx(
                frame,
                self.operation_map[self.operation],
                self.kernel,
                iterations=self.iterations,
            )

    def set_operation(self, operation):
        """
        Change the morphological operation

        Args:
            operation (str): New operation to perform
        """
        valid_operations = [
            "opening",
            "closing",
            "erosion",
            "dilation",
            "gradient",
            "tophat",
            "blackhat",
        ]
        if operation.lower() not in valid_operations:
            raise ValueError(
                f"Invalid operation '{operation}'. Must be one of: {valid_operations}"
            )

        self.operation = operation.lower()

    def set_kernel_size(self, kernel_size):
        """
        Update the kernel size and recreate the kernel

        Args:
            kernel_size (int): New kernel size
        """
        self.kernel_size = kernel_size
        self.kernel = self._create_kernel()

    def set_kernel_shape(self, kernel_shape):
        """
        Update the kernel shape and recreate the kernel

        Args:
            kernel_shape (str): New kernel shape ('rect', 'ellipse', 'cross')
        """
        valid_shapes = ["rect", "ellipse", "cross"]
        if kernel_shape.lower() not in valid_shapes:
            raise ValueError(
                f"Invalid kernel shape '{kernel_shape}'. Must be one of: {valid_shapes}"
            )

        self.kernel_shape = kernel_shape.lower()
        self.kernel = self._create_kernel()

    def set_iterations(self, iterations):
        """
        Update the number of iterations

        Args:
            iterations (int): Number of iterations to perform
        """
        self.iterations = iterations

    def __str__(self):
        """
        String representation of the processor
        """
        return (
            f"MorphologyProcessor(operation='{self.operation}', "
            f"kernel_size={self.kernel_size}, kernel_shape='{self.kernel_shape}', "
            f"iterations={self.iterations})"
        )
