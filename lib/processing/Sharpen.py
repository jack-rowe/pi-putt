# lib/processing/Sharpen.py
import cv2
import numpy as np


class Sharpen:
    def __init__(
        self, method="unsharp_mask", intensity=1.0, kernel_type="laplacian", sigma=1.0
    ):
        """
        Initialize Sharpening processor

        Args:
            method (str): Sharpening method to use
                         Options: 'unsharp_mask', 'kernel', 'high_boost'
            intensity (float): Sharpening intensity (0.0 to 3.0, where 1.0 is normal)
            kernel_type (str): Type of kernel for kernel-based sharpening
                              Options: 'laplacian', 'edge_enhance', 'custom'
            sigma (float): Standard deviation for Gaussian blur in unsharp masking
        """
        valid_methods = ["unsharp_mask", "kernel", "high_boost"]
        if method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

        valid_kernels = ["laplacian", "edge_enhance", "custom"]
        if kernel_type.lower() not in valid_kernels:
            raise ValueError(
                f"Invalid kernel_type '{kernel_type}'. Must be one of: {valid_kernels}"
            )

        self.method = method.lower()
        self.intensity = max(0.0, min(3.0, intensity))  # Clamp between 0 and 3
        self.kernel_type = kernel_type.lower()
        self.sigma = sigma

        # Create sharpening kernels
        self.kernels = self._create_kernels()

    def _create_kernels(self):
        """
        Create different sharpening kernels

        Returns:
            dict: Dictionary of kernels for different sharpening methods
        """
        kernels = {}

        # Standard Laplacian kernel (good for general sharpening)
        kernels["laplacian"] = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )

        # Edge enhancement kernel (more aggressive)
        kernels["edge_enhance"] = np.array(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
        )

        # Custom kernel optimized for small details (good for golf ball edges)
        kernels["custom"] = np.array(
            [[0, -1, 0], [-1, 6, -1], [0, -1, 0]], dtype=np.float32
        )

        return kernels

    def process(self, frame):
        """
        Apply sharpening to the input frame

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Sharpened image
        """
        if frame is None:
            return frame

        if self.method == "unsharp_mask":
            return self._unsharp_mask(frame)
        elif self.method == "kernel":
            return self._kernel_sharpen(frame)
        elif self.method == "high_boost":
            return self._high_boost_filter(frame)

    def _unsharp_mask(self, frame):
        """
        Apply unsharp masking for sharpening
        This is often the best method for natural-looking sharpening

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Sharpened image
        """
        # Convert to float for precise calculations
        frame_float = frame.astype(np.float32)

        # Create Gaussian blur
        blurred = cv2.GaussianBlur(frame_float, (0, 0), self.sigma)

        # Create unsharp mask: original + intensity * (original - blurred)
        sharpened = frame_float + self.intensity * (frame_float - blurred)

        # Clip values and convert back to uint8
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened

    def _kernel_sharpen(self, frame):
        """
        Apply kernel-based sharpening

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Sharpened image
        """
        kernel = self.kernels[self.kernel_type]

        # Adjust kernel intensity
        center_value = kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]
        adjusted_kernel = kernel.copy()
        adjusted_kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = center_value + (
            center_value - 1
        ) * (self.intensity - 1)

        # Apply convolution
        sharpened = cv2.filter2D(frame, -1, adjusted_kernel)

        return sharpened

    def _high_boost_filter(self, frame):
        """
        Apply high-boost filtering
        Formula: sharpened = A * original - low_pass
        where A > 1 (amplification factor)

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Sharpened image
        """
        # Convert to float
        frame_float = frame.astype(np.float32)

        # Create low-pass filter (Gaussian blur)
        low_pass = cv2.GaussianBlur(frame_float, (5, 5), self.sigma)

        # High-boost filter: A * original - low_pass
        A = 1.0 + self.intensity  # Amplification factor
        sharpened = A * frame_float - low_pass

        # Clip and convert back
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened

    def set_method(self, method):
        """
        Change the sharpening method

        Args:
            method (str): New sharpening method
        """
        valid_methods = ["unsharp_mask", "kernel", "high_boost"]
        if method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

        self.method = method.lower()

    def set_intensity(self, intensity):
        """
        Update the sharpening intensity

        Args:
            intensity (float): New intensity (0.0 to 3.0)
        """
        self.intensity = max(0.0, min(3.0, intensity))

    def set_kernel_type(self, kernel_type):
        """
        Update the kernel type for kernel-based sharpening

        Args:
            kernel_type (str): New kernel type
        """
        valid_kernels = ["laplacian", "edge_enhance", "custom"]
        if kernel_type.lower() not in valid_kernels:
            raise ValueError(
                f"Invalid kernel_type '{kernel_type}'. Must be one of: {valid_kernels}"
            )

        self.kernel_type = kernel_type.lower()

    def set_sigma(self, sigma):
        """
        Update the sigma value for Gaussian operations

        Args:
            sigma (float): New sigma value
        """
        self.sigma = sigma

    def __str__(self):
        """
        String representation of the processor
        """
        return (
            f"Sharpen(method='{self.method}', intensity={self.intensity}, "
            f"kernel_type='{self.kernel_type}', sigma={self.sigma})"
        )
