# lib/processing/FrameStabilizer.py
import cv2
import numpy as np
from collections import deque


class FrameStabilizer:
    def __init__(
        self, method="temporal_average", buffer_size=3, threshold=10, alpha=0.8
    ):
        """
        Initialize Frame Stabilizer to reduce TV static effect between consecutive frames

        Args:
            method (str): Stabilization method to use
                         Options: 'temporal_average', 'weighted_average', 'background_subtraction', 'threshold_filter'
            buffer_size (int): Number of frames to keep in buffer for averaging (3-7 recommended)
            threshold (int): Pixel difference threshold for noise detection (0-50)
            alpha (float): Blending factor for weighted averaging (0.1-0.9)
        """
        valid_methods = [
            "temporal_average",
            "weighted_average",
            "background_subtraction",
            "threshold_filter",
        ]
        if method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

        self.method = method.lower()
        self.buffer_size = max(2, min(10, buffer_size))  # Clamp between 2-10
        self.threshold = max(0, min(255, threshold))
        self.alpha = max(0.1, min(0.9, alpha))

        # Frame buffer to store recent frames
        self.frame_buffer = deque(maxlen=self.buffer_size)

        # Background model for background subtraction
        self.background_model = None

        # Previous frame for comparison
        self.previous_frame = None

        # Running average for weighted method
        self.running_average = None

        # Frame counter
        self.frame_count = 0

    def process(self, frame):
        """
        Apply frame stabilization to reduce static effect

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            numpy.ndarray: Stabilized frame
        """
        if frame is None:
            return frame

        self.frame_count += 1

        # For the first few frames, just return the original
        if self.frame_count <= 2:
            self._update_buffers(frame)
            return frame

        if self.method == "temporal_average":
            result = self._temporal_average(frame)
        elif self.method == "weighted_average":
            result = self._weighted_average(frame)
        elif self.method == "background_subtraction":
            result = self._background_subtraction(frame)
        elif self.method == "threshold_filter":
            result = self._threshold_filter(frame)

        # Update buffers for next frame
        self._update_buffers(frame)

        return result

    def _temporal_average(self, frame):
        """
        Average current frame with recent frames to reduce noise

        Args:
            frame (numpy.ndarray): Current frame

        Returns:
            numpy.ndarray: Averaged frame
        """
        # Add current frame to buffer
        temp_buffer = list(self.frame_buffer)
        temp_buffer.append(frame)

        # Convert frames to float for averaging
        frame_stack = np.stack([f.astype(np.float32) for f in temp_buffer])

        # Calculate temporal average
        averaged = np.mean(frame_stack, axis=0)

        # Convert back to uint8
        return np.clip(averaged, 0, 255).astype(np.uint8)

    def _weighted_average(self, frame):
        """
        Use exponential weighted average to stabilize frames

        Args:
            frame (numpy.ndarray): Current frame

        Returns:
            numpy.ndarray: Stabilized frame
        """
        if self.running_average is None:
            self.running_average = frame.astype(np.float32)
            return frame

        # Exponential moving average: new_avg = alpha * current + (1-alpha) * old_avg
        frame_float = frame.astype(np.float32)
        self.running_average = (
            self.alpha * frame_float + (1 - self.alpha) * self.running_average
        )

        return np.clip(self.running_average, 0, 255).astype(np.uint8)

    def _background_subtraction(self, frame):
        """
        Use background subtraction to identify and reduce static noise

        Args:
            frame (numpy.ndarray): Current frame

        Returns:
            numpy.ndarray: Stabilized frame
        """
        if self.background_model is None:
            # Initialize background model with current frame
            self.background_model = frame.astype(np.float32)
            return frame

        frame_float = frame.astype(np.float32)

        # Calculate difference from background
        diff = np.abs(frame_float - self.background_model)

        # Create mask for pixels that changed significantly
        if len(frame.shape) == 3:
            # For color images, use magnitude of change across all channels
            change_mask = np.sqrt(np.sum(diff**2, axis=2)) > self.threshold
            change_mask = np.expand_dims(change_mask, axis=2)
            change_mask = np.repeat(change_mask, 3, axis=2)
        else:
            change_mask = diff > self.threshold

        # Update background model slowly for static areas
        update_rate = 0.05
        self.background_model = np.where(
            change_mask,
            frame_float,  # Keep new pixels for changed areas
            update_rate * frame_float + (1 - update_rate) * self.background_model,
        )  # Slow update for static areas

        # Return the current frame but with reduced noise in static areas
        stabilized = np.where(change_mask, frame_float, self.background_model)

        return np.clip(stabilized, 0, 255).astype(np.uint8)

    def _threshold_filter(self, frame):
        """
        Filter out small pixel changes between consecutive frames

        Args:
            frame (numpy.ndarray): Current frame

        Returns:
            numpy.ndarray: Filtered frame
        """
        if self.previous_frame is None:
            return frame

        frame_float = frame.astype(np.float32)
        prev_float = self.previous_frame.astype(np.float32)

        # Calculate pixel-wise difference
        diff = np.abs(frame_float - prev_float)

        # Create mask for significant changes
        if len(frame.shape) == 3:
            # For color images, consider change across all channels
            significant_change = np.sqrt(np.sum(diff**2, axis=2)) > self.threshold
            significant_change = np.expand_dims(significant_change, axis=2)
            significant_change = np.repeat(significant_change, 3, axis=2)
        else:
            significant_change = diff > self.threshold

        # Keep new pixels only where change is significant, otherwise use previous frame
        stabilized = np.where(significant_change, frame_float, prev_float)

        return np.clip(stabilized, 0, 255).astype(np.uint8)

    def _update_buffers(self, frame):
        """
        Update internal buffers with current frame

        Args:
            frame (numpy.ndarray): Current frame to add to buffers
        """
        # Add to frame buffer
        self.frame_buffer.append(frame.copy())

        # Update previous frame
        self.previous_frame = frame.copy()

    def reset(self):
        """
        Reset all buffers and models (useful when switching scenes)
        """
        self.frame_buffer.clear()
        self.background_model = None
        self.previous_frame = None
        self.running_average = None
        self.frame_count = 0

    def set_method(self, method):
        """
        Change the stabilization method

        Args:
            method (str): New stabilization method
        """
        valid_methods = [
            "temporal_average",
            "weighted_average",
            "background_subtraction",
            "threshold_filter",
        ]
        if method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

        self.method = method.lower()
        self.reset()  # Reset buffers when changing methods

    def set_threshold(self, threshold):
        """
        Update the change detection threshold

        Args:
            threshold (int): New threshold value (0-255)
        """
        self.threshold = max(0, min(255, threshold))

    def set_alpha(self, alpha):
        """
        Update the blending factor for weighted averaging

        Args:
            alpha (float): New alpha value (0.1-0.9)
        """
        self.alpha = max(0.1, min(0.9, alpha))

    def set_buffer_size(self, buffer_size):
        """
        Update the buffer size for temporal averaging

        Args:
            buffer_size (int): New buffer size (2-10)
        """
        self.buffer_size = max(2, min(10, buffer_size))
        # Update the deque maxlen
        self.frame_buffer = deque(
            list(self.frame_buffer)[-self.buffer_size :], maxlen=self.buffer_size
        )

    def __str__(self):
        """
        String representation of the processor
        """
        return (
            f"FrameStabilizer(method='{self.method}', buffer_size={self.buffer_size}, "
            f"threshold={self.threshold}, alpha={self.alpha})"
        )
