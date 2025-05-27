# lib/processing/ProcessingPipeline.py
import cv2
import numpy as np
import time


class ProcessingPipeline:
    def __init__(self):
        """
        Initialize an empty processing pipeline
        """
        self.processors = []
        self.debug = False
        self.last_process_time = 0

    def add_processor(self, processor):
        """
        Add a processor to the pipeline

        Args:
            processor: A processing object with a process() method
        """
        self.processors.append(processor)
        return self  # Allow chaining

    def set_debug(self, debug=True):
        """
        Enable or disable debug mode

        Args:
            debug (bool): Whether to enable debug outputs
        """
        self.debug = debug
        return self

    def process(self, frame):
        """
        Run the frame through all processors in the pipeline

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Processed image after running through the entire pipeline
        """
        start_time = time.time()

        # Make a copy of the input frame to avoid modifying the original
        result = frame.copy()

        # Apply each processor in sequence
        for i, processor in enumerate(self.processors):
            result = processor.process(result)

            # if self.debug:
            #     processor_name = processor.__class__.__name__
            #     print(f"Applied {processor_name}")

            #     # You could save intermediate results here if needed
            #     cv2.imwrite(f"debug_{i}_{processor_name}.jpg", result)

        self.last_process_time = time.time() - start_time

        if self.debug:
            print(f"Pipeline processing time: {self.last_process_time * 1000:.2f} ms")

        return result

    def get_last_process_time(self):
        """
        Get the time taken for the last processing run

        Returns:
            float: Processing time in seconds
        """
        return self.last_process_time
