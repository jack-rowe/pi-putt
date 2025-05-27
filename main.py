#!/usr/bin/env python3
import time
import cv2
from lib.processing.FrameStabilizer import FrameStabilizer
from lib.processing.MedianFilter import MedianFilter
from lib.processing.Sharpen import Sharpen
from lib.processing.MorphologyProcessor import MorphologyProcessor
from lib.Camera import Camera
from lib.processing.Brightness import Brightness
from lib.processing.GaussianBlur import GaussianBlur
from lib.processing.EdgeDetection import EdgeDetection
from lib.processing.Mask import Mask
from lib.processing.PerspectiveCorrector import PerspectiveCorrector
from lib.processing.ProcessingPipeline import ProcessingPipeline


def process_video(input_source, output_file=None, duration=3, show_preview=True):
    """
    Process video from camera or file using the processing pipeline

    Args:
        input_source: Camera object or path to video file
        output_file (str): Path to save the processed video
        duration (int): Recording duration in seconds if using camera
        show_preview (bool): Whether to show preview window
    """
    # Setup the processing pipeline
    pipeline = ProcessingPipeline()

    pipelineProcessors = [
        GaussianBlur(kernel_size=3),  # Noise reduction
        # PerspectiveCorrector(),           # Geometric correction
        Brightness(alpha=2.0, beta=40),  # Enhanced visibility
        Sharpen("unsharp_mask", intensity=2.2, sigma=1.0),  # Enhance image details
        FrameStabilizer(method="background_subtraction", alpha=0.7),
        # Mask(
        #     lower_bound=[0, 0, 200],
        #     upper_bound=[180, 120, 255],
        # ),  # White ball
        MedianFilter(
            kernel_size=3, iterations=2
        ),  # Apply filter twice for stubborn noise
        # MorphologyProcessor("opening", 3),
        # MorphologyProcessor("closing", 5),
        # EdgeDetection(low_threshold=30, high_threshold=100),
        # HistogramEqualization()  # Improve contrast
        # WhiteBalancing()  # Correct color temperature
        # FrameAlignment()  # Align consecutive frames
        # BilateralFilter()  # Edge-preserving noise reduction
        # MedianFilter()  # Remove salt-and-pepper noise
    ]

    # Add processing modules in the desired order
    for processor in pipelineProcessors:
        pipeline.add_processor(processor)

    # Enable debug mode to see timing information
    pipeline.set_debug(True)

    # Initialize video source
    if isinstance(input_source, Camera):
        # Using camera
        camera = input_source
        camera.start_camera()

        if not camera.isRecording:
            print("Failed to start camera. Exiting...")
            return

        # Get first frame to determine dimensions
        if camera.picam2 is None:
            print("Camera not initialized properly")
            return

        frame = camera.picam2.capture_array()
        height, width = frame.shape[:2]
        fps = camera.framerate
    else:
        # Using video file
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer if output file is specified
    video_writer = None
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"Processing video at {width}x{height} @ {fps} fps")

    # Process frames
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            # Check if duration exceeded for camera recording
            if (
                isinstance(input_source, Camera)
                and time.time() - start_time >= duration
            ):
                break

            # Get frame from source
            if isinstance(input_source, Camera):
                if camera.picam2 is None or not camera.isRecording:
                    break
                frame = camera.picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:  # End of video file
                    break

            # Process the frame
            processed_frame = pipeline.process(frame)

            # Write frame to output video
            if video_writer:
                video_writer.write(processed_frame)

            frame_count += 1

    except KeyboardInterrupt:
        print("Processing interrupted")
    finally:
        # Clean up resources
        if isinstance(input_source, Camera):
            camera.stop_camera()
        else:
            cap.release()

        if video_writer:
            video_writer.release()

        cv2.destroyAllWindows()

        # Show stats
        total_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count / total_time:.2f}")

        if output_file:
            print(f"Saved processed video to: {output_file}")


if __name__ == "__main__":
    camera = Camera()
    output_file = "processed_video.mp4"

    process_video(
        input_source=camera, output_file=output_file, duration=5, show_preview=True
    )

    # Example 2: Process from video file
    # process_video(
    #     input_source="recordings/test.mp4",
    #     output_file="processed_output.mp4",
    #     show_preview=True,
    # )
