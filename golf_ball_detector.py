#!/usr/bin/env python3
from picamera2 import Picamera2
import cv2
import time
import numpy as np
import threading
import queue

class GolfBallDetector:
    def __init__(self):
        # Detection parameters - adjusted for smaller resolution and lower light conditions
        self.detection_params = {
            'hsv_lower': np.array([0, 0, 80]),  # Reduced brightness threshold (from 160 to 120)
            'hsv_upper': np.array([180, 100, 255]),  # Increased saturation tolerance (from 60 to 80)
            'circularity_threshold': 0.85,  # Further relaxed for lower light conditions
            'hough_param1': 35,  # Reduced for better detection in lower light
            'hough_param2': 20,  # Reduced for more sensitivity in lower light
            'min_radius': 5,     # Kept the same
            'max_radius': 25,    # Kept the same
            'min_area': 40,      # Reduced for lower light conditions
            'temporal_consistency': True,  # Enable temporal filtering
            'consistency_threshold': 4,    # Reduced to be more lenient
            'position_tolerance': 12,      # Increased for better tracking in lower light
        }
       
        # For temporal consistency
        self.previous_detections = []
        self.consistent_detections = []
       
        # Set up frame queue for communication with server
        self.frame_queue = queue.Queue(maxsize=2)
       
        # Initialize camera
        self.picam2 = None
        self.running = False
       
    def start_camera(self):
        """Initialize and start the camera with higher framerate and lower resolution"""
        self.picam2 = Picamera2()
        preview_config = self.picam2.create_video_configuration(
            main={"size": (320, 120), "format": "BGR888"}, 
            controls={
                "FrameRate": 100.0,  # Request 120 FPS
                "FrameDurationLimits": (10000, 10000),  # ~120fps in microseconds
                "ExposureTime": 8000,  # Slightly longer exposure time for lower light (increased from 5000)
                "AnalogueGain": 3.0    # Increased gain for lower light conditions (from 2.0 to 3.0)
            }
        )
        self.picam2.configure(preview_config)
        self.picam2.start()
        print("Camera started with resolution 320x240 at target 120 FPS (optimized for lower light)")
       
    def stop_camera(self):
        """Stop the camera"""
        if self.picam2:
            self.picam2.stop()
            print("Camera stopped")
           
    def detect_golf_ball(self, frame):
        """Optimized detection for small resolution and high FPS, adjusted for lower light"""
        # Skip format conversion if not needed
        if frame.shape[2] == 4:  # If RGBA/BGRA format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            frame_bgr = frame  # No need to copy, just reference
       
        # Apply slight brightness adjustment to help with low light
        adjusted_frame = cv2.convertScaleAbs(frame_bgr, alpha=1.1, beta=10)
       
        # Convert to HSV color space
        hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
       
        # Create a mask for white/light colors using current parameters
        mask = cv2.inRange(hsv, self.detection_params['hsv_lower'], self.detection_params['hsv_upper'])
       
        # Simplified morphological operations - reduce iterations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
       
        # Additional dilation to help connect edges in lower light
        mask = cv2.dilate(mask, kernel, iterations=1)
       
        # Simplified edge detection for performance - adjusted for lower light
        edges = cv2.Canny(mask, 30, 100)  # Reduced thresholds for lower light
       
        # Alternate approach: Parallel processing of both detection methods
        detected_circles = []
       
        # For small resolution, HoughCircles can be more optimized with different parameters
        circles = cv2.HoughCircles(
            mask,  # Use mask directly instead of edges for better detection at small resolution
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,  # Reduced for better detection in low light
            param1=self.detection_params['hough_param1'],
            param2=self.detection_params['hough_param2'],
            minRadius=self.detection_params['min_radius'],
            maxRadius=self.detection_params['max_radius']
        )
       
        # If circles are found
        if circles is not None:
            # Convert circles to integers
            circles = np.uint16(np.around(circles))
           
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
               
                # For lower light, check a small region around the center rather than just the center point
                center_x, center_y = center
                # Create a mask to check a small region around the center
                check_radius = max(3, radius // 4)  # Use a reasonable minimum size
                
                # Create an ROI around the center point
                min_x = max(0, center_x - check_radius)
                max_x = min(mask.shape[1] - 1, center_x + check_radius)
                min_y = max(0, center_y - check_radius)
                max_y = min(mask.shape[0] - 1, center_y + check_radius)
                
                # Check if there are enough white pixels in the region
                roi = mask[min_y:max_y, min_x:max_x]
                if roi.size > 0:
                    white_ratio = cv2.countNonZero(roi) / roi.size
                    if white_ratio > 0.3:  # More lenient threshold for lower light
                        detected_circles.append((center, radius, 'hough', white_ratio))
       
        # Find contours (can be processed in parallel with HoughCircles)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < self.detection_params['min_area']:
                continue
               
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
               
            circularity = 4 * np.pi * area / (perimeter * perimeter)
           
            # Relaxed circularity threshold for lower light
            if circularity > self.detection_params['circularity_threshold']:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
               
                # Check if this contour is already detected by Hough Circles - use faster distance calculation
                is_duplicate = False
                for existing_center, existing_radius, _, _ in detected_circles:
                    # Manhattan distance is faster than Euclidean
                    distance = abs(center[0] - existing_center[0]) + abs(center[1] - existing_center[1])
                    if distance < (radius + existing_radius) / 2:
                        is_duplicate = True
                        break
                       
                if not is_duplicate:
                    detected_circles.append((center, radius, 'contour', circularity))
       
        # Simplified temporal consistency for performance
        # Only use if we have enough previous detections
        current_detections = []
        for center, radius, method, score in detected_circles:
            current_detections.append((center[0], center[1], radius))
       
        # Update previous detections
        self.previous_detections.append(current_detections)
        if len(self.previous_detections) > self.detection_params['consistency_threshold']:
            self.previous_detections.pop(0)
       
        # Simplified temporal consistency logic for performance
        if self.detection_params['temporal_consistency'] and len(self.previous_detections) >= 2:  # Reduced from 3 to 2
            position_tolerance = self.detection_params['position_tolerance']
            consistent_detections = []
            
            for center, radius, method, score in detected_circles:
                detection = (center[0], center[1], radius)
                
                # Check only in a few recent frames instead of all previous detections
                recent_frames = self.previous_detections[-3:]  # Just use last 3 frames
                appearances = 0
                
                for prev_frame in recent_frames:
                    for prev in prev_frame:
                        # Manhattan distance is faster than Euclidean
                        if (abs(prev[0] - detection[0]) < position_tolerance and
                            abs(prev[1] - detection[1]) < position_tolerance):
                            appearances += 1
                            break
                
                if appearances >= 1:  # Only require 1 appearance for high framerate
                    consistent_detections.append((center, radius, method, score))
                    
            # Update consistent detections
            self.consistent_detections = consistent_detections
        else:
            # If we don't have enough previous frames, use all detections
            self.consistent_detections = detected_circles
        
        return self.consistent_detections, mask, edges
           
    def create_display_frame(self, original_frame, circles, mask, edges):
        """Create the display frame with adjusted sizes for smaller resolution"""
        display_frame = original_frame.copy()
        
        # Draw detected circles
        for (center, radius, method, score) in circles:
            # Use different colors based on detection method
            color = (0, 255, 0) if method == 'hough' else (0, 165, 255)  # Green for Hough, Orange for contour
           
            # Draw circle around the ball - thinner line for small resolution
            cv2.circle(display_frame, center, radius, color, 1)
           
            # Draw center point
            cv2.circle(display_frame, center, 1, (0, 0, 255), -1)
           
            # Add text with coordinates, radius - smaller font for small resolution
            text = f"({center[0]},{center[1]}),r={radius}"
            cv2.putText(display_frame, text, (center[0] - radius, center[1] - radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add stats - smaller font
        cv2.putText(
            display_frame, f"Balls: {len(circles)}", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
       
        # Add mask and edge overlays - smaller for small resolution
        h, w = display_frame.shape[:2]
       
        # Create smaller versions of the mask and edges
        mask_small = cv2.resize(mask, (64, 48))
        mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
       
        # Add mask in the top-left corner
        display_frame[5:5+mask_small.shape[0], 5:5+mask_small.shape[1]] = mask_color
       
        # Add small label
        cv2.putText(display_frame, "MASK", (5, 5+mask_small.shape[0]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
       
        # Create a smaller version of the edges
        edges_small = cv2.resize(edges, (64, 48))
        edges_color = cv2.cvtColor(edges_small, cv2.COLOR_GRAY2BGR)
       
        # Add edges in the top-right corner
        display_frame[5:5+edges_small.shape[0], w-5-edges_small.shape[1]:w-5] = edges_color
       
        # Add small label
        cv2.putText(display_frame, "EDGES", (w-5-edges_small.shape[1], 5+edges_small.shape[0]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add FPS counter - smaller font
        curr_time = time.time()
        processing_fps = 1 / (curr_time - self.prev_time) if hasattr(self, 'prev_time') else 0
        self.prev_time = curr_time
        
        cv2.putText(
            display_frame, f"FPS: {int(processing_fps)}", (5, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
       
        return display_frame
   
    def update_detection_params(self, param_name, value):
        """Update detection parameters from UI"""
        if param_name in self.detection_params:
            # Handle special case for numpy arrays
            if param_name in ['hsv_lower', 'hsv_upper']:
                # Parse array values from string like "0,0,200"
                components = [int(x) for x in value.split(',')]
                self.detection_params[param_name] = np.array(components)
            elif isinstance(self.detection_params[param_name], bool):
                self.detection_params[param_name] = value.lower() == 'true'
            else:
                # Try to convert to the same type as the existing parameter
                param_type = type(self.detection_params[param_name])
                self.detection_params[param_name] = param_type(value)
            return True
        return False
   
    def process_frames(self):
        """Main processing loop - optimized for high FPS"""
        self.running = True
        self.prev_time = time.time()
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.picam2.capture_array()
               
                # Detect golf ball with current parameters
                circles, mask, edges = self.detect_golf_ball(frame)
               
                # Create display frame with fixed display options
                display_frame = self.create_display_frame(frame, circles, mask, edges)
               
                # Put frame in queue for server to access
                try:
                    # Non-blocking put to avoid slowdowns if server is not consuming frames
                    self.frame_queue.put(display_frame, block=False)
                except queue.Full:
                    # Skip this frame if queue is full
                    pass
               
                # Print stats occasionally
                frame_count += 1
                if frame_count % 500 == 0:  # Less frequent reporting at high FPS
                    elapsed = time.time() - start_time
                    print(f"Processed {frame_count} frames at {frame_count/elapsed:.2f} FPS")
                    print(f"Detected {len(circles)} golf balls")
               
                # Remove delay to maximize FPS, especially for >100FPS target
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_camera()
   
    def start_processing(self):
        """Start the frame processing in a thread"""
        self.start_camera()
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("Golf ball detection started")
   
    def stop_processing(self):
        """Stop the processing thread"""
        self.running = False
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.stop_camera()
        print("Golf ball detection stopped")
   
    def get_latest_frame(self):
        """Get the latest processed frame for the web server"""
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None

# For direct testing of the detector
if __name__ == '__main__':
    detector = GolfBallDetector()
    detector.start_processing()
   
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detector.stop_processing()
        print("Detector stopped by user")