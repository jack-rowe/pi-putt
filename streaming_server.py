#!/usr/bin/env python3
from http import server
import socketserver
import threading
import cv2
import numpy as np
import io
import json
import time
import urllib.parse
from golf_ball_detector import GolfBallDetector

# HTML template with UI controls
PAGE = """\
<!DOCTYPE html>
<html>
<head>
    <title>Golf Ball Tracking Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
            background-color: #f0f0f0;
            color: #333;
        }
        h1 {
            color: #2c3e50;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .controls {
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stream {
            margin-top: 20px;
            border: 3px solid #2c3e50;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            overflow: hidden;
            display: inline-block;
        }
        .control-group {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .control-group h3 {
            margin-top: 0;
            color: #2980b9;
        }
        .btn {
            padding: 8px 16px;
            margin: 5px;
            border: none;
            border-radius: 4px;
            background-color: #3498db;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .btn.active {
            background-color: #16a085;
        }
        .slider-container {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        .slider-container label {
            min-width: 150px;
            text-align: right;
            margin-right: 10px;
        }
        .slider-container input[type="range"] {
            flex-grow: 1;
            max-width: 300px;
        }
        .slider-container .value {
            min-width: 50px;
            text-align: left;
            margin-left: 10px;
        }
        .toggle-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
        }
        .status {
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
        }
        .collapsible {
            cursor: pointer;
            padding: 10px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 16px;
            background-color: #3498db;
            color: white;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .active, .collapsible:hover {
            background-color: #2980b9;
        }
        .collapsible:after {
            content: '+';
            float: right;
            margin-left: 5px;
            font-weight: bold;
        }
        .active:after {
            content: '-';
        }
        .content {
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            background-color: #f9f9f9;
            border-radius: 0 0 4px 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Golf Ball Tracking System</h1>
        
        <div class="stream">
            <img src="stream.mjpg" width="1024" height="576" />
        </div>
    </div>
    

</body>
</html>
"""

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    # Get the latest frame from the detector
                    frame = self.server.detector.get_latest_frame()
                    
                    if frame is not None:
                        # Convert frame to JPEG
                        ret, jpeg = cv2.imencode('.jpg', frame)
                        
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(jpeg.tobytes()))
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b'\r\n')
                    else:
                        # If no frame is available, send a placeholder
                        time.sleep(0.1)
            except Exception as e:
                print(f"Streaming error: {e}")
        elif self.path.startswith('/update_param'):
            self.handle_parameter_update()
        elif self.path.startswith('/update_display'):
            self.handle_display_update()
        elif self.path == '/reset':
            self.handle_reset()
        else:
            self.send_error(404)
            self.end_headers()
    
    def handle_parameter_update(self):
        """Handle parameter update requests"""
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        
        name = params.get('name', [''])[0]
        value = params.get('value', [''])[0]
        
        success = False
        if name and value:
            success = self.server.detector.update_detection_params(name, value)
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        if success:
            self.wfile.write(f"Updated {name} to {value}".encode('utf-8'))
        else:
            self.wfile.write(f"Failed to update {name}".encode('utf-8'))
    
    
    def handle_reset(self):
        """Handle reset to defaults request"""
        # Reset detection parameters to defaults
        self.server.detector.detection_params = {
            'hsv_lower': np.array([0, 0, 200]),
            'hsv_upper': np.array([180, 30, 255]),
            'circularity_threshold': 0.75,
            'hough_param1': 50,
            'hough_param2': 30,
            'min_radius': 10,
            'max_radius': 50,
            'min_area': 150,
            'temporal_consistency': True,
            'consistency_threshold': 3,
        }
        
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write("Reset to defaults".encode('utf-8'))

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    
    def __init__(self, address, handler, detector):
        self.detector = detector
        super().__init__(address, handler)

def start_server(detector, port=8000):
    """Start the streaming server with the given detector"""
    address = ('', port)
    server = StreamingServer(address, StreamingHandler, detector)
    print(f"Starting golf ball tracking web server on port {port}")
    print(f"Open a browser and navigate to http://[raspberry-pi-ip]:{port}")
    server.serve_forever()

# Main execution
if __name__ == '__main__':
    # Create and start the detector
    detector = GolfBallDetector()
    detector.start_processing()
    
    # Start the web server
    try:
        start_server(detector)
    except KeyboardInterrupt:
        print("Server stopped by user")
        detector.stop_processing()
    except Exception as e:
        print(f"Server error: {e}")
        detector.stop_processing()