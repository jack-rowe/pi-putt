#!/usr/bin/env python3
from golf_ball_detector import GolfBallDetector
from streaming_server import start_server
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Golf Ball Tracking System')
    parser.add_argument('--port', type=int, default=8000, help='Port for the web server (default: 8000)')
    parser.add_argument('--no-temporal', action='store_true', help='Disable temporal consistency filtering')
    parser.add_argument('--view-mode', type=str, default='all', 
                        choices=['all', 'circles_only', 'mask_only', 'edges_only', 'raw'],
                        help='Initial view mode (default: all)')
    args = parser.parse_args()
    
    # Create and start the detector
    detector = GolfBallDetector()
    
    # Apply command line configurations
    if args.no_temporal:
        detector.detection_params['temporal_consistency'] = False
    
    # Start the detector
    detector.start_processing()
    
    # Start the web server
    try:
        print(f"Golf Ball Tracking System started")
        print(f"Access the interface at http://[raspberry-pi-ip]:{args.port}")
        start_server(detector, port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop_processing()
        print("System stopped")

if __name__ == '__main__':
    main()