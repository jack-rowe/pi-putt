#!/usr/bin/env python3
from EventBus import EventBus
from golf_ball_detector import GolfBallDetector
from streaming_server import start_server

eventbus = EventBus()


def boot_handler():
    print("System booting up...")
    print("Initializing hardware...")
    print("Running diagnostics...")
    # When initialization is complete, change to the IDLE mode
    eventbus.change_mode("IDLE")


def idle_handler():
    print("System is idle and ready for commands")


eventbus.register_mode_handler("BOOT", boot_handler)
eventbus.register_mode_handler("IDLE", idle_handler)


def main():
    # Create and start the detector
    detector = GolfBallDetector()
    # Start the detector
    detector.start_processing()

    # Start the web server
    try:
        print(f"Golf Ball Tracking System started")
        start_server(detector, port=8000)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop_processing()
        print("System stopped")


if __name__ == "__main__":
    eventbus.change_mode("BOOT")
    # main()
    print("DONE")
