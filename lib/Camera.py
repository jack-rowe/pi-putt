from picamera2 import Picamera2


class Camera:
    def __init__(self):
        self.picam2 = None
        self.framerate = 60.0
        self.exposure_time = 8000
        self.analogue_gain = 3.0
        self.resolution = (320, 120)
        self.format = "BGR888"
        self.isRecording = False

    def start_camera(self):
        """Initialize and start the camera with higher framerate and lower resolution"""
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_video_configuration(
                main={"size": self.resolution, "format": self.format},
                controls={
                    "FrameRate": self.framerate,
                    # "FrameDurationLimits": (10000, 10000),  # ~100fps in microseconds
                    "ExposureTime": self.exposure_time,
                    "AnalogueGain": self.analogue_gain,
                },
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            self.isRecording = True
        except:
            self.isRecording = False
            print("Error: Unable to start camera")

    def stop_camera(self):
        """Stop the camera"""
        try:
            if self.picam2:
                self.picam2.stop()
                print("Camera stopped")
                self.isRecording = False
        except:
            print("Error: Unable to stop camera")
