import json

class CalibrationPipeline:
    def __init__(self, camera, light_controller=None):
        self.camera = camera
        self.light_controller = light_controller
        self.calibration_data = {}

    def run_full_calibration(self):
        pass

    def save_calibration(self):
        """Save calibration to file"""
        with open("golf_tracking_calibration.json", "w") as f:
            json.dump(self.calibration_data, f, indent=2)
        print("📁 Calibration saved to golf_tracking_calibration.json")

    def load_calibration(self):
        """Load previous calibration"""
        try:
            with open("golf_tracking_calibration.json", "r") as f:
                self.calibration_data = json.load(f)
            print("📁 Previous calibration loaded")
            return True
        except FileNotFoundError:
            print("⚠️ No previous calibration found")
            return False
