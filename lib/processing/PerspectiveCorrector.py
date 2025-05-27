# lib/processing/ArucoPerspectiveCorrector.py
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerspectiveCorrector:
    def __init__(
        self,
        dictionary_id=cv2.aruco.DICT_6X6_250,
        marker_ids=None,
        output_size=(320, 120),
        margin=20,
        marker_size=0.05,
        verbose_logging=True,
        enhance_detection=True,
    ):
        """
        Combined ArUco detection and perspective correction processor
        Args:
            dictionary_id: ArUco dictionary to use (default: DICT_6X6_250)
            marker_ids (list): List of marker IDs to use for correction (None for all)
            output_size (tuple): Size of output image (width, height)
            margin (int): Margin to add around detected markers
            marker_size (float): Physical size of markers in meters (for pose estimation)
            verbose_logging (bool): Enable detailed logging of marker detection
            enhance_detection (bool): Use enhanced detection parameters for low-res/difficult conditions
        """
        # Set instance variables first
        self.marker_size = marker_size
        self.verbose_logging = verbose_logging
        self.enhance_detection = enhance_detection

        # ArUco detection setup
        self.dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Enhanced detection parameters for low resolution and difficult conditions
        if enhance_detection:
            self._configure_enhanced_detection()

        # Configuration
        self.marker_ids = marker_ids
        self.output_size = output_size
        self.margin = margin

        # State tracking
        self.corners = None
        self.ids = None
        self.rejected = None
        self.last_src_points = None
        self.transform_matrix = None

        # Logging state
        self.last_detected_ids = set()
        self.frame_count = 0

        # Define destination points (rectangle in output image)
        self.dst_points = np.array(
            [
                [0, 0],
                [output_size[0], 0],
                [output_size[0], output_size[1]],
                [0, output_size[1]],
            ],
            dtype=np.float32,
        )

    def _configure_enhanced_detection(self):
        """
        Configure detector parameters for better detection in challenging conditions
        """
        # More aggressive corner detection
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23  # Increased from default
        self.parameters.adaptiveThreshWinSizeStep = 4  # Smaller steps

        # More lenient thresholds
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.minMarkerPerimeterRate = (
            0.01  # Allow smaller markers (default: 0.03)
        )
        self.parameters.maxMarkerPerimeterRate = 4.0  # Allow larger markers

        # Corner detection tuning
        self.parameters.polygonalApproxAccuracyRate = (
            0.1  # More flexible polygon approximation
        )
        self.parameters.minCornerDistanceRate = 0.05  # Allow closer corners
        self.parameters.minDistanceToBorder = 1  # Allow markers closer to border

        # Marker validation - be more permissive
        self.parameters.minOtsuStdDev = 2.0  # Lower threshold (default: 5.0)
        self.parameters.perspectiveRemovePixelPerCell = (
            8  # Higher resolution perspective correction
        )
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.1

        # Error correction
        self.parameters.maxErroneousBitsInBorderRate = (
            0.5  # Allow more errors in border
        )
        self.parameters.errorCorrectionRate = 1.0  # Maximum error correction

        if self.verbose_logging:
            logger.info(
                "Enhanced detection parameters configured for low-resolution/challenging conditions"
            )

    def process(self, frame):
        """
        Detect ArUco markers and apply perspective correction
        Args:
            frame (numpy.ndarray): Input image
        Returns:
            numpy.ndarray: Perspective corrected image
        """
        self.frame_count += 1

        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply image preprocessing for better detection
        if self.enhance_detection:
            gray = self._preprocess_for_detection(gray)

        # Detect markers - call the function with the image
        self.corners, self.ids, self.rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

        # Log marker detection details
        self._log_marker_detection()

        # Get bounding box from detected markers
        src_points = self._get_bounding_box_from_markers()

        if src_points is not None:
            # Add margin around the detected area
            if self.margin > 0:
                src_points = self._add_margin_to_points(src_points)

            # Update transformation matrix
            self.transform_matrix = cv2.getPerspectiveTransform(
                np.float32(src_points), self.dst_points
            )
            self.last_src_points = src_points

        elif self.last_src_points is not None:
            # Use last known good points if no markers detected
            self.transform_matrix = cv2.getPerspectiveTransform(
                np.float32(self.last_src_points), self.dst_points
            )

        # Apply perspective correction if we have a valid transformation
        if self.transform_matrix is not None:
            corrected = cv2.warpPerspective(
                frame, self.transform_matrix, (self.output_size[0], self.output_size[1])
            )

            # Add marker ID labels to the corrected output if any markers were detected
            if self.ids is not None:
                corrected = self._add_marker_labels_to_output(corrected, frame)

            return corrected
        else:
            # Return original frame resized if no transformation available
            return cv2.resize(frame, self.output_size)

    def _add_marker_labels_to_output(self, corrected_frame, original_frame):
        """
        Add marker ID labels to the perspective-corrected output
        Args:
            corrected_frame: The perspective corrected frame
            original_frame: The original input frame
        Returns:
            numpy.ndarray: Corrected frame with marker labels
        """
        if self.ids is None or self.transform_matrix is None:
            return corrected_frame

        result = corrected_frame.copy()

        for i, marker_id in enumerate(self.ids.flatten()):
            # Get marker center in original frame
            corners = self.corners[i][0]
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])

            # Transform the center point to corrected frame coordinates
            original_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(
                original_point, self.transform_matrix
            )

            # Extract transformed coordinates
            tx = int(transformed_point[0][0][0])
            ty = int(transformed_point[0][0][1])

            # Only draw if the point is within the corrected frame bounds
            if 0 <= tx < self.output_size[0] and 0 <= ty < self.output_size[1]:
                # Draw marker ID label
                label = f"ID:{marker_id}"

                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

                # Position text (adjust if near edges)
                text_x = max(
                    5,
                    min(tx - text_size[0] // 2, self.output_size[0] - text_size[0] - 5),
                )
                text_y = max(
                    text_size[1] + 5,
                    min(ty + text_size[1] // 2, self.output_size[1] - 5),
                )

                # Draw background rectangle
                cv2.rectangle(
                    result,
                    (text_x - 3, text_y - text_size[1] - 3),
                    (text_x + text_size[0] + 3, text_y + 3),
                    (0, 0, 0),
                    -1,
                )

                # Draw border
                cv2.rectangle(
                    result,
                    (text_x - 3, text_y - text_size[1] - 3),
                    (text_x + text_size[0] + 3, text_y + 3),
                    (255, 255, 255),
                    1,
                )

                # Draw text
                cv2.putText(
                    result,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness,
                )

                # Draw small center dot
                cv2.circle(result, (tx, ty), 3, (0, 255, 255), -1)

        return result

    def _preprocess_for_detection(self, gray):
        """
        Apply image preprocessing to improve marker detection
        Args:
            gray (numpy.ndarray): Grayscale input image
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Optional: Apply slight Gaussian blur to reduce noise
        # enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Optional: Sharpen the image slightly
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

    def test_detection_on_frame(self, frame, show_preprocessing=False):
        """
        Test detection on a single frame with detailed output
        Args:
            frame: Input frame
            show_preprocessing: Whether to show preprocessing steps
        Returns:
            dict: Detection results and debug info
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        original_gray = gray.copy()

        if self.enhance_detection:
            gray = self._preprocess_for_detection(gray)

        # Try detection with current parameters
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

        results = {
            "detected_count": 0 if ids is None else len(ids),
            "detected_ids": [] if ids is None else ids.flatten().tolist(),
            "rejected_count": len(rejected),
            "frame_size": frame.shape[:2],
            "preprocessing_applied": self.enhance_detection,
        }

        if show_preprocessing and self.enhance_detection:
            # Show before/after preprocessing
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            ax1.imshow(original_gray, cmap="gray")
            ax1.set_title("Original Grayscale")
            ax1.axis("off")

            ax2.imshow(gray, cmap="gray")
            ax2.set_title("After Preprocessing")
            ax2.axis("off")

            # Show detection overlay
            overlay = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            for rej in rejected:
                cv2.polylines(overlay, [rej.astype(np.int32)], True, (0, 0, 255), 2)

            ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax3.set_title(
                f'Detection Results\nFound: {results["detected_count"]}, Rejected: {results["rejected_count"]}'
            )
            ax3.axis("off")

            plt.tight_layout()
            plt.show()

        logger.info(f"Detection test results: {results}")
        return results

    def _log_marker_detection(self):
        """
        Log detailed information about detected markers
        """
        if not self.verbose_logging:
            return

        current_ids = set()
        if self.ids is not None:
            current_ids = set(self.ids.flatten())

        # Log new detections
        new_detections = current_ids - self.last_detected_ids
        lost_detections = self.last_detected_ids - current_ids

        if new_detections:
            logger.info(
                f"Frame {self.frame_count}: NEW MARKERS DETECTED: {sorted(new_detections)}"
            )

        if lost_detections:
            logger.info(
                f"Frame {self.frame_count}: MARKERS LOST: {sorted(lost_detections)}"
            )

        # Log detailed coordinates for all currently detected markers
        if self.ids is not None and len(self.ids) > 0:
            logger.info(f"Frame {self.frame_count}: DETECTED {len(self.ids)} MARKERS")

            for i, marker_id in enumerate(self.ids.flatten()):
                corners = self.corners[i][0]  # Get the 4 corners of this marker

                # Calculate center point
                center_x = np.mean(corners[:, 0])
                center_y = np.mean(corners[:, 1])

                # Calculate marker area (approximate)
                area = cv2.contourArea(corners)

                logger.info(f"  Marker ID {marker_id}:")
                logger.info(f"    Center: ({center_x:.1f}, {center_y:.1f})")
                logger.info(f"    Area: {area:.1f} pixels")
                logger.info(f"    Corners: {self._format_corners(corners)}")

                # Calculate marker orientation (angle of top edge)
                top_left = corners[0]
                top_right = corners[1]
                angle = np.degrees(
                    np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0])
                )
                logger.info(f"    Orientation: {angle:.1f}°")

        elif self.frame_count % 30 == 0:  # Log every 30 frames when no markers found
            logger.info(f"Frame {self.frame_count}: NO MARKERS DETECTED")
            if len(self.rejected) > 0:
                logger.info(f"  Rejected candidates: {len(self.rejected)}")

        self.last_detected_ids = current_ids

    def _format_corners(self, corners):
        """
        Format corner coordinates for logging
        """
        formatted = []
        corner_names = [
            "TL",
            "TR",
            "BR",
            "BL",
        ]  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        for i, (x, y) in enumerate(corners):
            formatted.append(f"{corner_names[i]}({x:.1f},{y:.1f})")
        return " | ".join(formatted)

    def get_marker_info(self, marker_id):
        """
        Get detailed information about a specific marker
        Args:
            marker_id (int): ID of the marker to get info for
        Returns:
            dict: Marker information or None if not found
        """
        if self.ids is None:
            return None

        marker_indices = np.where(self.ids.flatten() == marker_id)[0]
        if len(marker_indices) == 0:
            return None

        corners = self.corners[marker_indices[0]][0]
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        area = cv2.contourArea(corners)

        # Calculate orientation
        top_left = corners[0]
        top_right = corners[1]
        angle = np.degrees(
            np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0])
        )

        return {
            "id": marker_id,
            "center": (center_x, center_y),
            "corners": corners.tolist(),
            "area": area,
            "orientation": angle,
        }

    def print_all_detected_markers(self):
        """
        Print summary of all currently detected markers
        """
        print(f"\n=== CURRENT MARKER DETECTION (Frame {self.frame_count}) ===")
        if self.ids is None or len(self.ids) == 0:
            print("No markers detected")
            return

        print(f"Total markers detected: {len(self.ids)}")
        print(f"Marker IDs: {sorted(self.ids.flatten().tolist())}")

        for marker_id in sorted(self.ids.flatten()):
            info = self.get_marker_info(marker_id)
            if info:
                print(f"\nMarker {marker_id}:")
                print(f"  Center: ({info['center'][0]:.1f}, {info['center'][1]:.1f})")
                print(f"  Area: {info['area']:.1f} pixels")
                print(f"  Orientation: {info['orientation']:.1f}°")

    def _get_bounding_box_from_markers(self):
        """
        Get bounding box that encompasses specified markers
        Returns:
            numpy.ndarray: Four corner points of bounding box
        """
        if self.ids is None or len(self.ids) == 0:
            return None

        # Collect all corner points
        all_points = []

        if self.marker_ids is None:
            # Use all detected markers
            for corner in self.corners:
                all_points.extend(corner[0])
        else:
            # Use only specified markers
            for marker_id in self.marker_ids:
                marker_corners = self._get_marker_corners(marker_id)
                if marker_corners is not None:
                    all_points.extend(marker_corners)

        if len(all_points) == 0:
            return None

        all_points = np.array(all_points)

        # Find bounding rectangle
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)

        # Return as four corners (top-left, top-right, bottom-right, bottom-left)
        return np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        )

    def _get_marker_corners(self, marker_id):
        """
        Get corners of a specific marker
        Args:
            marker_id (int): Marker ID to get corners for
        Returns:
            numpy.ndarray: Marker corners
        """
        if self.ids is None or len(self.ids) == 0:
            return None

        # Find specific marker
        marker_indices = np.where(self.ids.flatten() == marker_id)[0]
        if len(marker_indices) == 0:
            return None

        return self.corners[marker_indices[0]][0]  # Return first occurrence

    def _add_margin_to_points(self, src_points):
        """
        Add margin around the source points
        Args:
            src_points (numpy.ndarray): Original source points
        Returns:
            numpy.ndarray: Source points with margin added
        """
        center = np.mean(src_points, axis=0)
        src_points_centered = src_points - center

        # Calculate scale factor based on margin
        width = np.max(src_points_centered[:, 0]) - np.min(src_points_centered[:, 0])
        height = np.max(src_points_centered[:, 1]) - np.min(src_points_centered[:, 1])
        scale_factor = 1 + (2 * self.margin) / min(width, height)

        return center + src_points_centered * scale_factor

    def get_detection_info(self):
        """
        Get information about the last detection
        Returns:
            dict: Detection information including marker count, IDs, etc.
        """
        info = {
            "markers_detected": 0 if self.ids is None else len(self.ids),
            "marker_ids": [] if self.ids is None else self.ids.flatten().tolist(),
            "has_transform": self.transform_matrix is not None,
            "using_last_known_transform": self.last_src_points is not None
            and self.ids is None,
        }
        return info

    def draw_detection_overlay(
        self, frame, draw_individual_boxes=True, draw_overall_bounding_box=True
    ):
        """
        Draw ArUco markers and bounding box on frame (for debugging/visualization)
        Args:
            frame (numpy.ndarray): Input image
            draw_individual_boxes (bool): Draw boxes around individual markers
            draw_overall_bounding_box (bool): Draw overall bounding box around all markers
        Returns:
            numpy.ndarray: Frame with detection overlay
        """
        result = frame.copy()

        # Draw detected markers
        if self.ids is not None:
            # Draw the standard ArUco marker outlines (green)
            cv2.aruco.drawDetectedMarkers(result, self.corners, self.ids)

            # Draw individual boxes around each marker
            if draw_individual_boxes:
                for i, marker_id in enumerate(self.ids.flatten()):
                    corners = self.corners[i][0]
                    center_x = int(np.mean(corners[:, 0]))
                    center_y = int(np.mean(corners[:, 1]))

                    # Draw a tight bounding box around the marker (cyan)
                    min_x = int(np.min(corners[:, 0]))
                    max_x = int(np.max(corners[:, 0]))
                    min_y = int(np.min(corners[:, 1]))
                    max_y = int(np.max(corners[:, 1]))

                    # Draw rectangular box around marker
                    cv2.rectangle(
                        result,
                        (min_x - 5, min_y - 5),
                        (max_x + 5, max_y + 5),
                        (255, 255, 0),
                        2,
                    )

                    # Draw center point
                    cv2.circle(result, (center_x, center_y), 5, (255, 0, 0), -1)

                    # Draw marker ID with background rectangle for better visibility
                    text = f"ID:{marker_id}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[
                        0
                    ]
                    text_x = center_x + 10
                    text_y = center_y - 10

                    # Background rectangle for text
                    cv2.rectangle(
                        result,
                        (text_x - 2, text_y - text_size[1] - 2),
                        (text_x + text_size[0] + 2, text_y + 2),
                        (0, 0, 0),
                        -1,
                    )

                    # Text
                    cv2.putText(
                        result,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                    # Draw coordinates
                    coord_text = f"({center_x},{center_y})"
                    cv2.putText(
                        result,
                        coord_text,
                        (text_x, text_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1,
                    )

        # Draw overall bounding box if available
        if draw_overall_bounding_box:
            src_points = self._get_bounding_box_from_markers()
            if src_points is not None:
                # Add margin for visualization
                if self.margin > 0:
                    src_points = self._add_margin_to_points(src_points)

                # Draw overall bounding box (green)
                pts = src_points.astype(np.int32)
                cv2.polylines(result, [pts], True, (0, 255, 0), 3)

                # Label corners
                corner_labels = ["TL", "TR", "BR", "BL"]
                for i, pt in enumerate(pts):
                    cv2.circle(result, tuple(pt), 7, (0, 255, 0), -1)
                    cv2.putText(
                        result,
                        corner_labels[i],
                        tuple(pt + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        # Add detection info text with background
        info = self.get_detection_info()

        # Background rectangle for info text
        cv2.rectangle(result, (5, 5), (300, 90), (0, 0, 0), -1)
        cv2.rectangle(result, (5, 5), (300, 90), (255, 255, 255), 2)

        cv2.putText(
            result,
            f"Markers: {info['markers_detected']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        if info["marker_ids"]:
            cv2.putText(
                result,
                f"IDs: {info['marker_ids']}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Show rejected candidates count
        cv2.putText(
            result,
            f"Rejected: {len(self.rejected) if self.rejected else 0}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 100, 100),
            2,
        )

        return result
