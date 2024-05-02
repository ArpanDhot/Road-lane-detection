import cv2
import numpy as np
import time
from collections import deque

# Constants
FRAME_RATE = 60  # Frame rate of the video, adjust according to your video
DISTANCE_BETWEEN_STRIPS = 1  # Distance between strips in meters, adjust as needed


def canny(image):
    """Applies Canny edge detection to an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def get_roi_and_detection_line_y(frame):
    """Applies an ROI mask to the frame and returns the y-coordinate for the detection line."""
    height, width = frame.shape[:2]
    # Define the ROI; adjust these points based on your actual ROI
    top_left = (450, height)
    top_right = (700, int(height * 0.79))
    bottom_right = (700, int(height * 0.73))
    bottom_left = (25, height)

    polygons = np.array([
        [bottom_left, bottom_right, top_right, top_left]
    ])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(frame, mask)

    # Calculate the middle y-coordinate of the ROI for the detection line
    top_y = int(height * 0.7)
    bottom_y = height
    detection_line_y = (top_y + bottom_y) // 2

    return masked_image, detection_line_y, top_left[0], bottom_right[0]


def detect_crossing(edges, line_y):
    """Checks if there are any edges crossing the detection line."""
    crossings = np.any(edges[line_y, :] > 0)
    return crossings


def calculate_speed(distance, time_seconds):
    """Calculates speed given distance and time."""
    if time_seconds > 0:
        speed_mps = distance / time_seconds  # Speed in meters per second
        return speed_mps * 2.23694  # Convert from m/s to mph
    return 0


class SpeedTracker:
    def __init__(self):
        self.speeds = deque(maxlen=FRAME_RATE)
        self.last_crossing_time = None

    def update_crossing(self, current_time):
        if self.last_crossing_time is not None:
            time_interval = current_time - self.last_crossing_time
            speed = calculate_speed(DISTANCE_BETWEEN_STRIPS, time_interval)
            self.speeds.append(speed)
        self.last_crossing_time = current_time

    def get_average_speed(self):
        if len(self.speeds) > 0:
            return sum(self.speeds) / len(self.speeds)
        return 0


def overlay_edges_on_image(original_image, edge_image):
    """Creates a color overlay of the detected edges on the original image."""
    edge_colored = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    edge_colored[edge_image == 255] = [0, 0, 255]  # Edge pixels marked in red
    overlay_image = cv2.addWeighted(original_image, 0.8, edge_colored, 0.2, 0)
    return overlay_image

def fill_strips(image, edges):
    """Finds and fills the detected strips in the image based on their size."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 40  # Minimum area for a contour to be considered a strip, adjust as needed
    max_area = 1000  # Maximum area to exclude too large contours, adjust as needed
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
    return image



def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    speed_tracker = SpeedTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        edges = canny(frame)
        roi_edges, detection_line_y, roi_start_x, roi_end_x = get_roi_and_detection_line_y(edges)
        filled_strips_image = fill_strips(frame.copy(), roi_edges)

        if detect_crossing(roi_edges, detection_line_y):
            speed_tracker.update_crossing(time.time())

        average_speed = speed_tracker.get_average_speed()
        if average_speed > 0:
            print(f"Average Speed: {average_speed:.2f} MPH")

        overlay_image = overlay_edges_on_image(filled_strips_image, roi_edges)
        cv2.line(overlay_image, (roi_start_x, detection_line_y), (roi_end_x, detection_line_y), (0, 255, 0), 2)
        cv2.imshow('Edges with Detection Line', overlay_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage
process_video('test_video.mp4')  # Replace with the path to your video