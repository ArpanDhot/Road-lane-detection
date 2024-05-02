import cv2
import time
import numpy as np
from collections import deque


# Constants
FRAME_RATE = 60  # Frame rate of the video, adjust according to your video
DISTANCE_BETWEEN_STRIPS = 1  # Distance between strips in meters, adjust as needed



class Line:
    def __init__(self, max_history=10):
        self.recent_fits = deque(maxlen=max_history)

    def add_fit(self, fit):
        if fit is not None:
            self.recent_fits.append(fit)

    def average_fit(self):
        if self.recent_fits:
            avg_fit = np.mean(self.recent_fits, axis=0)
            return avg_fit
        return None

# Initialize line objects for left and right lane lines
left_line = Line()
right_line = Line()


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



def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height, width = image.shape[:2]
    # Modify these coordinates to adjust the trapezoid
    top_left = [int(width * 0.37), int(height * 0.7)]  # Move right to decrease width, down to decrease height
    top_right = [int(width * 0.46), int(height * 0.7)]  # Move left to decrease width, down to decrease height
    bottom_left = [int(width * 0.04), height]  # Move right to decrease bottom width
    bottom_right = [int(width * 0.65), height]  # Move left to decrease bottom width
    polygons = np.array([
        [bottom_left, bottom_right, top_right, top_left]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def detect_lines(image):
    lines = cv2.HoughLinesP(image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    return lines

def average_lines(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < -0.3:
                left_fit.append((slope, intercept))
            elif slope > 0.3:
                right_fit.append((slope, intercept))

    if left_fit:
        left_line.add_fit(np.mean(left_fit, axis=0))
    if right_fit:
        right_line.add_fit(np.mean(right_fit, axis=0))

    left_avg = left_line.average_fit()
    right_avg = right_line.average_fit()

    return calculate_lines(image, left_avg), calculate_lines(image, right_avg)

def calculate_lines(image, line_params):
    if line_params is not None:
        slope, intercept = line_params
        y1 = image.shape[0]
        y2 = int(y1 * 0.8)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]
    return None

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    overlay = np.copy(image)  # Create an overlay for semi-transparency
    if lines[0] is not None and lines[1] is not None:
        h, w = image.shape[:2]
        lane_center_x = (lines[0][2] + lines[1][2]) / 2  # Midpoint x-coordinate between the end points of lane lines
        min_y = min(lines[0][3], lines[1][3])  # Use the lowest y-value from the endpoints of the lane lines

        # Adjusting the starting point for both lines to the left of the frame center
        shift_left = int(w * 0.089)  # Shift 10% of the width to the left
        bottom_start_x = (w // 2) - shift_left  # Move starting point 10% to the left from the center

        # Drawing the blue line from the lane center
        cv2.line(overlay, (bottom_start_x, h), (int(lane_center_x), min_y), (255, 0, 0), 5)

        # Drawing the yellow line vertically centered
        cv2.line(overlay, (bottom_start_x, h), (w // 2 - shift_left, min_y), (0, 255, 255), 5)

        # Draw lines and fill polygon between them
        cv2.line(overlay, (lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), (255, 0, 0), 20)
        cv2.line(overlay, (lines[1][0], lines[1][1]), (lines[1][2], lines[1][3]), (255, 0, 0), 20)
        pts = np.array([[lines[0][0], lines[0][1]], [lines[0][2], lines[0][3]],
                        [lines[1][2], lines[1][3]], [lines[1][0], lines[1][1]]], np.int32)
        #cv2.fillPoly(overlay, [pts], (0, 255, 0))

        # Calculate the deviation angle from vertical
        delta_x = (w // 2 - shift_left) - int(lane_center_x)
        delta_y = h - min_y
        angle_rad = np.arctan2(delta_y, abs(delta_x))  # Angle in radians
        angle_deg = np.degrees(angle_rad)  # Convert radians to degrees
        deviation = abs(90 - angle_deg)  # Absolute deviation from 90 degrees

        # Steering direction based on angle
        if angle_deg == 90:
            steering_text = "1- Steering Straight"
        elif angle_deg < 90:
            steering_text = "1- Steering Left"
        else:
            steering_text = "1- Steering Right"

        # Display the calculated angle and steering direction
        deviation_text = f"Deviation: {deviation:.0f} degrees"
        text_size = cv2.getTextSize(deviation_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = bottom_start_x - text_size[0] // 2  # Center the text horizontally at the adjusted start point
        text_y = h - 10  # Set just above the bottom
        cv2.rectangle(overlay, (text_x - 10, text_y + 10), (text_x + text_size[0] + 10, text_y - text_size[1] - 10),
                      (0, 0, 0), -1)
        cv2.putText(overlay, deviation_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, steering_text, (text_x - 500, text_y - 930), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Blend overlay with original image
        alpha = 0.4  # Set transparency factor
        line_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return line_image


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


def overlay_edges_on_image(original_image, edge_image):
    """Creates a color overlay of the detected edges on the original image."""
    edge_colored = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    edge_colored[edge_image == 255] = [0, 0, 255]  # Edge pixels marked in red
    overlay_image = cv2.addWeighted(original_image, 0.8, edge_colored, 0.2, 0)
    return overlay_image

def fill_strips(image, edges):
    """Finds and fills the detected strips in the image based on their size."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50  # Minimum area for a contour to be considered a strip, adjust as needed
    max_area = 1000  # Maximum area to exclude too large contours, adjust as needed
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
    return image



# Process video for both speed and lane detection
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    speed_tracker = SpeedTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Speed detection logic
        edges = canny(frame)
        roi_edges, detection_line_y, roi_start_x, roi_end_x = get_roi_and_detection_line_y(edges)
        filled_strips_image = fill_strips(frame.copy(), roi_edges)

        if detect_crossing(roi_edges, detection_line_y):
            speed_tracker.update_crossing(time.time())

        average_speed = speed_tracker.get_average_speed()
        if average_speed > 0:
            print(f"Average Speed: {average_speed:.2f} MPH")

        # Lane detection logic
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = detect_lines(cropped_image)
        averaged_lines = average_lines(frame, lines)
        line_image = draw_lines(frame, averaged_lines)

        # Combine overlays
        final_image = cv2.addWeighted(line_image, 0.5, filled_strips_image, 0.5, 0)
        cv2.line(final_image, (roi_start_x, detection_line_y), (roi_end_x, detection_line_y), (0, 255, 0), 2)
        cv2.imshow('Result', cv2.resize(final_image, (960, 540)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
process_video('test_video.mp4')  # Replace with the path to your video