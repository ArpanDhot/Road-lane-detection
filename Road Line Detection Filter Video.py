import cv2
import numpy as np
from collections import deque

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
            if slope < -0.3:  # Left lane
                left_fit.append((slope, intercept))
            elif slope > 0.3:  # Right lane
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
        y2 = int(y1 * 0.8)  # Adjust to reduce line length
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]
    return None

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    overlay = np.copy(image)  # Create an overlay for semi-transparency
    if lines[0] is not None and lines[1] is not None:
        # Draw lines and fill polygon between them
        cv2.line(overlay, (lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), (255, 0, 0), 10)
        cv2.line(overlay, (lines[1][0], lines[1][1]), (lines[1][2], lines[1][3]), (255, 0, 0), 10)
        pts = np.array([[lines[0][0], lines[0][1]], [lines[0][2], lines[0][3]],
                        [lines[1][2], lines[1][3]], [lines[1][0], lines[1][1]]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        # Blend overlay
        alpha = 0.4  # Transparency factor
        line_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return line_image

cap = cv2.VideoCapture('test_video.mp4')  # Replace 'test_video.mp4' with your video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = detect_lines(cropped_image)
    averaged_lines = average_lines(frame, lines)
    line_image = draw_lines(frame, averaged_lines)
    cv2.imshow('result', line_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
