import cv2
import numpy as np

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

def average_lines(image, lines, last_lines):
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
        left_line = calculate_line_average(image, left_fit)
    else:
        left_line = last_lines[0]

    if right_fit:
        right_line = calculate_line_average(image, right_fit)
    else:
        right_line = last_lines[1]

    return [left_line, right_line]

def calculate_line_average(image, line_parameters):
    if line_parameters:
        slope, intercept = np.average(line_parameters, axis=0)
        y1 = image.shape[0]
        y2 = int(y1 * 0.72)  # Adjust to reduce line length
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]
    else:
        return None

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    overlay = np.copy(image)  # Create a copy of the image to use as an overlay
    if lines is not None:
        # Ensure we have two lines and all points are present
        if lines[0] is not None and lines[1] is not None:
            x1, y1, x2, y2 = lines[0]  # Line 1
            x3, y3, x4, y4 = lines[1]  # Line 2
            # Define polygon to fill between lines
            polygon_points = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int32)
            cv2.fillPoly(overlay, [polygon_points], (0, 255, 0))  # Fill polygon with green color on overlay
            # Draw the lines on the overlay
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 15)
            cv2.line(overlay, (x3, y3), (x4, y4), (255, 0, 0), 15)
        # Blend the overlay with the original image using weighted sum
        alpha = 0.4  # Transparency factor
        line_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return line_image





cap = cv2.VideoCapture('test_video.mp4')
last_known_lines = [None, None]  # Initialize last known lines

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = detect_lines(cropped_image)
    averaged_lines = average_lines(frame, lines, last_known_lines)
    if any(averaged_lines):  # Update last known lines if current lines are detected
        last_known_lines = averaged_lines
    line_image = draw_lines(frame, last_known_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
