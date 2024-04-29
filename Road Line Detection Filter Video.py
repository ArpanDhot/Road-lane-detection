import cv2
import numpy as np


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]

    # Define new vertices for the trapezoid based on your specific needs
    polygons = np.array([
        [(width * 0.1, height),  # bottom left
         (width * 0.9, height),  # bottom right
         (width * 0.6, height * 0.6),  # top right
         (width * 0.4, height * 0.6)]  # top left
    ], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def canny_edge_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def process_frame(frame):
    canny_image = canny_edge_detector(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combo_image

# Load the video
cap = cv2.VideoCapture('test_video.mp4')  # Replace 'path_to_video.mp4' with your video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Lane Detection', processed_frame)

    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
