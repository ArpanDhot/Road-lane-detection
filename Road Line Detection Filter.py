import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to apply processing steps to an image
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    return image, gray, blur, edges


# Function to apply a trapezoidal mask to the image
def mask_trapezoid(edges):
    # Define a mask that is the same size as the edges image
    mask = np.zeros_like(edges)

    # Get the image dimensions
    imshape = edges.shape
    height, width = imshape[0], imshape[1]

    # Define the trapezoid dimensions
    lower_width = width
    upper_width = int(0.4 * width)
    height = int(0.5 * height)

    # Define the polygon for the mask based on the trapezoid dimensions
    vertices = np.array([[(width // 2 - upper_width // 2, height),
                          (width // 2 + upper_width // 2, height),
                          (width, imshape[0]), (0, imshape[0])]], dtype=np.int32)

    # Fill the defined polygon with white color
    cv2.fillPoly(mask, vertices, 255)

    # Mask the edges image
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges


# Function to apply the Hough Transform and filter the lines by angle and length
def hough_transform(masked_edges, image, min_line_length_threshold):
    # Define the Hough transform parameters
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on the edge-detected image
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Create an empty image to draw lines on
    line_image = np.zeros_like(image)

    # Find the midpoint at the bottom of the trapezoid
    midpoint_x = image.shape[1] // 2

    # Prepare lists to hold the qualifying lines on the left and right
    left_lines = []
    right_lines = []

    # Iterate over the output lines to sort them into left and right
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate the length and angle as before
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)

            # Calculate the slope and the intercept
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            intercept = y1 - slope * x1

            # Check if the line is of sufficient length and has an acceptable angle
            if length > min_line_length_threshold and (angle > 20 and angle < 160):
                # Determine if the line is on the left or right
                if slope < 0 and x1 < midpoint_x and x2 < midpoint_x:
                    left_lines.append((slope, intercept))
                elif slope > 0 and x1 > midpoint_x and x2 > midpoint_x:
                    right_lines.append((slope, intercept))

    # Function to find the line with the highest intercept on the image, which
    # is equivalent to the line that appears first from the bottom
    def find_first_line(lines):
        if lines:
            # Sort the lines based on the intercept
            lines = sorted(lines, key=lambda line: line[1], reverse=True)
            # Return the first line
            return lines[0]
        return None

    # Find the first line on the left and right
    first_left_line = find_first_line(left_lines)
    first_right_line = find_first_line(right_lines)

    # Draw the first left line
    if first_left_line is not None:
        slope, intercept = first_left_line
        y1 = image.shape[0]
        x1 = int((y1 - intercept) / slope)
        y2 = int(y1 / 2)
        x2 = int((y2 - intercept) / slope)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # Draw the first right line
    if first_right_line is not None:
        slope, intercept = first_right_line
        y1 = image.shape[0]
        x1 = int((y1 - intercept) / slope)
        y2 = int(y1 / 2)
        x2 = int((y2 - intercept) / slope)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image


# Path to the image file
image_path = 'um_000005.png'

# Process the image to get the edges
image, gray, blur, edges = process_image(image_path)

# Apply the trapezoidal mask to the Canny edges
trapezoid_masked_edges = mask_trapezoid(edges)

# Define the minimum line length threshold
min_line_length_threshold = 100  # Adjust this value as needed

# Use the Hough Transform to detect lines and filter them by length and angle
line_image = hough_transform(trapezoid_masked_edges, image, min_line_length_threshold)

# Draw the lines on the original image
combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

# Visualize the final image
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title('Lane Lines')
plt.axis('off')
plt.show()
