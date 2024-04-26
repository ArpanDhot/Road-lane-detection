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


# Path to the image file
image_path = 'um_000005.png'

# Process the image to get the edges
image, gray, blur, edges = process_image(image_path)

# Apply the trapezoidal mask to the Canny edges
trapezoid_masked_edges = mask_trapezoid(edges)

# Visualize the trapezoidal masked edges
plt.figure(figsize=(10, 5))
plt.imshow(trapezoid_masked_edges, cmap='gray')
plt.title('Trapezoid Masked Edges')
plt.axis('off')
plt.show()
