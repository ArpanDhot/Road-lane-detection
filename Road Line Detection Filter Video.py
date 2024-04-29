import cv2
import numpy as np

def preprocess_frame(frame):
    """Convert frame to grayscale and blur it."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)     # Apply Gaussian Blur
    return blurred

def apply_trapezoid_mask(image):
    """Apply a trapezoid mask to the image to focus on the lane area."""
    mask = np.zeros_like(image)  # Create a mask that's the same dimensions as the image
    height, width = image.shape

    # Points are defined in the following order: bottom left, top left, top right, bottom right
    polygon = np.array([
        [(int(width * 0.1), height), (int(width * 0.4), int(height * 0.6)),
         (int(width * 0.6), int(height * 0.6)), (int(width * 0.9), height)]
    ], np.int32)

    cv2.fillPoly(mask, polygon, 255)  # Fill the polygon with white color
    masked_image = cv2.bitwise_and(image, mask)  # Apply the mask to the grayscale image
    return masked_image

def apply_perspective_transform(image):
    """Apply a perspective transformation to the image to get a bird's eye view."""
    height, width = image.shape
    src = np.float32([
        [int(width * 0.4), int(height * 0.6)],
        [int(width * 0.6), int(height * 0.6)],
        [int(width * 0.9), height],
        [int(width * 0.1), height]
    ])
    dst = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (width, height))
    return warped

def main(video_source=0):
    # Open the video source (0 for default camera, or "path/to/video" for a file)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frame is captured

        processed_frame = preprocess_frame(frame)
        masked_frame = apply_trapezoid_mask(processed_frame)
        bird_eye_frame = apply_perspective_transform(masked_frame)

        # Display the bird's eye view frame
        cv2.imshow('Bird\'s Eye View', bird_eye_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("test_video.mp4")
