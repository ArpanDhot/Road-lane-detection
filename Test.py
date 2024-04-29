import cv2
import numpy as np


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    # Define vertices for the trapezoid
    polygons = np.array([
        [(int(width * 0.1), height),  # bottom left
         (int(width * 0.9), height),  # bottom right
         (int(width * 0.6), int(height * 0.65)),  # top right
         (int(width * 0.4), int(height * 0.65))]  # top left
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def main():
    # Load the video
    cap = cv2.VideoCapture('test_video.mp4')  # Replace with your video file path
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        return

    masked_frame = region_of_interest(frame)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Masked Frame', masked_frame)

    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
