
# Speed and Lane Detection for Vehicles

## Overview

This project processes a video to detect lane lines and calculate the speed of vehicles using computer vision techniques. The primary objective is to analyze the movement of vehicles by detecting lane lines and tracking vehicle speed through predefined strips on the road. This system can be beneficial for traffic monitoring and automated vehicle guidance systems.

## Features

- **Lane Line Detection**: Utilizes edge detection and Hough Transform to detect lane lines in the video.
- **Speed Calculation**: Tracks the vehicle speed by calculating the time taken to cross predefined strips on the road.
- **Dynamic Overlays**: Draws lane lines, speed information, and deviation angles on the video frames.
- **Real-time Processing**: Processes video frames in real-time and displays the results with overlays.

## Demo

![Demo GIF](path/to/demo.gif)

## Getting Started

### Prerequisites

- Python 3.7+
- OpenCV
- NumPy

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/speed-and-lane-detection.git
    cd speed-and-lane-detection
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Run the main script with your video file:
    ```sh
    python src/main.py path/to/your/video.mp4
    ```

2. The application will process the video and display real-time lane detection and speed calculations.

## Detailed Description of Key Components

### Data Preparation and Model Training

**Calibration Data**: The system uses predefined distances between strips to calculate speed.

```python
# Constants
FRAME_RATE = 60  # Frame rate of the video, adjust according to your video
DISTANCE_BETWEEN_STRIPS = 1.7  # Distance between strips in meters, adjust as needed
```

### Real-time Processing

**Canny Edge Detection**: Detects edges in the video frames.

```python
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny
```

**Region of Interest**: Masks the region of interest in the video frames to focus on the road.

```python
def region_of_interest(image):
    height, width = image.shape[:2]
    top_left = [int(width * 0.37), int(height * 0.7)]
    top_right = [int(width * 0.46), int(height * 0.7)]
    bottom_left = [int(width * 0.04), height]
    bottom_right = [int(width * 0.65), height]
    polygons = np.array([[bottom_left, bottom_right, top_right, top_left]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
```

**Detect Lines**: Uses the Hough Transform to detect lines in the edge-detected frames.

```python
def detect_lines(image):
    lines = cv2.HoughLinesP(image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    return lines
```

**Average Lines**: Averages the detected lines for more stable lane detection.

```python
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
```

### Speed Tracking

**Speed Calculation**: Calculates the speed of the vehicle based on the time interval between crossings of predefined strips.

```python
def calculate_speed(distance, time_seconds):
    if time_seconds > 0:
        speed_mps = distance / time_seconds
        return speed_mps * 2.23694  # Convert from m/s to mph
    return 0
```

**SpeedTracker Class**: Tracks and maintains the vehicle speeds.

```python
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
```

### Process Video

Processes the video to detect lanes and calculate speed.

```python
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    speed_tracker = SpeedTracker()
    last_time = time.time()

    target_fps = 60
    frame_duration = 0.03 / target_fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - last_time

        if elapsed_time < frame_duration:
            time.sleep(frame_duration - elapsed_time)

        last_time = current_time

        edges = canny(frame)
        roi_edges, detection_line_y, roi_start_x, roi_end_x = get_roi_and_detection_line_y(edges)
        filled_strips_image = fill_strips(frame.copy(), roi_edges)

        if detect_crossing(roi_edges, detection_line_y):
            speed_tracker.update_crossing(time.time())

        average_speed = speed_tracker.get_average_speed()
        if average_speed > 0:
            speed_text = f"2- Speed: {average_speed:.2f} MPH"
            cv2.putText(filled_strips_image, speed_text, (1450, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = detect_lines(cropped_image)
        averaged_lines = average_lines(frame, lines)
        line_image = draw_lines(frame, averaged_lines)

        final_image = cv2.addWeighted(line_image, 0.5, filled_strips_image, 0.5, 0)
        cv2.line(final_image, (roi_start_x, detection_line_y), (roi_end_x, detection_line_y), (0, 255, 0), 2)
        cv2.imshow('Result', cv2.resize(final_image, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

## Contributions

Feel free to open issues or submit pull requests if you have suggestions for improving this project.

## License

This project is licensed under the MIT License.
