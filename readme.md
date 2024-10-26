# Laser Pointer Detection

This project is designed to receive a video stream from an RTSP camera, detect a laser pointer in the video feed, and display the detected laser point in real-time.

## Features

- Real-time video streaming from an RTSP camera.
- Detection of laser pointers based on color thresholds.
- Visualization of detected laser points on the video feed.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vvamanda/LaserPointerDetection.git
   cd laser-pointer-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Update the RTSP URL in the `receive()` function of `main.py` to match your camera's stream.
2. Run the script:
   ```bash
   python main.py
   ```
3. Press `q` to exit the video display.



## Code Explanation

### 1. Importing Libraries

```python
import cv2
import numpy as np
```

- `cv2`: The OpenCV library used for computer vision tasks.
- `numpy`: Used for handling arrays and matrix operations.

### 2. Defining Color Thresholds

```python
lower_color = np.array([0, 0, 255])
upper_color = np.array([100, 100, 255])
```

- `lower_color` and `upper_color` define the color range for the laser pointer (red) to be detected in the image.

### 3. Receiving Video Stream

```python
def receive():
    cap = cv2.VideoCapture('rtsp://your_camera_url')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
```

- `cv2.VideoCapture`: Opens the RTSP stream.
- `cap.read()`: Reads frames from the video stream.

### 4. Processing Each Frame

```python
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
```

- `cv2.cvtColor`: Converts the BGR image to HSV color space for better color detection.
- `cv2.inRange`: Creates a mask that filters out pixels within the specified color range.

### 5. Finding Laser Points

```python
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

- `cv2.findContours`: Finds contours in the mask.
- `cv2.boundingRect`: Calculates the bounding box for the contour and draws a rectangle around it.

### 6. Displaying Results

```python
        cv2.imshow('Laser Pointer Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

- `cv2.imshow`: Displays the processed frame.
- `cv2.waitKey`: Waits for user input; press `q` to exit.

### 7. Releasing Resources

```python
    cap.release()
    cv2.destroyAllWindows()
```

- Releases the video capture object and closes all OpenCV windows.
