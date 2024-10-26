import cv2
import queue
import threading
import numpy as np

# Create a queue to store video frames
q = queue.Queue()

class LaserPointerDetector:
    def __init__(self):
        # Define the HSV range for the laser
        self.laser_min = np.array([0, 0, 230], np.uint8)
        self.laser_max = np.array([8, 115, 255], np.uint8)

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_threshed = cv2.inRange(hsv_img, self.laser_min, self.laser_max)

        contours, _ = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                laserx = int(M['m10'] / M['m00'])
                lasery = int(M['m01'] / M['m00'])
                return laserx, lasery
        return None, None

def receive():
    print("Starting to receive video stream")
    cap = cv2.VideoCapture("rtsp://admin:admin_123@172.0.0.0")
    while True:
        ret, frame = cap.read()
        if ret:
            q.put(frame)
        else:
            print("Failed to grab frame")
            break
    cap.release()

def display(detector):
    print("Starting to display video stream")
    while True:
        if not q.empty():
            frame = q.get()
            laserx, lasery = detector.detect(frame)

            if laserx is not None and lasery is not None:
                cv2.circle(frame, (laserx, lasery), 10, (0, 255, 0), -1)  # Draw the laser point

            cv2.imshow("Laser Pointer Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    detector = LaserPointerDetector()
    p1 = threading.Thread(target=receive)
    p2 = threading.Thread(target=display, args=(detector,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    cv2.destroyAllWindows()
