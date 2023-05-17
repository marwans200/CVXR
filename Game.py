import ursina.window
from ursina import *
import cv2
import numpy as np
import threading

app = Ursina()

window.fullscreen = False

player = Entity(model='cube', color=color.orange, scale=(1, 2, 1))
ground = Entity(model='plane', color=color.green, scale=(10, 1, 10), collider='box')

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(0)

center_x1 = 0
center_y1 = 0
center_x2 = 0
center_y2 = 0

def update():
    global center_x2,center_y2
    player.x = center_x2 * -0.01 + (window.size.x / 2 * 0.01)
    player.y = center_y2 * -0.01 + (window.size.y / 2 * 0.01)
def tracking():

    while True:
        global center_y1,center_x1,center_x2,center_y2

        lower_green = np.array([50, 50, 50])
        upper_green = np.array([70, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            quit()

        hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_frame1, lower_yellow, upper_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask1 = cv2.erode(mask1, kernel, iterations=2)
        mask1 = cv2.dilate(mask1, kernel, iterations=2)
        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours1) > 0:
            largest_contour1 = max(contours1, key=cv2.contourArea)
            x1, y1, w1, h1 = cv2.boundingRect(largest_contour1)
            cv2.rectangle(frame1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

            center_x1 = x1 + (w1 // 2)
            center_y1 = y1 + (h1 // 2)

        # Process frames from the second camera to track blue
        hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv_frame2, lower_green, upper_green)
        mask2 = cv2.erode(mask2, kernel, iterations=1)
        mask2 = cv2.dilate(mask2, kernel, iterations=1)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours2) > 0:
            largest_contour2 = max(contours2, key=cv2.contourArea)
            x2, y2, w2, h2 = cv2.boundingRect(largest_contour2)
            cv2.rectangle(frame2, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

            center_x2 = x2 + (w2 // 2)
            center_y2 = y2 + (h2 // 2)

        # Resize frames to have the same height
        frame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Combine frames horizontally
        combined_frame = np.concatenate((frame1, frame2), axis=1)

        # Display the combined frame
        cv2.imshow('Color Tracking (Combined)', combined_frame)

        if cv2.waitKey(1) == ord('q'):
            quit()
        time.sleep(0.02)

thread = threading.Thread(target=tracking)
thread.start()

app.run()
