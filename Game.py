import time

import ursina.window
from ursina import *
import cv2
import numpy as np
import threading

app = Ursina()
window.fullscreen = False

Right_Controller = Entity(model='cube', color=color.green, scale=(1, 2, 1))
Left_Controller = Entity(model='cube', color=color.yellow, scale=(1, 2, 1))

ground = Entity(model='plane', color=color.green, scale=(10, 1, 10), collider='box')

Z_Tracker = cv2.VideoCapture(1)
XY_Tracker = cv2.VideoCapture(0)

lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

Left_Controller_X = 0
Left_Controller_Y = 0
Right_Controller_X = 0
Right_Controller_Y = 0


def update():
    global Right_Controller_Y, Right_Controller_X

    Right_Controller.x = Right_Controller_X * -0.01 + (window.size.x / 2 * 0.01)
    Right_Controller.y = Right_Controller_Y * -0.01 + (window.size.y / 2 * 0.01)

    Left_Controller.x = Left_Controller_X * -0.01 + (window.size.x / 2 * 0.01)
    Left_Controller.y = Left_Controller_Y * -0.01 + (window.size.x / 2 * 0.01)

    if held_keys['q']:
        Z_Tracker.release()
        XY_Tracker.release()
        cv2.destroyAllWindows()
        quit()


def tracking():
    while True:

        global Right_Controller_Y, Right_Controller_X, Left_Controller_X, Left_Controller_Y

        ret1, frame1 = XY_Tracker.read()
        ret3, frame3 = XY_Tracker.read()

        if not ret1:
            break

        hsv_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        mask2 = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask2 = cv2.erode(mask2, kernel, iterations=1)
        mask2 = cv2.dilate(mask2, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _2 = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)

            kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
            kalman.measurementNoiseCov = np.array([[w, 0],
                                                   [0, h]], np.float32) * 0.01
            measurement = np.array([[x + w / 2],
                                    [y + h / 2]], np.float32)
            kalman.correct(measurement)

            prediction = kalman.predict()
            Right_Controller_X = prediction[0, 0]
            Right_Controller_Y = prediction[1, 0]

        if len(contours2) > 0:

            largest_contour2 = max(contours2, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour2)

            kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
            kalman.measurementNoiseCov = np.array([[w, 0],
                                                   [0, h]], np.float32) * 0.01
            measurement = np.array([[x + w / 2],
                                    [y + h / 2]], np.float32)
            kalman.correct(measurement)

            prediction = kalman.predict()
            Left_Controller_X = prediction[0, 0]
            Left_Controller_Y = prediction[1, 0]

        time.sleep(0.02)


thread = threading.Thread(target=tracking)
thread.start()

app.run()
