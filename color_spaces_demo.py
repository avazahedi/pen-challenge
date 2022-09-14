import cv2 as cv
import numpy as np

cap = cv.VideoCapture()
while(1):

    _, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_blue = np.array([110,50,50])
    