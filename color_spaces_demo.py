import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
while True:

    _, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    # mask = cv.inRange(hsv, lower_blue, upper_blue)

    # lower_purple = np.array([100,50,50])
    # upper_purple = np.array([280,200,200])
    # mask = cv.inRange(hsv, lower_purple, upper_purple)

    lower = np.array([120,50,50])
    upper = np.array([250,200,200])
    mask = cv.inRange(hsv, lower, upper)


    res = cv.bitwise_and(frame, frame, mask = mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()