import os
import cv2
import numpy as np

def process1(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100, param1=200, param2=100, minRadius=50, maxRadius=120)
    try:
        circles = np.uint16(np.around(circles))
    except:
        return []

    return circles[0, :]
