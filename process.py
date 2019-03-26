import os
import cv2
import numpy as np

def process1(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 4.2, minDist=80, minRadius=10, maxRadius=120)
    circles = np.uint16(np.around(circles))
    return circles[0, :]