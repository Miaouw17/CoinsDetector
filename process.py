import os
import cv2
import numpy as np
from config import *


def process1(img):
    circles = cv2.HoughCircles(img, **HOUGH_CIRCLE)
    circles = np.uint16(np.around(circles))
    return circles[0, :]


def process2(img):
    ret, labels = cv2.connectedComponents(img)
    hist, bins = np.histogram(labels.ravel(), 256, [0, 256])

    valid_surface_id = []
    for i in range(1, len(hist)):
        if hist[i] > 2000:
            valid_surface_id.append(i)

    circles = []
    for i in valid_surface_id:
        img = labels.copy()
        mask = img == i
        nmask = img != i
        img[mask] = 255
        img[nmask] = 0
        img = np.uint8(img)

        # cv2.imshow("img", img)
        # cv2.waitKey()

        ret, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        x = int(x)
        y = int(y)
        radius = int(radius * EXTENDED_RADIUS)

        # hcircles = cv2.HoughCircles(img, **HOUGH_CIRCLE)
        # hcircles = np.uint16(np.around(hcircles))[0, :]

        circles.extend([[x, y, radius]])

    return circles
