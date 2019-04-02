import os
import cv2
import numpy as np

from config import *

def preprocess1(img):
    blured = cv2.blur(img, **BLUR)
    # cv2.imshow("blured", blured)

    gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    threshold = gray.copy()
    mask = (threshold < MIN_THRESHOLD) | (threshold > MAX_THRESHOLD)
    notMask = np.bitwise_not(mask)
    threshold[mask] = 0
    threshold[notMask] = 255
    # cv2.imshow("threshold", threshold)

    kernel = np.ones((KERNEL_SIZE_CLOSE, KERNEL_SIZE_CLOSE), np.uint8)

    morphclose = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("morphopen", morphclose)

    kernelErode = np.ones((KERNEL_SIZE_ERODE, KERNEL_SIZE_ERODE), np.uint8)
    erode = cv2.morphologyEx(morphclose, cv2.MORPH_ERODE, kernelErode)
    # cv2.imshow("erode", erode)

    return erode