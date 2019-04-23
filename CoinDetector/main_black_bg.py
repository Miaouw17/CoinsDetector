# SOURCE ; https://www.pyimagesearch.com/2015/11/02/watershed-opencv/

# import the necessary packages
from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import imutils
import numpy as np
import cv2


def process_s(image):
    # cv2.imshow("shifted", shifted)
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for (i, c) in enumerate(cnts):
        ((x, y), _) = cv2.minEnclosingCircle(c)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for (i, c) in enumerate(cnts):
        ((x, y), _) = cv2.minEnclosingCircle(c)

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    circles = []
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        ((x, y), r) = cv2.minEnclosingCircle(c)

        circles.append((int(x), int(y), int(r)))

    return circles
