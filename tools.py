import os
import cv2
import numpy as np

def roi_circle(img, circle):
    output = img.copy()
    x = circle[0]
    y = circle[1]
    r = circle[2]
    left = x - r
    top = y - r
    right = x + r
    bottom = y + r

    hImg, wImg = img.shape[0], img.shape[1]

    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right >= wImg:
        right = wImg
    if bottom >= hImg:
        bottom = hImg

    roi = output[top:bottom, left:right]

    hRoi, wRoi = roi.shape[0], roi.shape[1]

    mask = create_circular_mask(hRoi, wRoi)

    maskedRoi = roi.copy()
    maskedRoi[mask] = 0

    return roi, maskedRoi

def create_circular_mask(h, w, center=None, radius=None):
    """https://stackoverflow.com/a/44874588"""
    if center is None: # use the middle of the image
        center = [w/2, h/2]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center > radius
    return mask