import os
import cv2
import numpy as np

import shutil

def clear_folder(f):
    try:
        shutil.rmtree(f)
    except:
        pass
    os.mkdir(f)

def file_extension(f):
    splited = f.split(".")
    extension = splited[-1]
    del splited[-1]
    return ".".join(splited), "." + extension

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

    masked_roi = roi.copy()
    masked_roi[mask] = 0

    return roi, masked_roi

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

def draw_circles_on_image(img, circles):
    output = img.copy()
    for x, y, r in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
    return output

def get_rois_from_image_and_circles(img, circles):
    rois = []
    rois_masked = []
    for x, y, r in circles:
        roi, roi_masked = roi_circle(img, (x, y, r))
        rois.append(roi)
        rois_masked.append(roi_masked)
    return rois, rois_masked