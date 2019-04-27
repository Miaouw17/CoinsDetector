import cv2

# Preprocess

BLUR = {
    "ksize": (15, 15),
    "anchor": (7, 7)
}

SCALE = 0.2
KERNEL_SIZE_CLOSE = 11
KERNEL_SIZE_ERODE = 5
MIN_THRESHOLD = 80
MAX_THRESHOLD = 130

EXTENDED_RADIUS = 1.0

# Process
HOUGH_CIRCLE = {
    "method": cv2.HOUGH_GRADIENT,
    "dp": 4.2,
    "minDist": 80,
    "minRadius": 10,
    "maxRadius": 120
}
