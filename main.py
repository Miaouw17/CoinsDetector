import os
import cv2
import numpy as np

from preprocess import *
from process import *
from tools import *
from config import *

def main():
    # imgPath = "img/IMG_1180.JPG"
    # imgPath = "img/IMG_1109.JPG"
    imgPath = "img/IMG_1111.JPG"
    img = cv2.imread(imgPath)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

    imgPreprocessed = preprocess1(img)
    cv2.imshow("imgPreprocessed", imgPreprocessed)
    circles = process1(imgPreprocessed)
    output = draw_circles_on_image(img, circles)
    cv2.imshow("output", output)

    rois, rois_masked = get_rois_from_image_and_circles(img, circles)
    cv2.waitKey(0)

    for roi in rois_masked:
        cv2.imshow("roi", roi)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
