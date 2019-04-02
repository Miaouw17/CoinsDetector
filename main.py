import os
import cv2
import numpy as np

from preprocess import *
from process import *
from tools import *
from config import *


def main(imgPath):

    img = cv2.imread(imgPath)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_CUBIC)

    imgPreprocessed = preprocess1(img)
    cv2.imshow("imgPreprocessed", imgPreprocessed)
    circles = process2(imgPreprocessed)
    output = draw_circles_on_image(img, circles)
    cv2.imshow("output", output)

    rois, rois_masked = get_rois_from_image_and_circles(img, circles)
    cv2.waitKey(0)

    for roi in rois_masked:
        cv2.imshow("roi", roi)
        cv2.waitKey(0)


if __name__ == '__main__':
    img = [
        "img/base/IMG_1180.JPG",
        "img/base/IMG_1109.JPG",
        # "img/base/IMG_1111.JPG",
        # "/home/raphael/Desktop/WhatsApp Image 2019-04-02 at 13.41.44.jpeg",
        # "/home/raphael/Desktop/WhatsApp Image 2019-04-02 at 13.53.20.jpeg"
    ]
    for i in img:
        main(i)
