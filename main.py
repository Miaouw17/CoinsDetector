import os
import cv2
import numpy as np

from preprocess import preprocess1
from process import process1

from tools import create_circular_mask, roi_circle

def main():
    # imgPath = "img/IMG_1180.JPG"
    # imgPath = "img/IMG_1109.JPG"
    imgPath = "img/IMG_1111.JPG"
    img = cv2.imread(imgPath)
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

    imgPreprocessed = preprocess1(img)
    circles = process1(imgPreprocessed)

    output = img.copy()

    coins_img = []

    for x, y, r in circles:
        r += 10

        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

        roi, maskedRoi = roi_circle(img, (x, y, r))

        coins_img.append(roi)
        coins_img.append(maskedRoi)

    cv2.imshow("output", output)

    for coin in coins_img:
        cv2.imshow("", coin)

        cv2.waitKey(0)



if __name__ == '__main__':
    main()
