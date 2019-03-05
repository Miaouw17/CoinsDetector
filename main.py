import os
import cv2
import numpy as np


def main():
    # imgPath = "img/IMG_1180.JPG"
    # imgPath = "img/IMG_1109.JPG"
    imgPath = "img/IMG_1111.JPG"
    imgOriginal = cv2.imread(imgPath)
    # cv2.imshow("Original", imgOriginal)
    img = cv2.resize(imgOriginal, None, fx=0.2, fy=0.2,
                     interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray", gray)
    mask = (gray < 80) | (gray > 130)
    gray[mask] = 0

    cv2.imshow("Masquage", gray)

    r, threshold = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold", threshold)

    kernel = np.ones((11, 11), np.uint8)




    morphclose = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("morphopen", morphclose)


    kernelErode = np.ones((5,5), np.uint8)
    erode = cv2.morphologyEx(morphclose, cv2.MORPH_ERODE, kernelErode)
    cv2.imshow("erode", erode)

    circles = cv2.HoughCircles(erode, cv2.HOUGH_GRADIENT, 4, minDist=50, minRadius=1, maxRadius=120)
    print(circles)
    circles = np.uint16(np.around(circles))

    output = img.copy()

    print(circles)
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("output", output)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
