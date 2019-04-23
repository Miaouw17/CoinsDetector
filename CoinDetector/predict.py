import cv2
import numpy as np

from preprocess import preprocess1
from process import process2
from tools import draw_circles_on_image, get_rois_from_image_and_circles


def predict_redest(rois):
    rois_hsv = list(map(lambda roi: cv2.cvtColor(roi, cv2.COLOR_BGR2HSV),rois))
    rois_hsv_hue = list(map(lambda roi_hsv: cv2.split(roi_hsv)[0],rois_hsv))
    averages = list(map(lambda h: h.mean(), rois_hsv_hue))
    return np.argmin(averages)


def predict(filepath):
    img = cv2.imread(filepath)
    imgPreprocessed = preprocess1(img)
    # cv2.imshow("imgPreprocessed", imgPreprocessed)
    # cv2.waitKey(0)
    circles = process2(imgPreprocessed)
    # print("circles", circles)

    rois, rois_masked = get_rois_from_image_and_circles(img, circles)
    redest_index = predict_redest(rois_masked)

    output = img.copy()
    # output = draw_circles_on_image(img, circles)

    redest_circles = circles[redest_index]
    x, y, r = redest_circles
    cv2.circle(output, (x, y), 30, (0, 0, 255, 0.5), 15)

    for x, y, r in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)

    cv2.imshow("output", cv2.resize(output,None,fx=0.5,fy=0.5))
    cv2.waitKey(0)
    #
    # for roi in rois_masked:
    #     cv2.imshow("roi", roi)
    #     cv2.waitKey(0)


if __name__ == '__main__':
    predict("img/redcoin/7.jpeg")
