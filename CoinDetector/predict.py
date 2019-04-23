import cv2
import numpy as np

from preprocess import preprocess1
from process import process2
from main_black_bg import process_s
from tools import draw_circles_on_image, get_rois_from_image_and_circles
from radius_recognition import predict_piece_by_radius_pourcentage


def predict_redest(rois):
    rois_hsv = list(map(lambda roi: cv2.cvtColor(roi, cv2.COLOR_BGR2HSV),rois))
    rois_hsv_hue = list(map(lambda roi_hsv: cv2.split(roi_hsv)[0],rois_hsv))
    averages = list(map(lambda h: h.mean(), rois_hsv_hue))
    print(averages)
    return np.argmin(averages)


def predict(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, None, fx=0.4, fy=0.4,
                           interpolation=cv2.INTER_CUBIC)

    # imgPreprocessed = preprocess1(img)
    # cv2.imshow("imgPreprocessed", imgPreprocessed)
    # cv2.waitKey(0)
    # circles = process2(imgPreprocessed)

    circles = process_s(img)

    # print("circles", circles)

    rois, rois_masked = get_rois_from_image_and_circles(img, circles)
    redest_index = predict_redest(rois_masked)

    output = img.copy()
    # output = draw_circles_on_image(img, circles)

    redest_circles = circles[redest_index]
    red_x, red_y, red_r = redest_circles
    cv2.circle(output, (red_x, red_y), 1, (0, 0, 255, 0.5), 55)

    for i in range(len(circles)):
        x, y, r = circles[i]
        ratio_r = r / red_r
        predicted = predict_piece_by_radius_pourcentage(ratio_r)
        cv2.putText(output, str(i), (x, y),
    		cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
        cv2.putText(output, predicted, (x + 30, y),
    		cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)

    cv2.imshow("output", cv2.resize(output,None,fx=0.5,fy=0.5))
    cv2.waitKey(0)


if __name__ == '__main__':
    predict("img/redcoin/13.jpeg")
