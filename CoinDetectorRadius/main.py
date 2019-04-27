import numpy as np
import cv2
import matplotlib.pyplot as plt
from tools import rois_from_circles

radius_lookup = [
    ("5fr", 1),
    ("2fr", 0.87122),
    ('1fr', 0.73767),
    ('20c', 0.66931),
    ('10c', 0.60890),
    ('50c', 0.57869),
    ('5c', 0.54531),
]


def show(i, s, bypass=False):
    if not bypass:
        return
    r = cv2.resize(i, None, fx=0.5, fy=0.5)
    cv2.imshow(s, r)
    # cv2.waitKey(0)


def predict_circle(radius):
    keys = []
    values = []
    for i in radius_lookup:
        keys.append(i[0])
        values.append(i[1])

    if radius >= values[0]:
        return keys[0]

    for index in range(len(keys) - 1):
        top_slice_key = keys[index]
        top_slice_value = values[index]

        bot_slice_key = keys[index + 1]
        bot_slice_value = values[index + 1]

        diff_top = abs(top_slice_value - radius)
        diff_bot = abs(radius - bot_slice_value)

        if radius <= top_slice_value and radius >= bot_slice_value:
            if diff_top < diff_bot:
                return top_slice_key
            else:
                return bot_slice_key

    return keys[-1]


def process(original):
    show(original, "original")

    blured = cv2.medianBlur(original, 19)
    show(blured, "blured")

    shifted = cv2.pyrMeanShiftFiltering(blured, 11, 60)
    show(shifted, "shifted")

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    show(gray, "gray")

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 20)
    show(thresh, "thresh")

    # ret, thresh = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # show(thresh, "thresh")

    k = np.ones((15, 15), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
    show(opening, "opening")

    circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, dp=3,
                               minDist=100, param1=60, param2=170, minRadius=0, maxRadius=0)
    if circles is None:
        circles = [[]]

    circles = circles[0, :]
    circles = np.uint16(np.around(circles))

    return circles, shifted


def draw_circles_predictions(img, circles, predictions):
    img_circles = img.copy()
    for circle, prediction in zip(circles, predictions):
        pos = (circle[0], circle[1])
        r = circle[2]
        cv2.circle(img_circles, pos, r, (0, 255, 0), 2)
        cv2.circle(img_circles, pos, 2, (0, 0, 255), 3)
        cv2.putText(img_circles, prediction, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
    return img_circles


def predict_circles(circles, refcircle):
    predictions = []

    refx, refy, refr = refcircle

    for circle in circles:
        x, y, r = circle
        if x == refx and y == refy and r == refr:
            predictions.append("ref")
        else:
            ratio = r / refr
            prediction = predict_circle(ratio)
            predictions.append(prediction)

    return predictions


def find_ref(img, circles):
    rois, rois_masked = rois_from_circles(img, circles)

    color_goal = (63, 63, 168)
    closest = 255 * 3
    closest_i = 0
    for i in range(len(rois_masked)):
        roi_masked = rois_masked[i]
        mask = np.any(roi_masked != [0, 0, 0], axis=-1)
        roi_masked = roi_masked[mask]
        # show(roi_masked, "m" + str(i), True)
        # roi_masked = roi_masked[roi_masked != [0,0,0]]
        # print(roi_masked)
        mean_color = np.mean(roi_masked, axis=(0))
        dist = color_dist(color_goal, mean_color)
        if dist < closest:
            closest = dist
            closest_i = i

    return circles[closest_i]


def color_dist(c1, c2):
    (b1, g1, r1) = c1
    (b2, g2, r2) = c2
    return (r1 - r2)**2 + (g1 - g2) ** 2 + (b1 - b2) ** 2


if __name__ == '__main__':
    img = cv2.imread("img/red/10.jpeg")
    circles, shifted = process(img)

    refcircle = find_ref(shifted, circles)
    predictions = predict_circles(circles, refcircle)

    img_circles = draw_circles_predictions(img, circles, predictions)
    show(img_circles, "circles", True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
