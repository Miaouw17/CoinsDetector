import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tools import rois_from_circles
import math

DEBUG = True
IMAGE_SCALE = 0.7

def show(i, s, by_pass=False):
    """Affichage d'une image resizé, si DEBUG et a true ou que le by_pass"""
    if by_pass or DEBUG:
        r = cv2.resize(i, None, fx=IMAGE_SCALE, fy=IMAGE_SCALE)
        cv2.imshow(s, r)


def process(original):
    """Traitement pour détecter les pièces"""
    show(original, "original")

    # Change to gray scale
    shifted = cv2.pyrMeanShiftFiltering(original, 23, 60)
    show(shifted, "shifted")

    # Passe en niveau de gris
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    show(gray, "gray")

    # Augemente le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    cl = clahe.apply(gray)
    show(cl, "clahe")

    # Effectue un seuillage adaptatif
    thresh = cv2.adaptiveThreshold(
        cl, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 20)
    show(thresh, "thresh")

    # ret, thresh = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # show(thresh, "thresh")

    # k = np.ones((15, 15), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
    # show(opening, "opening")

    # Cherche les pièces avec hought circles
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=3,
                               minDist=100, param1=10, param2=165, minRadius=0, maxRadius=180)

    # Traitement des cercles retournée par houghcircles pour eviter des erreurs
    if circles is None:
        circles = [[]]

    circles = circles[0, :]
    circles = np.uint16(np.around(circles))

    return circles


# Table de ratio des pièces de franc Suisse
radius_lookup = [
    ("5fr", 1),
    ("2fr", 0.87122),
    ('1fr', 0.73767),
    ('50c', 0.57869),
    ('20c', 0.66931),
    ('10c', 0.60890),
    ('5c', 0.54531),
]


def predict_circle(ratio):
    """Pour le rayon normalizé donnée trouve la pièce la plus probable, distance minimale du ratio en utilisant la table ci-dessus (étbli avec la taille des pièce de wikipédia)
    https://en.wikipedia.org/wiki/Swiss_franc#Coins_of_the_Swiss_Confederation"""
    ratio_dists = list(map(lambda r: (r[0], abs(ratio - r[1])), radius_lookup))
    sorted_ratios = sorted(ratio_dists, key=lambda c: c[1])
    return sorted_ratios[0][0]


def predict_circles(circles, refcircle):
    """Prédiction de la pièce en fonction du rayon"""
    predictions = []

    for circle in circles:
        if np.array_equal(circle, refcircle):
            predictions.append("ref")
        else:
            ratio = circle[2] / refcircle[2] # radius ratio
            prediction = predict_circle(ratio)
            predictions.append(prediction)

    return predictions


def color_dist(c1, c2):
    """https://en.wikipedia.org/wiki/Color_difference"""
    b1, g1, r1 = c1
    b2, g2, r2 = c2

    r_mean = (r1 + r2) / 2
    delta_r = r2 - r1
    delta_g = g2 - g1
    delta_b = b2 - b1

    r_dist = (r_mean / 256.0 + 2) * delta_r**2
    g_dist = 4 * (delta_g**2)
    b_dist = ((255.0 - r_mean) / 256.0 + 2) * delta_b**2

    return math.sqrt(r_dist + g_dist + b_dist)


def find_ref(img, circles):
    """Essaye de trouver la pièce de référence sur l'image donnée avec les cercles données, dans notre cas la pièce la plus rouge"""
    rois, rois_masked = rois_from_circles(img, circles)

    color_goal = (0, 0, 255)
    closest = math.inf
    closest_i = None

    for i in range(len(rois_masked)):
        roi_masked = rois_masked[i]
        # retire les zone masqué
        mask = np.any(roi_masked != [0, 0, 0], axis=-1)
        roi_masked = roi_masked[mask]
        mean_color = np.mean(roi_masked, axis=(0))
        dist = color_dist(color_goal, mean_color)
        if dist < closest:
            closest = dist
            closest_i = i

    return circles[closest_i]

def text_center(img, text, org, fontFace, fontScale, color, thickness):
    retval, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
    org = (org[0] - retval[0] // 2, org[1] - retval[1] // 2 + baseLine // 2)
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness)

def draw_circles_predictions(img, circles, predictions):
    """Retourne une image avec un overlay des pièces détectés"""
    fontFace = cv2.FONT_HERSHEY_DUPLEX

    img_circles = img.copy()
    for circle, prediction, i in zip(circles, predictions, range(len(circles))):
        pos = (circle[0], circle[1])
        r = circle[2]
        cv2.circle(img_circles, pos, r, (0, 255, 255), 4)
        cv2.circle(img_circles, pos, 3, (0, 255, 255), 4)
        text_center(img_circles, str(i) + "#", (pos[0], pos[1] - r), fontFace, 2, (0, 255, 0), 4)
        text_center(img_circles, prediction, (pos[0], pos[1] + r), fontFace, 4, (0, 0, 255), 6)
    return img_circles


def use_case(filepath):
    """Application de la pipe line de détection de pièce"""
    img = cv2.imread(filepath)
    if img is None:
        print(filepath + " is not a valid image")
        exit()
    circles = process(img)

    refcircle = find_ref(img, circles)
    predictions = predict_circles(circles, refcircle)

    img_circles = draw_circles_predictions(img, circles, predictions)
    return img_circles


if __name__ == '__main__':
    cwd = os.getcwd()

    DEBUG = "--debug" in sys.argv
    if DEBUG:
        del sys.argv[sys.argv.index("--debug")]

    for files in sys.argv[1:]:
        fullpath = os.path.join(cwd, files)
        if os.path.exists(fullpath):
            show(use_case(fullpath), fullpath, True)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("this file doesn't exist :", fullpath)
