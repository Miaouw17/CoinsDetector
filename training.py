from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import os
import cv2
import numpy as np

from preprocess import *
from process import *
from tools import *
from config import *

class ML:
    def __init__(self):
        self.clf = MLPClassifier(solver="lbfgs")

    def predict(self, img):
        data = ML.img_to_histo(img)
        return self.clf.predict([data])

    def train(self, paths):
        print("start training")
        X = []
        Y = []

        for path in paths:
            input_files = os.listdir(path)
            for input_file in input_files:
                path_input_file = path + input_file
                img = cv2.imread(path_input_file)

                data = ML.img_to_histo(img)

                X.append(data)
                Y.append(path)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, shuffle=True, random_state=42)

        self.clf.fit(X_train, Y_train)
        score = self.clf.score(X_test, Y_test) * 100
        print(score)

    @staticmethod
    def img_to_histo(img):
        hsv
        histB,bins = np.histogram(img[0].ravel(), 256, [0,256], normed=True)
        histG,bins = np.histogram(img[1].ravel(), 256, [0,256], normed=True)
        histR,bins = np.histogram(img[2].ravel(), 256, [0,256], normed=True)

        data = []
        data.extend(histB[::5])
        data.extend(histG[::5])
        data.extend(histR[::5])
        return data

if __name__ == '__main__':
    ml = ML()
    paths = ["img/rotated/5c/", "img/rotated/2f/"]
    ml.train(paths)

    imgPath = "img/base/IMG_1109.JPG"

    img = cv2.imread(imgPath)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE,interpolation=cv2.INTER_CUBIC)

    imgPreprocessed = preprocess1(img)
    cv2.imshow("imgPreprocessed", imgPreprocessed)
    circles = process2(imgPreprocessed)
    output = draw_circles_on_image(img, circles)
    cv2.imshow("output", output)

    rois, rois_masked = get_rois_from_image_and_circles(img, circles)
    cv2.waitKey(0)

    for roi in rois_masked:
        cv2.imshow("roi", roi)
        print(ml.predict(roi))
        cv2.waitKey(0)