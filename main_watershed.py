import numpy as np
import cv2
from matplotlib import pyplot as plt
from preprocess import *

img = cv2.imread('img/IMG_1109.JPG')
img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imshow("threshold", thresh)

imgPreprocessed = preprocess1(img)
cv2.imshow("imgPreprocessed", imgPreprocessed)

# # noise removal
# kernel = np.ones((3,3),np.uint8)
# # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# # cv2.imshow("opening", opening)
#
# # sure background area
# sure_bg = cv2.dilate(imgPreprocessed,kernel,iterations=20)
#
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(imgPreprocessed,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# cv2.imshow("sure_fg", sure_fg)
#
# unknown = cv2.subtract(sure_bg,sure_fg)
# cv2.imshow("unknown", unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(imgPreprocessed)
print(ret)
print(markers)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
# markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
cv2.imshow("markers", markers)

img[markers == -1] = [0,0,255]

cv2.imshow("output", img)

cv2.waitKey(0)
