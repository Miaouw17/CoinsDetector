import cv2

from preprocess import preprocess1
from process import process2

def predict(filepath):
    img = cv2.imread(filepath)
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
