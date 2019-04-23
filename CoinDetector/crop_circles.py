import os

import cv2

from preprocess import preprocess1
from process import process1

from tools import create_circular_mask, roi_circle, draw_circles_on_image, get_rois_from_image_and_circles, clear_folder

def crop_circles(input_folder, ouput_folder):
    clear_folder(ouput_folder)
    input_files = os.listdir(input_folder)
    i = 0
    for input_file in input_files:
        path_input_file = input_folder + input_file
        img = cv2.imread(path_input_file)
        # print(path_input_file)
        imgResize = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        imgPreprocessed = preprocess1(imgResize)
        circles = process1(imgPreprocessed)
        # only one coin per image, filter bad recognition
        if len(circles) == 1:
            path = ouput_folder + str(i) + ".jpeg"
            i += 1
            rois, rois_masked = get_rois_from_image_and_circles(
                imgResize, circles)
            # print("write " + path)
            cv2.imwrite(path, rois_masked[0])
