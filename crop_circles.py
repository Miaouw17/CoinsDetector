import os

import cv2

from preprocess import preprocess1
from process import process1

from tools import create_circular_mask, roi_circle

if __name__ == '__main__':
    input_folder = "img"
    ouput_folder = "output"

    input_files = os.listdir(input_folder)

    for input_file in input_files:
        path_input_file = input_folder + "/" + input_file
        img = cv2.imread(path_input_file)
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        
        cv2.imshow("", img)
        cv2.waitKey(0)
        print(input_file)
