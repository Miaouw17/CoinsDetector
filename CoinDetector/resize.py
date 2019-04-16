import os
import cv2

from tools import clear_folder, file_extension

def resize(input_folder, output_folder, size):
    clear_folder(output_folder)

    input_files = os.listdir(input_folder)
    for input_file in input_files:
        path_input_file = input_folder + input_file
        filename, extension = file_extension(input_file)
        img = cv2.imread(path_input_file)

        resized = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

        path = output_folder + filename + "-" + extension
        cv2.imwrite(path, resized)
