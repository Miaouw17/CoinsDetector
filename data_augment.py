import os
import cv2

from tools import clear_folder, file_extension

if __name__ == '__main__':
    subfolder = "5c"

    input_folder = "img/cropped/" + subfolder + "/"
    output_folder = "img/rotated/" + subfolder + "/"
    clear_folder(output_folder)

    nRotations = 10
    stepRotation = 360.0 / nRotations

    input_files = os.listdir(input_folder)
    for input_file in input_files:
        path_input_file = input_folder + input_file
        filename, extension = file_extension(input_file)
        print(path_input_file)
        img = cv2.imread(path_input_file)
        if img is not None:
            cols, rows, depth = img.shape

            for i in range(nRotations):
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), stepRotation * i, 1)
                imgRotated = cv2.warpAffine(img, M, (cols, rows))
                path = output_folder + filename + "-" + str(i) + extension
                print(path)
                cv2.imwrite(path, imgRotated)
