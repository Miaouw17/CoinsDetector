import os
import cv2
import numpy as np

from preprocess import *
from process import *
from tools import *
from config import *

from crop_circles import crop_circles
from data_augment import data_augment
from resize import resize

def prepare_images():
    print("Crop circles 2f")
    crop_circles("img/original/2f/", "img/cropped/2f/")
    print("Crop circles 5c")
    crop_circles("img/original/5c/", "img/cropped/5c/")

    print("Augment 2f")
    data_augment("img/cropped/2f/", "img/augmented/2f/")
    print("Augment 5c")
    data_augment("img/cropped/5c/", "img/augmented/5c/")

    size = (28, 28)
    print("Resize 2f")
    resize("img/augmented/2f/", "img/resized/2f/", size)
    print("Resize 5c")
    resize("img/augmented/5c/", "img/resized/5c/", size)

if __name__ == '__main__':
    prepare_images()
