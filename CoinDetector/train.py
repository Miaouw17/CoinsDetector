import os
import cv2
import numpy as np
import datetime
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def train(folders):
    x_train = []
    y_train = []
    for i in range(len(folders)):
        folder = folders[i]
        files = os.listdir(folder)
        for file in files:
            path = folder + file
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # cv2.imshow("i", img)
            # cv2.imshow("c", cl1)
            # cv2.waitKey(0)
            x_train.append(img)
            y_train.append(i)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = x_train.astype('float32')
    x_train /= 255

    shape = x_train[0].shape
    size = shape[0]

    model = Sequential()
    model.add(Conv2D(size, kernel_size=(3, 3), input_shape=shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(40, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, shuffle=True, epochs=2000)

    print("eval")
    print(model.evaluate(x=x_train, y=y_train))
    model.save("models/" + str(datetime.date.today()) + "_model.h5")
    # image_index = 4444
    # pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
    # print(pred.argmax())
