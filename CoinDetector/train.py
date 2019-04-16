import os
import cv2
import numpy as np
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
            x_train.append(img)
            y_train.append(i)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = x_train.astype('float32')
    x_train /= 255

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(20, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, shuffle=True, epochs=10)

    # model.evaluate(x=x_test, y=y_test)
    # image_index = 4444
    # pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
    # print(pred.argmax())
