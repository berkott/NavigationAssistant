import csv
import glob
import time

import cv2
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import load_model

from functools import partial

conv3 = partial(layers.Conv2D,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu')


class vgg:
    def __init__(self, epochs, batch_size):
        self.model = Sequential()
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

    def create_model(self, in_shape=(285, 380, 3), opt='SGD'):
        self.model = Sequential([
            Conv2D(32, (3, 3), input_shape=in_shape, padding='same', activation='relu'),
            # Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            # Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            # Dense(128, activation='relu'),
            Dense(4, activation='linear')
        ])

        self.model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mae"])

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)

    def evaluate(self, x_test, y_test):
        loss_and_metrics = self.model.evaluate(x_test, y_test, batch_size=self.BATCH_SIZE)

        print("mse, accuracy:", loss_and_metrics)

        with open('performance.csv', mode='a') as per_file:
            per_writer = csv.writer(per_file, delimiter=',')

            per_writer.writerow(['cnn',
                                 self.EPOCHS,
                                 self.BATCH_SIZE,
                                 int(time.time()),
                                 loss_and_metrics[0],
                                 loss_and_metrics[1]])

        y_pred = self.model.predict(x_test)

        print("Y Test:", y_test)
        print("Y Pred:", y_pred)
        # print(confusion_matrix(y_test, y_pred.argmax(axis=1)))
        # print(classification_report(y_test, y_pred.argmax(axis=1)))

    def save(self):
        self.model.save("../data/models/models" + str(int(time.time())) + ".h5")

    def load(self, model_time):
        list_of_files = glob.glob('../data/models/model*')
        print(list_of_files)

        try:
            file_name = list_of_files[list_of_files.index("../data/models/models" + model_time + ".h5")]
        except ValueError:
            print(ValueError)
            list_of_files.sort()
            file_name = list_of_files[len(list_of_files) - 1]

        self.model = load_model(file_name)
        print(self.model.summary())

    def predict(self, data):
        return self.model.predict(data, batch_size=None)
