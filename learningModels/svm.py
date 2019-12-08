import csv
import glob
import time

import cv2
import pandas as pd
import numpy as np


class svm:
    def __init__(self, c, gamma):
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_RBF)
        self.svm.setC(c)
        self.svm.setGamma(gamma)

    def train(self, x_train, y_train):
        self.svm.train  (x_train, cv2.ml.ROW_SAMPLE, y_train)
    #
    # def evaluate(self, x_test, y_test):
    #     loss_and_metrics = self.model.evaluate(x_test, y_test, batch_size=self.BATCH_SIZE)
    #
    #     print("mse, accuracy:", loss_and_metrics)
    #
    #     with open('performance.csv', mode='a') as per_file:
    #         per_writer = csv.writer(per_file, delimiter=',')
    #
    #         per_writer.writerow(['cnn',
    #                              self.EPOCHS,
    #                              self.BATCH_SIZE,
    #                              int(time.time()),
    #                              loss_and_metrics[0],
    #                              loss_and_metrics[1]])
    #
    #     y_pred = self.model.predict(x_test)
    #
    #     print("Y Test:", y_test)
    #     print("Y Pred:", y_pred)
    #     # print(confusion_matrix(y_test, y_pred.argmax(axis=1)))
    #     # print(classification_report(y_test, y_pred.argmax(axis=1)))

    def save(self):
        self.svm.save("../data/models/svm" + str(int(time.time())) + ".dat")

    def load(self, model_time):
        list_of_files = glob.glob('../data/models/svm*')
        print(list_of_files)

        try:
            file_name = list_of_files[list_of_files.index("../data/models/svm" + model_time + ".dat")]
        except ValueError:
            print(ValueError)
            list_of_files.sort()
            file_name = list_of_files[len(list_of_files) - 1]

        self.svm = cv2.ml.SVM_load(file_name)

    def predict(self, data):
        return self.svm.predict(data)
