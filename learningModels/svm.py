import csv
import glob
import os
import time

import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
import pickle
import pandas as pd
import numpy as np


class svm:
    def __init__(self):
        self.model = SVC(decision_function_shape='ovo')

    def train(self, x_train, y_train):
        print("training")
        self.model.fit(x_train, y_train)

    def save(self):
        filename = 'data/' + str(int(time.time())) + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        # self.model.save("../../data/models/svm" + str(int(time.time())) + ".dat")

    def load(self, model_time):
        list_of_files = glob.glob('data/*sav')
        print(list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        # file_name = None
        #
        # if not model_time == 0:
        #     file_name = list_of_files[list_of_files.index("../../data/models/svm" + str(model_time) + ".sav")]
        # else:
        #     list_of_files.sort()
        #     file_name = list_of_files[len(list_of_files) - 1]

        self.model = pickle.load(open(latest_file, 'rb'))
        print("loaded")

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, x, y):
        y_pred = self.model.predict(x)
        print("Accuracy: " + str(accuracy_score(y, y_pred)))
        print('\n')
        print(classification_report(y, y_pred))
