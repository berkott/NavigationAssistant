import glob
import sys

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
import time

sys.path.append("../..")
from learningModels import svm
sys.path.remove("../..")

START_SAMPLE = 0
TOTAL_SAMPLES = 20


def get_candidates():
    filenames = glob.glob("../data/annotatedData/good/*.png")
    filenames.sort()

    images = []

    for i in range(TOTAL_SAMPLES):
        images.append(cv2.imread(filenames[i], 1))

    return np.asarray(images)


def generate_histogram_features(img):
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    return hog.compute(img)


def get_features(candidates):
    features = []

    for candidate in candidates:
        red_histogram_features = generate_histogram_features(candidate[:][:][2])
        blue_histogram_features = generate_histogram_features(candidate[:][:][0])
        features.append(np.concatenate((red_histogram_features, blue_histogram_features), axis=None))

    return features


def loadAnnotations():
    df = pd.read_csv('../../data/trafficAnnotatedData/labels.csv', usecols=["label"])
    annotations = df.to_numpy()

    return annotations[START_SAMPLE:START_SAMPLE + TOTAL_SAMPLES]


def unison_shuffled_copies(a, b):
    randomize = np.arange(len(a))
    np.random.shuffle(randomize)
    return a[randomize], b[randomize]


def main():
    candidates = get_candidates()
    x = get_features(candidates)
    y = loadAnnotations()
    rand_x, rand_y = unison_shuffled_copies(x, y)

    model = svm.svm(2.5, 5.4)
    model.train(rand_x, rand_y)
    model.save()


if __name__ == "__main__":
    main()
