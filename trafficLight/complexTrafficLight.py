import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time

sys.path.append("../..")
from learningModels import svm
sys.path.remove("../..")

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960

MAX_ASPECT = 2
MIN_ASPECT = 0.8
MAX_FILL = 0.99
MIN_FILL = 0.3
MAX_AREA = 1000
MIN_AREA = 100


def check_size(w, h):
    return MIN_AREA < w * h < MAX_AREA


def check_fill(img, x, y, w, h):
    count = 0
    for i in range(h):
        for j in range(w):
            if img[i + y][j + x] == 255:
                count += 1

    return MIN_FILL < count / (w * h) < MAX_FILL


def check_aspect(w, h):
    return MIN_ASPECT < h / w < MAX_ASPECT


def color_extraction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_lower_color = np.array([0, 0, 23])
    white_upper_color = np.array([175, 50, 91])
    white_mask = cv2.inRange(hsv, white_lower_color, white_upper_color)
    # cv2.imshow("white_mask", white_mask)

    red_lower_color = np.array([0, 161, 32])
    red_upper_color = np.array([196, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower_color, red_upper_color)
    # cv2.imshow("red_mask", red_mask)

    combined_mask = white_mask + red_mask
    combined_img = cv2.bitwise_and(img, img, mask=combined_mask)

    return combined_img, combined_mask


def candidate_extraction(img):
    combined_img, combined_mask = color_extraction(img)

    contours_img = combined_mask.copy()
    # contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 3)
    cv2.imshow("contours", contours_img)
    cv2.imshow("combined", combined_mask)

    candidates = []
    for i in contours:
        bx, by, bw, bh = cv2.boundingRect(i)
        # if check_size(bw, bh) or check_fill(combined_mask, bx, by, bw, bh) or check_aspect(bw, bh):
        #     print(check_size(bw, bh), " ", check_fill(combined_mask, bx, by, bw, bh), " ", check_aspect(bw, bh))

        if check_size(bw, bh) and check_fill(combined_mask, bx, by, bw, bh) and check_aspect(bw, bh):
            candidates.append([bx, by, bw, bh])
    # print(candidates)

    for bx, by, bw, bh in candidates:
        cv2.imshow("rect", cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2))

    return candidates


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


def candidate_selection(img, candidates, svm_model):
    features = []

    for i in range(len(candidates)):
        x = candidates[i][0]
        y = candidates[i][1]
        w = candidates[i][2]
        h = candidates[i][3]
        red_histogram_features = generate_histogram_features(cv2.resize(img[y:y + h][x:x + w][2], (20, 20),
                                                                        interpolation=cv2.INTER_AREA))
        blue_histogram_features = generate_histogram_features(cv2.resize(img[y:y + h][x:x + w][0], (20, 20),
                                                                         interpolation=cv2.INTER_AREA))

        features.append(np.concatenate((red_histogram_features, blue_histogram_features), axis=None))

    predictions = []
    for feature in features:
        predictions.append(svm_model.predict(feature))

    return predictions


videoName = 'trafficLight'
cap = cv2.VideoCapture('../../data/trafficLight/' + videoName + '.mkv')
skip = False

if not cap.isOpened():
    print("Error opening video stream or file")

model = svm.svm(2.5, 5.4)
model.load(0)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Frame', frame)

        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        potential_candidates = candidate_extraction(filtered)
        final_selection = candidate_selection(filtered, potential_candidates)
        print(final_selection)

    else:
        break

    k = cv2.waitKey()

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
