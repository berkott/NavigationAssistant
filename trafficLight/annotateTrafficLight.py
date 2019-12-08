import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time

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

    red_lower_color = np.array([0, 161, 32])
    red_upper_color = np.array([196, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower_color, red_upper_color)

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
    candidate_imgs = []
    for i in contours:
        bx, by, bw, bh = cv2.boundingRect(i)

        if check_size(bw, bh) and check_fill(combined_mask, bx, by, bw, bh) and check_aspect(bw, bh):
            candidates.append([bx, by, bw, bh])
            candidate_imgs.append(cv2.resize(img[by:by + bh, bx:bx + bw], (20, 20), interpolation=cv2.INTER_AREA))

    for bx, by, bw, bh in candidates:
        cv2.imshow("rect", cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2))

    return candidates, candidate_imgs


videoName = 'my_video-2'
cap = cv2.VideoCapture('../../data/trafficLight/rawVideos' + videoName + '.mkv')
skip = False

if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    k = ''
    if ret:
        cv2.imshow('Frame', frame)

        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        potential_candidates, potential_candidate_imgs = candidate_extraction(filtered)

        for cand in potential_candidate_imgs:
            classification = 'n'
            while True:
                cv2.imshow("Candidate", cand)

                k = cv2.waitKey()
                current_time = time.time()

                if k == ord('r'):
                    classification = 'r'
                if k == ord('w'):
                    classification = 'w'
                if k == ord('e'):
                    classification = 'n'
                if k == ord('s'):
                    cv2.imwrite('../../data/trafficLight/annotatedData/goodTraffic/' + videoName + '_' + str(current_time) + '.png',
                                cand)
                    with open('../../data/trafficLight/annotatedData/trafficClassifications.csv', mode='a') as lines_file:
                        lines_writer = csv.writer(lines_file, delimiter=',')
                        lines_writer.writerow([str(current_time), classification])
                    break
                if k == ord('d') or k == ord('q'):
                    break
            if k == ord('q'):
                break
    else:
        break

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
