# Alright so detect traffic light first using color filters for the red and the white
# To find accurate traffic lights check Aspect ratio, filling ratio
# 0.25 width and height is occupied by black pixels: get percentage as a number

import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960

WHITE_RATIO = 1.5
WHITE_FILL = 0.6
RED_RATIO = 1.363636364
RED_FILL = 0.303030303


def basic_filtering(img):
    image1 = cv2.GaussianBlur(img, (11, 11), 0)
    image2 = cv2.bilateralFilter(image1, 9, 75, 75)
    image3 = cv2.bilateralFilter(image2, 9, 75, 75)

    return image2


def candidate_extraction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_lower_color = np.array([0, 0, 23])
    white_upper_color = np.array([175, 50, 91])

    white_mask = cv2.inRange(hsv, white_lower_color, white_upper_color)

    white_img = cv2.bitwise_and(img, img, mask=white_mask)

    red_lower_color = np.array([0, 203, 32])
    red_upper_color = np.array([196, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower_color, red_upper_color)

    red_img = cv2.bitwise_and(img, img, mask=red_mask)

    return white_img, red_img, white_mask, red_mask


def candidate_selection(filtered_img, img, mask, color):
    # kernel = np.ones((10, 10), np.uint8)
    #
    # filtered = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. find contours and  draw the green lines on the white strips
    cv2.imshow("img_" + color, img)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)
    cv2.imshow("contours_" + color, mask)

    candidates = []

    for i in contours:
        bx, by, bw, bh = cv2.boundingRect(i)

        if 200 < bw * bh < 1000:
            candidates.append([bx, by, bw, bh])

        cv2.imshow("rect_" + color, cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2))

    scores = []
    for candidate in candidates:
        fill = get_fill(mask, candidate) / (candidate[3] * candidate[2])
        percent_black = get_black(filtered_img, candidate)
        if color == 'w':
            score = gauss(candidate[3] / candidate[2], WHITE_RATIO) + gauss(fill, WHITE_FILL) + percent_black
        else:
            score = gauss(candidate[3] / candidate[2], RED_RATIO) + gauss(fill, RED_FILL) + percent_black
        scores.append(score)
    print(len(scores))

    try:
        selected = scores.index(max(scores))

        cv2.imshow("selected_" + color, cv2.rectangle(img, (candidates[selected][0], candidates[selected][1]),
                                                      (candidates[selected][0] + candidates[selected][2],
                                                       candidates[selected][1] + candidates[selected][3]),
                                                      (0, 0, 255), 2))
    except Exception:
        print(Exception)

    return img


def get_fill(img, coords):
    count = 0
    for i in range(coords[3]):
        for j in range(coords[2]):
            if img[i + coords[1]][j + coords[0]] == 255:
                count += 1

    return count


def gauss(x, shift):
    return np.e ** ((-2) * ((x - shift) ** 2))


def get_black(img, coords):
    total_area = coords[2] * coords[3] * 0.5625

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    black_lower_color = np.array([75, 10, 0])
    black_upper_color = np.array([187, 163, 30])

    black_mask = cv2.inRange(hsv, black_lower_color, black_upper_color)

    black_area = 0

    for i in range(coords[3]):
        for j in range(coords[2]):
            if black_mask[i + coords[1]][j + coords[0]] == 255:
                black_area += 1

    return black_area / total_area

# def candidate_selection(white_img, red_img, white_bw, red_bw):
#     # kernel = np.ones((10, 10), np.uint8)
#     #
#     # filtered = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # 3. find contours and  draw the green lines on the white strips
#     white_contours, white_hierarchy = cv2.findContours(white_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#     cv2.drawContours(white_img, white_contours, -1, (0, 255, 0), 3)
#     cv2.imshow("contoursW", white_img[2])
#
#     for i in white_contours:
#         bx, by, bw, bh = cv2.boundingRect(i)
#
#         rect = cv2.minAreaRect(i)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         im = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)
#
#         cv2.imshow("rectW", cv2.rectangle(white_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3))
#
#     red_contours, red_hierarchy = cv2.findContours(red_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#     cv2.drawContours(red_bw, red_contours, -1, (0, 255, 0), 3)
#     cv2.imshow("contoursR", red_bw)
#
#     red_candidates =
#     for i in red_contours:
#         bx, by, bw, bh = cv2.boundingRect(i)
#         # rect = cv2.minAreaRect(i)
#         # box = cv2.boxPoints(rect)
#         # box = np.int0(box)
#         # im = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)
#         cv2.imshow("rectR", cv2.rectangle(red_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3))
#
#
#     return white_img, red_img


videoName = 'trafficLight'
cap = cv2.VideoCapture('../../data/trafficLight/' + videoName + '.mkv')
skip = False

if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Frame', frame)

        filtered = basic_filtering(frame)
        white_extracted, red_extracted, white_mask, red_mask = candidate_extraction(filtered)
        white_selected = candidate_selection(filtered, white_extracted, white_mask, "w")
        red_selected = candidate_selection(filtered, red_extracted, red_mask, "r")
        # white_selected, red_selected = candidate_selection(white_extracted, red_extracted, white_mask, red_mask)

        cv2.imshow("Final White", white_selected)
        cv2.imshow("Final Red", red_selected)

        # Maybe also candidate identification (white or red)

    else:
        break

    k = cv2.waitKey()

    if k == ord('q'):
        break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
