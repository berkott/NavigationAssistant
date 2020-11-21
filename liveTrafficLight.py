import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time
from skimage.feature import hog
from skimage import exposure

sys.path.append("..")
from learningModels import svm
sys.path.remove("..")

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960

MAX_ASPECT = 2
MIN_ASPECT = 0.8
MAX_FILL = 0.99
MIN_FILL = 0.4
MAX_AREA = 1000
MIN_AREA = 50


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


def min_zero(num):
    if num < 0:
        return 0
    return num


def max_num(num, max_num):
    if num > max_num:
        return max_num
    return num


def color_extraction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # white_lower_color = np.array([0, 0, 23])
    # white_upper_color = np.array([175, 50, 91])
    white_lower_color = np.array([70, 0, 118])
    white_upper_color = np.array([180, 50, 255])
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

        nl = bh + 10
        nx = max_num(min_zero(int(bx + (bw / 2) - (bh / 2)) - 5), 959-nl)
        ny = max_num(min_zero(by - 5), 719-nl)

        # if check_size(bw, bh) or check_fill(combined_mask, bx, by, bw, bh) or check_aspect(bw, bh):
        #     print(check_size(bw, bh), " ", check_fill(combined_mask, bx, by, bw, bh), " ", check_aspect(bw, bh))

        if check_size(bw, bh) and check_fill(combined_mask, bx, by, bw, bh) and check_aspect(bw, bh):
            candidates.append([nx, ny, nl])
        # cv2.imshow("rect", cv2.rectangle(img, (nx, ny), (nx + nl, ny + nl), (255, 0, 0), 2))
    # print(candidates)

    for bx, by, bl in candidates:
        cv2.imshow("rect", cv2.rectangle(img, (bx, by), (bx + bl, by + bl), (0, 255, 0), 2))

    return candidates


def candidate_selection(img, candidates, svm_model):
    features = []
    hogs = []
    hogs2 = []

    for i in range(len(candidates)):
        x = candidates[i][0]
        y = candidates[i][1]
        l = candidates[i][2]

        resized = cv2.resize(img[y:y + l, x:x + l], (20, 20), interpolation=cv2.INTER_AREA)
        b = resized.copy()
        # set green and red channels to 0
        b[:, :, 1] = 0
        b[:, :, 2] = 0

        r = resized.copy()
        # set blue and green channels to 0
        r[:, :, 0] = 0
        r[:, :, 1] = 0


        red_histogram_feature, red_img = hog(r, orientations=8, pixels_per_cell=(5, 5), cells_per_block=(2, 2),
                                             block_norm='L2', visualize=True)
        blue_histogram_feature, blue_img = hog(b, orientations=8, pixels_per_cell=(5, 5), cells_per_block=(2, 2),
                                               block_norm='L1', visualize=True)
        _, red_img2 = hog(cv2.resize(r, (40, 40), interpolation=cv2.INTER_AREA),
                          orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                          block_norm='L1', visualize=True)
        _, blue_img2 = hog(cv2.resize(b, (40, 40), interpolation=cv2.INTER_AREA), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                                               block_norm='L1', visualize=True)

        hogs.append(np.hstack((exposure.rescale_intensity(red_img2, out_range=(0, 255)),
                               exposure.rescale_intensity(blue_img2, out_range=(0, 255)))))
        hogs2.append(np.hstack((exposure.rescale_intensity(red_img, out_range=(0, 255)),
                               exposure.rescale_intensity(blue_img, out_range=(0, 255)))))
        features.append(np.concatenate((red_histogram_feature, blue_histogram_feature), axis=None))

    hog_features = np.array(features, np.float32)

    predictions = []

    for i in range(len(hog_features)):
        x = candidates[i][0]
        y = candidates[i][1]
        l = candidates[i][2]

        resized = cv2.resize(img[y:y + l, x:x + l], (20, 20), interpolation=cv2.INTER_AREA)

        print([hog_features[i]])

        prediction = svm_model.predict([hog_features[i]])[0]
        print(f"Prediction: {prediction}")

        if not prediction == 0:
            print(prediction)

            # hogImage = exposure.rescale_intensity(hog_features[i], out_range=(0, 255))
            # hogImage = hogImage.astype("uint8")
            cv2.imshow("HOG Image", hogs[i])
            cv2.imshow("HOG Image2", hogs2[i])

            if prediction == 1:
                white_image = np.zeros((20, 20, 3), np.uint8)
                white_image[:] = (255, 255, 255)
                combined_img = np.hstack((resized, white_image))
            else:
                red_image = np.zeros((20, 20, 3), np.uint8)
                red_image[:] = (0, 0, 255)
                combined_img = np.hstack((resized, red_image))

            cv2.imshow("candidate", combined_img)
            cv2.waitKey()

        predictions.append(prediction)

    return predictions


def live_video_main():
    cap = cv2.VideoCapture('data/video6.mkv')

    model = svm.svm()
    model.load(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)

        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        potential_candidates = candidate_extraction(filtered.copy())
        predictions = candidate_selection(filtered, potential_candidates, model)
        # print(predictions)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_video_main()
