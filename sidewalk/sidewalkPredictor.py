import glob
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append("..")
from learningModels import vgg
sys.path.remove("..")

SAMPLE_START = 0
TOTAL_SAMPLES = 100


def load_images():
    filenames = glob.glob("../data/annotatedData/good/*.png")
    filenames.sort()

    # images = np.zeros((NUMBER_OF_SAMPLES, 720, 960, 3))
    # final_images = np.zeros((NUMBER_OF_SAMPLES, 285, 380, 3))

    images = []
    final_images = []

    for i in range(SAMPLE_START, SAMPLE_START + TOTAL_SAMPLES):
        images.append(cv2.imread(filenames[i], 1))
        # images[i] = cv2.imread('../data/annotatedData/good/my_video-1e_1570837657.7385182.png', 1)
        # img = cv2.imread('../data/annotatedData/good/my_video-1e_1570837657.7385182.png', 1)

        # cv2.imshow("Original", images[i])

        final_images.append(np.divide(cv2.resize(images[i-SAMPLE_START], (380, 285),
                                                 interpolation=cv2.INTER_AREA), 255))
        # cv2.imshow("Downsized", final_images[i])

        # _ = cv2.waitKey()
        # final_images[i][:] = [x / 255 for x in final_images]
        # final_images[i] /= 255

    # images = [cv2.imread(img) for img in filenames[:NUMBER_OF_SAMPLES]]
    # wait(5)
    print('loaded')
    return np.asarray(final_images), np.asarray(images)


def load_annotations():
    df = pd.read_csv('../data/annotatedData/lines.csv', usecols=["rhoL", "thetaL", "rhoR", "thetaR"])
    annotations = df.to_numpy()
    annotations.astype(float)

    print(annotations.size / 4)
    print(annotations[0])

    for i in range(int(annotations.size / 4)):

        if annotations[i][3] < 0:
            annotations[i][3] = annotations[i][3] + np.pi

        annotations[i][0] = abs(annotations[i][0] / 760)
        annotations[i][2] = abs(annotations[i][2] / 760)
        annotations[i][1] /= (2 * np.pi)
        annotations[i][3] /= (2 * np.pi)

    return annotations[SAMPLE_START:SAMPLE_START + TOTAL_SAMPLES]


def unison_shuffled_copies(a, b):

    randomize = np.arange(len(a))
    np.random.shuffle(randomize)
    return a[randomize], b[randomize]


def getData():
    init_x, orig_x = load_images()
    init_y = load_annotations()

    print(len(init_x))
    print(len(init_x[0]))
    print(len(init_x[1]))
    print(np.shape(init_y))

    # rand_x, rand_y = unison_shuffled_copies(init_x, init_y)

    return init_x, orig_x, init_y


def get_lined_image(img, lines, color):
    for i in range(2):
        a = np.cos(lines[1 + (i*2)])
        b = np.sin(lines[1 + (i*2)])
        x0 = a * lines[i*2]
        y0 = b * lines[i*2]

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    return img


def reverse_transform(lines):
    lines[1] *= (2 * np.pi)
    lines[3] *= (2 * np.pi)
    if lines[1] > np.pi / 2:
        lines[0] *= -760
    else:
        lines[0] *= 760

    if lines[3] > np.pi/2:
        lines[2] *= -760
    else:
        lines[2] *= 760

    return lines


def main():
    x, xo, y = getData()

    model = vgg.vgg(0, 0)
    # model.load("1574810490")
    model.load("1574833282")

    for i in range(TOTAL_SAMPLES):
        x_mod = [x[i]]
        prediction = model.predict(np.asarray(x_mod))
        cv2.imshow('Input', np.multiply(x[i], 255))

        transformed_prediction = reverse_transform(prediction[0])
        print("Prediction:", transformed_prediction)
        prediction_img = get_lined_image(xo[i], transformed_prediction, (0, 255, 255))
        # cv2.imshow('Prediction', prediction_img)

        transformed_actual = reverse_transform(y[i])
        print("Actual:", transformed_actual)
        actual_img = get_lined_image(xo[i], transformed_actual, (0, 0, 255))
        cv2.imshow('Res', actual_img)

        _ = cv2.waitKey()


if __name__ == "__main__":
    main()
