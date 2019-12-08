# Could try feature extraction instead of using CNN. Maybe HOG??

import glob
import sys
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append("..")
from learningModels import vgg

sys.path.remove("..")

EPOCHS = 10
BATCH_SIZE = 2
TOTAL_SAMPLES = 20
N_SPLITS = 10


def load_images():
    filenames = glob.glob("../data/annotatedData/good/*.png")
    filenames.sort()

    # images = np.zeros((NUMBER_OF_SAMPLES, 720, 960, 3))
    # final_images = np.zeros((NUMBER_OF_SAMPLES, 285, 380, 3))

    images = []
    final_images = []

    for i in range(TOTAL_SAMPLES):
        images.append(cv2.imread(filenames[i], 1))
        # images[i] = cv2.imread('../data/annotatedData/good/my_video-1e_1570837657.7385182.png', 1)
        # img = cv2.imread('../data/annotatedData/good/my_video-1e_1570837657.7385182.png', 1)

        # cv2.imshow("Original", images[i])

        final_images.append(np.divide(cv2.resize(images[i], (380, 285), interpolation=cv2.INTER_AREA), 255))
        # cv2.imshow("Downsized", final_images[i])

        # _ = cv2.waitKey()
        # final_images[i][:] = [x / 255 for x in final_images]
        # final_images[i] /= 255

    # images = [cv2.imread(img) for img in filenames[:NUMBER_OF_SAMPLES]]
    # wait(5)
    print('loaded')
    return np.asarray(final_images)


def load_annotations():
    df = pd.read_csv('../data/annotatedData/lines.csv', usecols=["rhoL", "thetaL", "rhoR", "thetaR"])
    annotations = df.to_numpy()
    annotations.astype(float)

    print(annotations.size / 4)

    for i in range(int(annotations.size / 4)):

        if annotations[i][3] < 0:
            annotations[i][3] = annotations[i][3] + np.pi

        annotations[i][0] = abs(annotations[i][0] / 760)
        annotations[i][2] = abs(annotations[i][2] / 760)
        annotations[i][1] /= (2 * np.pi)
        annotations[i][3] /= (2 * np.pi)

    return annotations[:TOTAL_SAMPLES]


def unison_shuffled_copies(a, b):
    randomize = np.arange(len(a))
    np.random.shuffle(randomize)
    return a[randomize], b[randomize]


def getData():
    init_x = load_images()
    init_y = load_annotations()

    print(len(init_x))
    print(len(init_x[0]))
    print(len(init_x[1]))
    print(np.shape(init_y))

    rand_x, rand_y = unison_shuffled_copies(init_x, init_y)

    return rand_x, rand_y


def main():
    x, y = getData()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

    for train, test in skf.split(x, np.zeros(TOTAL_SAMPLES)):
        print("TRAIN:", train, "TEST:", test)
        model = vgg.vgg(EPOCHS, BATCH_SIZE)

        model.create_model()

        model.train(x[train], y[train])
        model.evaluate(x[test], y[test])
        model.save()


if __name__ == "__main__":
    main()
