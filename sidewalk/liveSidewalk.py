import sys

import cv2
import numpy as np

sys.path.append("..")
from learningModels import vgg
sys.path.remove("..")


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


def live_video_main():
    cap = cv2.VideoCapture(2)

    model = vgg.vgg(0, 0)
    # model.load("1574810490")
    model.load("1574833282")

    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        orig_img = frame.copy()

        downscaled_frame = np.divide(cv2.resize(frame, (380, 285), interpolation=cv2.INTER_AREA), 255)

        x_mod = [downscaled_frame]
        prediction = model.predict(np.asarray(x_mod))

        transformed_prediction = reverse_transform(prediction[0])
        print("Prediction:", transformed_prediction)
        prediction_img = get_lined_image(orig_img, transformed_prediction, (0, 0, 255))

        cv2.imshow('Result', prediction_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_video_main()