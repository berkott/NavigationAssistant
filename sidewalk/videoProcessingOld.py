import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960

def warpImage(img):
    dst_size = (960, 720)
    src = np.float32([(250, 400), (710, 400), (0, 720), (960, 720)])
    dst = np.float32([(0, 0), (960, 0), (0, 720), (960, 720)])

    M = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, M, dst_size)


def inverseWarpImage(img):
    dst_size = (960, 720)
    src = np.float32([(250, 400), (710, 400), (0, 720), (960, 720)])
    dst = np.float32([(0, 0), (960, 0), (0, 720), (960, 720)])

    M = cv2.getPerspectiveTransform(dst, src)

    return cv2.warpPerspective(img, M, dst_size)


def filtering(img):
    # cv2.medianBlur(img, 5)
    image1 = cv2.GaussianBlur(img, (11, 11), 0)
    image2 = cv2.blur(image1, (5, 5))
    image3 = cv2.medianBlur(image2, 5)
    image4 = cv2.bilateralFilter(image3, 9, 75, 75)
    image5 = cv2.bilateralFilter(image4, 9, 75, 75)
    image = cv2.bilateralFilter(image5, 9, 75, 75)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower_color = np.array([0, 0, 20])
    # upper_color = np.array([255, 200, 250])

    lower_color = np.array([0, 0, 0])
    upper_color = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    res = cv2.bitwise_and(image, image, mask=mask)

    return image


def imageWithLines(img):
    edge = cv2.Canny(img, 100, 200, 15)
    cv2.imshow('edge', edge)

    # modified_edge = cv2.rectangle(edge, (0, 0), (960, 300), (0, 0, 0), -1)

    lines = cv2.HoughLines(edge, 1, np.pi / 180, 100)

    for i in range(6):
        try:
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        except Exception:
            print(Exception)
    return img


def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:, :], axis=0)
    return hist


def separateColors(img):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    kernel = np.ones((20, 20), np.uint8)

    return cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel)


def getSides(img):
    sidewalkColor = img[int(IMAGE_HEIGHT-1), int(IMAGE_WIDTH/2)]

    leftBoundary = []
    rightBoundary = []

    middle = int(IMAGE_WIDTH/2)

    for y in range(IMAGE_HEIGHT-1, -1, -1):
        for i in range(middle, -1, -1):
            if (img[y, i] != sidewalkColor).all():
                leftBoundary.append([i, y])
                break
            if i == 0:
                leftBoundary.append([0, y])

        for i in range(middle, IMAGE_WIDTH):
            if (img[y, i] != sidewalkColor).all():
                rightBoundary.append([i, y])
                break
            if i == IMAGE_WIDTH - 1:
                rightBoundary.append([IMAGE_WIDTH - 1, y])

        middle = int((leftBoundary[IMAGE_HEIGHT - 1 - y][0] + rightBoundary[IMAGE_HEIGHT - 1 - y][0]) / 2)

    return leftBoundary, rightBoundary


def getLinesFromBoundary(left, right):
    boundaryImageLeft = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), np.uint8)
    boundaryImageRight = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), np.uint8)

    for i in left:
        boundaryImageLeft[i[1], i[0]] = 255

    for i in right:
        boundaryImageRight[i[1], i[0]] = 255

    # cv2.imshow("boundaryImageLeft", boundaryImageLeft)
    # cv2.imshow("boundaryImageRight", boundaryImageRight)

    leftLines = cv2.HoughLines(boundaryImageLeft, 1, np.pi / 180, 50)
    rightLines = cv2.HoughLines(boundaryImageRight, 1, np.pi / 180, 50)

    lines = []

    if leftLines is not None:
        lines.append(leftLines[0])

    if rightLines is not None:
        lines.append(rightLines[0])

    leftLinesOG = cv2.HoughLines(inverseWarpImage(boundaryImageLeft), 1, np.pi / 180, 50)
    rightLinesOG = cv2.HoughLines(inverseWarpImage(boundaryImageRight), 1, np.pi / 180, 50)

    originalLines = []

    if leftLinesOG is not None:
        originalLines.append(leftLinesOG[0])

    if rightLinesOG is not None:
        originalLines.append(rightLinesOG[0])

    # print(lines)

    return lines, originalLines


def getLinedImage(img, lines):
    for i in range(len(lines)):
        try:
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        except Exception:
            print(Exception)
    return img

videoName = 'my_video-2n'
cap = cv2.VideoCapture('../data/noon/' + videoName + '.mkv')

if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame', frame)
        originalImage = frame.copy()

        warped = warpImage(frame)
        # cv2.imshofw('warped', warped)

        filtered = filtering(warped)
        cv2.imshow('filtered', filtered)

        colorSeparated = separateColors(filtered)
        # cv2.imshow("separated", colorSeparated)

        leftSidewalk, rightSidewalk = getSides(colorSeparated)
        lines, linesOG = getLinesFromBoundary(leftSidewalk, rightSidewalk)
        linedImage = getLinedImage(colorSeparated, lines)
        cv2.imshow("linedImage", linedImage)

        linedOriginal = getLinedImage(frame, linesOG)
        cv2.imshow("linedOriginal", linedOriginal)

        # lined = imageWithLines(colorSeparated)
        # cv2.imshow('lined', lined)
        key = cv2.waitKey()

        current_time = time.time()
        if key == ord('k'):
            cv2.imwrite('../data/annotatedData/good/' + videoName + '_' + str(current_time) + '.png', originalImage)
            with open('../data/annotatedData/lines.csv', mode='a') as lines_file:
                lines_writer = csv.writer(lines_file, delimiter=',')
                # videoName, time, rhoL, thetaL, rhoR, thetaR
                lines_writer.writerow([videoName, current_time, linesOG[0][0][0],
                                       linesOG[0][0][1], linesOG[1][0][0], linesOG[1][0][1]])
        elif key == ord('l'):
            cv2.imwrite('../data/annotatedData/bad/' + videoName + '_' + str(current_time) + '.png', originalImage)
        elif key == ord('q'):
            break

    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
