import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(*arg):
    pass


# icol = (0, 203, 32, 196, 255, 255)  # Red
icol = (0, 0, 23, 175, 50, 91)  # White

cv2.namedWindow('colorTest')
# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

img1 = cv2.imread('../data/trafficLight/whiteTrafficLight.png', 1)
# img1 = cv2.imread('../data/annotatedData/good/my_video-1e_1570837657.7385182.png', 1)
img2 = cv2.resize(img1, (960, 720))

img3 = cv2.GaussianBlur(img2, (3, 3), 0)

hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)

while True:
    # Get HSV values from the GUI sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
    lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
    lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
    highHue = cv2.getTrackbarPos('highHue', 'colorTest')
    highSat = cv2.getTrackbarPos('highSat', 'colorTest')
    highVal = cv2.getTrackbarPos('highVal', 'colorTest')

    # Show the original image.
    cv2.imshow('img', img3)

    # HSV values to define a colour range.
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    # Show the first mask
    cv2.imshow('mask-plain', mask)

    # kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    # Show morphological transformation mask
    # cv2.imshow('mask', mask)

    # Put mask over top of the original image.
    result = cv2.bitwise_and(img3, img3, mask=mask)

    # Show final output image
    cv2.imshow('colorTest', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()