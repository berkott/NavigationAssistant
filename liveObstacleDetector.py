# Great videos:
# 0:00 of sidewalk 4n - people
# 9:00, 11:00  of sidewalk 5e - people

import numpy as np
import time
import cv2
import os


CONFIDENCE = 0.3
LABELS = open(os.path.abspath("data/coco.names")).read().strip().split("\n")


def getDist(px, co):
    # return ((px*(-.15)) + co)/100  # m
    return ((px * (-3.7)) + co) / 100  # m

def getVel(d1, d2):
    return (d2-d1)/.2  # m/s


def getCars(_labels, _boxes):
    newCars = []
    for i in range(len(_labels)):
        if LABELS[_labels[i]] == "car":
            newCars.append(_boxes[i])
    return newCars


# (x, y) = (boxes[i][0], boxes[i][1])
# (w, h) = (boxes[i][2], boxes[i][3])
def getClosestCar(cx, cy, boxz):
    closestDistance = 9999
    closestIndex = 0
    for i in range(len(boxz)):
        distance = np.sqrt(np.square(cx - boxz[i][0]) + np.square(cy - boxz[i][1]))
        if distance < closestDistance:
            closestDistance = distance
            closestIndex = i
    return boxz[closestIndex][3]


# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join(["data", "yolov3-tiny.weights"])
configPath = os.path.sep.join(["data", "yolov3-tiny.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture("/home/berk/code/general/eliminator/videos/video.mp4")
# cap = cv2.VideoCapture("/home/berk/code/projects/blindAssist/data/evening/my_video-4e.mkv")
# cap = cv2.VideoCapture("/home/berk/code/projects/blindAssist/data/trafficLight/rawVideos/my_video-7.mkv")
cap = cv2.VideoCapture("data/video4.mkv")
ret, frame = cap.read()
(W, H) = (None, None)
k = 0
oldCarBoxes = []

# loop over frames from the video file stream
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(layer_names)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, CONFIDENCE)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            dist = 0
            vel = 0

            if LABELS[classIDs[i]] == "person":
                dist = getDist(h, 1270)

            if LABELS[classIDs[i]] == "car":
                dist = getDist(h, 3000)
                if len(oldCarBoxes) > 0:
                    closestCarHeight = getClosestCar(x, y, oldCarBoxes)
                    vel = getVel(dist, getDist(closestCarHeight, 3000))

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if LABELS[classIDs[i]] == "car":
               text = "{}: {:.4f}, {:.4f}m, {:.4f}m/s".format(LABELS[classIDs[i]], confidences[i], dist, vel)
            elif LABELS[classIDs[i]] == "person":
               text = "{}: {:.4f}, {:.4f}m".format(LABELS[classIDs[i]], confidences[i], dist)
            else:
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    oldCarBoxes = getCars(classIDs, boxes)

    if k == ord(' '):
        k = cv2.waitKey()

    if k == ord('q') or not ret:
        break

# release the file pointers
print("[INFO] cleaning up...")

cap.release()
cv2.destroyAllWindows()
