# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:01:49 2020

@author: Saurabh
"""
import cv2 as cv
import numpy as np
from imutils.object_detection import non_max_suppression
newW=int(320*3.5)
newH=int(320*10.2)
model="./frozen_east_text_detection.pb"
net = cv.dnn.readNet(model)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
img = cv.imread(r'i1.png')
img=cv.resize(img,(720,540))
orig = img.copy()
(H, W) = img.shape[:2]
# set the new width and height and then determine the ratio in change
# for both the width and height
#(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)
# resize the image and grab the new image dimensions
img = cv.resize(img, (newW, newH))
(H, W) = img.shape[:2]


#im2 = cv.resize(img, (960, 540))

blob = cv.dnn.blobFromImage(img, 1.0, (newW, newH), (123.68, 116.78, 103.94), True, False)
opLayer = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
net.setInput(blob)
output = net.forward(opLayer)
scores = output[0]
geometry = output[1]
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []
for y in range(numRows):
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]
    for x in range(0, numCols):
       # print(scoresData[x])
        if scoresData[x] < 0.5:
            continue
        (offsetX, offsetY) = (x * 4.0, y * 4.0)
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])
boxes = non_max_suppression(np.array(rects), probs=confidences)
#print(boxes.len())
for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    cv.rectangle(orig, (startX-5, startY-5), (endX+5, endY+5), (0, 255, 0), 2)
cv.imshow('image',orig)
cv.waitKey(0)