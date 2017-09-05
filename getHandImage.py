#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:21:55 2017

@author: miyashitayudai
"""

import os
import sys
import scipy.io
import numpy as np
import cv2

labelFolderPath = "C:/Users/murakamiryosuke/Documents/Pose_Estimation/cnn-sample/hand_dataset/training_dataset/training_data/annotations"
imageFolderPath = "C:/Users/murakamiryosuke/Documents/Pose_Estimation/cnn-sample/hand_dataset/training_dataset/training_data/images"
saveFolderPath = "C:/Users/murakamiryosuke/Documents/Pose_Estimation/cnn-sample/hand_dataset/training_dataset/training_data/handImages"

labels = os.listdir(labelFolderPath)
imageFolderList = os.listdir(imageFolderPath)

imageNum = 0
labelArray = []
for file in labels:
    #隠しファイルは無視
    if 0 is file.find("."):
        continue

    #ラベルデータ読込
    matdata = scipy.io.loadmat(labelFolderPath + "/" + file)

    #画像読込
    image = cv2.imread(imageFolderPath + "/" + file.replace(".mat", ".jpg"))
   
    max_x = 0;
    max_y = 0;
    min_x = 9999;
    min_y = 9999;
    
    xList = [int(matdata["boxes"][0][0][0][0][0][0][1]), 
             int(matdata["boxes"][0][0][0][0][1][0][1]), 
             int(matdata["boxes"][0][0][0][0][2][0][1]), 
             int(matdata["boxes"][0][0][0][0][3][0][1])]

    yList = [int(matdata["boxes"][0][0][0][0][0][0][0]), 
             int(matdata["boxes"][0][0][0][0][1][0][0]), 
             int(matdata["boxes"][0][0][0][0][2][0][0]), 
             int(matdata["boxes"][0][0][0][0][3][0][0])]
    
    xList.sort()
    yList.sort()
    
    bbox = [xList[0], yList[0], xList[3], yList[3]]
    dst = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    #cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
#    cv2.circle(image, (int(matdata["boxes"][0][0][0][0][0][0][1]), int(matdata["boxes"][0][0][0][0][0][0][0])), 3, (255, 0, 0), -1)
#    cv2.circle(image, (int(matdata["boxes"][0][0][0][0][1][0][1]), int(matdata["boxes"][0][0][0][0][1][0][0])), 3, (255, 0, 0), -1)
#    cv2.circle(image, (int(matdata["boxes"][0][0][0][0][2][0][1]), int(matdata["boxes"][0][0][0][0][2][0][0])), 3, (255, 0, 0), -1)
#    cv2.circle(image, (int(matdata["boxes"][0][0][0][0][3][0][1]), int(matdata["boxes"][0][0][0][0][3][0][0])), 3, (255, 0, 0), -1)
    
        
    # show the output images
    #cv2.imshow("hand", image)
    #cv2.waitKey(-1)
    cv2.imwrite(saveFolderPath + "/" + "{0:04d}".format(imageNum) + ".png", dst)
    imageNum = imageNum + 1
    print(file)