# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:00:51 2022

@author: Khoa
"""

'''
------------------Harris Cornor
import cv2
import numpy as np

img = cv2.imread("khoa.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

harris = cv2.cornerHarris(gray, 5, 7, 0.04)
        
img[harris>0.001*harris.max()] = [255, 0, 0]

cv2.imshow('Khoa', img)

cv2.waitKey(0)

'''
'''
#------------------Good feature to track
import cv2
import numpy as np

img = cv2.imread("khoa.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)
    

cv2.imshow('Corners', img)

cv2.waitKey(0)

'''
import cv2
import numpy as np
import pylab as pl

img = cv2.imread("khoa.png")

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
brisk = cv2.BRISK_create(80)

keypoints, des = brisk.detectAndCompute(img, None)

im2 = cv2.drawKeypoints(img, keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Corners', im2)
pl.matshow(im2)
    
cv2.waitKey(0)



















