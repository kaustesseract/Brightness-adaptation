# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:45:44 2021

@author: kaust
"""

import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
img_1 = cv.imread('input1.jpg',0)
img_1 = rescaleFrame(img_1)
# global thresholding

# Otsu's thresholding
ret2,th1_otsu = cv.threshold(img_1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
#blur = cv.GaussianBlur(img,(3,3),0)
#ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms

cv.imshow("th1_otsu",th1_otsu)
cv.imwrite("th1_otsu.jpg",th1_otsu)
#cv.imshow("th3",th3)


img_2 = cv.imread('input2.jpg',0)
img_2 = rescaleFrame(img_2)
# global thresholding
# Otsu's thresholding
ret2,th2_otsu = cv.threshold(img_2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
#blur = cv.GaussianBlur(img,(3,3),0)
#ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms

cv.imshow("th2_otsu",th2_otsu)
cv.imwrite("th2_otsu.jpg",th2_otsu)


threshibo_L1 = cv.norm(img_1 - img_2, cv.NORM_L1)
print(threshibo_L1)
threshibo_L2 = cv.norm(img_1 - img_2, cv.NORM_L2)
print(threshibo_L2)
thresh_L1 = cv.norm(th1_otsu  - th2_otsu , cv.NORM_L1)
print(thresh_L1)
thresh_L2 = cv.norm(th1_otsu - th2_otsu , cv.NORM_L2)
print(thresh_L2)


cv.waitKey(0)
cv.destroyAllWindows() 