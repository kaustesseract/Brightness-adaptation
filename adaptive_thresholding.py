# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:13:28 2021

@author: kaust
"""

import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img_1 = cv.imread('img_1_1.jpg',0)
img_1 = rescaleFrame(img_1)
img_1 = cv.medianBlur(img_1,5)
th2_i1 = cv.adaptiveThreshold(img_1,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3_i1 = cv.adaptiveThreshold(img_1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
#print(th2_i1)
cv.imshow('at_gauss_1', th3_i1) 
#cv.imshow('gaussian_1', th3)
cv.imwrite('at_gauss_1.jpg', th3_i1) 



img_2 = cv.imread('img_1_2.jpg',0)
img_2 = rescaleFrame(img_2)
img_2 = cv.medianBlur(img_2,5)

th2_i2 = cv.adaptiveThreshold(img_2,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3_i2 = cv.adaptiveThreshold(img_2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
ll = cv.cvtColor(img_2, cv.COLOR_GRAY2RGB)
#cv.imshow('at_gauss_2', img_2) 
#cv.imwrite('at_gauss_2.jpg', th3_i2) 

"""
threshibo_L1 = cv.norm(img_1 - img_2, cv.NORM_L1)
print(threshibo_L1)
threshibo_L2 = cv.norm(img_1 - img_2, cv.NORM_L2)
print(threshibo_L2)
thresh_L1 = cv.norm(th2_i1 - th2_i2, cv.NORM_L1)
print(thresh_L1)
thresh_L2 = cv.norm(th2_i1 - th2_i2, cv.NORM_L2)
print(thresh_L2)
"""
threshibo_L1 = cv.norm(img_1 - img_2, cv.NORM_L1)
print(threshibo_L1)
threshibo_L2 = cv.norm(img_1 - img_2, cv.NORM_L2)
print(threshibo_L2)
thresh_L1 = cv.norm(th3_i1 - th3_i2, cv.NORM_L1)
print(thresh_L1)
thresh_L2 = cv.norm(th3_i1 - th3_i2, cv.NORM_L2)
print(thresh_L2)
#cv2.imshow('Set to 0 Inverted', thresh5) 
cv.waitKey(0)
cv.destroyAllWindows() 

