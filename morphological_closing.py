# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:49:46 2021

@author: kaust
"""

import cv2 as cv
import numpy as np


def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img=cv.imread("input1.jpg",0)
image_resized = rescaleFrame(img)
kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,2))
close = cv.morphologyEx(image_resized, cv.MORPH_CLOSE,kernel1)
div = np.float32(image_resized)/(close)
res = np.uint8(cv.normalize(div,div,0,255,cv.NORM_MINMAX))
cv.imshow("mc_1", res)
cv.imwrite("mc_1.jpg", res)



img=cv.imread("input2.jpg",0)
image_resized = rescaleFrame(img)
kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,2))
close = cv.morphologyEx(image_resized, cv.MORPH_CLOSE,kernel1)
div = np.float32(image_resized)/(close)
res = np.uint8(cv.normalize(div,div,0,255,cv.NORM_MINMAX))
cv.imshow("mc_2", res)
cv.imwrite("mc_2.jpg", res)

cv.waitKey(0) 
cv.destroyAllWindows() 