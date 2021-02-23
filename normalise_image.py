# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:57:47 2021

@author: kaust
"""

import cv2 as cv
import numpy as np
def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#pic1 = cv.imread('pic1.jpg')
#resized_image1 = rescaleFrame(pic1)
original_image = cv.imread('home.jpg',0) # reading the image
gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY) # original image
kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)) # setting up the kernel of 5*5 
close = cv.morphologyEx(gray,cv.MORPH_CLOSE,kernel1) # morphing the image
div = np.float32(gray)/(close) 
min_max_image = np.uint8(cv.normalize(div,div,0,255,cv.NORM_MINMAX)) # applying the min max normalisation
cv.imshow('Image1', min_max_image) # displaying the image 


#pic2 = cv.imread('pic2.jpg')
#resized_image2 = rescaleFrame(pic2)
#cv.imshow('Image2', resized_image2)


cv.waitKey(0)
cv.destroyAllWindows()