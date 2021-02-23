# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:48:10 2021

@author: kaust
"""



import numpy as np
import cv2
 
def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
# Load the image in greyscale
img = cv2.imread('img_1_1.jpg',0)
img = rescaleFrame(img)
 
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
out = clahe.apply(img)
 
# Display the images side by side using cv2.hconcat

cv2.imshow('a1',out)


img2 = cv2.imread('img_1_2.jpg',0)
img2 = rescaleFrame(img2)
 
# create a CLAHE object (Arguments are optional).
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
out2 = clahe.apply(img2)
 
# Display the images side by side using cv2.hconcat



cv2.imshow('a2',out2)

original_img_L1 = cv2.norm(img - img2, cv2.NORM_L1)
print("L1 distance of before normalization: ",original_img_L1)


normalized_img_L1 = cv2.norm(out - out2, cv2.NORM_L1)
print("L1 distance of after normalization: ",normalized_img_L1)


cv2.waitKey(0)
cv2.destroyAllWindows()
