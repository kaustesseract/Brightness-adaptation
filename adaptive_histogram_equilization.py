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

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = clahe.apply(image)
    return out
# Load the image in greyscale
img1 = cv2.imread('img_1_1.jpg',0)
img1 = rescaleFrame(img1)
out1 = clahe(img1)
cv2.imshow('a1',out1)
cv2.imwrite('ahe_1.jpg',out1)


img2 = cv2.imread('img_1_2.jpg',0)
img2 = rescaleFrame(img2)
out2 = clahe(img2) 
cv2.imshow('a2',out2)
cv2.imwrite('ahe_2.jpg',out2)

original_img_L1 = cv2.norm(img1 - img2, cv2.NORM_L1)
print("L1 distance of before normalization: ",original_img_L1)
original_img_L2 = cv2.norm(img1 - img2, cv2.NORM_L2)
print("L2 distance of before normalization: ",original_img_L2)
normalized_img_L1 = cv2.norm(out1 - out2, cv2.NORM_L1)
print("L1 distance of after normalization: ",normalized_img_L1)
normalized_img_L2 = cv2.norm(out1 - out2, cv2.NORM_L2)
print("L2 distance of after normalization: ",normalized_img_L2)


cv2.waitKey(0)
cv2.destroyAllWindows()
