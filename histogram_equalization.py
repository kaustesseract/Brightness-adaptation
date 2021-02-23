# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:03:16 2021

@author: kaust
"""
"""
import cv2 as cv
import numpy as np
def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


#image = cv2.imread('../data/Lena.png').astype(np.float32) / 255
gray_image = cv.imread("input1.jpg") # uint8 image

gray_resized = rescaleFrame(gray_image)

dst = cv.equalizeHist(gray_resized)

cv.imshow('Equalized Image', dst)


cv.waitKey(0)
cv.destroyAllWindows()
"""
import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
  
# read a image using imread 
img_one = cv.imread("img_1_1.jpg",0)    

#img = cv.imread("input1.jpg", 0)

img_resized_one = rescaleFrame(img_one) 
  
# creating a Histograms Equalization 
# of a image using cv2.equalizeHist() 
equ_one = cv.equalizeHist(img_resized_one) 
  

  
# show image input vs output 
cv.imshow("he_1",equ_one) 
cv.imwrite("he_1.jpg", equ_one)
print("MAX image 1: ",np.amax(equ_one))
print("MIN image 1: ",np.amin(equ_one))

#img_rgb_two = cv.imread("img_1_2.jpg")

img_two = cv.imread("img_1_2.jpg", 0)

img_resized_two = rescaleFrame(img_two) 
  
# creating a Histograms Equalization 
# of a image using cv2.equalizeHist() 
equ_two = cv.equalizeHist(img_resized_two) 
  
  
# show image input vs output 
cv.imshow("he_2",equ_two) 
cv.imwrite("he_2.jpg", equ_two)
print("MAX imaage 2: ",np.amax(equ_two))
print("MIN imaage 2: ",np.amin(equ_two))

distance_original_one = cv.norm(img_resized_one - img_resized_two, cv.NORM_L1)
print(distance_original_one)
distance_original_one = cv.norm(img_resized_one - img_resized_two, cv.NORM_L2)
print(distance_original_one)
distance_normalized_two = cv.norm(equ_one - equ_two, cv.NORM_L1)
print(distance_normalized_two)
distance_normalized_two = cv.norm(equ_one - equ_two, cv.NORM_L2)
print(distance_normalized_two)
cv.waitKey(0) 
cv.destroyAllWindows() 