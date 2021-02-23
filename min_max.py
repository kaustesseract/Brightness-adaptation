# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:26:57 2021

@author: kaust
"""

import cv2 as cv
import numpy as np


def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def min_max_normalization(image):
    dst = np.zeros((500, 500))
    norm_image = cv.normalize(image, dst, alpha=0, beta=255, norm_type=cv.NORM_MINMAX) # normalising the image with min max method 
    return norm_image
    


original_image_1 = cv.imread("img_1_1.jpg", cv.IMREAD_COLOR)  # reading image
original_image_1 = rescaleFrame(original_image_1) # rescaling the image
norm_image_1 = min_max_normalization(original_image_1) # getting the normalized image

cv.imshow('Normalized Image 1', norm_image_1) # dispayling the image
cv.imwrite('norm_image_1.jpg', norm_image_1) # writing the image
print("MAX value of image 1: ", np.amax(norm_image_1)) # displaying the max value of the image 1
print("MIN value of image 1: ", np.amin(norm_image_1)) # displaying the max value of the image 1


original_image_2 = cv.imread("img_1_2.jpg", cv.IMREAD_COLOR)  # reading image
original_image_2 = rescaleFrame(original_image_2) # rescaling the image
norm_image_2 = min_max_normalization(original_image_2) # getting the normalized image

cv.imshow('Normalized Image 2', norm_image_2) # dispayling the image
cv.imwrite('norm_image_2.jpg', norm_image_2) # writing the image
print("MAX value of image 2: ",np.amax(norm_image_2)) # displaying the max value of the image 2
print("MIN value of image 2: ",np.amin(norm_image_2)) # displaying the max value of the image 2


original_image_L1 = cv.norm(original_image_1 - original_image_2, cv.NORM_L1) # calculating L1 for original image
print("L1 distance of before normalization: ",original_image_L1)
original_image_L2 = cv.norm(original_image_1 - original_image_2, cv.NORM_L2) # calculating L1 for original image
print("L2 distance of before normalization: ",original_image_L2)
normalised_image_L1 = cv.norm(norm_image_1 - norm_image_2, cv.NORM_L1) # calculating L1 for normalized image
print("L1 distance of after normalization: ",normalised_image_L1)
normalised_image_L2 = cv.norm(norm_image_1 - norm_image_2, cv.NORM_L2) # calculating L2 for normalized image
print("L2 distance of after normalization: ",normalised_image_L2)


cv.waitKey(0)
cv.destroyAllWindows()

"""
def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img1 = cv.imread('input1.jpg')
img_1 = rescaleFrame(img1)
normalizedImg1 = np.zeros((500, 500))
normalizedImg1 = cv.normalize(img_1, None , 0, 255, cv.NORM_MINMAX)
cv.imshow('dst_rt1', normalizedImg1)
print(np.amax(normalizedImg1))


img2 = cv.imread('input2.jpg')
img_2 = rescaleFrame(img2)
normalizedImg2 = np.zeros((500, 500))
normalizedImg2 = cv.normalize(img_2, None,  0, 255, cv.NORM_MINMAX)
cv.imshow('dst_rt2', normalizedImg2)

original_image_L1 = cv.norm(img1 - img2, cv.NORM_L1) # calculating L1 for original image
print(original_image_L1)
original_image_L2 = cv.norm(img1 - img2, cv.NORM_L2) # calculating L1 for original image
print(original_image_L2)
normalised_image_L1 = cv.norm(normalizedImg1- normalizedImg2, cv.NORM_L1) # calculating L1 for normalized image
print(normalised_image_L1)
normalised_image_L2 = cv.norm(normalizedImg1 - normalizedImg2, cv.NORM_L2) # calculating L2 for normalized image
print(normalised_image_L2)


cv.waitKey(0)
cv.destroyAllWindows()
"""