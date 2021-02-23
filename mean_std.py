# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:29:00 2021

@author: kaust
"""

import cv2 as cv
import numpy as np
def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def mean_std(normalizing_image):
    normalizing_image -= normalizing_image.mean() # calculting the minus of mean of the image
    normalizing_image /= normalizing_image.std() # dividing it with standard deviation
    normalized_image = normalizing_image.astype(np.uint8)*255
    return normalized_image

#image = cv2.imread('../data/Lena.png').astype(np.float32) / 255
image_1 = cv.imread("img_1_1.jpg").astype(np.float64) / 255 # reading the image and converting into float values
original_image_resized_1 = rescaleFrame(image_1) # rescaling the image
normalizing_image_1 = rescaleFrame(image_1)

normalized_image_1 = mean_std(normalizing_image_1) # returning the value from the function
print(normalized_image_1)
 
cv.imshow('Normalized Image 1', normalized_image_1) # displaying the image
cv.imwrite("mean_std_1.jpg", normalized_image_1) # writing the image
print("MAX: ",np.amax(normalized_image_1)) # calculating the max value of the image 1


image_2 = cv.imread("img_1_2.jpg").astype(np.float64) / 255  # reading the image and converting into float values
original_image_resized_2 = rescaleFrame(image_2) # rescaling the image
normalizing_image_2 = rescaleFrame(image_2)

normalized_image_2 = mean_std(normalizing_image_2) # returning the value from the function

cv.imshow('Normalized Image 2', normalized_image_2 ) # displaying the image
cv.imwrite("mean_std_2.jpg", normalized_image_2) # writing the image
print("MAX: ",np.amax(normalized_image_2)) # calculating the max value of the image 2
#print(np.amin(inputimage1*255))


original_img_L1 = cv.norm(original_image_resized_1 - original_image_resized_2, cv.NORM_L1)
print("L1 distance of before normalization: ",original_img_L1)

original_img_L2 = cv.norm(original_image_resized_1 - original_image_resized_2, cv.NORM_L2)
print("L2 distance of before normalization: ",original_img_L2)
normalized_img_L1 = cv.norm(normalized_image_1 - normalized_image_2, cv.NORM_L1)
print("L1 distance of after normalization: ",normalized_img_L1)

normalized_img_L2 = cv.norm(normalized_image_1 - normalized_image_2, cv.NORM_L2)
print("L2 distance of after normalization: ",normalized_img_L2)


cv.waitKey(0)
cv.destroyAllWindows()

