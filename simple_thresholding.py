# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:27:19 2021

@author: kaust
"""
import cv2  

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
  
# path to input image is specified and   
# image is loaded with imread command  
image1 = cv2.imread('input1.jpg')  
image1 = rescaleFrame(image1)
image2 = cv2.imread('input2.jpg')
image2 = rescaleFrame(image2)
# cv2.cvtColor is applied over the 
# image input with applied parameters 
# to convert the image in grayscale  
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  
# applying different thresholding  
# techniques on the input image 
# all pixels value above 120 will  
# be set to 255 
ret, thresh1_img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY) 
ret, thresh1_img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY)

ret, thresh2_img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY_INV) 
ret, thresh2_img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY_INV) 

ret, thresh3_img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_TRUNC) 
ret, thresh3_img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_TRUNC) 

ret, thresh4_img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_TOZERO) 
ret, thresh4_img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_TOZERO) 

ret, thresh5_img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_TOZERO_INV) 
ret, thresh5_img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_TOZERO_INV) 

  
# the window showing output images 
# with the corresponding thresholding  
# techniques applied to the input images 
#cv2.imshow('Binary Threshold', thresh1) 
#cv2.imshow('Binary Threshold Inverted', thresh2) 
#cv2.imshow('Truncated Threshold', thresh3) 
cv2.imshow('s_th_thresh_tozeroi_1', thresh5_img1) 
cv2.imwrite('s_th_thresh_tozeroi_1.jpg', thresh5_img1)

cv2.imshow('s_th_thresh_tozeroi_2', thresh5_img2) 
cv2.imwrite('s_th_thresh_tozeroi_2.jpg', thresh5_img2)
"""
threshbo_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print(threshbo_L1)
threshbo_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print(threshbo_L2)
thresh_L1 = cv2.norm(thresh1_img1 - thresh1_img2, cv2.NORM_L1)
print(thresh_L1)

thresh_L2 = cv2.norm(thresh1_img1 - thresh1_img2, cv2.NORM_L2)
print(thresh_L2)


threshibo_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print(threshibo_L1)
threshibo_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print(threshibo_L2)
thresh_L1 = cv2.norm(thresh2_img1 - thresh2_img2, cv2.NORM_L1)
print(thresh_L1)
thresh_L2 = cv2.norm(thresh2_img1 - thresh2_img2, cv2.NORM_L2)
print(thresh_L2)


threshibo_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print(threshibo_L1)
threshibo_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print(threshibo_L2)
thresh_L1 = cv2.norm(thresh3_img1 - thresh3_img2, cv2.NORM_L1)
print(thresh_L1)
thresh_L2 = cv2.norm(thresh3_img1 - thresh3_img2, cv2.NORM_L2)
print(thresh_L2)
"""

threshibo_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print(threshibo_L1)
threshibo_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print(threshibo_L2)
thresh_L1 = cv2.norm(thresh5_img1 - thresh5_img2, cv2.NORM_L1)
print(thresh_L1)
thresh_L2 = cv2.norm(thresh5_img1 - thresh5_img2, cv2.NORM_L2)
print(thresh_L2)

#cv2.imshow('Set to 0 Inverted', thresh5) 
cv2.waitKey(0)
cv2.destroyAllWindows() 