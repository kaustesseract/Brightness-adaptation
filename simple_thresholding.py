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
image1 = cv2.imread('img_3_1.jpg')  
image1 = rescaleFrame(image1)
image2 = cv2.imread('img_3_2.jpg')
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
cv2.imshow('s_th_bi_1', thresh2_img1) 
cv2.imwrite('s_th_bi_1.jpg', thresh2_img1)

cv2.imshow('s_th_bi_2', thresh2_img2) 
cv2.imwrite('s_th_bi_2.jpg', thresh2_img2)

"""
threshb_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print("L1 distance of before normalization: ",threshb_L1)
threshb_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print("L2 distance of before normalization: ",threshb_L2)
threshb_L1 = cv2.norm(thresh1_img1 - thresh1_img2, cv2.NORM_L1)
print("L1 distance of after normalization: ",threshb_L1)
threshb_L2 = cv2.norm(thresh1_img1 - thresh1_img2, cv2.NORM_L2)
print("L2 distance of after normalization: ",threshb_L2)

"""
threshbi_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print("L1 distance of before normalization: ",threshbi_L1)
threshbi_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print("L2 distance of before normalization: ",threshbi_L2)
threshbi_L1 = cv2.norm(thresh2_img1 - thresh2_img2, cv2.NORM_L1)
print("L1 distance of after normalization: ",threshbi_L1)
threshbi_L2 = cv2.norm(thresh2_img1 - thresh2_img2, cv2.NORM_L2)
print("L2 distance of after normalization: ",threshbi_L2)

"""
thresht_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print("L1 distance of before normalization: ",thresht_L1)
thresht_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print("L2 distance of before normalization: ",thresht_L2)
thresht_L1 = cv2.norm(thresh3_img1 - thresh3_img2, cv2.NORM_L1)
print("L1 distance of after normalization: ",thresht_L1)
thresht_L2 = cv2.norm(thresh3_img1 - thresh3_img2, cv2.NORM_L2)
print("L2 distance of after normalization: ",thresht_L2)

threshtz_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print("L1 distance of before normalization: ",threshtz_L1)
threshtz_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print("L2 distance of before normalization: ",threshtz_L2)
threshtz_L1 = cv2.norm(thresh4_img1 - thresh4_img2, cv2.NORM_L1)
print("L1 distance of after normalization: ",threshtz_L1)
threshtz_L2 = cv2.norm(thresh4_img1 - thresh4_img2, cv2.NORM_L2)
print("L2 distance of after normalization: ",threshtz_L2)


threshtzi_L1 = cv2.norm(image1 - image2, cv2.NORM_L1)
print("L1 distance of before normalization: ",threshtzi_L1)
threshtzi_L2 = cv2.norm(image1 - image2, cv2.NORM_L2)
print("L2 distance of before normalization: ",threshtzi_L2)
threshtzi_L1 = cv2.norm(thresh5_img1 - thresh5_img2, cv2.NORM_L1)
print("L1 distance of after normalization: ",threshtzi_L1)
threshtzi_L2 = cv2.norm(thresh5_img1 - thresh5_img2, cv2.NORM_L2)
print("L2 distance of after normalization: ",threshtzi_L2)
"""
#cv2.imshow('Set to 0 Inverted', thresh5) 
cv2.waitKey(0)
cv2.destroyAllWindows() 