# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:55:43 2021

@author: kaust
"""

import cv2 
import numpy as np 
  
# Reading the image from the present directory 
image = cv2.imread("img_1_1.jpg") 
# Resizing the image for compatibility 
image = cv2.resize(image, (500, 600)) 
  
# The initial processing of the image 
# image = cv2.medianBlur(image, 3) 
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# The declaration of CLAHE  
# clipLimit -> Threshold for contrast limiting 
clahe = cv2.createCLAHE(clipLimit = 5) 
final_img = clahe.apply(image_bw) + 30
  
# Ordinary thresholding the same image 
#_, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY) 
  
# Showing all the three images 
#cv2.imshow("ordinary threshold", ordinary_img) 
cv2.imshow("CLAHE image", final_img)

image2 = cv2.imread("img_1_2.jpg") 
# Resizing the image for compatibility 
image2 = cv2.resize(image2, (500, 600)) 
  
# The initial processing of the image 
# image = cv2.medianBlur(image, 3) 
image_bw2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) 
  
# The declaration of CLAHE  
# clipLimit -> Threshold for contrast limiting 
clahe2 = cv2.createCLAHE(clipLimit = 5) 
final_img2 = clahe2.apply(image_bw2) + 30

cv2.imshow("CLAHE image2", final_img2)


original_img_L1 = cv2.norm(image - image2, cv2.NORM_L1)
print("L1 distance of before normalization: ",original_img_L1)


normalized_img_L1 = cv2.norm(final_img - final_img2, cv2.NORM_L1)
print("L1 distance of after normalization: ",normalized_img_L1)


cv2.waitKey(0)
cv2.destroyAllWindows()