# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:18:45 2021

@author: kaust
"""
#https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale) # 1 is the width of image
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

image_1 = cv2.imread('input1.jpg')
image_1 = rescaleFrame(image_1)
auto_result_1, alpha_1, beta_1 = automatic_brightness_and_contrast(image_1)
#print('alpha', alpha_1)
#print('beta', beta_1)
cv2.imshow('cohc_1', auto_result_1)
cv2.imwrite('cohc_1.png', auto_result_1)


image_2 = cv2.imread('input2.jpg')
image_2 = rescaleFrame(image_2)
auto_result_2, alpha_2, beta_2 = automatic_brightness_and_contrast(image_2)
#print('alpha', alpha_2)
#print('beta', beta_2)
cv2.imshow('cohc_2', auto_result_2)
cv2.imwrite('cohc_2.png', auto_result_2)

cohc_original_L1 = cv2.norm(image_1 - image_2, cv2.NORM_L1)
print(cohc_original_L1)
cohc_original_L2 = cv2.norm(image_1 - image_2, cv2.NORM_L2)
print(cohc_original_L2)
cohc_input_L1 = cv2.norm(auto_result_1 - auto_result_2, cv2.NORM_L1)
print(cohc_input_L1)
cohc_input_L2 = cv2.norm(auto_result_1 - auto_result_2, cv2.NORM_L2)
print(cohc_input_L2)



cv2.waitKey(0)
cv2.destroyAllWindows()