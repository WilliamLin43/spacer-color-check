# -*- coding: utf-8 -*-
from PIL import Image
import cv2 #載入opencv模組
import os
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt

img0 = cv2.imread("./lcd5-spacer/test2.jpg") #讀取圖片, 讀成原始照片

gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #轉灰階 COLOR_BGR2GRAY ; THRESH_BINARY_INV ; THRESH_TRUNC ; THRESH_TOZERO ; THRESH_TOZERO_INV

#cv2.imshow("image gray", gray) #顯示圖片
#cv2.waitKey(0)

#if don't use a floating point data type when computing
#the gradient magnitude image, you will miss edges
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

#display two images in a figure
#cv2.imshow("Edge detection by Laplacaian", np.hstack([lap,gray]))
#cv2.waitKey(0)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))
sobelcombine = cv2.bitwise_or(sobelx,sobely)
#display two images in a figure
#cv2.imshow("Edge detection by Sobel", np.hstack([sobelcombine,gray]))
#cv2.waitKey(0)


#30 and 150 is the threshold, larger than 150 is considered as edge,
#less than 30 is considered as not edge
canny = cv2.Canny(gray, 30, 150)
canny = np.uint8(np.absolute(canny))


gaussian_laplace1 = ndimage.gaussian_laplace(gray, sigma=1.5)
cv2.namedWindow('Egaussian_laplace', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Egaussian_laplace', 900, 300)
cv2.imshow("Egaussian_laplace", np.vstack([gray,gaussian_laplace1]))
cv2.waitKey(0)



#display two images in a figure
cv2.namedWindow('Edge detection by lap,sobel,Canny', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Edge detection by lap,sobel,Canny', 900, 300)
cv2.imshow("Edge detection by lap,sobel,Canny", np.vstack([lap,sobelcombine,canny]))
cv2.waitKey(0)



ret, binary = cv2.threshold(gray, 180, 220, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(gray,180,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(gray,180,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

cv2.namedWindow('Threshold OTSU+BINAR, MEAN_C ,GAUSSIAN', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Threshold OTSU+BINAR, MEAN_C ,GAUSSIAN', 1200, 900)
cv2.imshow("Threshold OTSU+BINAR, MEAN_C ,GAUSSIAN", np.vstack([binary,th2,th3]))
cv2.waitKey(0)



contours1, hierarchy1 = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
contours2, hierarchy2 = cv2.findContours(th2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) 
contours3, hierarchy3 = cv2.findContours(th3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) 
#contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


#draw_img1 = cv2.drawContours(img0.copy(), contours1, -1, (0, 0, 255), 3)
#draw_img2 = cv2.drawContours(img0.copy(), contours2, -1, (0, 0, 255), 3)
#draw_img3 = cv2.drawContours(img0.copy(), contours3, -1, (0, 0, 255), 3)

for i in range(len(contours1)):
    cnt = contours1[i]
    # 計算該輪廓的面積
    area = cv2.contourArea(cnt) 
    # 面積小的都篩選掉
    if(area > 10) or (area < 1000000):
            continue
        

draw_img1 = cv2.drawContours(img0.copy(), contours1, -1, (0, 0, 255), 3)
draw_img2 = cv2.drawContours(img0.copy(), contours2, -1, (0, 0, 255), 3)
draw_img3 = cv2.drawContours(img0.copy(), contours3, -1, (0, 0, 255), 3)



cv2.namedWindow('Draw Contours', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Draw Contours', 1200, 500)
cv2.imshow("Draw Contours",  np.vstack([draw_img1,draw_img2,draw_img3]))
cv2.waitKey(0)                 #等待按下任何按鍵

