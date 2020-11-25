# -*- coding: utf-8 -*-
import cv2 #載入opencv模組
import time
import numpy as np
import argparse
from VideoStream import VideoStream#, VideoSave
from numpy import genfromtxt
import find_color


def croppoly(img,p):
    #global points
    pts=np.asarray(p)
    pts = pts.reshape((-1,1,2))
    ##Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.fillPoly(mask,pts=[pts],color=(255,255,255))
    #cv2.copyTo(cropped,mask)
    dst=cv2.bitwise_and(cropped,cropped,mask=mask)
    bg=np.ones_like(cropped,np.uint8)*255 #fill the rest with white
    cv2.bitwise_not(bg,bg,mask=mask)
    dst2=bg+dst
    return dst2


parser = argparse.ArgumentParser()
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720') #1280x720;1920x1080
parser.add_argument('--fps', help='Frames per second',
                    default=15)

args = parser.parse_args()
resW, resH = args.resolution.split('x')
w, h = int(resW), int(resH)
FPS = int(args.fps)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(w,h),framerate=FPS).start()
out=None

time.sleep(1)

while(True):
    
    t1 = cv2.getTickCount()
    
    frame = videostream.read()
    
    frame = cv2.flip(frame, 0)
    cv2.namedWindow('Image frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image frame', 600, 300)
    cv2.imshow("Image frame", np.hstack([frame])) #顯示圖片
        
    img=frame
   
      
    #read points record from file
    dataPath=r'./cutpoints.csv'
    points=genfromtxt(dataPath,delimiter=',').astype(int).tolist()
    
    dst=croppoly(img,points)
    cv2.imwrite("./cut.png", dst)
    filename = cv2.imread("./cut.png")
       
    
    #filename = cv2.imread("./images/429.png")
    color = find_color.get_color(filename)
    #print(color)
    
    imgcheck = cv2.putText(filename, color, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2,
                                (0, 0, 0), 3)
    cv2.imwrite("imgcheck.png", imgcheck)
            
    cv2.namedWindow('Image Process', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image Process', 900, 300)
    cv2.imshow("Image Process", np.hstack([dst,imgcheck])) #顯示圖片
    
    
    
    
    time.sleep(0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
videostream.stop()


