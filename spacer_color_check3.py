# -*- coding: utf-8 -*-
import cv2 #載入opencv模組
import time
import numpy as np
import argparse
from VideoStream import VideoStream#, VideoSave
from numpy import genfromtxt
import find_color
import configparser
import os


# Define and parse input arguments
config=configparser.ConfigParser()
config.read("config.txt")

P_COLOR = str(config["DEFAULT"]["ProcessColor"])#args.which color in process

#print(P_COLOR)

#cut image sub program
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


#camera setting
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

#read images from test video
camera = cv2.VideoCapture("spacer-3.avi") 


while(True):
    
    t1 = cv2.getTickCount()
    #read images from camera
    #frame = videostream.read()
    #frame = cv2.flip(frame, 0) #Rotate image
    
    #read images from test video
    ret, frame = camera.read()
    if not ret:
        break
    
    img=frame
   
    #read points record from file
    dataPath=r'./cutpoints.csv'
    points=genfromtxt(dataPath,delimiter=',').astype(int).tolist()
    
    #cut image & save log
    CheckArea=croppoly(img,points)
    cv2.imwrite("./cut.png", CheckArea)
    filename = cv2.imread("./cut.png")
    
    #Draw point area
    pts=np.asarray(points)
    pts = pts.reshape((-1,1,2)) 
    img2=cv2.polylines(img, [pts], True, (0, 0, 255), 4)
    
    cv2.namedWindow('Image frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image frame', 600, 300)
    cv2.imshow("Image frame", np.hstack([img2])) #show frame & mark cut area
       
    
    #filename = cv2.imread("./images/429.png")
    
    #call find cut image color
    color = find_color.get_color(filename)
    #print(color)
    
    puttingText=""
    
    
    #check color correct or not
    if color == P_COLOR:
        puttingText="Color is correct"
        #print("Color is correct")
    else:
        puttingText="Color is incorrect"
        #os.system("colorincorrect.mp3") #paly warning audio message
        #time.sleep(3) #waiting for play finish
        #print("Color is incorrect")

    #put some text message into image 
    imgcheck1 = cv2.putText(CheckArea, "Setting:"+P_COLOR, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    imgcheck2 = cv2.putText(filename, "Result:"+color, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    imgcheck2 = cv2.putText(filename, puttingText, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
    #save log image
    cv2.imwrite("imgcheck.png", imgcheck2)
            
    cv2.namedWindow('Image Process', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image Process', 900, 300)
    cv2.imshow("Image Process", np.hstack([imgcheck1,imgcheck2])) #show check information


    
    time.sleep(0) #delay a little bit, if require
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
videostream.stop()


