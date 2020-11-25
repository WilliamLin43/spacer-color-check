# -*- coding: utf-8 -*-
import cv2 #load opencv
import time
import numpy as np
import argparse
from VideoStream import VideoStream#, VideoSave
from numpy import genfromtxt
import find_color
import configparser
import os

'''
#set raspberry pi GPIO
import RPi.GPIO as gpio
gpio.setmode(gpio.BCM)
gpio.setup(14,gpio.OUT)
'''

# Define and parse input arguments
config=configparser.ConfigParser()
config.read("config.txt")

P_COLOR = str(config["DEFAULT"]["ProcessColor"])#args. which color in process
MOTION_VALUE =  float(config["DEFAULT"]["MotionDetect"])#args. Motion detection value
FRAME_FILTER_VALUE = int(config["DEFAULT"]["FrameFilter"])#args. Frame filter

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
camera = cv2.VideoCapture("spacer_green.avi") 

#firstFrame = None
frame_count=0

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
    
    cv2.namedWindow('Input Images Frame & Detect Area', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Input Images Frame & Detect Area', 500, 300)
    cv2.imshow("Input Images Frame & Detect Area", np.hstack([img2])) #show frame & mark cut area
    
    #check motion process
    gray = cv2.cvtColor(CheckArea, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.Canny(gray, 50, 150) # find Edged
    
    ''' 
    #check motion method 1
    if firstFrame is None:
        firstFrame = gray
        continue

    #frameDelta = cv2.absdiff(firstFrame, gray)
    #Motion_Check=np.mean(gray)
    '''
    #check motion method 2
    Motion_Check=np.sum(gray) #
    #print(Motion_Check)
    cv2.namedWindow('Motion Detection Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Motion Detection Image', 500, 300)
    cv2.imshow("Motion Detection Image", np.hstack([gray])) #show drawing
        
    imgsize = gray.shape #check image size
    pixelssize=imgsize[0]*imgsize[1]*255 #maximun number
        
    if Motion_Check < (pixelssize*MOTION_VALUE): #check cramp or another device over the spacer, base on edge percentage
        #firstFrame = None
        frame_count=frame_count+1 #add more continuous frame static
    else:
        frame_count=0
    
    
    if frame_count > FRAME_FILTER_VALUE: #check how many frame static
        frame_count=0
        #cv2.waitKey(0)             #stop and waiting
  
        #filename = cv2.imread("./images/429.png")
        
        #call find cut image color
        color = find_color.get_color(filename)
        #print(color)
        
        puttingText=""
        
        
        #check color correct or not
        if color == P_COLOR:
            puttingText="Color is correct"
            COLOR_ON=True
            #print("Color is correct")
        else:
            puttingText="Color is incorrect"
            COLOR_ON=False
            #print("Color is incorrect")
    
        #put some text message into image 
        imgcheck2 = cv2.putText(filename, "Setting:"+P_COLOR, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        imgcheck2 = cv2.putText(filename, "Result:"+color, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        imgcheck2 = cv2.putText(filename, puttingText, (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        #save log image
        cv2.imwrite("imgcheck.png", imgcheck2)
                
        cv2.namedWindow('Image Process', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image Process', 500, 300)
        cv2.imshow("Image Process", np.hstack([imgcheck2])) #show check information
    
    
        
        if COLOR_ON:
            #gpio.output(14,gpio.LOW)
            print("Color is correct")
        else:
            #gpio.output(14,gpio.HIGH)
            #os.system("mpg321 ./colorincorrect.mp3 &") #paly warning audio message
            #time.sleep(3) #waiting for play finish
            print("Color is incorrect")
        
    time.sleep(0) #delay a little bit, if require
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
videostream.stop()
#gpio.output(14,gpio.LOW)

