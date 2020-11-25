# -*- coding: utf-8 -*-
import cv2 #載入opencv模組
import numpy as np
import collections
import time
import find_color
from gtts import gTTS
import os

camera = cv2.VideoCapture(1)
#camera = cv2.VideoCapture("Spacer_test.avi")

firstframe = None
a=0
ret0,frame0 = camera.read()
cv2.imwrite("1.png",frame0)
x, y, w, h = 10,10,100,100
    

while True:
    ret, frame = camera.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    a=a+1
    if a%5==0:
        cv2.imwrite("1.png", frame)
    firstframe=cv2.imread("1.png")
    firstframe= cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    firstframe= cv2.GaussianBlur(firstframe, (21, 21), 0)
    frameDelta = cv2.absdiff(firstframe, gray)
    
    #Check motion
    Motion_Check=np.mean(frameDelta)
    print(Motion_Check)
    if Motion_Check > 30: 
        cv2.imshow("cat image", frameDelta) #顯示圖片
        cv2.waitKey(0)                 #等待按下任何按鍵

    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    # cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(thresh)
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("frame", frame)


    try:
            ret0, frame0 = camera.read()
            cropped = frame0[y:y+h,x:x+w ]  # [y0:y1, x0:x1]
            cv2.imwrite("2.png", cropped)
    
            filename = cv2.imread("2.png")
            #filename = cv2.imread("./images/429.png")
            color = find_color.get_color(filename)
            print(color)
            
            

            #plt.title(label[model.predict_classes(image)], fontproperties=myfont)
            imgzi = cv2.putText(frame, color, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2,
                                (255, 255, 255), 2)
            cv2.imwrite("3.png", imgzi)
            cv2.imshow("frame", cv2.imread("3.png"))
            
            
            if color == "gray" or color == "white":
                print("Color is white")
                result = "1"
                #tts = gTTS(text="Color is white", lang='en')
                #tts.save("color.mp3")
                #break
            elif color == "green":
                print("Color is green")
                result = "2"
                #tts = gTTS(text="Color is green", lang='en')
                #tts.save("color.mp3")
                #break
            elif color == "blue":
                print("Color is blue")
                result = "3"
                #tts = gTTS(text="Color is blue", lang='en')
                #tts.save("color.mp3")                
                #break
            
            #os.system("color.mp3")
        
    except:
        pass
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

camera.release()


