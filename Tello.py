# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:52:07 2022

@author: Abdalmalek Aburaddaha
"""

from tensorflow.keras.models import Model

from djitellopy import tello 
import time
from time import sleep 
import cv2
import numpy as np

#%%
# To connet to the tello drone, lets create our tello object
tello = tello.Tello()
tello.connect()

# # To make sure that it is running we will print the battery level. 
print(tello.get_battery())

# tello.streamon()
#%% The drone can translate in 3 directions and rotate in 1 direction 
tello.streamon()
tello.takeoff()
# Going up with speed of 25
tello.send_rc_control(0, 0, 23, 0)
time.sleep(2.1)
# tello.send_rc_control(0, 0, 0, 0)
# tello.land()
#%% Run the webcam with open cv

# Loading the face detection model
facetracker = load_model('facetracker.h5')

# Width and height
w, h = 360, 240

# Defined the forward, backward range
fbrange = [6000, 6800]

# P (propotional), I (Integral), D (Derivative) values for our controller
pid = [0.4, 0.4 ,0]

# Previous error value
pError = 0

def faceme(img):
    # faceCas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceCas = facetracker
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    faces = faceCas.detectMultiScale(gray, 1.2, 8)
    
    face_list_c = []
    face_list_A = [] 
    
    # To draw a rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0, 0), 2) 
        # Finding the center of x and y and the area
        x_c = x + w // 2
        y_c = y + h // 2
        area = w * h
        # Showing the center as a circle in the middle of the image
        cv2.circle(img, (x_c, y_c), 5, (0, 0, 255), cv2.FILLED)
        face_list_c.append([x_c, y_c])
        face_list_A.append(area)
        
    # We want to find the maximum area when detecting faces
    if len(face_list_A) !=0:
        i = face_list_A.index(max(face_list_A))
        # Getting the maximum area and the center of that area from i
        return img, [face_list_c[i], face_list_A[i]]  
        
    else:
        return img, [[0, 0], 0]
    
   
#%% Tracking a face
def trackFace(data, w, pid, pError):
    area = data[1]
    x, y = data [0]
    fb_s = 0
    # To calculate the error which is to find how far is our object from the center
    error = x - w//2
    
    # We are basically changing the sensitivity of our error by this value
    # This will eventually give us the speed
    speed = pid[0] * error + pid[1] * (error - pError)
    
    # To restrict the speed of moving over 100 or below -100
    speed = int(np.clip(speed, -100, 100))
    
    # To make the dron stay staionary if it is in the range of view
    if area > fbrange[0] and area < fbrange[1]:
        fb = 0
        
    elif area > fbrange[1]:
        # Then the forward and backward speed will be as follows
        # based on the area seen by the drone
        fb_s = -20
    
    # We add a safty factor that the area should not be 0 if it is less than 6200    
    elif area < fbrange[0] and area != 0:
        fb_s = 20
        
        
    # Do not do anything if you did not detect a face
    if x == 0:
        speed = 0
        error = 0
     
    # print (speed, fb_s)   
        
    tello.send_rc_control(0, fb_s, speed, 0)   
    
    return error
        
    
   
#%%
# cap = cv2.VideoCapture(0)

while True:
    # _, img = cap.read()
    img = tello.get_frame_read().frame
    # Resize the image based on the width and height values
    img = cv2.resize(img, (w,h))
    image, data = faceme(img)
    # Track after detecting the face
    pError = trackFace(data, w, pid, pError)

    print('Center', data[0], 'Area', data[1])
    cv2.imshow('Output', img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break
    
# # After the loop release the cap object
tello.release()
# # Destroy all the windows
cv2.destroyAllWindows()    

#%%
 