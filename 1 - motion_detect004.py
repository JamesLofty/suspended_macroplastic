#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:43:39 2023

@author: jameslofty
"""
#%%
"""
Pro libraries
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
# from EuclideanDistTracker import *
#%%
"""
define lists 
"""
previous_frame = None
tp_flash_list = []
tp_list = []
xp_list = []
yp_list = []
object_ids = []
area_list = []
px_intensity_list = []
frame_ID = 0
surfaced_list = []
rot_list = []
trans_list = []
caps = []
t_end = []

# tracker = EuclideanDistTracker()
#%%
"""
Capture video from file
"""
file  ="example_video/Cam1/GX010391_trim.mp4" #cam1
# file = "example_video/Cam2/GH010578_trim.mp4" #cam2

cap = cv2.VideoCapture(file)
t_end =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("t_end = ", t_end)

#%%
"""
Data saving 
"""
# camera = file[95:96]
folder = "1 - coordinates"
#%%
"""
lets find what video we are analysing
"""

volumes = ["05", "50", "98", "10"]
velocities = ["V1", "V2", "V3", "V4", "V5"]


# for i in volumes:
#     if i in file[85:90]:
#         print("volume =", i)
#         volume = i
        
# for i in velocities:
#     if i in file[85:95]:
#         print("velocty =", i)
#         velocity = i
        
if "Cam1" in file:
    print ("camera = 1")
    camera = "Cam_1_"
    camera_roi = "1"
else:
    print("camera = 2")
    camera = "Cam_2_"
    camera_roi = "2"

volume = "10"
velocity = "V5"


# video = file[100:101]
video = "1"
print ("video = ", video)

file_name = "cup_ps_" + volume +"_"+ velocity +"_"+ camera + video
file_name = file_name + ".xlsx"
print(file_name)

#%%
""""waterline"""

df_waterline = pd.read_excel("waterline.xlsx")


# exp_name = df_waterline["experiment"].tolist()
exp_name = 'cup_PS_10/V5'

for i in exp_name:
    if i in file:
        print(f"Found '{i}' at index {exp_name.index(i)}")
        loc = exp_name.index(i)

waterline_loc = df_waterline.iloc[loc]

waterline_x1 = int(waterline_loc[1])
waterline_x2 = int(waterline_loc[2])
#%%
"""
Read video
define roi
convert to grey scale 
flash sync values
add gaussian blur
background subtraction from prepared and prvious frame
differences in movement between frame
"""

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame_ID += 1
    print(frame_ID)
    
    """ when the video ends, save results"""
    
    if ret == False:
        results = pd.DataFrame(list(zip(tp_list, xp_list, yp_list,
                                        surfaced_list)),
                              columns=['tp', 'xp', 'yp',  
                                        'surfaced'])
        # results.to_excel(folder + "//" + file_name)
        # results.to_excel("heloooo.xlsx")
        print("all data saved !!!")
        break
    
    #roi
    if camera_roi == "1": 
        framesize = np.shape(frame)
        roi = np.zeros(framesize[:2], dtype = 'uint8')
        roi = cv2.rectangle(roi, (0, 0), (2704,1500), (255,0,0), -1)   
    else:
        framesize = np.shape(frame)
        roi = np.zeros(framesize[:2], dtype = 'uint8')
        roi = cv2.rectangle(roi, (0, 0), (2704,1200), (255,0,0), -1)

    frame_roi = cv2.bitwise_and(frame, frame, mask = roi)
                                
    # convert vdieo to RGB
    img_rgb = cv2.cvtColor(src=frame_roi, code=cv2.COLOR_BGR2RGB)
    
    # convert vdieo to gray
    # prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # prepared_frame = img_rgb[:,:,1]
    
    prepared_frame =np.copy(img_rgb)

    
    # average pixel value for flash sync
    px_intensity = np.average(prepared_frame)
    px_intensity_list.append(px_intensity)
    
    #time stamp by frame for flash sync
    tp_flash = frame_ID
    tp_flash_list.append(tp_flash)
    df = pd.DataFrame(list(zip(tp_flash_list, px_intensity_list)),
                          columns=['tp', 'px_intensity'])
 
    # add gaussian blur 
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

    """ We want now to filter the sediment bed that gives poor contrast with the cups"""
    
    # prepared_frame[:,:,2][(prepared_frame[:,:,2] < 150) * (prepared_frame[:,:,0] > 150)*(prepared_frame[:,:,1] < 170)] = 0
    # Apply CLAHE over the blue channel which has larger contrast:
    prepared_frame = clahe.apply(prepared_frame[:,:,2])
    
    # prepared_frame = prepared_frame[:,:,2]
    
    # prepared_frame = cv2.cvtColor(prepared_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the difference between frames
    if (previous_frame is None):
        # First frame; there is no previous one yet
        previous_frame = prepared_frame
        continue

    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame

    # Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((25, 25))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=55, maxval=255, type=cv2.THRESH_BINARY)[1]
    # thresh_frame = cv2.threshold(src=diff_frame, thresh=65, maxval=255, type=cv2.THRESH_BINARY)[1]

    # Find contours
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    detections = []
#%%
    """
    Object tracker
    """ 
    for contour in contours:
        if cv2.contourArea(contour) < 3000:
            # Too small? skip
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        detections.append([x,y,w,h])
 #%%
        """
        coordinates of tracked objects 
        """ 
        # frame ID
        tp = frame_ID
        
        # mean contour points (polygon) detected in x and y
        xp = np.mean(contour[:, 0], axis=0)[0]
        yp = np.mean(contour[:, 0], axis=0)[1]
        area = cv2.contourArea(contour)
    
        # list of tp, xp and yp
        tp_list.append(tp)
        xp_list.append(xp)
        yp_list.append(yp)
        area_list.append(area)
#%%        
        """
        To check if a particle has surfaced
        THIS ONLY WORKS FOR CAM1
        """
        #coordinates for waterline
        waterline = np.array([0, waterline_x1, 2704, waterline_x2])
        #show waterline in video
        cv2.line(img_rgb,(waterline[0],waterline[1]),(waterline[2],waterline[3]),(0,255,0),2)
        # cv2.line(prepared_frame,(waterline[0],waterline[1]),(waterline[2],waterline[3]),(0,255,0),2)

        #highest y coordinate in the contour
        max_contour_yp = np.min(contour[:, 0], axis=0)[1]
        
        x1, y1 = waterline[0], waterline[1]
        x2, y2 = waterline[2], waterline[3]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        if max_contour_yp > m*xp + b:
           surfaced = 0
           col = (0,255, 0)
        else:
            surfaced = 1
            col = (0,0, 255)

        #red boxs surfaced--- green boxesnnt surfaced 
        cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=col, thickness=3)
        
        # print(surfaced)
        surfaced_list.append(surfaced)
#%%
        """
        Store results in a nice dataframe (not saved automatically)
        """
        results_i = pd.DataFrame(list(zip(tp_list, xp_list, yp_list, area_list,
                                        surfaced_list)),
                              columns=['tp', 'xp', 'yp', "area",  
                                        'surfaced'])
#%%
    """
    release video
    """ 
    # thres
    # thresh_frame = cv2.flip(thresh_frame, 0)
    # thresh_frame = cv2.resize(thresh_frame, (800, 800))
    
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 940, 780)

    # # show image
    # cv2.imshow('thres', thresh_frame)
    cv2.imshow('Image', img_rgb)
    
    
    # Wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

#%%
"""
figures and plots
""" 
#print highest px intensity wirh tp
sync_timestamp = df.sort_values(by="px_intensity", ascending=False).head(n=1)
print(sync_timestamp)
plt.figure()
plt.scatter(tp_flash_list, px_intensity_list)

#print end frame number
print("t_end = ", t_end)

print(file_name)

#tracjectory figure
plt.figure()
plt.scatter(tp_list, xp_list)
plt.show()

plt.figure()
plt.scatter(tp_list, xp_list)
plt.xlim(0,7000)
plt.show()

#when surfaced figure
plt.figure()
plt.scatter(tp_list, xp_list, c = surfaced_list)
plt.show()
#%%
"""
SAVE DATA !!!!!
""" 

    


