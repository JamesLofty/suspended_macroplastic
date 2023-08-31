#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:02:44 2023

@author: jameslofty
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import cv2.aruco as aruco
import imutils
import pandas as pd
import json 
from tqdm import tqdm
import math

#%%

file  ="example_video/Cam1/GX010391_trim.mp4" #cam1
# file = "example_video/Cam2/GH010578_trim.mp4" #cam2


if "Cam 1" in file:
    Camera = 1
    print("Camera 1")
else:
    Camera = 2
    print("Camera 2")
    

cap = cv2.VideoCapture(file)
fps = cap.get(cv2.CAP_PROP_FPS) 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('FrameCount:', frame_count, 'fps:', fps)


# CREATE AOI MASK
ret, frame = cap.read()
h, w = frame.shape[:2]
frame = cv2.rotate(frame, cv2.ROTATE_180)

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
        
if "Cam 1" in file:
    print ("camera = 1")
    camera = "Cam_1_"
    camera_roi = "1"
else:
    print("camera = 2")
    camera = "Cam_2_"
    camera_roi = "2"


volume = "10"
velocity = "V5"

video = "1"
print ("video = ", video)



file_name = "cup_ps_" + volume +"_"+ velocity +"_"+ camera + video
file_name = file_name + ".xlsx"
print(file_name)

folder = "2 - calibration"

#%% CALIBRATION DATA
#GET intrinsic Calibration
PathIntCal = r'IntrinsicCalibration/Camera {0}/CalibrationCAM{0}.json'.format(Camera)
with open(PathIntCal) as json_file:
    Cal = json.load(json_file)
    
IntMatrix = np.array(Cal['mtx'])
DistPar = np.array(Cal['dist'])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(IntMatrix,DistPar,(w,h),1,(w,h))

# GET extrinsic Calibration
### Aruco parameterization
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000) # loading ARUCO fiduciary marker dictionary (4x4 code)
parameters_aruco =  aruco.DetectorParameters_create()
parameters_aruco.adaptiveThreshWinSizeMin = 100
parameters_aruco.adaptiveThreshWinSizeMax = 200
parameters_aruco.adaptiveThreshConstant = 10

parameters_aruco.minMarkerPerimeterRate = 0.03
parameters_aruco.maxMarkerPerimeterRate = 4.0

parameters_aruco.polygonalApproxAccuracyRate = 0.03

parameters_aruco.minCornerDistanceRate = 0.1
parameters_aruco.minDistanceToBorder = 3
parameters_aruco.minMarkerDistanceRate = 0.1

parameters_aruco.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters_aruco.cornerRefinementWinSize = 5
parameters_aruco.cornerRefinementMaxIterations = 30
parameters_aruco.cornerRefinementMinAccuracy = 0.1

parameters_aruco.markerBorderBits = 1
parameters_aruco.perspectiveRemovePixelPerCell = 8
parameters_aruco.perspectiveRemoveIgnoredMarginPerCell = 0.13
parameters_aruco.maxErroneousBitsInBorderRate = 0.35
parameters_aruco.errorCorrectionRate = 0.6

## Calibrate extrinsics
# import world coordinates
ArucoWorld_exterior = pd.read_excel(r"ExtrinsicCalibration/ArUcos_PhaseII.xlsx",
              sheet_name = 'outsideArucos',
              skiprows = [0,1,2],
              usecols=range(16),
              names = ['id','x0', 'y0', 'z0','x4', 'y4', 'z4', 'x3', 'y3', 'z3', 'x2', 'y2', 'z2', 'x1', 'y1', 'z1'], # Aruco coordinates of Biruk are oposite to the aruco convention, so we rename for convenience
              index_col = 'id')

ArucoWorld_interior = pd.read_excel(r"ExtrinsicCalibration/ArUcos_PhaseII.xlsx",
              sheet_name = 'insideArucos',
              skiprows = [0,1,2],
              usecols=range(16),
              names = ['id','x0', 'y0', 'z0', 'x4', 'y4', 'z4', 'x3', 'y3', 'z3', 'x2', 'y2', 'z2' ,'x1', 'y1', 'z1'],
              index_col = 'id')

ArucoWorld_exterior = ArucoWorld_exterior
ArucoWorld_interior = ArucoWorld_interior

def Get_Extrinsic_frame_to_world_pointsOnlyCentre(ids, corners, WorldCoords):
    """
    Gets the corresponding points in the world and frame CRS for the markers coordinates:[centre, top_left, top_right, bottom_right, bottom_left]
    """
    idsInDic = [int(s.split('_')[1]) for s in WorldCoords.index]
    indexes = [int(np.where(ids == i)[0]) for i in idsInDic if i in ids]
    WorldCoordinates = []
    FrameCoordinates = []
    for ii in indexes:
        FrameCoordinates.append([corners[ii].mean(axis = 1)])
        WorldCoordinates.append(ArucoWorld_exterior.loc[f'id_{ids[ii][0]}', ['x0','y0','z0']].values.reshape(1,3))
    return np.array(WorldCoordinates).reshape(len(indexes),3,1).astype(float), np.array(FrameCoordinates).reshape(len(indexes),2).astype(float)

def Get_Extrinsic_frame_to_world_points(ids, corners, WorldCoords):
    """
    Gets the corresponding points in the world and frame CRS for the markers coordinates:[centre, top_left, top_right, bottom_right, bottom_left]
    """
    
    idsInDic = [int(s.split('_')[1]) for s in WorldCoords.index]
    indexes = [int(np.where(ids == i)[0]) for i in idsInDic if i in ids]
    
    WorldCoordinates = []
    FrameCoordinates = []
    
    for ii in indexes:
        FrameCoordinates.append(np.vstack([corners[ii].mean(axis = 1), corners[ii][0]]))
        WorldCoordinates.append(WorldCoords.loc[f'id_{ids[ii][0]}'].values.reshape(5,3))

    return np.array(WorldCoordinates).reshape(5*len(indexes),3,1).astype(float), np.array(FrameCoordinates).reshape(5*len(indexes),2).astype(float)


def ExtrinsicCalibration(frame_gray, frame_display):
    global aruco_dic, parameters_aruco,Corresponding_W2pixCoords, P2, p, ids, corners
    
    # Extrinsic Calibration
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=parameters_aruco)
    #print(ids)
    
    ids_f = []
    corners_f = []
    
    ids_f_int = []
    corners_f_int = []
    
    for i, id_i in enumerate(ids):
        if id_i in [13,14,15,16,17,11]: # Center Arucos
            pts = corners[i].astype(int).reshape((-1,1,2))
            cv2.circle(frame_display, tuple(pts.mean(axis = 0)[0].astype(int)), 1, (0,255,0), 2)
            #cv2.polylines(frame_display,[pts],True,(0,255,0), 3)
            cv2.putText(frame_display, f'Id_{id_i}', (pts[0,0][0], pts[0,0][1]), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0, 255, 0) , 2, cv2.LINE_AA) 
    
            ids_f.append(id_i)
            corners_f.append(corners[i])
            
        else: # PLOT EXTRA ARUCOS
            pts = corners[i].astype(int).reshape((-1,1,2))
            cv2.circle(frame_display, tuple(pts.mean(axis = 0)[0].astype(int)), 1, (0,255,0), 2)
            #cv2.polylines(frame_display,[pts],True,(0,255,0), 3)
            cv2.putText(frame_display, f'Int_{id_i}', (pts[0,0][0], pts[0,0][1]), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0, 255, 0) , 2, cv2.LINE_AA) 

    for id_jj in [13,14,15,16,17,11,9,12,8,10]:
        if id_jj in [s[0] for s in ids]:
            ids_f_int.append(id_jj)
            pts = np.array(corners)[np.array(ids).T[0] == id_jj].astype(int).reshape((-1,1,2))
            corners_f_int.append(tuple(pts.mean(axis = 0)[0].astype(int)))
        else:
            ids_f_int.append(id_jj)
            corners_f_int.append((np.nan, np.nan))        

    Corresponding_W2pixCoords = Get_Extrinsic_frame_to_world_points(ids, corners, ArucoWorld_exterior)
    
    DistPar = np.array([0.,0.,0.,0.,0.])
    retval, rvec_ran, tvec_ran = cv2.solvePnP(Corresponding_W2pixCoords[0], Corresponding_W2pixCoords[1], IntMatrix, DistPar)

    ## Validate projection
    P2, _ = cv2.projectPoints(np.array(Corresponding_W2pixCoords[0]).astype(float), rvec_ran, tvec_ran, IntMatrix, DistPar)
    
    for p in P2:
        cv2.circle(frame_display,tuple(p.astype(int)[0]), 4, (255,190,0), -1)

    return rvec_ran, tvec_ran, ids_f_int, corners_f_int



#%% Processing:
     
Results = []
    
StartingFrame = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, StartingFrame) # index base 0

FrameIDSync = 0
ObjectDetected = False

while True:
    ret, frame = cap.read()

    if frame is not None:
        None

        try:    
              
            #Undistort
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.undistort(frame, IntMatrix, DistPar, None, newcameramtx)
            frame_display = frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            
            if (FrameIDSync%50 == 0) | (FrameIDSync < 55):
                rot, trans, ids_I, corners_I = ExtrinsicCalibration(frame_gray, frame_display)
            
          
            # Display
            cv2.putText(frame_display, r'Frame {:0}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0),2)  
            frame_display = imutils.resize(frame_display, width = 1000)
            cv2.imshow('Frame', frame_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            Results.append([FrameIDSync, FrameIDSync + StartingFrame, *rot.T[0], *trans.T[0], *corners_I])

            if FrameIDSync == frame_count :
                Results = pd.DataFrame(Results, columns = ['FrameIDSync', 'FrameUDAbs', 'rot1', 'rot2', 'rot3', 'trans1', 'trans2', 'trans3'] + ['pt_{}'.format(i) for i in ids_I])
                Results = pd.DataFrame(Results, columns = ['FrameIDSync', 'FrameUDAbs', 'rot1', 'rot2', 'rot3', 'trans1', 'trans2', 'trans3'])
                Results.to_excel(folder + "//" + file_name)
                print("all data saved !!!")
                break

            FrameIDSync += 1
            print(FrameIDSync)
            
        except Exception as e:
            print(e)
            FrameIDSync += 1
            continue
    else:
        None

cv2.destroyAllWindows()
                                
# Results = pd.DataFrame(Results, columns = ['FrameIDSync', 'FrameUDAbs', 'rot1', 'rot2', 'rot3', 'trans1', 'trans2', 'trans3'] + ['pt_{}'.format(i) for i in ids_I])

# Results.to_excel(folder + "//" + file_name)
#%%

# plt.figure(figsize = (12,8))

# plt.imshow(frame_display[:,:,::-1])
# plt.axis('off')
# plt.savefig('FrameArucos.png')
print(file_name)
#%%
# Results.to_csv('Camera1_Test_ErrorAnalysis.csv')

















