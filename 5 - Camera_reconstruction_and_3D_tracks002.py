# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:15:00 2023

@author: Moreno-Rodenas / Moreno-Rodenas / Lofty / Valero
"""


# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:26:25 2021 in Delft
@author: Antonio Moreno-Rodenas
Example for a 2-camera 3D positioning of tracks within a water tank
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json 
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
import cv2
##

def L2P_intersect(p_line, v_line, p_plane, n_plane):
    """
    Line to plane intersection
    """
    n_d_u = n_plane.dot(v_line)
    if abs(n_d_u) < 1e-6: 
        return []
    else:
        w = p_line - p_plane
        si = - n_plane.dot(w) / n_d_u
        return np.array([w + si * v_line + p_plane])
    
def ref_ray(i, n_sur, n1, n2):
    """
    3D ray refraction
    """
    i = i/(i**2).sum()**0.5
    n_sur = n_sur/(n_sur**2).sum()**0.5
    r = n1/n2
    c = -n_sur.dot(i)
    return r*i + (r*c - np.sqrt(1 - r**2*(1 - c**2)))*n_sur

def ray_tra(particle_loc, IntrinsicM, cameraPosition, Pose_matrix_camtoworld, p_plane, n_plane, n1, n2):
    """
    Propagation of visual ray
    """
    hom_p = np.array([particle_loc[0], particle_loc[1], 1]) # normalized particle coordinates in frame CRS, [xp,yp,1]
    v_ray_f = np.hstack([np.linalg.inv(IntrinsicM).dot(hom_p), 1]) 
    v_ray_w = np.linalg.inv(Pose_matrix_camtoworld).dot(v_ray_f)[:-1] - np.array(cameraPosition).reshape(3)  # get visual ray vector in world coordinates
    P_watsurf_i = L2P_intersect(np.array(cameraPosition).reshape(3), v_ray_w, p_plane, n_plane) # intersect waterplane
    r_i = ref_ray(v_ray_w , n_plane, n1, n2) # compute the refracted ray

    return v_ray_w, P_watsurf_i, r_i


def ray_intersection(points, dirs):
    """
    Compute intersection between N crossing 3D lines
    """
    dirs_mat = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
    points_mat = points[:, :, np.newaxis]
    I = np.eye(3)
    return np.linalg.lstsq(
        (I - dirs_mat).sum(axis=0),
        ((I - dirs_mat) @ points_mat).sum(axis=0),rcond=None)[0]

ArucoWorld_exterior = pd.read_excel(r"ExtrinsicCalibration/ArUcos_PhaseII.xlsx",
              sheet_name = 'outsideArucos',
              skiprows = [0,1,2],
              usecols=range(16),
              names = ['id','x0', 'y0', 'z0','x4', 'y4', 'z4', 'x3', 'y3', 'z3', 'x2', 'y2', 'z2', 'x1', 'y1', 'z1'], # Aruco coordinates of Biruk are oposite to the aruco convention, so we rename for convenience
              index_col = 'id')

# Import calibration files
#%% CAMERA 1 
Camera = 1
#GET intrinsic Calibration
PathIntCal = r'./IntrinsicCalibration/Camera {0}/CalibrationCAM{0}.json'.format(Camera)
with open(PathIntCal) as json_file:
    Cal = json.load(json_file)
    
IntMatrix1 = np.array(Cal['mtx'])
DistPar1 = np.array(Cal['dist'])

#%% CAMERA 2
Camera = 2
#GET intrinsic Calibration
PathIntCal = r'./IntrinsicCalibration/Camera {0}/CalibrationCAM{0}.json'.format(Camera)
with open(PathIntCal) as json_file:
    Cal = json.load(json_file)
    
IntMatrix2 = np.array(Cal['mtx'])
DistPar2 = np.array(Cal['dist'])


#%% IMPORT DATASETS and synchronization data

# plt.close()

folder = '4 - particle_matching'
# filename = 'Cup_PP_05_V5_d5.xlsx'
filename = 'cup_ps_10_V5.xlsx'

path = folder + '//' + filename

df = pd.read_excel(path)

# ExperimentsPath = '.\input_test_3D_reconstruction'

# ExpList_cam1 = [s for s in os.listdir(ExperimentsPath) if 'Cam1' in s]
# ExpList_cam2 = [s for s in os.listdir(ExperimentsPath) if 'Cam2' in s]

# exp1 = 'ManualDet_Exp_F2_Cam1_01_Transport_test _Cups(+ve)_1_PP_undef_100'
# Replicatei = 5

# exp2 = [s for s in ExpList_cam2 if ('_').join(exp1.split('_')[4:]) in s][0]

# ExpName1 = f'{exp1}_V{Replicatei}.csv'
# ExpName2 = f'{exp2}_V{Replicatei}.csv'

# Camera1 = pd.read_csv(os.path.join(ExperimentsPath, exp1, ExpName1), index_col = [0])
# Camera2 = pd.read_csv(os.path.join(ExperimentsPath, exp2, ExpName2), index_col = [0])

# #% ReSync cameras
# Sync = pd.read_excel(r'./ExtrinsicCalibration/DataFrameSync_Phase2_01_Transport_test _Cups(+ve).xls')

# #%GET FLASH SYNC
# Synked = Sync[Sync['Experiment'] == r'E:\01_Transport_test _Cups(+ve)\1_PP_undef_100\V5'] 


# SynkedDelay = Synked['Sync12'].values[0]
# Camera2['FrameIDSync'] = Camera2['FrameIDSync'] + SynkedDelay 

#% Crop to corresponding detection

""" Split the data """
Camera1 = df[df.camera == 1]
Camera2 = df[df.camera == 2]

Camera1.index = Camera1['tp']
Camera2.index = Camera2['tp']

notnan1 = Camera1[~np.isnan(Camera1['xp'])]['tp'].values
notnan2 = Camera2[~np.isnan(Camera2['xp'])]['tp'].values

notnanjoint = list(notnan1) + list(notnan2)
notnanjointc = np.unique([item for item, count in collections.Counter(notnanjoint).items() if count == 2])

Camera1 = Camera1[~Camera1.index.duplicated(keep='first')].copy()
Camera2 = Camera2[~Camera2.index.duplicated(keep='first')].copy()

Camera1 = Camera1.loc[Camera1.index.intersection(notnanjointc),:]
Camera2 = Camera2.loc[Camera2.index.intersection(notnanjointc),:]

#%%
### 3D Reconstruction
### DEFINE WALL PLANE
W_p1 = ArucoWorld_exterior.iloc[0][['x0', 'y0', 'z0']]
W_p2 = ArucoWorld_exterior.iloc[1][['x0', 'y0', 'z0']]
W_p3 = ArucoWorld_exterior.iloc[2][['x0', 'y0', 'z0']]
W_n = - np.cross(W_p2 - W_p1, W_p3 - W_p1)
W_n = W_n/np.linalg.norm(W_n)

Tracking = []
Trackingid = []
Trackingpart = []
Surfacedtrack = []

list_lbls = Camera1["label"]
list_lbls = list_lbls.append(Camera2["label"])
list_lbls_unique = np.unique(list_lbls) 

for lbl in list_lbls_unique:
    
    print("Reconstructing particle: ", lbl)

    #Iterate over all frames in which lbl is visible:
        
    for framei in tqdm(Camera1[Camera1['label'] == lbl]['tp']):
    
        Trackingid.append(framei)
        Trackingpart.append(lbl)
        
        try: 
            # Query the results that match particle label and framei
            
            Result1 = Camera1[(Camera1['label'] == lbl) & (Camera1['tp'] == framei)]
            Result2 = Camera2[(Camera2['label'] == lbl) & (Camera2['tp'] == framei)]
            
            
            ### get tracked coordinates and camera rot-trans vectors
            Part_loc1 = Result1[['xp', 'yp']].values[0]
            rot1 = Result1[['rot1','rot2','rot3']].values
            tvec1 = Result1[['trans1','trans2','trans3']].values
        
            Part_loc2 = Result2[['xp', 'yp']].values[0]
            rot2 = Result2[['rot1','rot2','rot3']].values
            tvec2 = Result2[['trans1','trans2','trans3']].values
            
            # set pose and position
            camPos1 = -cv2.Rodrigues(rot1)[0].T.dot(tvec1.T)
            Pose_matrix_camtoworld_cam1 = np.vstack([np.hstack([cv2.Rodrigues(rot1)[0], tvec1.T]), [0,0,0,1]])
    
            # Propagate rays
            v_ray_w_1, P_watsurf_i_1, r_i_1 = ray_tra(Part_loc1, IntMatrix1, camPos1, Pose_matrix_camtoworld_cam1, W_p1, W_n, 1.0003, 1.333)
            
            # set pose and position
            # camPos2 = -np.matrix(cv2.Rodrigues(rot2)[0]).T * np.matrix(tvec2)
            camPos2 = -cv2.Rodrigues(rot2)[0].T.dot(tvec2.T)
            Pose_matrix_camtoworld_cam2 = np.vstack([np.hstack([cv2.Rodrigues(rot2)[0], tvec2.T]), [0,0,0,1]])
            
            # Propagate rays
            v_ray_w_2, P_watsurf_i_2, r_i_2 = ray_tra(Part_loc2, IntMatrix2, camPos2, Pose_matrix_camtoworld_cam2, W_p1, W_n, 1.0003, 1.333)
    
            # Intersect
            ObjectIntersected = ray_intersection(np.array([P_watsurf_i_1[0],P_watsurf_i_2[0]]),np.array([r_i_1,r_i_2]))
            
            Tracking.append(ObjectIntersected)
            Surfacedtrack.append(Result1.surfaced.values[0])
            
        except:
            Tracking.append(np.array([[np.nan,np.nan,np.nan]]).T)
            Surfacedtrack.append(np.nan)
#%%
# arrange outputs
Tracking = np.array(Tracking)
TrackingDF = pd.DataFrame(np.hstack([np.array(Trackingid).reshape(len(Trackingid),1), 
                                     np.array(Trackingpart).reshape(len(Trackingpart),1), 
                                     Tracking[:,:,0], 
                                     np.array(Surfacedtrack).reshape(len(Surfacedtrack), 1)]), 
                          columns = ['FrameId', 'label', 'x','y','z','Surfaced'])

#% Visual testing of estimated tracks
Tx = Tracking[:,0]
Ty = Tracking[:,1]
Tz = Tracking[:,2]

fig = plt.figure(figsize = (12, 6))
plt.axis('off')

# plot tracks x coordinate 
ax0 = fig.add_subplot(321)  
plt.plot(Trackingid, Tx, '.')
plt.ylabel('x')

# plot tracks y coordinate 
fig.add_subplot(323, sharex = ax0) 
plt.plot(Trackingid, Ty, '.')
plt.ylabel('y')

# plot tracks z coordinate 
fig.add_subplot(325, sharex = ax0)  
plt.plot(Trackingid, Tz, '.')
plt.ylim(-100,100)
plt.ylabel('z')


# display histograms of traces projected at each coordinate
fig.add_subplot(322)  
plt.hist(Tx, bins = 100)
fig.add_subplot(324)  
plt.hist(Ty, bins = 100)
fig.add_subplot(326)  
plt.hist(Tz, bins = 100)
plt.show()

plt.scatter(Tx, Tz, s=1)
plt.ylim(0,40)

folder = '5 - 3d_coordinates'
filename = filename

path = folder + '//' + filename

TrackingDF.to_excel(path)
print("all data saved")
