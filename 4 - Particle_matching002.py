# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 08:54:13 2023

@author: Lofty!! & Valero
"""

#%%
# Pro libraries:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import os

#%%
# Our functions:
def order_the_labels(tp_cam1, xp_cam1, yp_cam1, surfaced_cam1, labels_cam1):
    """
    This function takes labelled data, in a random order, and renumbers them
    following the stream of tp.
    """    
    
    labels_pos = labels_cam1[labels_cam1>=0]
    tp_pos = tp_cam1[labels_cam1>=0]
    xp_pos = xp_cam1[labels_cam1>=0]
    yp_pos = yp_cam1[labels_cam1>=0]
    surfaced_pos = surfaced_cam1[labels_cam1>=0]
    
    # labels_list = np.arange(500)
    labels_new = []
    # labels_assigned = []
    labels_assigned = np.asarray([], dtype=int)
    
    lbl_k_new = 0
    for i in range(0,len(labels_pos)):
        if labels_pos[i] in labels_assigned:
            
            # I retrieve which is the position within "labels_assigned", = 23
            # Then the new label is: labels.nea.append(23)
            # print("This particle already existed...")
            
            location = labels_assigned == labels_pos[i]
            # the corresponding new label of that old label is: labels_assigned[position][0]
            position = np.argmax(location)        
            labels_new.append(position)
    
        else:
            labels_new.append(lbl_k_new)
            # Keep track:
            # labels_assigned.append(labels_aa[i])
            labels_assigned = np.append(labels_assigned, labels_pos[i])
            # For the next one...
            lbl_k_new = lbl_k_new+1
            
    labels_new = np.asarray(labels_new)            
    
    return tp_pos, xp_pos, yp_pos, surfaced_pos, labels_new

#%%
"""" working folders"""
folder_coords = "3 - coordinates_cleaned"
folder_calibration = "2 - calibration"

"""flash sync files"""
file_sync = "flash_sync.xlsx"
df_sync = pd.read_excel(folder_coords + "//" + file_sync)
# Dt = df_sync["tpstart"][0] - df_sync["tpstart"][1]

"""video length files"""
file_vid_len = "length_vids.xlsx"
df_vid_len = pd.read_excel(folder_coords + "//" + file_vid_len)

"""coordinate and calibration files"""
file_coords = os.listdir(folder_coords)
file_cali = os.listdir(folder_calibration)
#%%
# =============================================================================
"""what video are we analysing ????"""
volume = '10'
velocity = '_V5_'
# =============================================================================
#%%
"""process and open coordinate and calibration files"""
#remove "cali" from files to so file names are the same
file_cali = [y.replace('cali', '') for y in file_cali]
#find matching files
same = set(file_coords) & set(file_cali)
try:
    same.remove('.DS_Store')
except Exception:
    print("No need to remove DS_Store folder...")
    
# Open files which are the same in both folders. store as a directory
df_cali= {}
df_coords = {}
for i in same:
    filecali = pd.read_excel(folder_calibration+"//"+i[0:20]+"cali.xlsx")
    print("Opening file: ", folder_calibration+"//"+i[0:20]+"cali.xlsx")
    filecoords = pd.read_excel(folder_coords+"//"+i)
    print("Opening file: ", folder_coords+"//"+i)
    df_cali[i] = filecali
    df_coords[i] = filecoords

#%%
"""merge coords and cali into one datafraeme set"""
df_merge = {}
for i in same:   
    AA = df_coords[i]    
    BB = df_cali[i]
    CC = pd.merge_asof(AA, BB, left_on = 'tp', right_on='FrameIDSync')
    df_merge[i] = CC
#%%
"""find how many videos are in each camera and extract the file names"""
cam1_videos = []
cam1_filenames = []

cam2_videos = []
cam2_filenames = []
    
for i in same:
    if velocity in i:
        if volume in i:
            if "Cam_1" in i:
                cam1_videos.append(int(i[19:20]))
                cam1_filenames.append(i[0:20])
            elif "Cam_2" in i:
                cam2_videos.append(int(i[19:20]))
                cam2_filenames.append(i[0:20])
             
cam1_filenames.sort(key=lambda x: int(x[-1]))
cam2_filenames.sort(key=lambda x: int(x[-1]))

cam1_filenames = [file + '.xlsx' for file in cam1_filenames]
cam2_filenames= [file + '.xlsx' for file in cam2_filenames]

print("___how many videos in camera 1____ =", len(cam1_videos))
print("filenames in camera 1 =", cam1_filenames)
print("___how many videos in camera 2____ =", len(cam2_videos))
print("filenames in camera 2 =", cam2_filenames)

#%%
"""
get length of video data 
"""
tend_cam1 = []
tend_cam2 = []

for i in range(0, len(df_vid_len)):
    if velocity in df_vid_len["Experiment"][i]:
        if volume in df_vid_len["Experiment"][i]:
            if "Cam_1" in df_vid_len["Experiment"][i]:
                print("tend cam 1=", df_vid_len["tend"][i])
                tend_cam1.append(df_vid_len["tend"][i])
            
            elif "Cam_2" in df_vid_len["Experiment"][i]:
                print("tend cam 2=", df_vid_len["tend"][i])
                tend_cam2.append(df_vid_len["tend"][i])
                
#%%

for i in range(0, len(df_sync)):
    if velocity in df_sync["Experiment"][i]:
        if volume in df_sync["Experiment"][i]:
            if "Cam 1" in df_sync["Experiment"][i]:
                print("flash_cam1=", df_sync["tpstart"][i])
                flash_cam1 = (df_sync["tpstart"][i])
            
            elif "Cam 2" in df_sync["Experiment"][i]:
                print("flash_cam2=", df_sync["tpstart"][i])
                flash_cam2 = (df_sync["tpstart"][i])

Dt = flash_cam1-flash_cam2
print(Dt)
#%%
"""append videos from camera 1 together"""

df_cam1 = pd.DataFrame()
tp_shift = 0

for i in range(len(cam1_videos)):
    filename = cam1_filenames[i]
    video_df = df_merge[filename]
    video_df['tp'] += tp_shift
    
    df_cam1 = pd.concat([df_cam1, video_df], ignore_index=True)
    
    tp_shift += tend_cam1[i]
    
print(f"total number of videos in cam 1: {len(cam1_videos)}")

# if len(cam1_videos) == 1:
#     print("only one video")
#     df_cam1 = df_merge[cam1_filenames[0]]
    
# elif len(cam1_videos) == 2:
#     print("only two videos")
#     df_merge[cam1_filenames[1]]["tp"] = df_merge[cam1_filenames[1]]["tp"] + tend_cam1[0]   
    
#     df_cam1 = df_merge[cam1_filenames[0]]
#     df_cam1 = df_cam1.append(df_merge[cam1_filenames[1]], ignore_index=True)
    
# elif len(cam1_videos) == 3:
#     df_merge[cam1_filenames[1]]["tp"] = df_merge[cam1_filenames[1]]["tp"] + tend_cam1[0]
#     df_merge[cam1_filenames[2]]["tp"] = df_merge[cam1_filenames[2]]["tp"] + tend_cam1[0] + tend_cam1[1]
       
#     df_cam1 = df_merge[cam1_filenames[0]]
#     df_cam1 = df_cam1.append(df_merge[cam1_filenames[1]], ignore_index=True)
#     df_cam1 = df_cam1.append(df_merge[cam1_filenames[2]], ignore_index=True)

# elif len(cam1_videos) == 4:
#     df_merge[cam1_filenames[1]]["tp"] = df_merge[cam1_filenames[1]]["tp"] + tend_cam1[0]
#     df_merge[cam1_filenames[2]]["tp"] = df_merge[cam1_filenames[2]]["tp"] + tend_cam1[0] + tend_cam1[1]
#     df_merge[cam1_filenames[3]]["tp"] = df_merge[cam1_filenames[3]]["tp"] + tend_cam1[0] + tend_cam1[1] + tend_cam1[2]

#     df_cam1 = df_merge[cam1_filenames[0]]
#     df_cam1 = df_cam1.append(df_merge[cam1_filenames[1]], ignore_index=True)
#     df_cam1 = df_cam1.append(df_merge[cam1_filenames[2]], ignore_index=True)
#     df_cam1 = df_cam1.append(df_merge[cam1_filenames[3]], ignore_index=True)

"""append videos from camera 2 together"""

df_cam2 = pd.DataFrame()
tp_shift = 0

for i in range(len(cam2_videos)):
    filename = cam2_filenames[i]
    video_df = df_merge[filename]
    video_df['tp'] += tp_shift
    
    df_cam2 = pd.concat([df_cam2, video_df], ignore_index=True)
    
    tp_shift += tend_cam2[i]
    
print(f"total number of videos in cam2: {len(cam2_videos)}")


# if len(cam2_videos) == 1:
#     print("only one video")
#     df_cam2 = df_merge[cam2_filenames[0]]
    
# elif len(cam2_videos) == 2:
#     print("only two videos")
#     df_merge[cam2_filenames[1]]["tp"] = df_merge[cam2_filenames[1]]["tp"] + tend_cam2[0]   
    
#     df_cam2 = df_merge[cam2_filenames[0]]
#     df_cam2 = df_cam2.append(df_merge[cam2_filenames[1]], ignore_index=True)
    
# elif len(cam2_videos) == 3:
#     df_merge[cam2_filenames[1]]["tp"] = df_merge[cam2_filenames[1]]["tp"] + tend_cam2[0]
#     df_merge[cam2_filenames[2]]["tp"] = df_merge[cam2_filenames[2]]["tp"] + tend_cam2[0] + tend_cam2[1]
       
#     df_cam2 = df_merge[cam2_filenames[0]]
#     df_cam2 = df_cam2.append(df_merge[cam2_filenames[1]], ignore_index=True)
#     df_cam2 = df_cam2.append(df_merge[cam2_filenames[2]], ignore_index=True)

# elif len(cam2_videos) == 4:
#     df_merge[cam2_filenames[1]]["tp"] = df_merge[cam2_filenames[1]]["tp"] + tend_cam2[0]
#     df_merge[cam2_filenames[2]]["tp"] = df_merge[cam2_filenames[2]]["tp"] + tend_cam2[0] + tend_cam2[1]
#     df_merge[cam2_filenames[3]]["tp"] = df_merge[cam2_filenames[3]]["tp"] + tend_cam2[0] + tend_cam2[1] + tend_cam2[2]

#     df_cam2 = df_merge[cam2_filenames[0]]
#     df_cam2 = df_cam2.append(df_merge[cam2_filenames[1]], ignore_index=True)
#     df_cam2 = df_cam2.append(df_merge[cam2_filenames[2]], ignore_index=True)
#     df_cam2 = df_cam2.append(df_merge[cam2_filenames[3]], ignore_index=True)

df_cam2["tp"] = df_cam2["tp"]+Dt ######bloody DT!!!!!!

plt.figure()
plt.plot(df_cam1["tp"])
plt.plot(df_cam2["tp"])
plt.title("tp merging been successful?")
plt.legend(["cam1", "cam2"])
plt.xlabel("tp")
plt.show()

#%%
"""
as the flash from the videos interfears wit the labelling, we cut each video
manually 


tpCutFrom needs to be altered mannually 
"""
tpCutFrom = 4800
trim1 = df_cam1[df_cam1["tp"]>tpCutFrom]
trim2 = df_cam2[df_cam2["tp"]>tpCutFrom]

cam1_all_trim = pd.DataFrame({
    'tp': trim1['tp'],
    'xp': trim1['xp'],
    'yp': trim1['yp'],
    'surfaced': trim1['surfaced'],
    'rot1': trim1['rot1'],
    'rot2': trim1['rot2'],
    'rot3': trim1['rot3'],
    'trans1': trim1['trans1'],
    'trans2': trim1['trans2'],
    'trans3': trim1['trans3']
})
cam1_all_trim['camera'] = 1

cam2_all_trim = pd.DataFrame({
    'tp': trim2['tp'],
    'xp': trim2['xp'],
    'yp': trim2['yp'],
    'surfaced': trim2['surfaced'],
    'rot1': trim2['rot1'],
    'rot2': trim2['rot2'],
    'rot3': trim2['rot3'],
    'trans1': trim2['trans1'],
    'trans2': trim2['trans2'],
    'trans3': trim2['trans3']
})
cam2_all_trim['camera'] = 2
#%%
dfs = [cam1_all_trim, cam2_all_trim]
all_cams_df = pd.concat(dfs)

tp_all = np.asarray(all_cams_df["tp"])
xp_all = np.asarray(all_cams_df["xp"])
yp_all = np.asarray(all_cams_df["yp"])
surfaced_all = np.asarray(all_cams_df["surfaced"])
cam_all_raw = np.asarray(all_cams_df["camera"])

# lets plot to make sure everything is goooood
plt.figure()
plt.scatter(tp_all[cam_all_raw == 1], xp_all[cam_all_raw == 1], c="r", label="cam 1")
plt.scatter(tp_all[cam_all_raw == 2], xp_all[cam_all_raw == 2], c="b", label="cam 2")
plt.legend()
plt.title("raw data")
plt.xlabel("tp")
plt.ylabel("xp")  # orginal data

#%%%
"""
Orthogonal rotation of data to make particle tracjectories clear to indentify
"""
Delta_xp = np.gradient(xp_all)
Delta_tp = np.gradient(tp_all)

# We calculate the angle of our data
theta = np.arctan2(np.median(Delta_xp), np.median(Delta_tp))

# We rotate against the angle.
theta = -theta

# Rotation matrix
R = np.zeros([2, 2])
R[0, 0] = np.cos(theta)
R[0, 1] = -np.sin(theta)
R[1, 0] = np.sin(theta)
R[1, 1] = np.cos(theta)

# Create vector to be rotated
vec_tp_xp = np.asarray([tp_all, xp_all])

# product of two arrays (rotation matrix and data vector)
ptxstar = np.dot(R, vec_tp_xp)

# re define tp and xp
tpx = ptxstar[1, :]
xpx = ptxstar[0, :]

#let plot the rotated data 
plt.figure()
plt.scatter(tpx[cam_all_raw == 1], xpx[cam_all_raw == 1], c="r", label="cam 1")
plt.scatter(tpx[cam_all_raw == 2], xpx[cam_all_raw == 2], c="b", label="cam 2")
plt.title("rotated data")
plt.legend()
plt.xlabel("tp*")
plt.ylabel("xp*")
plt.show()

#%%%
"""
Clustering of data to label and id each tracjectory in a both videos. This is done
easily as the trajcectories have been rotated to align vertically (tp)

Difficult part - estimating how many labels to assign to the data i.e how many
objects were detected in the video (n-conponents).

To do this we take three characteristic positions of the particles during their
travel and count how many particles there are. Some particles travel together
but, hopefully, they may separate while travelling and then we identify them
either at: 1- entry 2- middle 3- exit

We take the largest amount of particles identied from the 3 positions
as our n-conponent value
"""

#lets plot where the cutting location will be 
plt.figure()
rotated_data_fig = plt.scatter(tpx, xpx)
plt.title("cut locations")
plt.xlabel("tp*")
plt.ylabel("xp*")  # rotated data

jump = 140 # threshold

# 1- entry: ------------------------------------------------------------------
tpxcut = []
xpxcut = []

for i in range(0, len(tpx)):
    if ((1900 + -tpx[i]*np.cos(theta)) < xpx[i]) and (2300 + -tpx[i]*np.cos(theta) > xpx[i]): 
        tpxcut.append(tpx[i])
        xpxcut.append(xpx[i])

tpxcut = np.asarray(tpxcut)
xpxcut = np.asarray(xpxcut)

countP1 = 0
for i in range(1, len(tpxcut)):
    if np.abs(tpxcut[i] - tpxcut[i-1]) > jump:
        countP1 += 1
        #print(tpxcut[i-1])
        plt.scatter(tpxcut[i-1], xpxcut[i-1], c="r", marker="+")
countP1 += 1
       
tpxlist = np.linspace(min(tpx),max(tpx))
plt.plot(tpxlist, 1900 + -tpxlist*np.cos(theta), "k:", c = "r")
plt.plot(tpxlist, 2300 + -tpxlist*np.cos(theta), "k:", c = "r")
print(countP1)

# 2- middle ------------------------------------------------------------------
tpxcut = []
xpxcut = []

for i in range(0, len(tpx)):
    if ((1300 + -tpx[i]*np.cos(theta)) < xpx[i]) and (1650 + -tpx[i]*np.cos(theta) > xpx[i]): 
        tpxcut.append(tpx[i])
        xpxcut.append(xpx[i])
        
tpxcut = np.asarray(tpxcut)
xpxcut = np.asarray(xpxcut)

countP2 = 0
for i in range(1, len(tpxcut)):
    if np.abs(tpxcut[i] - tpxcut[i-1]) > jump:
        countP2 += 1
        plt.scatter(tpxcut[i-1], xpxcut[i-1], c="black", marker="+")
countP2 += 1
plt.plot(tpxlist, 1650 + -tpxlist*np.cos(theta), "k:")
plt.plot(tpxlist, 1300 + -tpxlist*np.cos(theta), "k:")
print(countP2)

# 3- exit------------------------------------------------------------------
tpxcut = []
xpxcut = []

for i in range(0, len(tpx)):
    if ((100 + -tpx[i]*np.cos(theta)) < xpx[i]) and (400 + -tpx[i]*np.cos(theta) > xpx[i]): 
        tpxcut.append(tpx[i])
        xpxcut.append(xpx[i])

tpxcut = np.asarray(tpxcut)
xpxcut = np.asarray(xpxcut)

countP3 = 0
for i in range(1, len(tpxcut)):
    if np.abs(tpxcut[i] - tpxcut[i-1]) > jump:
        countP3 += 1
        plt.scatter(tpxcut[i-1], xpxcut[i-1], c="c", marker="+")
countP3 += 1
plt.plot(tpxlist, 100 + -tpxlist*np.cos(theta), "k:", c = "c")
plt.plot(tpxlist, 400 + -tpxlist*np.cos(theta), "k:", c = "c")
print(countP3)
plt.show()

#%%
"""
Here we choose the best n-conponent for our gaussian mixture model clustering
"""
# choose n conponents based on highest number of tracjectories identidied
# n_components = np.min([countP1, countP2, countP3])
# n_components = np.max([countP1, countP2, countP3])
n_components = 144

"""
Here we do the gaussian mixture model clustering to label trajectories 
"""
# define X as rotated tp (tpx) - only as data is rotated, we only consider
# tp as a the clustering vraiable
X = np.zeros([len(tp_all), 1])
X[:, 0] = tpx

# gaussian mixture model clustering
gmm = GMM(n_components=n_components,covariance_type='full', random_state=8).fit(X)
labels_all_gmm = gmm.predict(X)

#lets plot the data with unordered labels
plt.figure()
clustered_data = plt.scatter(tpx, xpx, c=labels_all_gmm)
plt.title("clustered data - unordered labels")
plt.xlabel("tp")
plt.ylabel("xp")  
plt.show()

#lets see if the clustering was successful in joining cam 1 and cam 2 trajectories
#by plotting a small section of the data 
plt.figure(figsize=(4,4))
plt.scatter(cam1_all_trim["tp"][cam1_all_trim["tp"]<7000], cam1_all_trim["xp"][cam1_all_trim["tp"]<7000], c="r", label="cam 1")
plt.scatter(cam2_all_trim["tp"][cam2_all_trim["tp"]<7000], cam2_all_trim["xp"][cam2_all_trim["tp"]<7000], c="b", label="cam 2")
# plt.title("section of data with trajectories coloured by camera number")
plt.legend()
plt.xlabel("tp")
plt.ylabel("xp") 
plt.show()

#fingers crossed, trajectories from cam 1 and cam 2 are clustered with the same label
plt.figure(figsize=(4,4))
# plt.scatter(tp_all[(tp_all>30000)&(tp_all<35000)], xp_all[(tp_all>30000)&(tp_all<35000)], c=labels_all_gmm[(tp_all>30000)&(tp_all<35000)])
plt.scatter(tp_all[tp_all<7000], xp_all[tp_all<7000], c=labels_all_gmm[tp_all<7000])
# plt.title("section of data with trajectories coloured by trajectory label")
plt.xlabel("tp")
plt.ylabel("xp") 
plt.show

#%%
"""
Here we take the unodered labels and order them by tp starting at 1
"""

tp_all, xp_all, yp_all, surfaced_all, labels_all_gmm = \
    order_the_labels(tp_all, xp_all, yp_all, surfaced_all, labels_all_gmm)
#%%
# =============================================================================
"""RESULTS"""
all_cams_df["label"] = labels_all_gmm
results = all_cams_df



folder = '4 - particle_matching'
# filename = 'Cup_PP_05_V5_d5.xlsx'

filename = cam1_filenames[1][0:12] + '.xlsx'
# filename = "hello.xlsx"
# filename = 'cups_ps_v1_100.xlsx'

# folder = 'Films_trajectories'
# filename = 'Film_HDPE_15_V5_d5.csv'

path = folder + "//" + filename

print(path)
all_cams_df.to_excel(path)
print("all data saved")
# =============================================================================

#lets plot the final results 
plt.figure()
plt.scatter(tp_all, xp_all, c=labels_all_gmm)
plt.title("clustered data - ordered labels")
plt.xlabel("tp")
plt.ylabel("xp") 
plt.show

plt.figure()
# plt.scatter(tp_all[tp_all<7000], xp_all[tp_all<7000], c=labels_all_gmm[tp_all<7000])
plt.scatter(tp_all[(tp_all>30000)&(tp_all<35000)], xp_all[(tp_all>30000)&(tp_all<35000)], c=labels_all_gmm[(tp_all>30000)&(tp_all<35000)])
plt.title("section of clustered data - ordered labels")
plt.xlabel("tp")
plt.ylabel("xp") 
plt.show
