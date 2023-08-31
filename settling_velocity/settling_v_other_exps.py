#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:46:42 2023

@author: jameslofty
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
# import joypy



y0 = 322 # px, point at which the normal from camera intersects grid
m = 0.15/365 # m/px

L = 1.28  # m, distance to the glass from the camera / normal dirc
W = 0.60 # m, flume width


## Here the analysis

folder_path = "settling velocities/05_ps_results"


filename = "settling velocities/cup_05_mode_transport.xlsx"

file = pd.read_excel(filename)


mode = file["mode"]

# List all files in the folder
files = os.listdir(folder_path)

files.sort()

data = {}  # Store the data for each file, using dictionary

vall = []
yall = []
tall = []
filename1 = []

#%%
plt.figure(figsize=(2,2))

# Iterate over each file
for file_name in files:
    # Get the absolute path of the file
    file_path = os.path.join(folder_path, file_name)

    # Check if the file is a .txt file
    if (file_name.endswith(".txt") and len(file_name)>5):
        
        print(file_name)
        
        data_import = np.loadtxt(file_path, delimiter=";")
        t = data_import[:,0]
        x = data_import[:,1]
        y = data_import[:,2]
        
        y = y - y0 # referenced to the cental cross of the grid.
        y = y*m # m
        
        # We correct for refraction
        alpha2 = np.arcsin(y / np.sqrt((L+W)**2+y**2))
        y1 = y - W*np.tan(alpha2)
        y2 = (W/2)*np.tan(alpha2) + y1
        
        y = y2 # This is what we wanted
        
        t = t/30 # we recorded at 30 Hz
        
        # Velocity!!!
        
        vy = np.gradient(y, t)
        
        z_plt = -(y - np.min(y))*100-10

        vall.append(vy)
        filename1.append(file_name)
        yall.append(y)
        tall.append(t)
        
        print(len(vy))


#%%

vall = np.asarray(vall)
yall = np.asarray(yall)
tall = np.asarray(tall)

plt.figure(figsize=(2,2))

Np = len(vall)  # Number of arrays in vall_cup

for i in range(Np):
    value = mode[i]  # Get the value for color
    if value == 0:
        color = "steelblue"  # Assign a color for value 0
    else:
        color = "dimgray"  # Assign a color for value 1
        
    plt.plot(vall[i]*100, z_plt, color=color, alpha=0.25,lw=1 )



plt.plot(np.nanmean(vall, axis=0)*100, z_plt, lw=2, color="k",
          label="Ensembled avg. velocity")

plt.vlines(np.mean(vall[mode==0]*100), np.min(z_plt), np.max(z_plt),linewidth = 3,  color="steelblue" , ls=':', label="Median velocity (mode 1)")
plt.vlines(np.mean(vall*100), np.min(z_plt), np.max(z_plt), linewidth = 3, color='black', ls=':', label="Median velocity (all)")
plt.vlines(np.mean(vall[mode==1]*100), np.min(z_plt), np.max(z_plt), linewidth = 3, color='dimgray', ls=':', label="Median velocity (mode 2)")

# plt.vlines(np.median(vall)*100, np.min(z_plt), np.max(z_plt), color='Red', linewidth = 3,ls=':', label="Median velocity")

plt.xlim([0, 4])
plt.ylabel("Dis. from waterlevel (cm)")
plt.xlabel("$w$ (cm/s)")
# plt.ylabel("Elevation (cm)")
# plt.legend(loc="upper right")
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig('lines50.svg', format='svg')

plt.show()


#%%

bins1 = round((max(vall[:,15][mode==0]*100) - min(vall[:,15][mode==0]*100))/0.2)
bins2 = round((max(vall[:,15][mode==1]*100) - min(vall[:,15][mode==1]*100))/0.2)


plt.figure(figsize=(3, 2))
plt.hist(vall[:,15][mode==0]*100, density=(True), alpha = 0.5, bins = bins1, label = "Mode 1", color="steelblue", ec='black')  
plt.hist(vall[:,15][mode==1]*100, density=(True), alpha = 0.5, bins = bins2, label = "Mode 2" ,color='dimgray', ec='black')  

# plt.vlines(mu, 0, 1, color='k', ls=':', label=" avg. velocity")
plt.vlines(np.mean(vall[:,15][mode==0]*100), 0, 2.5,linewidth = 3,  color="steelblue" , ls=':', label="Median velocity (mode 1)")
plt.vlines(np.mean(vall[:,15][mode==1]*100), 0, 2.5, linewidth = 3, color='dimgray', ls=':', label="Median velocity (mode 2)")
plt.vlines(np.mean(vall[:,15]*100), 0, 2.5, linewidth = 3, color='black', ls=':', label="Median velocity (all)")

# plt.vlines(np.median(v_cup_100), 0, 1.2, lw=3, color="k",
#           label="Ensembled avg. velocity")
sns.despine(top=True, right=True, left=False, bottom=False)


plt.xlim([0, 5])
plt.ylim([0, 2.5])
plt.xlabel("$w$ (cm/s)")
plt.ylabel("Frequency")
plt.savefig('hist05.svg', format='svg')

#%%

print("mean v = ", vall[:,15].mean())
print("mean v mode1= ", np.mean(vall[:,15][mode==1]))
print("mean v mode2= ", np.mean(vall[:,15][mode==0]))


# print("median v = ", np.median(vall))
# print("median v std= ", np.std(vall))

# print("median v mode2= ", np.median(vall[mode==0]))
# print("median v mode2 std= ", np.std(vall[mode==0]))

# print("median v mode1= ", np.median(vall[mode==1]))
# print("median v mode1 std= ", np.std(vall[mode==1]))

# print("10 percentile=", np.percentile(vall, 10))
# print("90 percentile=", np.percentile(vall, 90))

# print("how many in mode 2", len(vall[mode==0]))
# print("how many in mode 1", len(vall[mode==1]))

#%%

# slice1=vall[:,0]
# slice2=vall[:,10]
# slice3=vall[:,15]
# slice4=vall[:,20]


# #%%

# df = pd.DataFrame({'0 slice' : slice1,
#                    '10 slice' : slice2,
#                    '15 slice': slice3,
#                    '20 slice': slice4
#                   })


# melted_df = df.melt(var_name='Variable', value_name='Value')

# colors = ["#FDAE61", "#FEE08B", "#FFFFBF"]

# #%%

# fig, ax = joypy.joyplot(melted_df, by="Variable", linecolor="white",
#                         hist = True, bins = 20, 
#                         fade = True, color = "steelblue",
#                         xlabels=True, figsize=(4, 4))

# plt.xlabel("$w$ (cm/s)")
# # plt.vlines(, 0, 1, color='k', ls=':', label=" avg. velocity")
# fig.tight_layout()
# # plt.savefig('plot.svg', format='svg')

#%%


results = pd.DataFrame(list(zip(vall[:,10]*100)))
folder = "settling velocities"
file = folder_path[20:]
# [tp, xp, yp, zp, vpx, vpy, vpz, surfp]
# folder = 'settling_velocity'

filename = folder+ "//"+file[0:12]+".xlsx"

results.to_excel(filename)


print("all data saved")

