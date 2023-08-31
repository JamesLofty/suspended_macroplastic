# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:53:41 2023
 
@author: Valero & Lofty
"""
 
# Pro libraries:
import os
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib.cm as cm

#%%
# Custom made

def get_vall(df_N1, elevation_refract):
    tN1 = np.asarray(df_N1[["t120", "t110","t100","t90","t80","t70","t60"]])
   
    DtN1 = tN1[:,1:-1] - tN1[:,0:-2] # DtN1 = np.gradient(tN1,axis=1)
    Dx_N1 = elevation_refract[1:-1] - elevation_refract[0:-2] # np.gradient(elevation_refract)
    vall_N1 = np.abs(Dx_N1/DtN1)
    return vall_N1
# %%
# Folders:
folder = "settling velocities"

file = "//Cup_PS_100.xlsx"

cup_100 = pd.read_excel(folder+file)


mode = cup_100["mode"]

# %%
# We need X-correction
xcenter = 95 # center vide point of the camera.
x = np.linspace(120, 60, 7) # markers on the wall
# x = [120, 110, 100, 90, 80, 70, 60]
# indx = 0,   1,  2,  3,  4,  5,  6
alpha = np.arcsin((x-xcenter)/71.5)  # projection angle with the horizontal
alpha_refract = np.arcsin(1*np.sin(alpha) / 1.33 )
elevation_refract = x+12*np.sin(alpha_refract)
 
Dx = elevation_refract[0] - elevation_refract[2] # consider index of each distance
 
# %%
 
Dt_cup_100 = cup_100["t100"] - cup_100["t120"]
Dx_cup_100 = Dx
v_cup_100 = np.abs(Dx_cup_100 / Dt_cup_100)

# %%
# Why not get all velocities?

vall_cup_100 = get_vall(cup_100, elevation_refract)

#%%
# label = cup_100["label"]
plt.figure()
elev_midpoint = (elevation_refract[1:-1] + elevation_refract[0:-2])/2

z_plt=-(elev_midpoint)

vall_cup = []  # Create an empty list
Np, Nt = np.shape(vall_cup_100)

for i in range(0, Np):
    values = vall_cup_100[i, :]  # Get the values for vall_cup_100[i, :]
    vall_cup.append(values)  # Append the values to the list

#%%
plt.figure(figsize=(2,2))

Np = len(vall_cup)  # Number of arrays in vall_cup

for i in range(Np):
    value = mode[i]  # Get the value for color
    if value == 0:
        color = "steelblue"  # Assign a color for value 0
    else:
        color = "dimgray"  # Assign a color for value 1
        
    plt.plot(vall_cup[i], z_plt, color=color, alpha=0.25,lw=1 )


plt.plot(np.nanmean(vall_cup_100, axis=0), z_plt, lw=2, color="k",
          label="Ensembled avg. velocity")


plt.vlines(np.mean(v_cup_100[mode==0]), np.min(z_plt), np.max(z_plt),linewidth = 3,  color="steelblue" , ls=':', label="Median velocity (mode 1)")
plt.vlines(np.mean(v_cup_100[mode==1]), np.min(z_plt), np.max(z_plt), linewidth = 3, color='dimgray', ls=':', label="Median velocity (mode 2)")
plt.vlines(np.mean(v_cup_100), np.min(z_plt), np.max(z_plt), linewidth = 3, color='black', ls=':', label="Median velocity (all)")


plt.xlabel("$w$ (cm/s)")
plt.ylabel("Dis. from waterlevel (cm)")
# plt.legend(loc="center left", bbox_to_anchor=(0.5, 0.8))
plt.ylim(-120, -70)
plt.xlim([0, 4])
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig('lines98.svg', format='svg')


plt.show()

#%%
plt.figure(figsize=(5,4))
 
plt.plot(v_cup_100, ls="None", marker="s", color="k")
plt.hlines(v_cup_100.mean(), 0, 100, ls=":", color="k", label=" avg. velocity") 
plt.xlabel("Exp ID")
plt.ylabel("$w$ (cm/s)")
plt.ylim([0, 4])
plt.legend(loc="best")

sns.despine(top=True, right=True, left=False, bottom=False)


#%%
plt.figure(figsize=(3, 2))

bins1 = round((max(v_cup_100[mode==0]) - min(v_cup_100[mode==0]))/0.2)
bins2 = round((max(v_cup_100[mode==1]) - min(v_cup_100[mode==1]))/0.2)

plt.hist(v_cup_100[mode==0], density=(True), alpha = 0.5, bins = bins1, label = "Mode 1", color="steelblue", ec='black')  
plt.hist(v_cup_100[mode==1], density=(True), alpha = 0.5, bins = bins2, label = "Mode 2" ,color='dimgray', ec='black')  

# mu, sigma = stats.norm.fit(v_cup_100)
# x = np.linspace(0.5, 4, 100)
# y = stats.norm.pdf(x, mu, sigma)/100

# plt.plot(x, y * len(v_cup_100), 'r-')

# plt.vlines(mu, 0, 1, color='k', ls=':', label=" avg. velocity")
plt.vlines(np.mean(v_cup_100[mode==0]), 0, 1.2,linewidth = 3,  color="steelblue" , ls=':', label="Median velocity (mode 1)")
plt.vlines(np.mean(v_cup_100[mode==1]), 0, 1.2, linewidth = 3, color='dimgray', ls=':', label="Median velocity (mode 2)")
plt.vlines(np.mean(v_cup_100), 0, 1.2, linewidth = 3, color='black', ls=':', label="Median velocity (all)")

# plt.vlines(np.median(v_cup_100), 0, 1.2, lw=3, color="k",
#           label="Ensembled avg. velocity")

plt.xlim([0, 5])
plt.ylim([0, 2.5])
plt.xlabel("$w$ (cm/s)")
plt.ylabel("Frequency")

# plt.legend(loc="center left", bbox_to_anchor=(0.5, 0.4))

sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig('hist98.svg', format='svg')

plt.show()

#%%

print("mean v (cm/s)= ", v_cup_100.mean()/100)
print("mean v mode1(cm/s)= ", np.mean(v_cup_100[mode==1])/100)
print("mean v mode2(cm/s)= ", np.mean(v_cup_100[mode==0])/100)




# print("median v mode2(cm/s)= ", np.median(v_cup_100[mode==0])/100)
# print("median v mode2 std(cm/s)= ", np.std(v_cup_100[mode==0])/100)
# print("median v mode1(cm/s)= ", np.median(v_cup_100[mode==1])/100)
# print("median v mode1 std(cm/s)= ", np.std(v_cup_100[mode==1])/100)
# print("median v (cm/s)= ", v_cup_100.median()/100)
# print("median v std(cm/s)= ", v_cup_100.std()/100)
# print("10 percentile=", np.percentile(v_cup_100, 10)/100)
# print("90 percentile=", np.percentile(v_cup_100, 90)/100)

# print("how many in mode 2", len(v_cup_100[mode==0]))
# print("how many in mode 1", len(v_cup_100[mode==1]))

plt.show()

#%%

results = pd.DataFrame(list(zip(v_cup_100, mode)))

# [tp, xp, yp, zp, vpx, vpy, vpz, surfp]
# folder = 'settling_velocity'

filename = folder+file[0:11]+"_result"+".xlsx"

results.to_excel(filename)


print("all data saved")
