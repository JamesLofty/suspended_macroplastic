#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:24:39 2023

@author: jameslofty
"""


#%%
# Pro libraries:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance


#%%
def Dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if Dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret


#%%

file = ("1 - coordinates/cup_ps_10_V5_Cam_2_1.xlsx")
df = pd.read_excel(file)

tp_list = np.asarray(df["tp"])
xp_list = np.asarray(df["xp"])
yp_list = np.asarray(df["yp"])
surfaced_list = np.asarray(df["surfaced"])


tp_cleaned = []
xp_cleaned = []
yp_cleaned = []
surfaced_cleaned = []

tp_unique = np.unique(tp_list)
xpi_list = []
ypi_list =[]
surfacedi_list =[]
a = []

for tpi in tp_unique:
    xpi = xp_list[tp_list == tpi]
    xpi_list.append(xpi)
    
    ypi = yp_list[tp_list == tpi]
    # ypi_list.append(ypi)
    
    surfacedi = surfaced_list[tp_list == tpi]
    # surfacedi_list.append(surfacedi)
    
    Np = len(xpi)
    
    a = []
    for i in range(0,Np):
        a.append((xpi[i], ypi[i]))
        
    
    a_fused = []
    a_fused = fuse(a, 100)
    
        
    surface_flag = 0
    if any(surfacedi == 1):
        surface_flag = 1
    
    for k in range(0, len(a_fused)):
        xp_cleaned.append(a_fused[k][0])
        yp_cleaned.append(a_fused[k][1])
        tp_cleaned.append(tpi)
        surfaced_cleaned.append(surface_flag)
        
plt.scatter(xp_list, yp_list, c = tp_list)
plt.scatter(xp_cleaned, yp_cleaned, c = tp_cleaned)
    
results = pd.DataFrame(list(zip(tp_cleaned, xp_cleaned, yp_cleaned,
                                surfaced_cleaned)),
                      columns=['tp', 'xp', 'yp',  
                                'surfaced'])


folder = "3 - coordinates_cleaned"
# filename = 
results.to_excel(folder + "//" + file[16:])


        