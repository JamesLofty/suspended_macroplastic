#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:19:34 2023

@author: jameslofty
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import math
# import joypy

data = pd.read_excel("plastic_dimensions.xlsx")

plastic =  np.array(data['Plastic'])
L1 = np.array(data['L1 (mm)'])/1000
L2 = np.array(data['L2 (mm)'])/1000
L3 = np.array(data['L3 (mm)'])/1000

angle = 20
angle = math.radians(20)

#%%
"""full"""

area_m1_full = L1[0] * L2[0]
area_m2_full = math.pi*(L2[0]/2)**2#area of a circle

d_m1_full = math.sqrt((4*area_m1_full)/math.pi)
d_m2_full = math.sqrt((4*area_m2_full)/math.pi)
print("d_m1_full", d_m1_full)
print("d_m2_full", d_m2_full)
#%%
"""def"""
area_m1_def = L1[1] * L2[1]
area_m2_def = math.pi*(L2[1]/2)**2#area of a circle

d_m1_def = math.sqrt((4*area_m1_def)/math.pi)
d_m2_def = math.sqrt((4*area_m2_def)/math.pi)
print("d_m1_def", d_m1_def)
print("d_m2_def", d_m2_def)
#%%
"""half"""
area_m1_half = L1[2] * L2[2]
area_m2_half = area_m2_full / 2 #area of a circle / 2

d_m1_half = math.sqrt((4*area_m1_half)/math.pi)
d_m2_half = math.sqrt((4*area_m2_half)/math.pi)
print("d_m1_half", d_m1_half)
print("d_m2_half", d_m2_half)

#%%
"""fragment"""
area_m1_frag = L1[3] * L2[3]
area_m2_frag = L1[3] * L3[3]   #L3 not L2

d_m1_frag = math.sqrt((4*area_m1_frag)/math.pi)
d_m2_frag = math.sqrt((4*area_m2_frag)/math.pi)
print("d_m1_frag", d_m1_frag)
print("d_m2_frag", d_m2_frag)

#%%
"""settling"""
w_full =  0.020731900859312976#MEAN!!!!
wM1_full =  0.017093726826651813
wM2_full =  0.02491193059896622

w_def =  0.01923034646366449#MEAN!!!!
wM1_def =  0.017237521980544054
wM2_def =  0.02221958318834516

w_half=  0.01445801632762716#MEAN!!!!
wM1_half =  0.012697714777368339
wM2_half =  0.02042773462850491


w_frag =  0.014186997567993756#MEAN!!!!
wM1_frag=  0.01216469525154998
wM2_frag=  0.03287183541664675

#%%
"""Reynolds analysis"""
v = 0.000001

Re_full_m1 = (wM1_full * d_m1_full) / v
Re_full_m2 = (wM2_full * d_m2_full) / v

Re_def_m1 = (wM1_def* d_m1_def) / v
Re_def_m2 = (wM2_def * d_m2_def) / v

Re_half_m1 = (wM1_half* d_m1_half) / v
Re_half_m2 = (wM2_half * d_m2_half) / v

Re_frag_m1 = (wM1_frag* d_m1_frag) / v
Re_frag_m2 = (wM2_frag * d_m2_frag) / v
#%%

"""drag coeff"""
p = 1000
pp = 1020
f = 4/3
rho = ((pp-p)/p)
g = 9.81

V_full = 3429.2/1e+9
V_def = 3399.3/1e+9
V_half = 1696.7/1e+9
V_frag = 277.6/1e+9

cd_full_m1 = (2 * (pp-p) * V_full * g) / (p * area_m1_full * wM1_full**2)
cd_full_m2 = (2 * (pp-p) * V_full * g) / (p * area_m2_full * wM2_full**2)

cd_def_m1 = (2 * (pp-p) * V_def * g) / (p * area_m1_def * wM1_def**2)
cd_def_m2 = (2 * (pp-p) * V_def * g) / (p * area_m2_def * wM2_def**2)

cd_half_m1 = (2 * (pp-p) * V_half * g) / (p * area_m1_half * wM1_half**2)
cd_half_m2 = (2 * (pp-p) * V_half * g) / (p * area_m2_half * wM2_half**2)

cd_frag_m1 = (2 * (pp-p) * V_frag * g) / (p * area_m1_frag * wM1_frag**2)
cd_frag_m2 = (2 * (pp-p) * V_frag * g) / (p * area_m2_frag * wM2_frag**2)


# cd_full_m1 = (f * rho * g * d_m1_full)/ wM1_full
# cd_full_m2 = (f * rho * g * d_m2_full) / wM2_full**2

# cd_def_m1 = (f * rho * g * d_m1_def)/ wM1_def**2
# cd_def_m2 = (f * rho * g * d_m2_def)/ wM2_def**2

# cd_half_m1 = f * rho * g * d_m1_half/ wM1_half**2
# cd_half_m2 = f * rho * g * d_m2_half/ wM2_half**2

# cd_frag_m1 = f * rho * g * d_m1_frag / wM1_frag**2
# cd_frag_m2 = f * rho * g * d_m2_frag / wM2_frag**2

#%%
import Standard_Drag_Curve as SDC
# -----------------------------------
def compute_Re(ds, vw, mu):
    Re = ds*vw/mu
    return Re
# ----------------------------------

N = 10000
Re = 10**np.linspace(-2, 6, N)
Cd = np.zeros(N)

for i in range(0, N):
    Cd[i] = SDC.StandardDragCurve(Re[i])
    
#%%    
    
    
# I_full_m1 = (math.pi * pp * L3[0]) / (64 * 1000 * d_m1_full)
# I_full_m2 = (math.pi * pp * L3[0]) / (64 * 1000 * d_m2_full)

# I_def_m1 = (math.pi * pp * L3[1]) / (64 * 1000 * d_m1_def)
# I_def_m2 = (math.pi * pp * L3[1]) / (64 * 1000 * d_m2_def)

# I_half_m1 = (math.pi * pp * L3[2]) / (64 * 1000 * d_m1_half)
# I_half_m2 = (math.pi * pp * L3[2]) / (64 * 1000 * d_m2_half)

# I_frag_m1 = (math.pi * pp * L3[3]) / (64 * 1000 * d_m1_frag)
# I_frag_m2 = (math.pi * pp * L3[3]) / (64 * 1000 * d_m2_frag)

#%%
plt.figure(figsize=(4, 3))
plt.loglog(Re, Cd, 'k', label = "Drag curve for sphere")

plt.scatter(Re_full_m1, cd_full_m1, label = "Full M1", facecolors="none", edgecolors="dimgray", marker="s")
plt.scatter(Re_full_m2, cd_full_m2, label = "Full M2", facecolors="none", edgecolors="steelblue", marker="s")

plt.scatter(Re_def_m1, cd_def_m1, label = "Def M1", facecolors="none", edgecolors="dimgray", marker="^")
plt.scatter(Re_def_m2, cd_def_m2, label = "Def M2", facecolors="none", edgecolors="steelblue", marker="^")

plt.scatter(Re_half_m1, cd_half_m1, label = "Half M1", facecolors="none", edgecolors="dimgray", marker="D")
plt.scatter(Re_half_m2, cd_half_m2, label = "Half M2", facecolors="none", edgecolors="steelblue", marker="D")

plt.scatter(Re_frag_m1, cd_frag_m1, label = "Frag M1", facecolors="none", edgecolors="dimgray", marker="o")
plt.scatter(Re_frag_m2, cd_frag_m2, label = "Frag M2", facecolors="none", edgecolors="steelblue", marker="o")


plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.xlabel('$Re_p$ (-)')
plt.ylabel('$C_D$ (-)')

# plt.grid(True)
sns.despine(top=True, right=True, left=False, bottom=False)

plt.legend(bbox_to_anchor=(-0.01, -0.2), loc='upper left', borderaxespad=0, ncol=2)

plt.xlim(100, 10000)
plt.ylim(0.1, 10)
plt.savefig('drag.svg', format='svg')


# plt.ticklabel_format(

# #%%
# plt.figure(figsize=(3, 5))
# # plt.loglog(Re, Cd, 'k', label = "Drag curve for sphere")

# plt.scatter(Re_full_m1, I_full_m1, label = "Full M1", facecolors="none", edgecolors="dimgray", marker="s")
# plt.scatter(Re_full_m2, I_full_m2, label = "Full M2", facecolors="none", edgecolors="steelblue", marker="s")

# plt.scatter(Re_def_m1, I_def_m1, label = "Def M1", facecolors="none", edgecolors="dimgray", marker="^")
# plt.scatter(Re_def_m2, I_def_m2, label = "Def M2", facecolors="none", edgecolors="steelblue", marker="^")

# plt.scatter(Re_half_m1, I_half_m1, label = "Half M1", facecolors="none", edgecolors="dimgray", marker="D")
# plt.scatter(Re_half_m2, I_half_m2, label = "Half M2", facecolors="none", edgecolors="steelblue", marker="D")

# plt.scatter(Re_frag_m1, I_frag_m1, label = "Frag M1", facecolors="none", edgecolors="dimgray", marker="o")
# plt.scatter(Re_frag_m2, I_frag_m2, label = "Frag M2", facecolors="none", edgecolors="steelblue", marker="o")


# plt.xscale('log')
# plt.yscale('log')

# # Labels and title
# plt.xlabel('$Re$ (-)')
# plt.ylabel('$C_D$ (-)')

# plt.grid(True)
# sns.despine(top=True, right=True, left=False, bottom=False)

# plt.legend(bbox_to_anchor=(0.05, -0.2), loc='upper left', borderaxespad=0, ncol=2)

# plt.xlim(100, 10000)
# plt.ylim(0.001, 0.1)
# # plt.ticklabel_format(

