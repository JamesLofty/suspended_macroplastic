#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:09:38 2023

@author: jameslofty
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


folder = '6 - concentration_profiles'

os.listdir(folder)

"""
what experiment do  we want to analysis 
"""

data_files = ['cup_ps_50_V2.xlsx', 'cup_ps_50_V3.xlsx', 'cup_ps_50_V4.xlsx', 'cup_ps_50_V5.xlsx']
rouse_files = ['cup_ps_50_V2rouse.xlsx', 'cup_ps_50_V3rouse.xlsx', 'cup_ps_50_V4rouse.xlsx', 'cup_ps_50_V5rouse.xlsx']
norm_files = ['cup_ps_50_V2norm.xlsx', 'cup_ps_50_V3norm.xlsx', 'cup_ps_50_V4norm.xlsx', 'cup_ps_50_V5norm.xlsx']
rouse1_files = ['cup_ps_50_V2rouse2.xlsx', 'cup_ps_50_V3rouse2.xlsx', 'cup_ps_50_V4rouse2.xlsx', 'cup_ps_50_V5rouse2.xlsx']
rouse2_files = ['cup_ps_50_V2rouse3.xlsx', 'cup_ps_50_V3rouse3.xlsx', 'cup_ps_50_V4rouse3.xlsx', 'cup_ps_50_V5rouse3.xlsx']
rouse3_files = [ 'cup_ps_50_V2rouse4.xlsx', 'cup_ps_50_V3rouse4.xlsx', 'cup_ps_50_V4rouse4.xlsx', 'cup_ps_50_V5rouse4.xlsx']
rouse4_files = [ 'cup_ps_50_V2rouse5.xlsx', 'cup_ps_50_V3rouse5.xlsx', 'cup_ps_50_V4rouse5.xlsx', 'cup_ps_50_V5rouse5.xlsx']

data1_rouse = pd.read_excel(folder + "//" + 'cup_ps_50_V2rouse.xlsx')

zC = []
surfC = []
C = []
C_10 = []
C_90 = []
C_M1 = []
C_M2 = []
ZRouse_true = []
ZRouse_10 = []
ZRouse_90 = []
ZRouse_M1 = []
ZRouse_M2 = []

zC_sus_conc = []
zC_surf_conc = []
zC_bed_conc = []
bins_sus_conc = []
bins_surf_conc = []
bins_bed_conc = []


binscum = []
zC_cum = []

# C_RSS = []
# C_10_RSS = []
# C_90_RSS = []

for data_file, rouse_file, rouse1_file, rouse2_file, rouse3_file, rouse4_file, norm_file in zip(data_files, rouse_files, rouse1_files, rouse2_files, rouse3_files, rouse4_files,norm_files):
    data = pd.read_excel(folder + "//" + data_file)
    rouse = pd.read_excel(folder + "//" + rouse_file)
    rouse2 = pd.read_excel(folder + "//" + rouse1_file)
    rouse3 = pd.read_excel(folder + "//" + rouse2_file)
    rouse4 = pd.read_excel(folder + "//" + rouse3_file)
    rouse5 = pd.read_excel(folder + "//" + rouse4_file)

    # norm = pd.read_excel(folder + "//" + norm_file)
    
    zC.append(data[0])
    surfC.append(data[1])
    C.append(rouse[0])
    C_10.append(rouse[1])
    C_90.append(rouse[2])
    
    C_M1.append(rouse[3])
    C_M2.append(rouse[4])
    
    ZRouse_true.append(rouse2[0])
    ZRouse_10.append(rouse2[1])
    ZRouse_90.append(rouse2[2])
    ZRouse_M1.append(rouse2[3])
    ZRouse_M2.append(rouse2[4])
    binscum.append(rouse3[0])
    zC_cum.append(rouse4[0])
    
    zC_sus_conc.append([rouse5[0]])
    zC_surf_conc.append([rouse5[1]])
    zC_bed_conc.append([rouse5[2]])
    bins_sus_conc.append([rouse5[3]])
    bins_surf_conc.append([rouse5[4]])
    bins_bed_conc.append([rouse5[5]])
    

zC2, zC3, zC4, zC5 = zC
surfC2, surfC3, surfC4, surfC5 = surfC
C2, C3, C4, C5 = C
C2_10, C3_10, C4_10, C5_10 = C_10
C2_90, C3_90, C4_90, C5_90 = C_90

C2_M1, C3_M1, C4_M1, C5_M1 = C_M1
C2_M2, C3_M2, C4_M2, C5_M2 = C_M2

ZRouse_true2, ZRouse_true3, ZRouse_true4, ZRouse_true5 = ZRouse_true
ZRouse_10_2, ZRouse_10_3, ZRouse_10_4, ZRouse_10_5 = ZRouse_10
ZRouse_90_2, ZRouse_90_3, ZRouse_90_4, ZRouse_90_5 = ZRouse_90
ZRouse_M1_2, ZRouse_M1_3, ZRouse_M1_4, ZRouse_M1_5 = ZRouse_M1
ZRouse_M2_2, ZRouse_M2_3, ZRouse_M2_4, ZRouse_M2_5 = ZRouse_M2

binscum2, binscum3, binscum4, binscum5 = binscum
zC_cum2, zC_cum3, zC_cum4, zC_cum5 = zC_cum

zC_sus_conc2, zC_sus_conc3, zC_sus_conc4, zC_sus_conc5 = zC_sus_conc
zC_surf_conc2, zC_surf_conc3, zC_surf_conc4, zC_surf_conc5 = zC_surf_conc
zC_bed_conc2, zC_bed_conc3, zC_bed_conc4, zC_bed_conc5 = zC_bed_conc
bins_sus_conc2, bins_sus_conc3,bins_sus_conc4, bins_sus_conc5 = bins_sus_conc
bins_surf_conc2, bins_surf_conc3, bins_surf_conc4, bins_surf_conc5 = bins_surf_conc
bins_bed_conc2, bins_bed_conc3, bins_bed_conc4, bins_bed_conc5 = bins_bed_conc



# C1_RSS, C2_RSS, C3_RSS, C4_RSS, C5_RSS = C_RSS
# C1_10_RSS, C2_10_RSS, C3_10_RSS, C4_10_RSS, C5_10_RSS = C_10_RSS
# C1_90_RSS ,C2_90_RSS, C3_90_RSS, C4_90_RSS, C5_90_RSS = C_90_RSS

z = data1_rouse[5]

# if "10" in data_files[1]: 
#     a = 5.858327406
# elif "98" in data_files[1]: 
#     a = 5.858869345

BedLev = 36.3# cm
WaterLev = BedLev-27.8 # cm
Hmax = BedLev-WaterLev

if "10" in data_files[1]: 
    Lx = 7.8
    Ly = 7.2    

elif "98" in data_files[1]: 
    Lx = 7.9
    Ly = 7.2    

elif "50" in data_files[1]:
    Lx = 7.9
    Ly = 7.1    

elif "05" in data_files[1]:
    Lx = 4.1
    Ly = 3.5    
    
Lmb = np.sqrt(Lx*Lx+Ly*Ly) # maximum dimension if the particle goes diagonally..
a =  Lmb
a_surf = a/2

# Define a bin si
 
binsize = Lmb/4 # 1.00 # 2.5 # cm
bins=np.arange(-BedLev+WaterLev, 0+binsize, binsize)

# bins = ([-27.8, -26.8, -25.8, -24.8, -23.8, -22.8, -21.8, -20.8, -19.8,
#         -18.8, -17.8, -16.8, -15.8, -14.8, -13.8, -12.8, -11.8, -10.8,
#         -9.8,  -8.8,  -7.8,  -6.8,  -5.8,  -4.8,  -3.8,  -2.8,  -1.8,
#         -0.8,   0.2])

# z = np.array([-26.4639026 , -23.79170781, -21.11951302, -18.44731823,
#        -15.77512344, -13.10292865, -10.43073386,  -7.75853907,
#         -5.08634428,  -2.41414949,   0.2580453 ])


#%%
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, sharex = True, figsize=(8, 2))

ax2.fill_between(zC_sus_conc2[0], bins_sus_conc2[0], step="post",  lw = 1.5, edgecolor = "k", facecolor = "none")
ax3.fill_between(zC_sus_conc3[0], bins_sus_conc3[0], step="post",  lw = 1.5, edgecolor = "k", facecolor = "none")
ax4.fill_between(zC_sus_conc4[0], bins_sus_conc4[0], step="post",  lw = 1.5, edgecolor = "k", facecolor = "none")
ax5.fill_between(zC_sus_conc5[0], bins_sus_conc5[0], step="post",  lw = 1.5, edgecolor = "k", facecolor = "none", label = "Suspended plastic")

ax2.fill_between(zC_surf_conc2[0], bins_surf_conc2[0], step="post", color='r' ,alpha=0.5, edgecolor = "none")
ax3.fill_between(zC_surf_conc3[0], bins_surf_conc3[0], step="post", color='r' ,alpha=0.5, edgecolor = "none")
ax4.fill_between(zC_surf_conc4[0], bins_surf_conc4[0], step="post", color='r' ,alpha=0.5, edgecolor = "none")
ax5.fill_between(zC_surf_conc5[0], bins_surf_conc5[0], step="post", color='r' ,alpha=0.5, edgecolor = "none", label = "Surfaced plastic")

ax2.fill_between(zC_bed_conc2[0], bins_bed_conc2[0], step="post", color='sienna', alpha=0.4, edgecolor = "none")
ax3.fill_between(zC_bed_conc3[0], bins_bed_conc3[0], step="post", color='sienna', alpha=0.4, edgecolor = "none")
ax4.fill_between(zC_bed_conc4[0], bins_bed_conc4[0], step="post", color='sienna', alpha=0.4, edgecolor = "none")
ax5.fill_between(zC_bed_conc5[0], bins_bed_conc5[0], step="post", color='sienna', alpha=0.4, edgecolor = "none", label = "Bed load plastic")



# plt.fill_between(zC_bed_conc, binedges_bed[1:], step="post", color='sienna', alpha=0.4, edgecolor = "none")

# ax1.hist(zC1[surfC1 < 0.5], bins=bins, histtype="step", color="blue", lw = 2, orientation="horizontal")
# ax2.hist(zC2[surfC2 < 0.5], bins=bins, histtype="step", color="blue", lw = 2,orientation="horizontal")
# ax3.hist(zC3[surfC3 < 0.5], bins=bins, histtype="step", color="blue", lw = 2,orientation="horizontal")
# ax4.hist(zC4[surfC4 < 0.5], bins=bins, histtype="step", color="blue", lw = 2,orientation="horizontal")
# ax5.hist(zC5[surfC5 < 0.5], bins=bins, histtype="step", color="blue", lw = 2,orientation="horizontal", label = "Suspended particle count")

# ax1.hist(zC1[surfC1 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5)
# ax2.hist(zC2[surfC2 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5)
# ax3.hist(zC3[surfC3 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5)
# ax4.hist(zC4[surfC4 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5)
# ax5.hist(zC5[surfC5 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5, label = "Surfaced particle count")

ax2.scatter(C2, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown")
ax3.scatter(C3, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown")
ax4.scatter(C4, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown")
ax5.scatter(C5, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown", label = "Rouse profile (based on $\overline{w}$ estimation)")

# ax1.plot(C1_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
# ax2.plot(C2_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
# ax3.plot(C3_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
# ax4.plot(C4_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
# ax5.plot(C5_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)

# ax1.plot(C1_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
# ax2.plot(C2_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
# ax3.plot(C3_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
# ax4.plot(C4_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
# ax5.plot(C5_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7, label = "10 and 90 percentile $w$ estimation")

ax2.scatter(C2_M1, z, lw=1.5, s=18, zorder=1, marker="+",c="dimgray")
ax3.scatter(C3_M1, z, lw=1.5, s=18, zorder=1, marker="+",c="dimgray")
ax4.scatter(C4_M1, z, lw=1.5, s=18, zorder=1, marker="+",c="dimgray")
ax5.scatter(C5_M1, z, lw=1.5, s=18, zorder=1, marker="+",c="dimgray", label = "Rouse profile (based on $w_{m1}$ estimation)")


ax2.scatter(C2_M2, z, lw=1.5, s=18, zorder=1, marker="+",c="steelblue")
ax3.scatter(C3_M2, z, lw=1.5, s=18, zorder=2, marker="+",c="steelblue")
ax4.scatter(C4_M2, z, lw=1.5, s=18, zorder=2, marker="+",c="steelblue")
ax5.scatter(C5_M2, z, lw=1.5, s=18, zorder=1, marker="+",c="steelblue", label = "Rouse profile (based on $w_{m2}$ estimation)")



Ca2 = C2_10[C2_10.notnull() & (C2_10 != 0.0)].iloc[0]
Ca3 = C3_10[C3_10.notnull() & (C3_10 != 0.0)].iloc[0]
Ca4 = C4_10[C4_10.notnull() & (C4_10 != 0.0)].iloc[0]
Ca5 = C5_10[C5_10.notnull() & (C5_10 != 0.0)].iloc[0]

# ax1.vlines(Ca1, bins_bed_conc1[0].iloc[0], a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6)
# ax2.vlines(Ca2, bins_bed_conc2[0].iloc[0], a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6)
# ax3.vlines(Ca3, bins_bed_conc3[0].iloc[0], a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6)
# ax4.vlines(Ca4, bins_bed_conc4[0].iloc[0], a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6)
# ax5.vlines(Ca5, bins_bed_conc5[0].iloc[0], a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6, label = "$C_a$")


ax2.vlines(Ca2, -27.8, a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6)
ax3.vlines(Ca3, -27.8, a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6)
ax4.vlines(Ca4, -27.8, a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6)
ax5.vlines(Ca5, -27.8, a-Hmax, color="sienna", ls="-", lw=2, alpha=0.6, label = "$C_a$")


ax3.set_xlabel("$C_p$ (Count/cm)") # , fontsize=8
ax1.set_ylabel('$z - H$ (cm)') # , fontsize=8

ax1.xaxis.set_ticks(np.arange(0, 16, 5))



for ax in [ax1, ax2, ax3, ax4, ax5]:
    # ax.hlines(0, 0, 60, color="cornflowerblue", lw=1, label = "Free surface")
    # ax.hlines(-27.8, 0, 60, color="k", lw=1, linestyle='--', label = "Roughness bed")
    ax.hlines(0, 0, 60, color="blue", lw=2, label = "Free surface")
    ax.hlines(a-Hmax, a-Hmax, 60, color="sienna", ls="--", lw=1, label = "$a_{bed}$")
    ax.hlines(-a_surf, -a_surf, 60, color="r", alpha = 0.7, ls="--", lw=1, label = "$a_{surf}$")
    ax.hlines(-27.8, 0, 60, color="k", lw=2, linestyle='-', label = "Bed")
    ax.set_xlim(0, 16) 
    ax.set_ylim(-30, 0) 
    
# plt.legend(bbox_to_anchor=(-6, -0.3), loca='upper left', borderaxespad=0, ncol=3)
sns.despine(top=True, right=True, left=False, bottom=False)

plt.savefig("profile.svg", format='svg')
#%%%
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, sharex = True, figsize=(8, 2))

ax2.hist(ZRouse_true2, binscum2, cumulative=True, zorder=6, histtype='step', lw=1.5, color="sandybrown", density=True, orientation='horizontal', alpha = 0.8)
ax3.hist(ZRouse_true3, binscum3, cumulative=True, zorder=6, histtype='step', lw=1.5, color="sandybrown", density=True, orientation='horizontal', alpha = 0.8)
ax4.hist(ZRouse_true4, binscum4, cumulative=True, zorder=6, histtype='step', lw=1.5, color="sandybrown", density=True, orientation='horizontal', alpha = 0.8)
ax5.hist(ZRouse_true5, binscum5, cumulative=True, zorder=6, histtype='step', lw=1.5, color="sandybrown", density=True, label='Rouse (based on $\overline{w}$ estimation)', alpha = 0.8, orientation='horizontal')

# ax1.hist(ZRouse_10_1, binscum1, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')
# ax2.hist(ZRouse_10_2, binscum2, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')
# ax3.hist(ZRouse_10_3, binscum3, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')
# ax4.hist(ZRouse_10_4, binscum4, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')
# ax5.hist(ZRouse_10_5, binscum5, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal', label='Rouse integral(0 and 90 percentile $w$ estimation)')

# ax1.hist(ZRouse_90_1, binscum1, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')
# ax2.hist(ZRouse_90_2, binscum2, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')
# ax3.hist(ZRouse_90_3, binscum3, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')
# ax4.hist(ZRouse_90_4, binscum4, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')
# ax5.hist(ZRouse_90_5, binscum5, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="sandybrown", density=True, orientation='horizontal')

ax2.hist(ZRouse_M1_2, binscum2, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="dimgray", density=True, orientation='horizontal')
ax3.hist(ZRouse_M1_3, binscum3, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="dimgray", density=True, orientation='horizontal')
ax4.hist(ZRouse_M1_4, binscum4, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="dimgray", density=True, orientation='horizontal')
ax5.hist(ZRouse_M1_5, binscum5, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="dimgray", density=True, orientation='horizontal', label='Rouse (based on $w_{m2}$ estimation)',)

ax2.hist(ZRouse_M2_2, binscum2, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="steelblue", density=True, orientation='horizontal')
ax3.hist(ZRouse_M2_3, binscum3, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="steelblue", density=True, orientation='horizontal')
ax4.hist(ZRouse_M2_4, binscum4, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="steelblue", density=True, orientation='horizontal')
ax5.hist(ZRouse_M2_5, binscum5, cumulative=True, histtype='step', lw=1, linestyle='dashed', color="steelblue", density=True, orientation='horizontal', label='Rouse (based on $w_{m1}$ estimation)',)


ax2.hist(zC_cum2, binscum2, density=True, 
                                histtype='step', lw=1.5, color="k",
                                cumulative=True, zorder = 7, orientation='horizontal')
ax3.hist(zC_cum3, binscum3, density=True, 
                                histtype='step', lw=1.5, color="k",
                                cumulative=True, zorder = 7, orientation='horizontal')
ax4.hist(zC_cum4, binscum4, density=True, 
                                histtype='step', lw=1.5, color="k",
                                cumulative=True, zorder = 7, orientation='horizontal')
ax5.hist(zC_cum5, binscum5, density=True, 
                                histtype='step', lw=1.5, color="k",
                                cumulative=True, zorder = 7, orientation='horizontal', label='Suspended particle count')

ax1.set_ylabel("$z-H$ (cm)") # , fontsize=8
ax3.set_xlabel("CDF (-)") # , fontsize=8

# ax1.xaxis.set_ticks(np.arange(-30, 0, 10))


for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.hlines(0, 0, 1, color="blue", lw=1, zorder = 8, label = "Free surface")
    ax.hlines(-a_surf, 0, 1,color="r", alpha = 0.7, zorder = 8,ls="--", lw=1, label = "$a_{surf}$")
    ax.hlines(a-Hmax, 0, 1, color="sienna", ls="--", zorder = 8,lw=1, label = "$a_{bed}$")
    ax.hlines(-27.8, 0, 1, color="k", lw=1,  zorder = 8,linestyle='-', label = "Bed")
    y_ticks = np.arange(-30, 1, 10)
    ax.set_yticks(y_ticks)
    x_ticks = np.arange(0, 1.1, 0.5)
    ax.set_xticks(x_ticks)
sns.despine(top=True, right=True, left=False, bottom=False)

plt.savefig("cdf.svg" , format='svg')
    
# plt.legend(bbox_to_anchor=(-4.5, -0.3), loc='upper left', borderaxespad=0, ncol=2)


#%%
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, sharex = True, figsize=(8, 3))
# ax1.step(zC_norm1, z_norm, where='pre', color='k', linewidth=1)
# ax2.step(zC_norm2, z_norm, where='pre', color='k', linewidth=1)
# ax3.step(zC_norm3, z_norm, where='pre', color='k', linewidth=1)
# ax4.step(zC_norm4, z_norm, where='pre', color='k', linewidth=1)
# ax5.step(zC_norm5, z_norm, where='pre', color='k', linewidth=1)

# ax1.fill_between(surfC_norm1, z_norm, step="pre", alpha=0.5, color="r",)
# ax2.fill_between(surfC_norm2, z_norm, step="pre", alpha=0.5, color="r",)
# ax3.fill_between(surfC_norm3, z_norm, step="pre", alpha=0.5, color="r",)
# ax4.fill_between(surfC_norm4, z_norm, step="pre", alpha=0.5, color="r",)
# ax5.fill_between(surfC_norm5, z_norm, step="pre", alpha=0.5, color="r",)

# ax1.scatter(C_norm1, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")
# ax2.scatter(C_norm2, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")
# ax3.scatter(C_norm3, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")
# ax4.scatter(C_norm4, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")
# ax5.scatter(C_norm5, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")

# ax3.set_xlabel("$C'$") # , fontsize=8
# ax1.set_ylabel('relarive depth (-)') # , fontsize=8

# # ax1.xaxis.set_ticks(np.arange(0, 21, 10))

# # ax1.set_title('V1', loc='center',  y=0.95)
# # ax2.set_title('V2', loc='center',  y=0.95)
# # ax3.set_title('V3', loc='center',  y=0.95)
# # ax4.set_title('V4', loc='center',  y=0.95)
# # ax5.set_title('V5', loc='center',  y=0.95)

# # ax1.set_title(r'$\beta$ = 2.26', loc='center',  y=0.95)
# # ax2.set_title(r'$\beta$ = 1.50', loc='center',  y=0.95)
# # ax3.set_title(r'$\beta$ = 1.08', loc='center',  y=0.95)
# # ax4.set_title(r'$\beta$ = 0.90', loc='center',  y=0.95)
# # ax5.set_title(r'$\beta$ = 0.78', loc='center',  y=0.95)

# # ax1.set_title(r'V1', loc='center',  y=0.95)
# # ax2.set_title(r'V2', loc='center',  y=0.95)
# # ax3.set_title(r'V3', loc='center',  y=0.95)
# # ax4.set_title(r'V4', loc='center',  y=0.95)
# # ax5.set_title(r'V5', loc='center',  y=0.95)


# for ax in [ax1, ax2, ax3, ax4, ax5]:
#     ax.hlines(0, 0, 1, color='cornflowerblue', lw=1)
#     ax.hlines(-1, 0, 1, color='k', lw=1, linestyle='--')
#     ax.set_xlim(0, 0.2) 


# sns.despine(top=True, right=True, left=False, bottom=False)

#%%
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, sharex = True, figsize=(8, 3))
# ax1.step(zC_norm1, z_norm, where='pre', color='k', linewidth=1)
# ax2.step(zC_norm2, z_norm, where='pre', color='k', linewidth=1)
# ax3.step(zC_norm3, z_norm, where='pre', color='k', linewidth=1)
# ax4.step(zC_norm4, z_norm, where='pre', color='k', linewidth=1)
# ax5.step(zC_norm5, z_norm, where='pre', color='k', linewidth=1)

# ax1.fill_between(surfC_norm1, z_norm, step="pre", alpha=0.5, color="r",)
# ax2.fill_between(surfC_norm2, z_norm, step="pre", alpha=0.5, color="r",)
# ax3.fill_between(surfC_norm3, z_norm, step="pre", alpha=0.5, color="r",)
# ax4.fill_between(surfC_norm4, z_norm, step="pre", alpha=0.5, color="r",)
# ax5.fill_between(surfC_norm5, z_norm, step="pre", alpha=0.5, color="r",)

# ax1.scatter(C_norm1, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")
# ax2.scatter(C_norm2, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")
# ax3.scatter(C_norm3, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")
# ax4.scatter(C_norm4, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")
# ax5.scatter(C_norm5, z_norm, lw=2, zorder=5, marker="+",c="sandybrown")

# ax3.set_xlabel("$C'$") # , fontsize=8
# ax1.set_ylabel('relarive depth (-)') # , fontsize=8

# # ax1.xaxis.set_ticks(np.arange(0, 21, 10))

# # ax1.set_title('V1', loc='center',  y=0.95)
# # ax2.set_title('V2', loc='center',  y=0.95)
# # ax3.set_title('V3', loc='center',  y=0.95)
# # ax4.set_title('V4', loc='center',  y=0.95)
# # ax5.set_title('V5', loc='center',  y=0.95)

# # ax1.set_title(r'$\beta$ = 2.26', loc='center',  y=0.95)
# # ax2.set_title(r'$\beta$ = 1.50', loc='center',  y=0.95)
# # ax3.set_title(r'$\beta$ = 1.08', loc='center',  y=0.95)
# # ax4.set_title(r'$\beta$ = 0.90', loc='center',  y=0.95)
# # ax5.set_title(r'$\beta$ = 0.78', loc='center',  y=0.95)

# # ax1.set_title(r'V1', loc='center',  y=0.95)
# # ax2.set_title(r'V2', loc='center',  y=0.95)
# # ax3.set_title(r'V3', loc='center',  y=0.95)
# # ax4.set_title(r'V4', loc='center',  y=0.95)
# # ax5.set_title(r'V5', loc='center',  y=0.95)


# for ax in [ax1, ax2, ax3, ax4, ax5]:
#     ax.hlines(0, 0, 1, color='cornflowerblue', lw=1)
#     ax.hlines(-1, 0, 1, color='k', lw=1, linestyle='--')
#     ax.set_xlim(0, 0.2) 


# sns.despine(top=True, right=True, left=False, bottom=False)


#%%





#%%
# # %%
# plt.figure()
# plt.hist(zC1, bins=bins, histtype="step", orientation="horizontal")
# plt.hist(zC1[surfC1 > 0.5], bins=bins, color="blue", orientation="horizontal")
# plt.ylabel('$z_p - H$ (cm)') # , fontsize=8
# plt.xlabel('count (-)') # , fontsize=8
# plt.xlim(0,22)
# plt.hlines(0, 0, 21, color="b", lw=2)
# plt.hlines(-27.8, 0, 21, color="k", lw=2)
# plt.figure()

# plt.hist(zC1, bins=bins, histtype="step", orientation="horizontal")


# plt.hist(zC2, bins=bins, histtype="step", orientation="horizontal")
# plt.hist(zC3, bins=bins, histtype="step", orientation="horizontal")
# plt.hist(zC4, bins=bins, histtype="step",orientation="horizontal")
# plt.hist(zC5, bins=bins, histtype="step", orientation="horizontal")

# # plt.hist(zC[surfC > 0.5], bins=bins, color="b", orientation="horizontal")
# plt.ylabel('$z_p - H$ (cm)') # , fontsize=8
# plt.xlabel('count (-)') # , fontsize=8
# plt.xlim(0,20)
# plt.hlines(0, 0, 20, color="b", lw=2)
# plt.hlines(-27.8, 0, 20, color="k", lw=2)



# # %%
# zC1p = np.histogram(zC1, bins=bins)
# zC2p = np.histogram(zC2, bins=bins)
# zC3p = np.histogram(zC3, bins=bins)
# zC4p = np.histogram(zC4, bins=bins)
# zC5p = np.histogram(zC5, bins=bins)

# zCp_norm1 = zC1p[0]/len(zC1)
# zCp_norm2 = zC2p[0]/len(zC2)
# zCp_norm3 = zC3p[0]/len(zC3)
# zCp_norm4 = zC4p[0]/len(zC4)
# zCp_norm5 = zC5p[0]/len(zC5)

# plt.figure()
# plt.plot(zCp_norm1, zC1p[1][:len(zC1p[0])])
# plt.plot(zCp_norm2, zC2p[1][:len(zC2p[0])])
# plt.plot(zCp_norm3, zC3p[1][:len(zC3p[0])])
# plt.plot(zCp_norm4, zC4p[1][:len(zC4p[0])])
# plt.plot(zCp_norm5, zC5p[1][:len(zC5p[0])])
# plt.ylabel('count (-)')
# plt.xlabel('$z_p - H$ (cm)')
# plt.legend()

# plt.figure()
# plt.plot(zC1p[0], zC1p[1][:len(zC1p[0])], label='V1')
# plt.plot(zC2p[0], zC2p[1][:len(zC2p[0])], label='V2')
# plt.plot(zC3p[0], zC3p[1][:len(zC3p[0])], label='V3')
# plt.plot(zC4p[0], zC4p[1][:len(zC4p[0])], label='V4')
# plt.plot(zC5p[0], zC5p[1][:len(zC5p[0])], label='V5')

# plt.ylabel('count (-)')
# plt.xlabel('$z_p - H$ (cm)')

# plt.legend()
