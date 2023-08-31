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


data_files = ['cup_ps_05_V1.xlsx', 'cup_ps_05_V2.xlsx', 'cup_ps_05_V3.xlsx', 'cup_ps_05_V4.xlsx', 'cup_ps_05_V5.xlsx']
rouse_files = ['cup_ps_05_V1rouse.xlsx', 'cup_ps_05_V2rouse.xlsx', 'cup_ps_05_V3rouse.xlsx', 'cup_ps_05_V4rouse.xlsx', 'cup_ps_05_V5rouse.xlsx']
norm_files = ['cup_ps_05_V1norm.xlsx', 'cup_ps_05_V2norm.xlsx', 'cup_ps_05_V3norm.xlsx', 'cup_ps_05_V4norm.xlsx', 'cup_ps_05_V5norm.xlsx']


zC = []
surfC = []
C = []
C_10 = []
C_90 = []
C_M1 = []
C_M2 = []

# C_RSS = []
# C_10_RSS = []
# C_90_RSS = []

for data_file, rouse_file, norm_file in zip(data_files, rouse_files, norm_files):
    data = pd.read_excel(folder + "//" + data_file)
    rouse = pd.read_excel(folder + "//" + rouse_file)
    norm = pd.read_excel(folder + "//" + norm_file)
    
    zC.append(data[0])
    surfC.append(data[1])
    C.append(rouse[0])
    C_10.append(rouse[1])
    C_90.append(rouse[2])
    
    C_M1.append(rouse[3])
    C_M2.append(rouse[4])

    
    # C_RSS.append(rouse[3])
    # C_10_RSS.append(rouse[4])
    # C_90_RSS.append(rouse[5])
    
    

zC1, zC2, zC3, zC4, zC5 = zC
surfC1, surfC2, surfC3, surfC4, surfC5 = surfC
C1, C2, C3, C4, C5 = C
C1_10, C2_10, C3_10, C4_10, C5_10 = C_10
C1_90, C2_90, C3_90, C4_90, C5_90 = C_90

C1_M1, C2_M1, C3_M1, C4_M1, C5_M1 = C_M1
C1_M2, C2_M2, C3_M2, C4_M2, C5_M2 = C_M2


data1_rouse = pd.read_excel(folder + "//" + 'cup_ps_05_V1rouse.xlsx')

z = data1_rouse[5]

# bins = np.linspace(-27.8, 0.2, 28)
bins = ([-27.8, -26.8, -25.8, -24.8, -23.8, -22.8, -21.8, -20.8, -19.8,
        -18.8, -17.8, -16.8, -15.8, -14.8, -13.8, -12.8, -11.8, -10.8,
        -9.8,  -8.8,  -7.8,  -6.8,  -5.8,  -4.8,  -3.8,  -2.8,  -1.8,
        -0.8,   0.2])



#%%
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, sharex = True, figsize=(8, 2))
ax1.hist(zC1[surfC1 < 0.5], bins=bins, histtype="step", color="k", orientation="horizontal")
ax2.hist(zC2[surfC2 < 0.5], bins=bins, histtype="step", color="k", orientation="horizontal")
ax3.hist(zC3[surfC3 < 0.5], bins=bins, histtype="step", color="k", orientation="horizontal")
ax4.hist(zC4[surfC4 < 0.5], bins=bins, histtype="step", color="k", orientation="horizontal")
ax5.hist(zC5[surfC5 < 0.5], bins=bins, histtype="step", color="k", orientation="horizontal", label = "Suspended particle count")

ax1.hist(zC1[surfC1 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5)
ax2.hist(zC2[surfC2 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5)
ax3.hist(zC3[surfC3 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5)
ax4.hist(zC4[surfC4 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5)
ax5.hist(zC5[surfC5 > 0.5], bins=bins, color="r", orientation="horizontal", alpha = 0.5, label = "Surfaced particle count")

ax1.scatter(C1, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown")
ax2.scatter(C2, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown")
ax3.scatter(C3, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown")
ax4.scatter(C4, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown")
ax5.scatter(C5, z, lw=1.5, s=18, zorder=5, marker="+",c="sandybrown", label = "Rouse profile (based on median $w$ estimation)")

ax1.plot(C1_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
ax2.plot(C2_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
ax3.plot(C3_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
ax4.plot(C4_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
ax5.plot(C5_10, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)

ax1.plot(C1_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
ax2.plot(C2_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
ax3.plot(C3_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
ax4.plot(C4_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7)
ax5.plot(C5_90, z, lw=1, zorder=5, linestyle='dashed', c="sandybrown", alpha = 0.7, label = "10 and 90 percentile $w$ estimation")

ax1.scatter(C1_M1, z, lw=1, s=15, zorder=1, marker="+",c="dimgray")
ax2.scatter(C2_M1, z, lw=1, s=15, zorder=1, marker="+",c="dimgray")
ax3.scatter(C3_M1, z, lw=1, s=15, zorder=2, marker="+",c="dimgray")
ax4.scatter(C4_M1, z, lw=1, s=15, zorder=2, marker="+",c="dimgray")
ax5.scatter(C5_M1, z, lw=1, s=15, zorder=1, marker="+",c="dimgray", label = "Rouse profile (based on mode 1 $w$ estimation)")

ax1.scatter(C1_M2, z, lw=1, s=15, zorder=1, marker="+",c="steelblue")
ax2.scatter(C2_M2, z, lw=1, s=15, zorder=1, marker="+",c="steelblue")
ax3.scatter(C3_M2, z, lw=1, s=15, zorder=1, marker="+",c="steelblue")
ax4.scatter(C4_M2, z, lw=1, s=15, zorder=1, marker="+",c="steelblue")
ax5.scatter(C5_M2, z, lw=1, s=15, zorder=1, marker="+",c="steelblue", label = "Rouse profile (based on mode 2 $w$ estimation)")



# ax1.scatter(C1_RSS, z, lw=1, s=15, zorder=5, marker="+",c="steelblue")
# ax2.scatter(C2_RSS, z, lw=1, s=15, zorder=5, marker="+",c="steelblue")
# ax3.scatter(C3_RSS, z, lw=1, s=15, zorder=5, marker="+",c="steelblue")
# ax4.scatter(C4_RSS, z, lw=1, s=15, zorder=5, marker="+",c="steelblue")
# ax5.scatter(C5_RSS, z, lw=1, s=15, zorder=5, marker="+",c="steelblue", label = "Rouse profile ($u_*$ from RSS fit)")


# ax1.plot(C1_10_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)
# ax2.plot(C2_10_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)
# ax3.plot(C3_10_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)
# ax4.plot(C4_10_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)
# ax5.plot(C5_10_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)

# # ax1.plot(C1_90_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)
# ax2.plot(C2_90_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)
# ax3.plot(C3_90_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)
# ax4.plot(C4_90_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5)
# ax5.plot(C5_90_RSS, z, lw=1, zorder=5, linestyle='dashed', c="steelblue", alpha = 0.5, label = "10 and 90 percentile $w$ estimation")


ax3.set_xlabel('Count (-)') # , fontsize=8
ax1.set_ylabel('$z_p - H$ (cm)') # , fontsize=8

ax1.xaxis.set_ticks(np.arange(0, 61, 20))

# ax1.set_title('V1', loc='center',  y=0.95)
# ax2.set_title('V2', loc='center',  y=0.95)
# ax3.set_title('V3', loc='center',  y=0.95)
# ax4.set_title('V4', loc='center',  y=0.95)
# ax5.set_title('V5', loc='center',  y=0.95)

# ax1.set_title(r'$\beta$ = 2.26', loc='center',  y=0.95)
# ax2.set_title(r'$\beta$ = 1.50', loc='center',  y=0.95)
# ax3.set_title(r'$\beta$ = 1.08', loc='center',  y=0.95)
# ax4.set_title(r'$\beta$ = 0.90', loc='center',  y=0.95)
# ax5.set_title(r'$\beta$ = 0.78', loc='center',  y=0.95)


# ax1.set_title(r'$\beta$ = 2.64', loc='center',  y=0.95)
# ax2.set_title(r'$\beta$ = 1.75', loc='center',  y=0.95)
# ax3.set_title(r'$\beta$ = 1.27', loc='center',  y=0.95)
# ax4.set_title(r'$\beta$ = 1.05', loc='center',  y=0.95)
# ax5.set_title(r'$\beta$ = 0.92', loc='center',  y=0.95)

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.hlines(0, 0, 30, color="cornflowerblue", lw=1, label = "Free surface")
    ax.hlines(-27.8, 0, 30, color="k", lw=1, linestyle='--', label = "Roughness bed")
    ax.set_xlim(0, 31) 
    
# plt.legend(bbox_to_anchor=(-6.1, -0.3), loc='upper left', borderaxespad=0, ncol=2)


sns.despine(top=True, right=True, left=False, bottom=False)
#%%%

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
