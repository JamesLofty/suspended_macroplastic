# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:06:56 2021
Updated on 07.06.2023

This code analyzes particles concentrations extracted from 3D trajectory
reconstructions.

@author: Daniel Valero, James Lofty
"""

#import os, glob
import numpy as np
import scipy.stats as stats
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kstest


"""
- bin size the same as the size of the particle 
 - bed layer size of a cup
- a = maximun dimension of the cup 
 - 3D coloured by startign position? gradient makers? 
"""

# In-house libraries
import Concentration_profiles_faux as cp

# Constants definition -------------------------------------------------------
# Geom
BedLev = 36.3# cm
WaterLev = BedLev-27.8 # cm
Hmax = BedLev-WaterLev
# Hydro
kappa = 0.41

# Choose a file --------------------------------------------------------------
folder = '5 - 3d_coordinates'
filename = 'cup_ps_10_V5.xlsx'

path = folder + '//' + filename


# Load data ------------------------------------------------------------------
data = pd.read_excel(path)
# Access shear velocity and rising velocity
# ushear, w = cp.get_flow_sample_chars(filename)

if "10" in filename: 
    w =  0.020731900859312976#MEAN!!!!
    wM1 =  0.017093726826651813
    wM2 =  0.02491193059896622
    
    
    # w = 0.02077457295469121  #MEDIAN!!!
    # w = 0.020731900859312976 #MEAN!!!!
    # wM1 = 0.025259383197797698
    # wM2 =  0.01678031180557054
    w10 = 0.012982760964595195 
    w90 = 0.02847508469971709
elif "98" in filename:
    w =  0.01923034646366449#MEAN!!!!
    wM1 =  0.017237521980544054
    wM2 =  0.02221958318834516
    
    # # w = 0.01949697246179795 #MEDIAN!!!
    # w = 0.01923034646366449#MEAN!!!!
    # wM1 = 0.02224872553599111
    # wM2 = 0.017371753514800595
    w10 = 0.012422537763891024 
    w90 = 0.026539876944575633 
elif "50" in filename:
    w =  0.01445801632762716#MEAN!!!!
    wM1 =  0.012697714777368339
    wM2 =  0.02042773462850491

    # w = 0.013400771172966654#MEDIAN!!!
    # w = 0.01440843747071984#MEAN!!!!
    # wM1 = 0.019012493875596427
    # wM2 = 0.012656617964882693
    w10 = 0.0088574248249577667
    w90 = 0.021356018653453813 
elif "05" in filename:
    w =  0.014186997567993756#MEAN!!!!
    wM1=  0.01216469525154998
    wM2=  0.03287183541664675
    
    # w = 0.010916247648795611#MEDIAN!!!
    # w = 0.015233436014994299#MEAN!!!!
    # wM1 = 0.032779252516646845
    # wM2 = 0.010244849266642661
    # w2 = 0.027
    w10 = 0.007749035782124365
    w90 = 0.03276650001121


ushear_list_log = [0.0213042562736631,
               0.0320206390412612,
               0.0443926526227695,
               0.0535055637118659,
               0.0612913658282144] # ushear from log-wake law fit to ADV data

ushear_list_RSS = [0.0174,
                   0.0285,
                   0.0388,
                   0.0484,
                   0.0555] # u

if "V1" in filename:
    ushear_log = ushear_list_log[0]
    ushear_RSS = ushear_list_RSS[0]
elif "V2" in filename:
    ushear_log = ushear_list_log[1]
    ushear_RSS = ushear_list_RSS[1]
elif "V3" in filename:
    ushear_log = ushear_list_log[2]
    ushear_RSS = ushear_list_RSS[2]
elif "V4" in filename:
    ushear_log = ushear_list_log[3]
    ushear_RSS = ushear_list_RSS[3]
elif "V5" in filename:
    ushear_log = ushear_list_log[4]
    ushear_RSS = ushear_list_RSS[4]
else:
    ushear_log = np.nan
    ushear_RSS = np.nan

    print("Flow case not defined in file: ", filename)


beta = w/(kappa*ushear_log)
beta10 = w10/(kappa*ushear_log)
beta90 = w90/(kappa*ushear_log)

betaM1 = wM1/(kappa*ushear_log)
betaM2 = wM2/(kappa*ushear_log)

beta_RSS= w/(kappa*ushear_RSS)
beta10_RSS = w10/(kappa*ushear_RSS)
beta90_RSS = w90/(kappa*ushear_RSS)




# Start ANALYSIS -------------------------------------------------------------

x = np.asarray(data['x'])
z = np.asarray(data['y']) # Note that the 3D reconstruction coordinates are different
y = np.asarray(data['z']) # Note that the 3D reconstruction coordinates are different

t = np.asarray(data["FrameId"])*1/60 # 60 Hz
labels = np.asarray(data["label"]) # The number of particle is the label
surf = np.asarray(data['Surfaced'])
# %%

"""
We need to count particles in the observation window. However, not everything 
in the observation window is always a particle (for instance, a person moving 
and their image being reflected in the glass or other artifacts). We know that
a particle moves from left to right, and that it is within the channel
coordinates.

We have defined several rules to make sure that we do not incoroporate into
data, things that are not data (e.g., reflections in the free surface, or
lights, or people moving). For that, we go with:
    - RULE 1: in the trasverse coordinate, if things are detected way off,
    this may mean we did not cross the right particles. Let us embrace data
    between p10 and p90 for "y".
   
    - RULE 2: in x direction, if the surface shines, or there is a human
    reflection, this only shows on one side of the observation window. So let
    us embrace data that passes from left to right of the obs. window.
"""
# RULE 1
p10 = np.nanpercentile(y, 10)
ymin = np.min([p10, 0]) # at 0, this the back glass of the flume.
p90 = np.nanpercentile(y, 90)
ymax = np.max([p90, 40 ]) # at 40, this the front glass of the flume.
mask = np.ones(len(y))
mask[y<ymin] = np.nan
mask[y>ymax] = np.nan

# RULE 2
xplane1 = 50
xplane2 = 60 
for lbl in np.unique(labels): # for each label (i.e., particle), do filter out...
    # is this particle, far upstream and far downstream of observation?
   
    xp = x[labels==lbl]
    condition1 = any(xp < xplane1)
    condition2 = any(xp > xplane2)
   
    if condition1*condition2 == False:
        mask[labels==lbl] = np.nan  
        
x = x*mask
y = y*mask
z = z*mask
surf = surf*mask

zC = [] # z-coordinate when crossing the measuring plane
surfC = [] # if it was surfaced, or not.

for lbl in np.unique(labels):
   
    xp, zp = x[labels==lbl], z[labels==lbl]
    surfp = surf[labels==lbl]
   
    xp = xp[~np.isnan(xp)]
    zp = zp[~np.isnan(zp)]
    surfp = surfp[~np.isnan(surfp)] # with xp..
   
    if len(xp) > 1:
       
        # We have already aplpied RULE 1 and RULE 2, and mask them over
        zC.append(np.interp(55, xp, zp)) # we want to measure at x = 55 cm, but exactly at 55 cm we do not measure anything because particles move from left to its right
        surfC.append(np.interp(55, xp, surfp)) # same as previous reasoning, but for flag "surfaced" [0,1]
    else:
        print("Could not interpolate, not enough records.")
       
zC = np.asarray(zC)
surfC = np.asarray(surfC)

zC = zC + WaterLev 
# zC = zC + WaterLev

# bins = np.linspace(-27.8, 0, 28)
# a = zC[surfC > 0.5]
# # Retrieve stats from trajectory data
# plane = 55 # cm, plane = 55 cm for midplane of the analysis window.
# [tp, xp, yp, zp, vpx, vpy, vpz, surfp] = cp.extract_particle_stats(data, plane)
# zp = zp + WaterLev # From here on, zero at the free surface (i.e.: zp = z-h)
# VERIFICATIONS ------------------------
print("number of plastics detected: ", len(zC))

# %%
# Let us bring Rouse into the plot
##############################
#%%

if "10" in filename: 
    Lx = 7.8
    Ly = 7.2    

elif "98" in filename: 
    Lx = 7.9
    Ly = 7.2    

elif "50" in filename:
    Lx = 7.9
    Ly = 7.1    

elif "05" in filename:
    Lx = 4.1
    Ly = 3.5    
Lmb = np.sqrt(Lx*Lx+Ly*Ly) # maximum dimension if the particle goes diagonally..
a =  Lmb
a_surf = a/2

#%%
zC_surf = zC[surfC > 0.5]
zC_notsurf = zC[surfC < 0.5]
zC_sus = zC_notsurf[zC_notsurf > a-Hmax]
zC_bed = zC_notsurf[zC_notsurf < a-Hmax]

#%%
# Define a bin size, and calculate the corresponding Ca for Rouse
binsize = Lmb/4 
# bins = np.arange(-BedLev+WaterLev, 0+binsize, binsize)

bins=np.array([-27.8       , -26.4523168 , -25.10463361, -23.75695041,
       -22.40926721, -21.06158401, -19.71390082, -18.36621762,
       -17.01853442, -15.67085123, -14.32316803, -12.97548483,
       -11.62780163, -10.28011844,  -8.93243524,  -7.58475204,
        -6.23706884,  -4.88938565,  -3.54170245,  -2.19401925,
        -0.84633606,   0.50134714])

bins_surf = np.arange(-BedLev+WaterLev, 0+binsize, a_surf)
bins_bed = np.arange(-BedLev+WaterLev-a, 0+binsize, a)


plt.figure(figsize = (3,4))

count_sus, binedges_sus = np.histogram(zC_sus, bins)
count_surf, binedges_surf= np.histogram(zC_surf, bins)
count_bed, binedges_bed = np.histogram(zC_bed, bins)

diff_sus = binedges_sus[1:] - binedges_sus[0:-1]
diff_surf = binedges_surf[1:] - binedges_surf[0:-1]
diff_bed = binedges_bed[1:] - binedges_bed[0:-1]

zC_sus_conc = count_sus/diff_sus
zC_surf_conc = count_surf/diff_surf
zC_bed_conc = count_bed/diff_bed

# binmiddles_sus = (binedges_sus[:-1] + binedges_sus[1:]) / 2
# binmiddles_surf = (binedges_surf[:-1] + binedges_surf[1:]) / 2
# binmiddles_bed = (binedges_bed[:-1] + binedges_bed[1:]) / 2

# plt.plot(zC_sus_conc, binmiddles_sus, drawstyle="steps", color='steelblue', lw=2 )
# plt.fill_between(zC_sus_conc, binmiddles_sus,  color='None' ,alpha=1, edgecolor = "steelblue")

# plt.scatter(zC_sus_conc, binmiddles_sus)

plt.fill_between(zC_sus_conc, binedges_sus[1:], step="post", color='b', lw = 2, edgecolor = "k", facecolor = "none")
plt.fill_between(zC_surf_conc, binedges_surf[1:], step="post", color='r' ,alpha=0.5, edgecolor = "none")
plt.fill_between(zC_bed_conc, binedges_bed[1:], step="post", color='sienna', alpha=0.4, edgecolor = "none")



# plt.show()


# [hist_count, hist_bins, axs] = plt.hist(zC_sus, bins, color='steelblue', lw=2, ls="-",  # surfC > 0.5 because we only want suspended in Rouse
#                                         alpha=1, histtype="step",
#                                         orientation="horizontal",
#                                         label="Total particles\' C.o.G. \ncount")


         
# plt.hist(zC_surf, bins = 1, color='r' ,alpha=0.5, histtype="stepfilled",
#          orientation="horizontal",label='Surfaced particles\' C.o.G. \ncount')

# plt.hist(zC_bed, bins = 1, color='sienna', histtype="stepfilled", alpha = 0.3, lw=2,
#          orientation="horizontal")

# # Total surfaced and suspended transport -----------
# # from 0 to a: surfaced. Beyond: suspended.
# z = (hist_bins[0:-1]+hist_bins[1:])/2
# binlist = hist_bins[1:]> (-BedLev + WaterLev + a) #     Starting over the level a, in absolute coordinates.
# Csus = np.sum(hist_count[binlist]) # Where binlist==TRUE, suspended transport.

"""
normal beta
"""

# WHERE TO EVALUATE ROUSE?? In the middle of each histogram bin....
ZRouse = Hmax + (bins[0:-1]  +  bins[1:])/2


try:
    # Ca = cp.get_Ca(ZRouse, a, Hmax, beta, Csus)
    
    # Ca = len(zC_bed_conc)
    # Ca = np.sum(zC_bed_conc[zC_bed_conc.nonzero()][-1])
    
    Ca = np.mean(zC_bed_conc[zC_bed_conc != 0.0])
    
    print(Ca)
    C_med = cp.Rouse(ZRouse, a, Ca, Hmax, beta)
    plt.scatter(C_med, ZRouse-Hmax, lw=2, zorder=5, marker="+",c="sandybrown",
                label="Rouse profile, $\\beta=w/\\kappa u_*$ ")

except Exception:
    print("Rouse profile could not be plotted.")


#%%


"""10 percentile beta"""

# Ca10 = cp.get_Ca(ZRouse, a, Hmax, beta10, Csus)
# Get the Rouse profile with physically-based beta and fit Ca.
ZRousefine = np.linspace(a, Hmax, 200)

# Ca10 = np.sum(zC_bed_conc[zC_bed_conc.nonzero()][-1])
Ca10 = Ca

C10 = cp.Rouse(ZRouse, a, Ca10, Hmax, beta10)

plt.plot(C10, ZRouse-Hmax, lw=1, zorder=5, linestyle='dashed', c="sandybrown", #alpha = 0.7,
            label="Rouse profile, $\\beta_{10}=w_{10}/\\kappa u_*$ ")

"""90 percentile beta"""    
# Ca90 = cp.get_Ca(ZRouse, a, Hmax, beta90, Csus)
# Get the Rouse profile with physically-based beta and fit Ca.

Ca90 = np.sum(zC_bed_conc[zC_bed_conc.nonzero()][-1])
Ca90 = Ca

C90 = cp.Rouse(ZRouse, a, Ca90, Hmax, beta90)

plt.plot(C90, ZRouse-Hmax, lw=1, zorder=5, linestyle='dashed',c="sandybrown", # alpha = 0.95,
            label="Rouse profile, $\\beta_{90}=w_{90}/\\kappa u_*$ ")


"""w mode 1"""    
# CaM1 = cp.get_Ca(ZRouse, a, Hmax, betaM1, Csus)
# Get the Rouse profile with physically-based beta and fit Ca.

# CaM1 = np.sum(zC_bed_conc[zC_bed_conc.nonzero()][-1])
CaM1 = Ca

CM1 = cp.Rouse(ZRouse, a, CaM1, Hmax, betaM1)

plt.scatter(CM1, ZRouse-Hmax, lw=2, zorder=1, marker="+",c="dimgray", alpha = 0.5,
            label="Rouse profile, $\\beta_{M1}=w_{M1}/\\kappa u_*$ ")


"""w mode 2"""    
# CaM2 = cp.get_Ca(ZRouse, a, Hmax, betaM2, Csus)
# CaM2 = np.sum(zC_bed_conc[zC_bed_conc.nonzero()][-1])
CaM2 = Ca

# Get the Rouse profile with physically-based beta and fit Ca.
CM2 = cp.Rouse(ZRouse, a, CaM2, Hmax, betaM2)


plt.scatter(CM2, ZRouse-Hmax, lw=2, zorder=1, marker="+",c="steelblue", # alpha = 0.5,
            label="Rouse profile, $\\beta_{M2}=w_{M2}/\\kappa u_*$ ")


# plt.hlines(0, 0, 60, color="cornflowerblue", lw=1, label = "Free surface")
# # plt.hlines(-27.8, 0, 60, color="k", lw=1, linestyle='--', label = "Roughness bed")
# plt.hlines(0, 0, 60, color="blue", lw=2, label = "Free surface")
# plt.hlines(a-Hmax, a-Hmplt, 60, color="sienna", ls="--", lw=1, label = "$a_{bed}$")
# plt.hlines(-a_surf, -a_surf, 60, color="r", alpha = 0.7, ls="--", lw=1, label = "$a_{surf}$")
# plt.hlines(-27.8, 0, 60, color="k", lw=2, linestyle='-', label = "Bed")
# plt.set_xlim(0, 16) 
# plt.set_ylim(-30, 0) 


plt.vlines(Ca, -27.8, a-Hmax, color="sienna", ls="--", lw=2)
plt.ylabel('$z - H$ (cm)') # , fontsize=8
plt.xlabel('$C_p$ (Count/cm)') # , fontsize=8
plt.xlim(0,15)
plt.ylim(-30,0)
plt.hlines(0, 0, 60, color="blue", lw=3)
plt.hlines(-a_surf, -a_surf, 60, color="red", ls="--", lw=2)
plt.hlines(a-Hmax, a-Hmax, 60, color="sienna", ls="--", lw=2)
plt.hlines(-27.8, 0, 60, color="k", lw=2, linestyle='-')
sns.despine(top=True, right=True, left=False, bottom=False)
# plt.xlim([0,15])


# # """RSS Rouse"""
# Ca_RSS = cp.get_Ca(z- (-BedLev + WaterLev), a, Hmax, beta_RSS, Csus)
# # Get the Rouse profile with physically-based beta and fit Ca.
# C_RSS = cp.Rouse(z- (-BedLev + WaterLev), a, Ca_RSS, Hmax, beta_RSS)

# plt.scatter(C_RSS, z, lw=2, zorder=5, marker="+", c = "magenta",# alpha = 0.7,
#             label="Rouse profile, $\\beta_{RSS}=w/\\kappa u_{*,RSS}$ ")

# # """90 RSS"""
# Ca90_RSS = cp.get_Ca(ZRouse, a, Hmax, beta90_RSS, Csus)
# # Get the Rouse profile with physically-based beta and fit Ca.
# C90_RSS  = cp.Rouse(ZRousefine, a, Ca90_RSS, Hmax, beta90_RSS)

# plt.plot(C90_RSS, ZRousefine-Hmax, lw=1, zorder=5, linestyle='dashed',c="magenta", # alpha = 0.7,
#             label="Rouse profile, $\\beta_{90,RSS}=w_{90}/\\kappa u_{*,RSS}$ ")

# # """10 RSS"""
# Ca10_RSS = cp.get_Ca(ZRouse, a, Hmax, beta10_RSS, Csus)
# # Get the Rouse profile with physically-based beta and fit Ca.
# C10_RSS  = cp.Rouse(ZRousefine, a, Ca10_RSS, Hmax, beta10_RSS)

# plt.plot(C10_RSS, ZRousefine-Hmax, lw=1, zorder=5, linestyle=':',c="magenta", # alpha = 0.7,
#             label="Rouse profile, $\\beta_{10,RSS}=w_{10}/\\kappa u_{*,RSS}$ ")

# """w2 (only for 05!!!!!!)"""
# Caw2 = cp.get_Ca(z- (-BedLev + WaterLev), a, Hmax, beta_w2, Csus)
# # Get the Rouse profile with physically-based beta and fit Ca.
# C_w2  = cp.Rouse(z- (-BedLev + WaterLev), a, Ca, Hmax, beta_w2)

# plt.scatter(C_w2, z, lw=2, zorder=5, linestyle='dashed',marker = "+", c="green", alpha = 0.7,
#             label="Rouse profile, $\\beta=w/\\kappa u_*$ ")

#%%

"""
Let us now look at CDF, because these are not subjective in binsize, and thus
allow to identify if data follows Rouse, or not!

To compare:
    1. Data: zC
    
    2. C.cumsum()
      at:  ZRouse-Hmax
"""


# fig, ax = plt.subplots(figsize=(3, 3))

# plot the cumulative histogram of data
binscum = np.linspace(-Hmax, 0, 400)
# n, binscum_, patches = ax.hist(zC[surfC < 0.5], binscum, density=True, 
#                                histtype='step', lw=2, color="steelblue",
#                                cumulative=True, label='Data')


# C = cp.Rouse(ZRousefine, a, Ca, Hmax, beta)
# plt.plot(ZRousefine-Hmax, np.nancumsum(C)/np.max(np.nancumsum(C)), c="k",
#          label="Rouse profile, $\\beta=w/\\kappa u_*$ ")

# plt.ylabel("CDF (-)")
# plt.xlabel("$z-H$ (cm)")
# plt.xlim([-30, 0])
# Formatting of characteristic levels


# %%
 # So we always include the last sample
# Nsamples[i] = len(zpsorted[0:i])




ZRouse_true = cp.inv_sampling_Rouse(a,Ca,Hmax,beta,10000) - Hmax

ZRouse_10 = cp.inv_sampling_Rouse(a,Ca,Hmax,beta10,10000) - Hmax
ZRouse_90 = cp.inv_sampling_Rouse(a,Ca,Hmax,beta90,10000) - Hmax
ZRouse_M1 = cp.inv_sampling_Rouse(a,Ca,Hmax,betaM1,10000) - Hmax
ZRouse_M2 = cp.inv_sampling_Rouse(a,Ca,Hmax,betaM2,10000) - Hmax


[st, pv] = stats.ks_2samp(zC_sus,
                          ZRouse_true,
                          alternative='two-sided')

[st_M1, pv_M1] = stats.ks_2samp(zC_sus,
                          ZRouse_M1,
                          alternative='two-sided')

[st_M2, pv_M2] = stats.ks_2samp(zC_sus,
                          ZRouse_M2,
                          alternative='two-sided')



print("beta, KS = ", beta, st)

# print("p-value med: ", pv)


print("betaM1, KS= ", betaM1, st_M1)
# print("KS stat M1 percentile: ", st_M1)
# print("p-value M1 percentile: ", pv_M1)


print("betaM2 , KS= ", betaM2, st_M2)
# print("KS stat M2 percentile: ", st_M2)
# print("p-value M2 percentile: ", pv_M2)


observed_values = zC_sus
predicted_values = ZRouse_true

differences = [abs(observed - predicted) for observed, predicted in zip(observed_values, predicted_values)]
ame = max(differences)



plt.figure(figsize=(3,4))
# TRUE ROUSE!!!!! at 10,000 samples
plt.hist(ZRouse_true, binscum, cumulative=True, zorder = 4, histtype='step', lw=2, color="sandybrown",density=True, orientation='horizontal')


# plt.hist(ZRouse_10, binscum, cumulative=True, histtype='step', lw=1, linestyle='dashed',color="sandybrown",density=True)
# plt.hist(ZRouse_90, binscum, cumulative=True, histtype='step', lw=1, linestyle='dashed',color="sandybrown",density=True)
plt.hist(ZRouse_M1, binscum, cumulative=True, histtype='step', lw=2, linestyle='dashed',color="dimgray",density=True, orientation='horizontal')
plt.hist(ZRouse_M2, binscum, cumulative=True, histtype='step', lw=2, linestyle='dashed', color="steelblue",density=True, orientation='horizontal')


# Sampled experiment!!!!! at 100 samples
plt.hist(zC_sus, binscum, density=True, 
                               histtype='step', zorder = 5,lw=2, color="k",
                               cumulative=True, label='Data', orientation='horizontal')


plt.hlines(0, 0, 1, color="blue", lw=1, zorder = 8, label = "Free surface")
plt.hlines(-a_surf, 0, 1,color="r", alpha = 0.7, zorder = 8,ls="--", lw=1, label = "$a_{surf}$")
plt.hlines(a-Hmax, 0, 1, color="sienna", ls="--", zorder = 8,lw=1, label = "$a_{bed}$")
plt.hlines(-27.8, 0, 1, color="k", lw=1,  zorder = 8,linestyle='-', label = "Bed")
plt.xlabel("CDF (-)")
plt.ylabel("$z-H$ (cm)")
y_ticks = np.arange(-30, 1, 10)
# plt.set_yticks(y_ticks)
x_ticks = np.arange(0, 1.1, 0.5)
# plt.set_xticks(x_ticks)
# sns.despine(top=True, right=True, left=False, bottom=False)

plt.show()



#%%
"""calculate RSME"""

#to get true rouse - but same length as observed values
# # Zrouse_RSME = cp.inv_sampling_Rouse(a,Ca,Hmax,beta,23) - Hmax

# squared_diff = (zC_sus_conc - C_med) ** 2

# # Calculate the mean of squared differences
# mean_squared_diff = np.mean(squared_diff)

# # Calculate the RMSE
# rmse = np.sqrt(mean_squared_diff)

# print("RMSE:", rmse)

#%%

#%%
# print("beta = ", beta)
# print("betaM1 = ", betaM1)
# print("betaM2 = ", betaM2)

# results = pd.DataFrame(list(zip(zC, surfC)))
# results_rouse = pd.DataFrame(list(zip(C_med, C10, C90, CM1, CM2, ZRouse-Hmax)))
# results_rouse2 = pd.DataFrame(list(zip(ZRouse_true, ZRouse_10, ZRouse_90, ZRouse_M1, ZRouse_M2)))
# results_rouse3 = pd.DataFrame(list(zip(binscum)))
# results_rouse4 = pd.DataFrame(list(zip(zC[(surfC < 0.5) * (zC > (a-Hmax))])))
# results_rouse5 = pd.DataFrame(list(zip(zC_sus_conc, zC_surf_conc, zC_bed_conc, binedges_sus[1:], binedges_surf[1:], binedges_bed[1:])))
# # results_norm = pd.DataFrame(list(zip(zCp_norm, z_r_norm, surfCp_norm, C_norm)))

# # [tp, xp, yp, zp, vpx, vpy, vpz, surfp]
# folder = '6 - concentration_profiles'
# filename1 = filename
# filename2 = filename[0:12]+"rouse"+".xlsx"
# filename3 = filename[0:12]+"rouse2"+".xlsx"
# filename4 = filename[0:12]+"rouse3"+".xlsx"
# filename5 = filename[0:12]+"rouse4"+".xlsx"
# filename6 = filename[0:12]+"rouse5"+".xlsx"


# path1 = folder + '//' + filename1
# path2 = folder + '//' + filename2
# path3 = folder + '//' + filename3
# path4 = folder + '//' + filename4
# path5 = folder + '//' + filename5
# path6 = folder + '//' + filename6

# results.to_excel(path1)
# results_rouse.to_excel(path2)
# results_rouse2.to_excel(path3)
# results_rouse3.to_excel(path4)
# results_rouse4.to_excel(path5)
# results_rouse5.to_excel(path6)
# # results_norm.to_excel(path3)

print("all data saved")
    
