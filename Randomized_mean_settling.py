# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:43:28 2023

@author: Valero
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(12051989)
filename = "sett_v_daniel.xlsx"


data = pd.read_excel(filename)

wfull = data["full"]
wfull.dropna(inplace=True) # no nan
wfull = np.asarray(wfull)
wfull_m = np.mean(wfull)

wdef = data["def"]
wdef.dropna(inplace=True) # no nan
wdef = np.asarray(wdef)
wdef_m = np.mean(wdef)


whalf = data["half"]
whalf.dropna(inplace=True) # no nan
whalf = np.asarray(whalf)
whalf_m = np.mean(whalf)

wfrag = data["frag"]
wfrag.dropna(inplace=True) # no nan
wfrag = np.asarray(wfrag)
wfrag_m = np.mean(wfrag)


plt.figure(figsize=(5,4))

N_list = []
a_mean = []
b_mean = []
c_mean = []
d_mean = []

for i in range(1,97):
    
    a = []
    b = []
    c = []
    d = []
    if i % 10 == 0:
        print("N = ", i)
        
    for j in range(0,40):
        np.random.shuffle(wfull)
        wfull_RAME = np.abs(np.mean(wfull[0:i]) - wfull_m) / wfull_m
        plt.scatter(i, wfull_RAME, color="k", marker="o", alpha=0.05)
        a.append(wfull_RAME)
        
        np.random.shuffle(wdef)
        wdef_RAME = np.abs(np.mean(wdef[0:i]) - wdef_m) / wdef_m
        plt.scatter(i, wdef_RAME, color="b", marker="o", alpha=0.05)
        
        np.random.shuffle(whalf)
        whalf_RAME = np.abs(np.mean(whalf[0:i]) - whalf_m) / whalf_m
        plt.scatter(i, whalf_RAME, color="g", marker="o", alpha=0.05)
        
        np.random.shuffle(wfrag)
        wfrag_RAME = np.abs(np.mean(wfrag[0:i]) - wfrag_m) / wfrag_m
        plt.scatter(i, wfrag_RAME, color="m", marker="o", alpha=0.05)

    a_mean.append(np.mean(wfull_RAME))
    b_mean.append(np.mean(wdef_RAME))
    c_mean.append(np.mean(whalf_RAME))
    d_mean.append(np.mean(wfrag_RAME))
    N_list.append(i)
    

plt.plot(N_list, a_mean, color="k", label="Full cups (average RAME)")
plt.plot(N_list, b_mean, color="b", label="Deformed cups (average RAME)")
plt.plot(N_list, c_mean, color="g", label="Half cups (average RAME)")
plt.plot(N_list, d_mean, color="m", label="Cup fragment (average RAME)")

plt.semilogy()
# plt.loglog()

#plt.ylabel("$\overline{w}_N$ (cm/s)")    
plt.ylabel("RAME($\overline{w}_N$) (-)")    
plt.xlabel("$N$ (-)")    

plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig("RAME_all.svg", dpi=600)
plt.savefig("RAME_all.pdf", dpi=600)
plt.savefig("RAME_all.png", dpi=600)

plt.show()




