#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:34:58 2023

@author: jameslofty
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data= pd.read_excel("beta_AME.xlsx",)

df = pd.DataFrame(data)

plastic_style_map = {
    # 'fullmmed': ('sandybrown', 's'),   # Square, red color
    'fullmmean': ('sandybrown', 's'),   # Square, red color
    'fullmm1': ('dimgray', 's'),  # Triangle, green color
    'fullmm2': ('royalblue', 's'),   # Circle, blue color
    # 'defmmed': ('sandybrown', '^'),   # Square, red color
    'defmmean': ('sandybrown', '^'),   # Square, red color
    'defmm1': ('dimgray', '^'),  # Triangle, green color
    'defmm2': ('royalblue', '^'),   # Circle, blue color
    # 'halfmmed': ('sandybrown', 'D'),   # Square, red color
    'halfmmean': ('sandybrown', 'D'),   # Square, red color
    'halfmm1': ('dimgray', 'D'),  # Triangle, green color
    'halfmm2': ('royalblue', 'D'),   # Circle, blue color
    # 'fragmmed': ('sandybrown', 'o'),   # Square, red color
    'fragmmean': ('sandybrown', 'o'),   # Square, red color
    'fragmm1': ('dimgray', 'o'),  # Triangle, green color
    'fragmm2': ('royalblue', 'o')   # Circle, blue color
}

full_plastics = df[df['plastic'].str.contains('full', case=False)]
def_plastics = df[df['plastic'].str.contains('def', case=False)]
half_plastics = df[df['plastic'].str.contains('half', case=False)]
frag_plastics = df[df['plastic'].str.contains('frag', case=False)]


fig, axs = plt.subplots(2, 2, figsize=(3, 2.5))

# Plot each type of plastic in a separate subplot
for data, title, ax in zip(
    [full_plastics, def_plastics, half_plastics, frag_plastics],
    ['Full Plastics', 'Def Plastics', 'Half Plastics', 'Frag Plastics'],
    axs.flat
):
    for _, row in data.iterrows():
        style = plastic_style_map[row['plastic']]
        ax.scatter(row['beta'], row['ΑΜΕ'], color=style[0], marker=style[1],facecolors = 'none',  s=12, label=row['plastic'])

    ax.set_xlabel('$β$(-)')
    # ax.set_ylabel('')
    # ax.set_title(title)
    ax.set_ylim(0, 0.6)  
    ax.set_xlim(0, 4)  
    sns.despine(top=True, right=True, left=False, bottom=False)
    
    if ax in axs[:, 1]:
     ax.yaxis.set_tick_params(which='both', labelright=False)

# plt.tight_layout()
plt.savefig('beta_AME.svg', format='svg')
# plt.legend()
plt.show()


#%%



plt.figure(figsize=(3, 2.5))

for plastic, group in df.groupby('plastic'):
    color, marker = plastic_style_map[plastic]
    plt.scatter(group['beta'], group['ΑΜΕ'], 
                edgecolors=color, 
                facecolors = 'none', 
                marker=marker, 
                label=plastic, 
                s = 20)

sandybrown_group = df[df['plastic'].str.contains('fullmmean|defmmean|halfmmean|fragmmean')]
dimgray_group = df[df['plastic'].str.contains('fullmm1|defmm1|halfmm1|fragmm1')]
royalblue_group = df[df['plastic'].str.contains('fullmm2|defmm2|halfmm2|fragmm2')]


# if not sandybrown_group.empty:
#     x_sb = sandybrown_group['beta']
#     y_sb = sandybrown_group['ΑΜΕ']
#     coefficients_sb = np.polyfit(x_sb, y_sb, 1)  # Fit a linear polynomial (degree 1)
#     line_sb = np.poly1d(coefficients_sb)
#     plt.plot(x_sb, line_sb(x_sb), color='sandybrown', alpha = 0.7)

# if not dimgray_group.empty:
#     x_dg = dimgray_group['beta']
#     y_dg = dimgray_group['ΑΜΕ']
#     coefficients_dg = np.polyfit(x_dg, y_dg, 1)  # Fit a linear polynomial (degree 1)
#     line_dg = np.poly1d(coefficients_dg)
#     plt.plot(x_dg, line_dg(x_dg), color='dimgray', alpha = 0.7)

# if not royalblue_group.empty:
#     x_dg = royalblue_group['beta']
#     y_dg = royalblue_group['ΑΜΕ']
#     coefficients_dg = np.polyfit(x_dg, y_dg, 1)  # Fit a linear polynomial (degree 1)
#     line_dg = np.poly1d(coefficients_dg)
#     plt.plot(x_dg, line_dg(x_dg), color='royalblue', alpha = 0.7)


# for plastic, group in df.groupby('plastic'):
#     color, marker = plastic_style_map[plastic]
#     plt.scatter(group['beta'], group['ΑΜΕ'], c=color, marker=marker, label=plastic)

plt.xlabel('$\it{β}$ (-)')
plt.ylabel('ΑΜΕ (-)')
# plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)

plt.xlim(0, 4)
plt.ylim(0, 0.6)

# plt.xticks()

sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig('beta_AME.svg', format='svg')

plt.show()

#%%
