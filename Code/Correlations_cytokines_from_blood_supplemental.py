#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot correlations between viral load, lung leukocytes, lung cytokines and hematological parameters
in various figures. Data from experiment 3 & 4

The plots are saved under "Plots/Correlations"

@author: Suneet Singh Jhutty
@date:   29.07.22
"""

# imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_excel('../Data/Testing_Data.xlsx', sheet_name='Blood Cytokines - LL_tissue')
data = raw_data.drop(['Mouse ID', 'Experiment', 'Sample '], axis=1)

for col in data.columns:
    data[col] = data[col].astype(str)
    data[col] = data[col].str.replace('<' , '')
    data[col] = data[col].str.replace(',' , '.')
    data[col] = data[col].astype(float)

data.drop(["Day"], axis=1, inplace=True)


data.columns = ['Viral Load', 'AMs', 'Neutrophils', 'NK cells', 'CD4$^+$ T cells', 'CD8$^+$ T cells',
                'IL-23', r'IL-1$\alpha$', r'IFN-$\gamma$', r'TNF-$\alpha$', 'MCP-1', 'IL-12p70', r'IL-1$\beta$',
                'IL-10', 'IL-6', 'IL-27', 'IL-17A', r'IFN-$\beta$', 'GM-CSF']

#swap columns
data = data[['NK cells', 'Neutrophils', 'CD4$^+$ T cells', 'CD8$^+$ T cells', 'AMs', 'Viral Load', 
             'IL-23', r'IL-1$\alpha$', r'IFN-$\gamma$', r'TNF-$\alpha$', 'MCP-1', 'IL-12p70', r'IL-1$\beta$',
             'IL-10', 'IL-6', 'IL-27', 'IL-17A', r'IFN-$\beta$', 'GM-CSF']]

sns.set(style="whitegrid", font_scale=1.5)

#%%
df = data.corr()
mask = np.zeros_like(df, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

# plot and save correlations
f, ax = plt.subplots(figsize=(20, 15))
ax = sns.heatmap(df,linewidths=0.25,vmax=0.7,square=True,cmap="Blues", #"BuGn_r" to reverse
                 linecolor='w',annot=True,annot_kws={"size":14},mask=mask,cbar_kws={"shrink": 0.9});

plt.savefig("../Plots/Supplemental/correlation_matrix_cytokine_from_blood_with_lung_V3_V4_supp", dpi=300, bbox_inches="tight")
plt.close()

#%%
vl_data = data[data['Viral Load'].notna()].reset_index(drop=True)
ll_data = data[data['AMs'].notna()]
vl_data.drop(['NK cells', 'Neutrophils', 'CD4$^+$ T cells', 'CD8$^+$ T cells', 'AMs'], axis=1, inplace=True)
ll_data.drop(['Viral Load', 'IL-23', 'IL-12p70' ], axis=1, inplace=True)

df_vl = vl_data.corr()
mask = np.zeros_like(df_vl, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

# plot and save correlations
f, ax = plt.subplots(figsize=(20, 15))
ax = sns.heatmap(df_vl,linewidths=0.25,vmax=0.7,square=True,cmap="Blues", #"BuGn_r" to reverse
                 linecolor='w',annot=True,annot_kws={"size":14},mask=mask,cbar_kws={"shrink": 0.9});

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
for label in ax.get_yticklabels():
    if label.get_text() == "Viral Load":
        label.set_color("white")
for label in ax.get_xticklabels():
    if label.get_text() == "GM-CSF":
        label.set_color("white")
        
# plt.show()
plt.savefig("../Plots/Supplemental/correlation_matrix_cytokine_from_blood_with_viral_load_supp", dpi=300, bbox_inches="tight")
plt.close()

#%%
df_ll = ll_data.corr()
mask = np.zeros_like(df_ll, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

# plot and save correlations
f, ax = plt.subplots(figsize=(20, 15))
ax = sns.heatmap(df_ll,linewidths=0.25,vmax=0.7,square=True,cmap="Blues", #"BuGn_r" to reverse
                  linecolor='w',annot=True,annot_kws={"size":14},mask=mask,cbar_kws={"shrink": 0.9});

for label in ax.get_yticklabels():
    if label.get_text() == "NK cells":
        label.set_color("white")
for label in ax.get_xticklabels():
    if label.get_text() == "GM-CSF":
        label.set_color("white")

# plt.show()
plt.savefig("../Plots/Supplemental/correlation_matrix_cytokine_from_blood_with_lung_leukocytes_supp", dpi=300, bbox_inches="tight")
plt.close()






















