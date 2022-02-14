#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot correlations between viral load, lung leukocytes, lung cytokines and hematological parameters
in various figures.

The plots are saved under "Plots/Correlations"

@author: Suneet Singh Jhutty
@date:   02.08.22
"""

# imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# %% load blood data (training data) ----------------------------------------------------------
raw_blood_data = pd.read_excel('Experiment1_Vetscan_and_Viral_load_data.xlsx',
                               sheet_name='Vetscan Data')
# get only relevant data columns
df = raw_blood_data.iloc[:, 2:23]
# delete unneccessary rows with no relevant information
df = df.drop(0)
df = df.drop(33)
df = df.drop(35).reset_index(drop=True)
# save days and corresponding indices
day_ind = df.iloc[:, 0:1]
# delete excess variables
df.drop("Day", axis=1, inplace=True)
df.drop("Lymphocytes (%)", axis=1, inplace=True)
df.drop("Monocytes (%)", axis=1, inplace=True)
df.drop("Granulocytes (%)", axis=1, inplace=True)
df.drop("PCT (%)", axis=1, inplace=True)
df.drop("PDWc (%)", axis=1, inplace=True)
df.drop("RDWc (%)", axis=1, inplace=True)
df = df.astype(float)

# %% load data of viral load in lungs (training data)
raw_vl_data = pd.read_excel('Experiment1_Vetscan_and_Viral_load_data.xlsx', sheet_name='Lung viral load')
df1 = raw_vl_data
# get only relevant data columns
df1 = raw_vl_data.iloc[:, 2:5]
# set NAN entries to zero
df1["Day"] = pd.to_numeric(df1.iloc[:, 0], errors='coerce').fillna(0)
df1[df1.columns[1]] = pd.to_numeric(df1.iloc[:, 1], errors='coerce').fillna(0)
df1["Viral load STD"] = pd.to_numeric(df1.iloc[:, 2], errors='coerce').fillna(0)
# delete data with no Vetscan equivalent
df1 = df1.drop([9, 19, 32, 42, 45, 46, 47, 48, 49])
# reset index
vl = df1.iloc[:, 1:2].reset_index(drop=True)
df["Viral Load (NP copies/50ng RNA)"] = vl

# %% load blood and viral load data (testing data) -------------------------------------------------------------
raw_blood_data_test = pd.read_excel('Experiment3_Vetscan_and_Viral_load_data.xlsx', sheet_name="Cleaned Data")

df2 = raw_blood_data_test
df2 = df2[pd.notnull(df2["Viral load [NP copies/50ng RNA]"])]
vl1 = df2["Viral load [NP copies/50ng RNA]"]
vl1 = vl1.reset_index(drop=True)
# relable column
vl1.columns = ["Viral Load"]
# delete unimportant columns    
df2 = df2.iloc[:, 1:55]
if df2.columns[-1] == 'Treatment':
    df2 = df2.iloc[:, :-1]
    
df2["Viral Load (NP copies/50ng RNA)"] = vl1
df2 = df2.reset_index(drop=True)

# %% rename columns
original_names = list(df.columns)
names_without_units = [s.split(" ")[0] for s in original_names[:-1]]
names_without_units.append("Viral Load")

# use simplified names
df.columns = ['leu', 'lym', 'mon', 'gra', 'ery', 'hgb', 'hct', 'mcv', 'mch', 'mchc',
 'rdws', 'plt', 'mpv', 'pdws', 'vl'] 
df2.columns = ['leu', 'lym', 'mon', 'gra', 'ery', 'hgb', 'hct', 'mcv', 'mch', 'mchc',
 'rdws', 'plt', 'mpv', 'pdws', 'vl']

# %% Create a complete dataframe with all data
df_data = pd.concat([df, df2])

# %% viral load and hematological data correlations
sns.set(style="whitegrid", font_scale=1.5)
df_corr = df_data.copy()
df_corr.columns = names_without_units
df_corr = df_corr.corr()
mask = np.zeros_like(df_corr, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

# plot and save figure
f, ax = plt.subplots(figsize=(20, 15))
ax = sns.heatmap(df_corr,linewidths=0.25,vmax=0.7,square=True,cmap="Blues", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":14},mask=mask,cbar_kws={"shrink": .9});
for label in ax.get_yticklabels():
    if label.get_text() == "Leukocytes":
        label.set_color("white")
for label in ax.get_xticklabels():
    if label.get_text() == "Viral Load":
        label.set_color("white")
plt.savefig("Plots/Correlations/correlation_matrix_VL-Blood_V1_supp", dpi=300, bbox_inches="tight")
plt.close()

# %% plot and save only selected correlations
f, ax = plt.subplots(figsize=(20, 15))
#plt.title('Pearson Correlation Matrix',fontsize=25)
df_corr_sel = df_data.copy()
df_corr_sel.columns = names_without_units
drop_columns = ["MCV", "MCH", "MCHC", "RDWs", "MPV"]
df_corr_sel.drop(axis=1, columns=drop_columns, inplace=True)
df_corr_sel = df_corr_sel.corr()
mask = np.zeros_like(df_corr_sel, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

ax = sns.heatmap(df_corr_sel,linewidths=0.25,vmax=0.7,square=True,cmap="Blues", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":14},mask=mask,cbar_kws={"shrink": .9});
for label in ax.get_yticklabels():
    if label.get_text() == "Leukocytes":
        label.set_color("white")
for label in ax.get_xticklabels():
    if label.get_text() == "Viral Load":
        label.set_color("white")
plt.savefig("Plots/Correlations/correlation_matrix_VL-Blood_V1_selected", dpi=300, bbox_inches="tight")
plt.close()

# %% load hematological data from 2nd experiment (training data)
raw_blood_data = pd.read_excel('Experiment2_Vetscan_Lung_Leukocyte_and_Cytokine_data.xlsx',
                               sheet_name='VetScan Data')

# get range of relevant data columns
df = raw_blood_data.iloc[:, 4:53]
# delete unneccessary columns with no relevant information
df = df[df.columns.drop(list(df.filter(regex='Hinweis')))]
df = df[df.columns.drop(list(df.filter(regex='EOS')))]
df = df[df.columns.drop(list(df.filter(regex='BAS')))]
df = df[df.columns.drop(list(df.filter(regex='% %')))]
df.drop("PCT %", axis=1, inplace=True)
df.drop("PDWc %", axis=1, inplace=True)
df.drop("RDWc %", axis=1, inplace=True)

# delete unneccessary rows with no relevant information
df = df.dropna(axis=0).reset_index(drop=True)

# get the mouse groups for the different days and the control group (0)
groups = df["Exp. Group"]
groups.replace("PBS d1", 0, inplace=True)
groups.replace("IAV d1", 1, inplace=True)
groups.replace("IAV d2", 2, inplace=True)
groups.replace("IAV d3", 3, inplace=True)
groups.replace("IAV d4", 4, inplace=True)
groups.replace("IAV d5", 5, inplace=True)
groups.replace("IAV d7", 7, inplace=True)
groups.replace("IAV d9", 9, inplace=True)
groups.replace("IAV d11", 11, inplace=True)
df.drop("Mouse#", axis=1, inplace=True)
df.drop("Exp. Group", axis=1, inplace=True)
df = df.astype(float)

# %% load data of leukocytes from lung tissue (training data)
raw_lung_data = pd.read_excel('Experiment2_Vetscan_Lung_Leukocyte_and_Cytokine_data.xlsx',
                              sheet_name='Cleaned lung tissue leukocytes')

raw_lung_data.head()
cells = raw_lung_data.copy()
cells_mean = cells.groupby(["Day"]).mean()
df_data_IC = df.copy()

# cell data from lungs
df_data_IC["Ams"] = cells[["Ams"]]
df_data_IC["PMNs"] = cells[["PMNs"]]
df_data_IC["NK"] = cells[["NK"]]
df_data_IC["CD4+ T"] = cells[["CD4+ T"]]
df_data_IC["CD8+ T"] = cells[["CD8+ T"]]

# %% rename columns
original_names = list(df.columns)
names_without_units = [s.split(" ")[0] for s in original_names]
names_without_units.append("AMs")
names_without_units.append("Neutrophils")
names_without_units.append("NK cells")
names_without_units.append(r"CD4$^+$ T cells")
names_without_units.append(r"CD8$^+$ T cells")

# %% load hematological data and data of lung leukocytes (testing data)
raw_V4 = pd.read_excel('Experiment4_Vetscan_and_Lung_Leukocyte_data.xlsx',
                               sheet_name='Data')
df_V4 = raw_V4.iloc[:, 2:]
df_V4.columns = df_data_IC.columns

# %% Create dataframe with combined data
df_data_IC = pd.concat([df_data_IC, df_V4])

# %% lung leukocytes and hematological data correlations
df_corr_IC = df_data_IC.copy()
df_corr_IC.columns = names_without_units
df_corr_IC = df_corr_IC.corr()
mask = np.zeros_like(df_corr_IC, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

# plot and save correlations
f, ax = plt.subplots(figsize=(20, 15))
ax = sns.heatmap(df_corr_IC,linewidths=0.25,vmax=0.7,square=True,cmap="Blues", #"BuGn_r" to reverse
            linecolor='w',annot=True,annot_kws={"size":14},mask=mask,cbar_kws={"shrink": 0.9});

for label in ax.get_yticklabels():
    if label.get_text() == "Leukocytes":
        label.set_color("white")
for label in ax.get_xticklabels():
    if label.get_text() == "CD8$^+$ T cells":
        label.set_color("white")
plt.savefig("Plots/Correlations/correlation_matrix_Leuko-Blood_V2_V4_supp", dpi=300, bbox_inches="tight")
plt.close()

# %% plot and save selected correlations
sns.set(style="whitegrid", font_scale=1.7)
f, ax = plt.subplots(figsize=(20, 15))
#plt.title('Pearson Correlation Matrix',fontsize=25)
df_corr_IC_sel = df_data_IC.copy()
df_corr_IC_sel.columns = names_without_units
drop_columns = ["MCV", "MCH", "MCHC", "RDWs", "MPV", "PDWs", "Hemoglobin"]
df_corr_IC_sel.drop(axis=1, columns=drop_columns, inplace=True)
df_corr_IC_sel = df_corr_IC_sel.corr()
mask = np.zeros_like(df_corr_IC_sel, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

ax = sns.heatmap(df_corr_IC_sel,linewidths=0.25,vmax=0.7,square=True,cmap="Blues", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":18},mask=mask,cbar_kws={"shrink": .9});

for label in ax.get_yticklabels():
    if label.get_text() == "Leukocytes":
        label.set_color("white")
for label in ax.get_xticklabels():
    if label.get_text() == "CD8$^+$ T cells":
        label.set_color("white")
plt.savefig("Plots/Correlations/correlation_matrix_Leuko-Blood_V2_V4_selected", dpi=300, bbox_inches="tight")
plt.close()

# %% merge selected viral load and lung leukocyte correlations in one plot
sns.set(style="whitegrid", font_scale=3)
f, ax = plt.subplots(figsize=(35, 25))
bottom =  0.3    # the bottom of the subplots of the figure
top    =  0.7    # the top of the subplots of the figure
plt.subplots_adjust(
    bottom  =  bottom, 
    top     =  top, 
)
plt.subplot(121)
mask = np.zeros_like(df_corr_sel, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

ax = sns.heatmap(df_corr_sel,linewidths=0.25,vmin=-0.35,vmax=0.7,square=True,cmap="Blues",
                 linecolor='w',annot=True,annot_kws={"size":25},mask=mask,cbar=False,cbar_kws={"shrink": .55});

# bolden VL and remove Leukocytes on y-label
for label in ax.get_yticklabels():
    if label.get_text() == "Viral Load":
        label.set_weight("bold")
    elif label.get_text() == "Leukocytes":
        label.set_color("white")

# remove VL from x-label
for label in ax.get_xticklabels():
    if label.get_text() == "Viral Load":
        label.set_color("white")
        
plt.subplot(122)
mask = np.zeros_like(df_corr_IC_sel, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

ax = sns.heatmap(df_corr_IC_sel,linewidths=0.25,vmin=-0.35,vmax=0.7,square=True,cmap="Blues",
                 linecolor='w',annot=True,annot_kws={"size":22},mask=mask,cbar_kws={"shrink": 0.885});

# bolden lung leukocytes and remove Leukocytes from y-label
for label in ax.get_yticklabels():
    if label.get_text() == "AMs" or label.get_text() == "Neutrophils" or label.get_text() == "NK cells" or label.get_text() == "CD4$^+$ T cells" or label.get_text() == "CD8$^+$ T cells":
        label.set_weight("bold")
    elif label.get_text() == "Leukocytes":
        label.set_color("white")
        
# remove CD8+ T cells from x-label
for label in ax.get_xticklabels():
    if label.get_text() == "CD8$^+$ T cells":
        label.set_color("white")
    if label.get_text() == "AMs" or label.get_text() == "Neutrophils" or label.get_text() == "NK cells" or label.get_text() == "CD4$^+$ T cells" or label.get_text() == "CD8$^+$ T cells":
        label.set_weight("bold")
plt.savefig("Plots/Correlations/correlation_matrix_merged_V1-4", dpi=300, bbox_inches="tight")
plt.close()

# %% load lung cytokine data (training data) 
data = pd.read_excel('Experiment2_Vetscan_Lung_Leukocyte_and_Cytokine_data.xlsx', sheet_name='Airway cytokines')
df_cyto = data.iloc[:, 2:]
df_cyto.drop(0, inplace=True)
df_cyto.reset_index(inplace=True, drop=True)
for col in df_cyto.columns:
        df_cyto[col] = df_cyto[col].astype(str)
        df_cyto[col] = df_cyto[col].str.replace('<' , '')
        df_cyto[col] = df_cyto[col].str.replace(',' , '.')
        df_cyto[col] = df_cyto[col].astype(float)

df_cyto.columns = ['IL-23', r'IL-1$\alpha$', r'IFN-$\gamma$', r'TNF-$\alpha$', 'MCP-1',
                   'IL-12p70', r'IL-1$\beta$', 'IL-10', 'IL-6', 'IL-27',
                   'IL-17A', r'IFN-$\beta$', 'GM-CSF']

# %% load lung cytokine data (testing data) 
cytokines_V3_V4 = pd.read_excel('Experiment3_and_4_Cytokine_data.xlsx')
df_cyto_eval = cytokines_V3_V4.iloc[:, 4:17]
df_cyto_eval.drop(0, inplace=True)
df_cyto_eval.reset_index(inplace=True, drop=True)

for col in df_cyto_eval.columns:
        df_cyto_eval[col] = df_cyto_eval[col].astype(str)
        df_cyto_eval[col] = df_cyto_eval[col].str.replace('<' , '')
        df_cyto_eval[col] = df_cyto_eval[col].str.replace(',' , '.')
        df_cyto_eval[col] = df_cyto_eval[col].astype(float)

df_cyto_eval.columns = ['IL-23', r'IL-1$\alpha$', r'IFN-$\gamma$', r'TNF-$\alpha$', 'MCP-1',
                   'IL-12p70', r'IL-1$\beta$', 'IL-10', 'IL-6', 'IL-27',
                   'IL-17A', r'IFN-$\beta$', 'GM-CSF']

# %% Create dataframe with combined data
df_cyto = pd.concat([df_cyto, df_cyto_eval], ignore_index=True)

# %% align rows between cytokine and blood data frames
df_data_cyto = df.copy()
for v in df_cyto:
    df_data_cyto[v] = df_cyto[v]
names_without_units_cyto = names_without_units
del names_without_units_cyto[-5:]
names_without_units_cyto.extend(list(df_cyto.columns))

# %% lung cytokine and hematological data correlations
sns.set(style="whitegrid", font_scale=1.5)
df_corr_cyto = df_data_cyto.copy()
df_corr_cyto.columns = names_without_units_cyto
df_corr_cyto = df_corr_cyto.corr()
mask = np.zeros_like(df_corr_cyto, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True

# plot and save correlations
f, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(df_corr_cyto,linewidths=0.25,vmax=0.7,square=True,cmap="Blues", #"BuGn_r" to reverse
            linecolor='w',annot=True,annot_kws={"size":10},mask=mask,cbar_kws={"shrink": .9});

for label in ax.get_yticklabels():
    if label.get_text() == "Leukocytes":
        label.set_color("white")
for label in ax.get_xticklabels():
    if label.get_text() == "GM-CSF":
        label.set_color("white")
plt.savefig("Plots/Correlations/correlation_matrix_Cyto-Blood_V2-4", dpi=300, bbox_inches="tight")
plt.close()












