#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the best r2 scores between the tested models for various lung parameter mappings next to 
each other for comparison.

The plot is saved under "Plots/R2_scores_comparison"

@author: Suneet Singh Jhutty
@date:   02.08.22
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# name of the mapped lung parameter
name = ["Viral Load", r"CD8$^+$ T", r"CD4$^+$ T", "AMs", "NK", "Neutrophils"]

# r"IFN-$\beta$", r"IFN-$\gamma$", "IL-6", r"TNF-$\alpha$",

# r2 scores for each parameter in following order: VL, CD8, CD4, AMs, NK, Neutrophils
r2_model = [-5.378427640896443, 0.4732350142461711, -31.550837468378493, -742.8831509759572, -32.0577478582599, -75.35527943628908]

# r2 scores using mean of the data for performance comparison
r2_mean = [-0.08381877590381737, -0.29916114712700004, -52.52636072043845, -0.04486817301142043, -0.471376367929376, -0.5020252942918068]

# create dataframe for plotting
df_plot = pd.DataFrame()
df_plot["r2 model"] = r2_model
df_plot["r2 mean"] = r2_mean
df_plot.index = name
df_plot.sort_values(by="r2 model", inplace=True)

# create plot
fig, ax = plt.subplots()
fig.set_size_inches(14, 10)
#plt.grid()
siz=25
plt.ylabel(r"$R^2$ score", size=siz+2)
plt.axhline(y=0, color="grey", linestyle='--')
plt.plot(df_plot["r2 mean"], 'kd', label="benchmark", markersize=11, color = "k")
plt.plot(df_plot["r2 model"], 'bd', label="model", markersize=11, color="#5974B3")
plt.xticks(np.arange(len(name)), size=13)
plt.yticks(size=16)
ax.set_xticklabels(df_plot.index, fontsize=siz-6)
plt.legend(loc="lower right", fontsize=siz-4, frameon=True)
plt.savefig("../Plots/Supplemental/R2_scores_comparison_for_supplemental.png", dpi=300, bbox_inches="tight")
