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
name = ["AMs", r"CD4$^+$ T", r"CD8$^+$ T", r"IFN-$\beta$", r"IFN-$\gamma$", "IL-6", "NK", "Neutrophils", r"TNF-$\alpha$",
        "Viral Load"]

# r2 scores for each parameter
r2_model = [-0.489990858632666, 0.009766414602115, -0.127755285203315, -0.394220244748988, 0.16434990632092983, 
            0.2547622069157439, -0.133331611688781, 0.341945303138773, -0.052029681954352, 0.203203796792608]

# r2 scores using mean of the data for performance comparison
r2_mean = [-0.653716730634632, -0.005479164050356, -0.050761035988531, -3.45195340083709, -0.14225998387426, 
           -0.042671972729998, -0.209548118188491, -0.009631590927169, -0.187343063998834, -0.368553171206015]

# create dataframe for plotting
df_plot = pd.DataFrame()
df_plot["r2 model"] = r2_model
df_plot["r2 mean"] = r2_mean
df_plot.index = name
df_plot.sort_values(by="r2 model", inplace=True)

# create plot
fig, ax = plt.subplots()
fig.set_size_inches(14, 10)
plt.grid()
siz=20
plt.ylabel(r"$R^2$ score", size=siz+2)
plt.axhline(y=0, color="grey", linestyle='--')
plt.plot(df_plot["r2 mean"], 'kd', label="benchmark", markersize=11, color = "k")
plt.plot(df_plot["r2 model"], 'bd', label="model", markersize=11, color="#5974B3")
plt.xticks(np.arange(len(name)), size=13)
ax.set_xticklabels(df_plot.index, fontsize=siz-6)
plt.legend(loc="lower right", fontsize=siz-4, frameon=True)
plt.savefig("../Plots/R2_scores_comparison/R2_scores_comparison.png", dpi=300, bbox_inches="tight")
