#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping of IL-6 via gradient boosted regression trees using hematological data. Addtionally, a 
feature important analysis is performed.
Results are saved in a text file and plotted for training and test data. 

The plots are saved under "Plots/IL-6"

@author: Suneet Singh Jhutty
@date:   02.08.22
"""

#%% imports
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, make_scorer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import time

# measure runtime of program
start = time.time()

#%% define labels for saving results
figname = "08-02-22"
targetname = "IL-6"
# switch for saving results, if True results are saved to file
save_results = True

#%% load training data data
data = pd.read_excel('Training_Data.xlsx', sheet_name='Blood-Cytokines')

X_train = data.iloc[:, 15:]  # blood data
y_train = data.iloc[:, 10]   # viral load

# remove repeatative variables
X_train.drop("Lymphocytes (%)", axis=1, inplace=True)
X_train.drop("Monocytes (%)", axis=1, inplace=True)
X_train.drop("Granulocytes (%)", axis=1, inplace=True)
X_train.drop("RDWc (%)", axis=1, inplace=True)
X_train.drop("PCT (%)", axis=1, inplace=True)
X_train.drop("PDWc (%)", axis=1, inplace=True)
X_train = X_train.astype(float)

#%% transform training data for better performing results
# scale blood data
mm_scaler = preprocessing.MinMaxScaler()
mm_scaler.fit(X_train)
X_train_scaled= mm_scaler.transform(X_train)

# convert back into pandas dataframe
X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

# log-transform viral load data
y_train = np.log(y_train.copy())
y_train = y_train.replace([np.inf, -np.inf], 0)  # set control group to zero

#%% load testing data
test_data = pd.read_excel('Testing_Data.xlsx', sheet_name='Blood-Cytokines')

X_test = test_data.iloc[:, 15:]  # blood data
y_test = test_data.iloc[:, 10]   # viral load
# remove "<" from data
y_test = y_test.replace("<*", "", regex=True)
y_test = y_test.astype(float)

# remove repeatative variables
X_test.drop("Lymphocytes (%)", axis=1, inplace=True)
X_test.drop("Monocytes (%)", axis=1, inplace=True)
X_test.drop("Granulocytes (%)", axis=1, inplace=True)
X_test.drop("RDWc (%)", axis=1, inplace=True)
X_test.drop("PCT (%)", axis=1, inplace=True)
X_test.drop("PDWc (%)", axis=1, inplace=True)
X_test = X_test.astype(float)

#%% transform testing data
# scale blood data
X_test_scaled = mm_scaler.transform(X_test)

# convert back into pandas dataframe
X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# log-transform viral load data
y_test = np.log(y_test.copy())
# set control group to zero
y_test = y_test.replace([np.inf, -np.inf], 0)

#%% GBRT
params = {'n_estimators': 150, 'max_depth': 3, 'min_samples_split': 4,
          'learning_rate': 0.01, 'random_state': 0, 'loss': 'ls'}

est = GradientBoostingRegressor(**params)
est.fit(X_train, np.array(y_train).flatten())
y_pred = est.predict(X_test)

#%% calculate mse, mae and r2 score from average prediction
mse_print = mean_squared_error(y_test, y_pred)
mae_print = mean_absolute_error(y_test, y_pred)
r2_print = r2_score(y_test, y_pred)

# print results in console
print("Testing GBRT:")
print("==================================")
print("MSE:", mse_print)
print("MAE:", mae_print)
print("R2 score:", r2_print)
print("==================================")

# save to file
if save_results:
    file2 = open(targetname+"_Mapping_Results_Testing"+figname+".txt","a")
    file2.write("=== "+targetname+" Mapping (GBRT, Testing, "+figname+") ===")
    file2.write("\n MSE:        "+str(mse_print))
    file2.write("\n MAE:        "+str(mae_print))
    file2.write("\n R2 score:   "+str(r2_print))
    file2.write("\n")
    file2.close()

#%% plot prediction and testing data

# backtransform data
y_pred_bt = np.exp(y_pred)
y_test_bt = np.exp(y_test)
# extract days
df_day = test_data["Day"]
# days for plotting
day_exp = np.array([0,2,4,6,9,11])

# loop to plot elementwise
for i,d in enumerate(day_exp):        
    # relevant part of days dataframe
    day = df_day[df_day == d]
    
    # set plotting options
    plt.style.use('seaborn-ticks')
    fig = plt.figure(1, figsize=(15, 10))  # Create a figure instance
    ms = 12                                # markersize
    col = "#5974B3"                        # color prediction data
    col_test = "#99B8FF"                   # color testing data
    col_vlin = "#B3C9FF"                   # color vertical line
    # plot
    if len(day) == 1:
        plt.plot(d, y_pred_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot(d, y_test_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d, y_pred_bt[day.index][0], y_test_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
    elif len(day) == 2:
        pos = 0.3
        plt.plot([d-pos, d+pos], y_pred_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-pos, d+pos], y_test_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-pos, y_pred_bt[day.index][0], y_test_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_bt[day.index][1], y_test_bt[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
    elif len(day) == 3 and d == 0:
        pos = 0.1
        plt.plot([d-pos, d, d+pos], y_pred_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-pos, d, d+pos], y_test_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.axvspan(day_exp[0]-1, (d+day_exp[i+1])/2-0.01, facecolor="lightgrey", alpha=0.5, zorder=-10)  # use axvspan to set grey area
        plt.vlines(d-pos, y_pred_bt[day.index][0], y_test_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d, y_pred_bt[day.index][1], y_test_bt[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_bt[day.index][2], y_test_bt[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
    elif len(day) == 3:
        pos = 0.3
        plt.plot([d-pos, d, d+pos], y_pred_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-pos, d, d+pos], y_test_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-pos, y_pred_bt[day.index][0], y_test_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d, y_pred_bt[day.index][1], y_test_bt[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_bt[day.index][2], y_test_bt[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
    elif len(day) == 4:
        pos = 0.1
        plt.plot([d-2*pos, d-pos, d+pos, d+2*pos], y_pred_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-2*pos, d-pos, d+pos, d+2*pos], y_test_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-2*pos, y_pred_bt[day.index][0], y_test_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d-pos, y_pred_bt[day.index][1], y_test_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_bt[day.index][2], y_test_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+2*pos, y_pred_bt[day.index][3], y_test_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
    elif len(day) == 5:
        pos = 0.1
        plt.plot([d-2*pos, d-pos, d, d+pos, d+2*pos], y_pred_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-2*pos, d-pos, d, d+pos, d+2*pos], y_test_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-2*pos, y_pred_bt[day.index][0], y_test_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d-pos, y_pred_bt[day.index][1], y_test_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d, y_pred_bt[day.index][2], y_test_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_bt[day.index][3], y_test_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+2*pos, y_pred_bt[day.index][4], y_test_bt.iloc[day.index].iloc[4], colors=col_vlin, linestyles="dashed")
    elif len(day) == 6:
        pos = 0.1
        plt.plot([d-3*pos, d-2*pos, d-pos, d+pos, d+2*pos, d+3*pos], y_pred_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-3*pos, d-2*pos, d-pos, d+pos, d+2*pos, d+3*pos], y_test_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-3*pos, y_pred_bt[day.index][0], y_test_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d-2*pos, y_pred_bt[day.index][1], y_test_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d-pos, y_pred_bt[day.index][2], y_test_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_bt[day.index][3], y_test_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+2*pos, y_pred_bt[day.index][4], y_test_bt.iloc[day.index].iloc[4], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+3*pos, y_pred_bt[day.index][5], y_test_bt.iloc[day.index].iloc[5], colors=col_vlin, linestyles="dashed")
        
    # separate data with vertical lines
    if d != 11:
        plt.axvline((d+day_exp[i+1])/2, color="grey", linestyle="dashed")
    
# general plotting setting
plt.xlim(day_exp[0]-1, day_exp[-1]+0.5)
plt.ylim(0.1, 10**4)
plt.plot([], 'D', alpha=1, markersize=ms, color=col, label='Model Prediction')
plt.plot([], 'o', alpha=1, markersize=ms, color=col_test, label='Experimental Data')
plt.yscale('log')
fs = 30
plt.xticks(day_exp, size=fs)
#plt.ylim(ymin_eval, ymax_eval)
plt.yticks(size=fs)
plt.title("Testing Performance", size=45)
plt.xlabel("Time/ Days", size=fs+2)
plt.ylabel("IL-6/ (pg/ml)", size=fs+2)
plt.legend(loc="upper right", fontsize=fs-7, frameon=True)#, bbox_to_anchor=(1.16, 1))
# save plots
if save_results:
    plt.savefig("Plots/IL-6/IL_GBRT_Testing_"+figname+".png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    plt.show()
    
# print runtime of program
end = time.time()
print("Time/min:", (end - start)/60)

#%% Permutation Importance
result = permutation_importance(est, X_test, y_test, n_repeats=100, 
                                scoring=make_scorer(mean_squared_error, greater_is_better=False),
                                random_state=42, n_jobs=1)
#%% sort according to importance mean
sorted_idx = result.importances_mean.argsort()
# create figure
fig = plt.figure(1, figsize=(15,12))
# Create an axes instance
ax = fig.add_subplot(122)
# define labels
feature_labels = [x.replace('10^9', '$10^9$') for x in list(X_test.columns[sorted_idx])]
feat_labels = [x.replace('10^12', '$10^{12}$') for x in feature_labels]
# create box plot
bppi=ax.boxplot(result.importances[sorted_idx].T, patch_artist=True,
            vert=False, labels=feat_labels)

col = "#5974B3"
col2 = "#B3C9FF"
fs = 25
## change outline color, fill color and linewidth of the boxes
for box in bppi['boxes']:
    # change outline color
    box.set(color=col, linewidth=2)
    # change fill color
    box.set(facecolor=col)

    ## change color and linewidth of the whiskers
    for whisker in bppi['whiskers']:
        whisker.set(color=col, linewidth=2)

    ## change color and linewidth of the caps
    for cap in bppi['caps']:
        cap.set(color=col, linewidth=2)

    ## change color and linewidth of the medians
    for median in bppi['medians']:
        median.set(color=col2, linewidth=2)

    ## change the style of fliers and their fill
    for flier in bppi['fliers']:
        flier.set(marker='o', color=col, markeredgecolor=col, alpha=0.5)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('Relative Importance', fontsize=fs+6)
    t=plt.title("IL-6", fontsize=fs+15)
    t.set_y(1.01)
    plt.subplots_adjust(top=0.86)
    fig.tight_layout()
    
# save plot
if save_results:
    plt.savefig("Plots/IL-6/Feature_Importance_IL-6_GBRT_"+figname+".png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    plt.show()

#%% Mean of data as benchmark to compare results

# calculate average prediction
y_mean = np.repeat(np.mean(y_train), len(y_test))
# calculate mse, mae and r2 score from average prediction
mse_print = mean_squared_error(y_test, y_mean)
mae_print = mean_absolute_error(y_test, y_mean)
r2_print = r2_score(y_test, y_mean)

# print results in console
print("Benchmark:")
print("==================================")
print("MSE:", mse_print)
print("MAE:", mae_print)
print("R2 score:", r2_print)
print("==================================")

# save to file
if save_results:
    file = open(targetname+"_Mapping_Results_"+figname+".txt","a")
    file.write("\n")
    file.write("=== "+targetname+" Mapping (Benchmark, "+figname+") ===")
    file.write("\n MSE:        "+str(mse_print))
    file.write("\n MAE:        "+str(mae_print))
    file.write("\n R2 score:   "+str(r2_print))
    file.write("\n")
    file.close()
