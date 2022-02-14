#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping of viral load via a neural network using different hematological features. 
Results are saved in a text file and plotted for training and test data.

Comment/Uncomment code on the top of this file to chose which features to use.
4 different cases are shown but any combination of these features can be used.

The plots are saved under "Plots/Supplemental/Viral_Load_different_Features"

@author: Suneet Singh Jhutty
@date:   02.08.22
"""

#%% imports
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time

# measure runtime of program
start = time.time()

### comment/uncomment features to use as input ###

# case 1: all 14 features ("14var")
selected_vars = ['Leukocytes (10^9/l)', 'Lymphocytes (10^9/l)', 'Monocytes (10^9/l)',
                  'Granulocytes (10^9/l)', 'Erythrocytes (10^12/l)', 'Hemoglobin (g/dl)',
                  'Hematocrit (%)', 'MCV (fl)', 'MCH (pg)', 'MCHC (g/dl)', 'RDWs (fl)',
                  'Platelets (10^9/l)', 'MPV (fl)', 'PDWs (fl)']

# case 2: 2 features ("2var")
# selected_vars = ['Lymphocytes (10^9/l)', 'Granulocytes (10^9/l)']

# case 3: 2 features of ratios ("2ratio")
# selected_vars = ['Gra/Lymph', 'Plt/Gra']

# case 4: 13 features with one feature being a ratio of two features ("13var")
# selected_vars = ['Leukocytes (10^9/l)', 'Monocytes (10^9/l)', 'Erythrocytes (10^12/l)', 
#                   'Hemoglobin (g/dl)', 'Hematocrit (%)', 'MCV (fl)', 'MCH (pg)', 'MCHC (g/dl)',
#                   'RDWs (fl)', 'Platelets (10^9/l)', 'MPV (fl)', 'PDWs (fl)', 'Gra/Lymph']

# case 5: 4 features with two features being ratios of other features ("4var")
# selected_vars = ['Erythrocytes (10^12/l)', 'Hemoglobin (g/dl)', 'Gra/Lymph', 'Plt/Gra']

#%% define labels for saving results
figname = "14vars_08-02-22"
targetname = "Viral Load"
# switch for saving results, if True results are saved to file
save_results = True

#%% function to calculate corrected Akaike criterium (AICc)
def aic(n, mse, m):
    """
    n:       number of data points
    m:       number of unknown parameter
    mse:     mean squared error or residual sum of squares from fitting routine divided by n
    return:  corrected Akaike criterium (AICc)
    
    For details, see Burham K, Anderson D. 2002. Model selection and multimodel inference. 
    Springer, New York, NY.
    """
    if n-m-1 != 0:
        return n*np.log(mse) + (2*n*m)/(n-m-1)
    else:
        return "n = m+1 !"
    
#%% load training data
data = pd.read_excel('../Data/Training_Data.xlsx', sheet_name='Blood-VL')

# blood data
X_train = data.iloc[:, 3:]
# viral load
y_train = data.iloc[:, 2]

# remove repetative variables
X_train.drop("Lymphocytes (%)", axis=1, inplace=True)
X_train.drop("Monocytes (%)", axis=1, inplace=True)
X_train.drop("Granulocytes (%)", axis=1, inplace=True)
X_train.drop("RDWc (%)", axis=1, inplace=True)
X_train.drop("PCT (%)", axis=1, inplace=True)
X_train.drop("PDWc (%)", axis=1, inplace=True)
X_train = X_train.astype(float)

# add combined features
X_train["Gra/Lymph"] = X_train["Granulocytes (10^9/l)"]/X_train["Lymphocytes (10^9/l)"]
X_train["Plt/Gra"] = X_train["Platelets (10^9/l)"]/X_train["Granulocytes (10^9/l)"]

# define input variables for training
X_train = X_train[selected_vars]

#%% preprocess training data for better performing results
# scale blood data
mm_scaler = preprocessing.MinMaxScaler()
mm_scaler.fit(X_train)
X_train_scaled= mm_scaler.transform(X_train)

# convert back into pandas dataframe
X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

# log-transform viral load data
y_train = np.log(y_train.copy())
y_train = y_train.replace([np.inf, -np.inf], 0)  # set control group to zero

# %% train neural network model with data

# number of simulations over which to average
num_runs = 10
str_num_runs = str(num_runs)

# number of days
num_days = 9
# number of neurons in the network
w=21
# empty array to save every model from each simulation
NN_mod_lst = []
# empty array to save AIC of every model from each simulation
aic_lst = []

for k in range(num_runs):
    # create model
    model = Sequential()
    model.add(Dense(w, input_dim=X_train.shape[1], activation='relu', kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(w, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # compile
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])    
    # callback
    cb = [EarlyStopping(monitor="val_loss", min_delta=10**(-4), patience=10000),
          ModelCheckpoint(filepath='vl_best_in_latest_run.h5', save_best_only=True, monitor='mse')]
    # fit
    history = model.fit(X_train, y_train, validation_split=0.1, batch_size=9, verbose=0, epochs=300, callbacks=cb)
    latest_model = load_model('vl_best_in_latest_run.h5')
    
    # calculate and save corrected AIC
    aic_lst.append(aic(len(y_train), np.min(history.history["mse"]), latest_model.count_params()))
    # save best model
    NN_mod_lst.append(latest_model)

#%% prediction on training data
y_pred_train_lst = []
for mod in NN_mod_lst:
    y_pred_train_lst.append(mod.predict(X_train))

# calculate individual mse, mae, r2 score for each run
mse_train = []
mae_train = []
r2_train = []
for y_p in y_pred_train_lst:
    mse_train.append(mean_squared_error(y_train, y_p.flatten()))
    mae_train.append(mean_absolute_error(y_train, y_p.flatten()))
    r2_train.append(r2_score(y_train, y_p.flatten()))

# calculate average mse, mae and r2 score
mse_train_mean = np.mean(mse_train)
mse_train_std = np.std(mse_train)
mae_train_mean = np.mean(mae_train)
mae_train_std = np.std(mae_train)
r2_train_mean = np.mean(r2_train)
r2_train_std = np.std(r2_train)
aic_mean = np.mean(aic_lst)
aic_std = np.std(aic_lst)

# calculate average prediction
y_pred_train = np.mean(y_pred_train_lst, axis=0).flatten()

# calculate mse, mae and r2 score from average prediction
mse_print = mean_squared_error(y_train, y_pred_train)
mae_print = mean_absolute_error(y_train, y_pred_train)
r2_print = r2_score(y_train, y_pred_train)

# print results to console
print("Training Score (average for each run):")
print("==================================")
print("MSE:", mse_train_mean, "(+-"+str(mse_train_std)+")")
print("MAE:", mae_train_mean, "(+-"+str(mae_train_std)+")")
print("R2 score:", r2_train_mean, "(+-"+str(r2_train_std)+")")
print("AIC:", aic_mean, "(+-"+str(aic_std)+")")
print("==================================")
print("Final Training Score:")
print("==================================")
print("MSE:", mse_print)
print("MAE:", mae_print)
print("R2 score:", r2_print)
print("AIC:", aic(len(y_train), mse_print, 10*latest_model.count_params()))
print("==================================")

# save results to file
if save_results:
    file1 = open("../Results/VL_Mapping_Results_"+figname+".txt","a")
    file1.write("=== VL Mapping (NN, Training, "+figname+") ===")
    file1.write("\n MSE:        "+str(mse_print))
    file1.write("\n MAE:        "+str(mae_print))
    file1.write("\n R2 score:   "+str(r2_print))
    file1.write("\n AIC:        "+str(aic_mean)+" (+-"+str(aic_std)+")")
    file1.write("\n")
    file1.close()

#%% plot training data

# backtransform data
y_pred_train_bt = np.exp(y_pred_train)
y_train_bt = np.exp(y_train)
# set control group to zero
y_train_bt = y_train_bt.replace(1, 0)

# extract days
df_day = data["Day"]

# days for plotting
day_exp = np.array([0,1,2,3,4,5,7,9,11])

#  to plot all days instead, uncomment the line below
# day_exp = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

# plotting
for i,d in enumerate(day_exp):        
    # relevant part of days dataframe
    day = df_day[df_day == d]

    # set plotting options
    plt.style.use('seaborn-ticks')
    fig = plt.figure(1, figsize=(15, 10))  # Create a figure instance
    ms = 12                                # markersize
    col = "#5974B3"                        # color prediction data
    col_test = "#99B8FF"                   # color training data
    col_vlin = "#B3C9FF"                   # color vertical line
    # plot
    if len(day) == 1:
        plt.plot(d, y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot(d, y_train_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d, y_pred_train_bt[day.index][0], y_train_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
    elif len(day) == 2:
        pos = 0.3
        plt.plot([d-pos, d+pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-pos, d+pos], y_train_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-pos, y_pred_train_bt[day.index][0], y_train_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_train_bt[day.index][1], y_train_bt[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
    elif len(day) == 3:
        pos = 0.3
        plt.plot([d-pos, d, d+pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-pos, d, d+pos], y_train_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-pos, y_pred_train_bt[day.index][0], y_train_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d, y_pred_train_bt[day.index][1], y_train_bt[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_train_bt[day.index][2], y_train_bt[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
    elif len(day) == 4:
        pos = 0.1
        plt.plot([d-2*pos, d-pos, d+pos, d+2*pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-2*pos, d-pos, d+pos, d+2*pos], y_train_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-2*pos, y_pred_train_bt[day.index][0], y_train_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d-pos, y_pred_train_bt[day.index][1], y_train_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_train_bt[day.index][2], y_train_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+2*pos, y_pred_train_bt[day.index][3], y_train_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
    elif len(day) == 5 and d == 0:
        pos = 0.1
        plt.plot([d-2*pos, d-pos, d, d+pos, d+2*pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.axvspan(day_exp[0]-0.5, (d+day_exp[i+1])/2, facecolor="lightgrey", alpha=0.5, zorder=-10)  # use axvspan to set grey area
    elif len(day) == 5:
        pos = 0.1
        plt.plot([d-2*pos, d-pos, d, d+pos, d+2*pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-2*pos, d-pos, d, d+pos, d+2*pos], y_train_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-2*pos, y_pred_train_bt[day.index][0], y_train_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d-pos, y_pred_train_bt[day.index][1], y_train_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d, y_pred_train_bt[day.index][2], y_train_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_train_bt[day.index][3], y_train_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+2*pos, y_pred_train_bt[day.index][4], y_train_bt.iloc[day.index].iloc[4], colors=col_vlin, linestyles="dashed")
    elif len(day) == 6:
        pos = 0.1
        plt.plot([d-3*pos, d-2*pos, d-pos, d+pos, d+2*pos, d+3*pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-3*pos, d-2*pos, d-pos, d+pos, d+2*pos, d+3*pos], y_train_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-3*pos, y_pred_train_bt[day.index][0], y_train_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d-2*pos, y_pred_train_bt[day.index][1], y_train_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d-pos, y_pred_train_bt[day.index][2], y_train_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_train_bt[day.index][3], y_train_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+2*pos, y_pred_train_bt[day.index][4], y_train_bt.iloc[day.index].iloc[4], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+3*pos, y_pred_train_bt[day.index][5], y_train_bt.iloc[day.index].iloc[5], colors=col_vlin, linestyles="dashed")
    # separate data with vertical lines
    if d != 11:
        plt.axvline((d+day_exp[i+1])/2, color="grey", linestyle="dashed")

# general plotting setting    
plt.xlim(day_exp[0]-0.5, day_exp[-1]+0.5)
plt.ylim(2, 10**8)
plt.plot([], 'D', alpha=1, markersize=ms, color=col, label='Model Prediction')
plt.plot([], 'o', alpha=1, markersize=ms, color=col_test, label='Experimental Data')
plt.yscale('log')
fs = 30
plt.xticks(day_exp, size=fs)
plt.yticks(size=fs)
plt.title("Training Performance", size=45)
plt.xlabel("Time/ Days", size=fs+2)
plt.ylabel("Viral Load/ (NP copies/50ng RNA)", size=fs+2)
plt.legend(loc="upper right", fontsize=fs-7, frameon=True)

# save plot
if save_results:
    plt.savefig("../Plots/Supplemental/Viral_Load_different_Features/Viral_load_NN_Training_"+figname+".png", dpi=300, bbox_inches="tight")
    plt.close()
    
#%% load testing data
test_data = pd.read_excel('../Data/Testing_Data.xlsx', sheet_name='Blood-VL')

X_test = test_data.iloc[:, 3:]  # blood data
y_test = test_data.iloc[:, 2]   # viral load

# remove repeatative variables
X_test.drop("Lymphocytes (%)", axis=1, inplace=True)
X_test.drop("Monocytes (%)", axis=1, inplace=True)
X_test.drop("Granulocytes (%)", axis=1, inplace=True)
X_test.drop("RDWc (%)", axis=1, inplace=True)
X_test.drop("PCT (%)", axis=1, inplace=True)
X_test.drop("PDWc (%)", axis=1, inplace=True)
X_test = X_test.astype(float)

# add combined features
X_test["Gra/Lymph"] = X_test["Granulocytes (10^9/l)"]/X_test["Lymphocytes (10^9/l)"]
X_test["Plt/Gra"] = X_test["Platelets (10^9/l)"]/X_test["Granulocytes (10^9/l)"]

# define input variables for testing
X_test = X_test[selected_vars]
#%% transform testing data
# scale blood data
X_test_scaled = mm_scaler.transform(X_test)

# convert back into pandas dataframe
X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# log-transform viral load data
y_test = np.log(y_test.copy())
y_test = y_test.replace([np.inf, -np.inf], 0)  # set control group to zero

#%% Evaluate prediction with testing data

# calculate predictions from all runs
y_pred_lst = []
for mod in NN_mod_lst:
    y_pred_lst.append(mod.predict(X_test))

# calculate individual mse, mae, r2 score for each run
mse = []
mae = []
r2 = []
for y_p in y_pred_lst:
    mse.append(mean_squared_error(y_test, y_p.flatten()))
    mae.append(mean_absolute_error(y_test, y_p.flatten()))
    r2.append(r2_score(y_test, y_p.flatten()))

# calculate average mse, mae and r2 score
mse_mean = np.mean(mse)
mse_std = np.std(mse)
mae_mean = np.mean(mae)
mae_std = np.std(mae)
r2_mean = np.mean(r2)
r2_std = np.std(r2)

# calculate average prediction
y_pred = np.mean(y_pred_lst, axis=0).flatten()
# calculate mse, mae and r2 score from average prediction
mse_print_test = mean_squared_error(y_test, y_pred)
mae_print_test = mean_absolute_error(y_test, y_pred)
r2_print_test = r2_score(y_test, y_pred)

# print results in console
print("Testing Score (average for each run):")
print("==================================")
print("MSE:", mse_mean, "(+-"+str(mse_std)+")")
print("MAE:", mae_mean, "(+-"+str(mae_std)+")")
print("R2 score:", r2_mean, "(+-"+str(r2_std)+")")
print("==================================")
print("Final Testing Score:")
print("==================================")
print("MSE:", mse_print_test)
print("MAE:", mae_print_test)
print("R2 score:", r2_print_test)
print("==================================")

# save to file
if save_results:
    file2 = open("../Results/VL_Mapping_Results_"+figname+".txt","a")
    file2.write("\n")
    file2.write("=== VL Mapping (NN, Testing, "+figname+") ===")
    file2.write("\n MSE:        "+str(mse_print_test))
    file2.write("\n MAE:        "+str(mae_print_test))
    file2.write("\n R2 score:   "+str(r2_print_test))
    file2.write("\n")
    file2.close()

#%% plot prediction and testing data

# backtransform data
y_pred_bt = np.exp(y_pred)
y_test_bt = np.exp(y_test)
# extract days
df_day = test_data["Day"]
# days for plotting
day_exp = np.array([2,4,6,9,11])

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
    elif len(day) == 3:
        pos = 0.3
        plt.plot([d-pos, d, d+pos], y_pred_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
        plt.plot([d-pos, d, d+pos], y_test_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
        plt.vlines(d-pos, y_pred_bt[day.index][0], y_test_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        plt.vlines(d, y_pred_bt[day.index][1], y_test_bt[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        plt.vlines(d+pos, y_pred_bt[day.index][2], y_test_bt[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
    
    # separate data with vertical lines
    if d != 11:
        plt.axvline((d+day_exp[i+1])/2, color="grey", linestyle="dashed")
    
# general plotting setting
plt.ylim(2, 10**8)
plt.plot([], 'D', alpha=1, markersize=ms, color=col, label='Model Prediction')
plt.plot([], 'o', alpha=1, markersize=ms, color=col_test, label='Experimental Data')
plt.yscale('log')
fs = 30
plt.xticks(day_exp, size=fs)
plt.yticks(size=fs)
plt.title("Testing Performance", size=45)
plt.xlabel("Time/ Days", size=fs+2)
plt.ylabel("Viral Load/ (NP copies/50ng RNA)", size=fs+2)
plt.legend(fontsize=fs-7, frameon=True)
# save plots
if save_results:
    plt.savefig("../Plots/Supplemental/Viral_Load_different_Features/Viral_load_NN_Testing_"+figname+".png", dpi=300, bbox_inches="tight")
    plt.close()
# print runtime of program
end = time.time()
print("Time/min:", (end - start)/60)