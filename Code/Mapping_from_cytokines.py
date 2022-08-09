#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping of viral load via a neural network using cytokines from blood. Addtionally, a feature important
analysis is performed.
Results are saved in a text file and plotted for training and test data. 

The plots are saved under "Plots/Supplemental"

@author: Suneet Singh Jhutty
@date:   28.07.22
"""

#%% imports
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, make_scorer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import time

# measure runtime of program
start = time.time()

#%% define labels for saving results
figname = "08-02-22"
targetname = "Viral Load"
# switch for saving results, if True results are saved to file
save_results = True

#%% load training data data
raw_data = pd.read_excel('../Data/Testing_Data.xlsx', sheet_name='Blood Cytokines - LL_tissue')
data = raw_data.drop(['Mouse ID', 'Experiment', 'Sample '], axis=1)

for col in data.columns:
    data[col] = data[col].astype(str)
    data[col] = data[col].str.replace('<' , '')
    data[col] = data[col].str.replace(',' , '.')
    data[col] = data[col].astype(float)

vl_data = data[data['Viral load [NP copies/50ng RNA]'].notna()].reset_index(drop=True)
ll_data = data[data['AMs'].notna()]

# %% train neural network model with virus data

# normalize
X = vl_data.iloc[:, 7:]
y = vl_data['Viral load [NP copies/50ng RNA]']
y = np.log(y.copy())

# number of simulations over which to average
num_runs = 1
str_num_runs = str(num_runs)

# number of days
num_days = 9
# number of neurons in the network
w=20
# empty array to save every model from each simulation
NN_mod_lst = []
# define 10-fold cross validation test harness
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=123)
cvscores = []
for train, test in kfold.split(X, y):
    # create model
    model = Sequential()
    model.add(Dense(w, input_dim=X.iloc[train].shape[1], activation='relu', kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(w, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # compile
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    # Fit the model
    model.fit(X.iloc[train], y.iloc[train], epochs=300, batch_size=10, verbose=0)
    # evaluate the model
    y_pred = model.predict(X.iloc[test])
    mse_value = mean_squared_error(y.iloc[test], y_pred)
    r2_value = r2_score(y.iloc[test], y_pred)
    y_benchmark = np.mean(y.iloc[train])
    benchmark = r2_score(y.iloc[test], np.repeat(y_benchmark, 3))
    print("benchmark:", benchmark)
    print("mse score:", mse_value)
    print("r2 score:", r2_value)
    cvscores.append([mse_value, r2_value, benchmark])

mse_vl = np.median(np.array(cvscores)[:, 0])
benchmark_lst_vl = np.array(cvscores)[:, 2]
r2_lst_vl = np.array(cvscores)[:, 1]
vl_ind = r2_lst_vl.argsort()    
r2_lst_vl_sorted = r2_lst_vl[vl_ind]
benchmark_sorted = benchmark_lst_vl[vl_ind]
r2_vl = np.mean(r2_lst_vl_sorted[24:26])
benchmark_vl = np.mean(benchmark_sorted[24:26])
    


# %% train neural network model with lung leukocyte data ############################################################

# normalize
X = ll_data.iloc[:, 7:]
y = ll_data['CD8+ T cells ']
y = np.log(y.copy())

# number of simulations over which to average
num_runs = 1
str_num_runs = str(num_runs)

# number of days
num_days = 9
# number of neurons in the network
w=20
# empty array to save every model from each simulation
NN_mod_lst = []
# define 10-fold cross validation test harness
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=123)
cvscores = []

def make_model():
    # create model
    model = Sequential()
    model.add(Dense(w, input_dim=X.iloc[train].shape[1], activation='relu', kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(w, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
for train, test in kfold.split(X, y):
    # create model
    make_model()
    # model = Sequential()
    # model.add(Dense(w, input_dim=X.iloc[train].shape[1], activation='relu', kernel_initializer="he_uniform"))
    # model.add(Dropout(0.2))
    # model.add(Dense(w, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1))
    # compile
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    # Fit the model
    model.fit(X.iloc[train], y.iloc[train], epochs=300, batch_size=10, verbose=0)
    # evaluate the model
    y_pred = model.predict(X.iloc[test])
    mse_value = mean_squared_error(y.iloc[test], y_pred)
    r2_value = r2_score(y.iloc[test], y_pred)
    y_benchmark = np.mean(y.iloc[train])
    benchmark = r2_score(y.iloc[test], np.repeat(y_benchmark, len(y.iloc[test])))
    print("benchmark:", benchmark)
    print("mse score:", mse_value)
    print("r2 score:", r2_value)
    cvscores.append([mse_value, r2_value, benchmark])

mse_cd8 = np.median(np.array(cvscores)[:, 0])    
benchmark_lst_cd8 = np.array(cvscores)[:, 2]
r2_lst_cd8 = np.array(cvscores)[:, 1]
cd8_ind = r2_lst_cd8.argsort()    
r2_lst_cd8_sorted = r2_lst_cd8[cd8_ind]
benchmark_sorted = benchmark_lst_cd8[cd8_ind]
r2_cd8 = np.mean(r2_lst_cd8_sorted[24:26])
benchmark_cd8 = np.mean(benchmark_sorted[24:26])


#%%
# normalize
X = ll_data.iloc[:, 7:]
y = ll_data['CD4+ T cells ']
y = np.log(y.copy())

# number of simulations over which to average
num_runs = 1
str_num_runs = str(num_runs)

# number of days
num_days = 9
# number of neurons in the network
w=20
# empty array to save every model from each simulation
NN_mod_lst = []
# define 10-fold cross validation test harness
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=123)
cvscores = []
for train, test in kfold.split(X, y):
    # create model
    model = Sequential()
    model.add(Dense(w, input_dim=X.iloc[train].shape[1], activation='relu', kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(w, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # compile
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    # Fit the model
    model.fit(X.iloc[train], y.iloc[train], epochs=300, batch_size=10, verbose=0)
    # evaluate the model
    y_pred = model.predict(X.iloc[test])
    mse_value = mean_squared_error(y.iloc[test], y_pred)
    r2_value = r2_score(y.iloc[test], y_pred)
    y_benchmark = np.mean(y.iloc[train])
    benchmark = r2_score(y.iloc[test], np.repeat(y_benchmark, len(y.iloc[test])))
    print("benchmark:", benchmark)
    print("mse score:", mse_value)
    print("r2 score:", r2_value)
    cvscores.append([mse_value, r2_value, benchmark])

mse_cd4 = np.median(np.array(cvscores)[:, 0])    
benchmark_lst_cd4 = np.array(cvscores)[:, 2]
r2_lst_cd4 = np.array(cvscores)[:, 1]
cd4_ind = r2_lst_cd4.argsort()    
r2_lst_cd4_sorted = r2_lst_cd4[cd4_ind]
benchmark_sorted = benchmark_lst_cd4[cd4_ind]
r2_cd4 = np.mean(r2_lst_cd4_sorted[24:26])
benchmark_cd4 = np.mean(benchmark_sorted[24:26])
    
#%% ams
# normalize
X = ll_data.iloc[:, 7:]
y = ll_data['AMs']
y = np.log(y.copy())

# number of simulations over which to average
num_runs = 1
str_num_runs = str(num_runs)

# number of days
num_days = 9
# number of neurons in the network
w=20
# empty array to save every model from each simulation
NN_mod_lst = []
# define 10-fold cross validation test harness
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=123)
cvscores = []
for train, test in kfold.split(X, y):
    # create model
    model = Sequential()
    model.add(Dense(w, input_dim=X.iloc[train].shape[1], activation='relu', kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(w, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # compile
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    # Fit the model
    model.fit(X.iloc[train], y.iloc[train], epochs=300, batch_size=10, verbose=0)
    # evaluate the model
    y_pred = model.predict(X.iloc[test])
    mse_value = mean_squared_error(y.iloc[test], y_pred)
    r2_value = r2_score(y.iloc[test], y_pred)
    y_benchmark = np.mean(y.iloc[train])
    benchmark = r2_score(y.iloc[test], np.repeat(y_benchmark, len(y.iloc[test])))
    print("benchmark:", benchmark)
    print("mse score:", mse_value)
    print("r2 score:", r2_value)
    cvscores.append([mse_value, r2_value, benchmark])

mse_AMs = np.median(np.array(cvscores)[:, 0])    
benchmark_lst_AMs = np.array(cvscores)[:, 2]
r2_lst_AMs = np.array(cvscores)[:, 1]
AMs_ind = r2_lst_AMs.argsort()    
r2_lst_AMs_sorted = r2_lst_AMs[AMs_ind]
benchmark_sorted = benchmark_lst_AMs[AMs_ind]
r2_AMs = np.mean(r2_lst_AMs_sorted[24:26])
benchmark_AMs = np.mean(benchmark_sorted[24:26])

#%% neutrophils
# normalize
X = ll_data.iloc[:, 7:]
y = ll_data['Neutrophils ']
y = np.log(y.copy())

# number of simulations over which to average
num_runs = 1
str_num_runs = str(num_runs)

# number of days
num_days = 9
# number of neurons in the network
w=20
# empty array to save every model from each simulation
NN_mod_lst = []
# define 10-fold cross validation test harness
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=123)
cvscores = []
for train, test in kfold.split(X, y):
    # create model
    model = Sequential()
    model.add(Dense(w, input_dim=X.iloc[train].shape[1], activation='relu', kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(w, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # compile
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    # Fit the model
    model.fit(X.iloc[train], y.iloc[train], epochs=300, batch_size=10, verbose=0)
    # evaluate the model
    y_pred = model.predict(X.iloc[test])
    mse_value = mean_squared_error(y.iloc[test], y_pred)
    r2_value = r2_score(y.iloc[test], y_pred)
    y_benchmark = np.mean(y.iloc[train])
    benchmark = r2_score(y.iloc[test], np.repeat(y_benchmark, len(y.iloc[test])))
    print("benchmark:", benchmark)
    print("mse score:", mse_value)
    print("r2 score:", r2_value)
    cvscores.append([mse_value, r2_value, benchmark])

mse_neutrophils = np.median(np.array(cvscores)[:, 0])    
benchmark_lst_neutrophils = np.array(cvscores)[:, 2]
r2_lst_neutrophils = np.array(cvscores)[:, 1]
neutrophils_ind = r2_lst_neutrophils.argsort()    
r2_lst_neutrophils_sorted = r2_lst_neutrophils[neutrophils_ind]
benchmark_sorted = benchmark_lst_neutrophils[neutrophils_ind]
r2_neutrophils = np.mean(r2_lst_neutrophils_sorted[24:26])
benchmark_neutrophils = np.mean(benchmark_sorted[24:26])

#%% NK
# normalize
X = ll_data.iloc[:, 7:]
y = ll_data['NK cells ']
y = np.log(y.copy())

# number of simulations over which to average
num_runs = 1
str_num_runs = str(num_runs)

# number of days
num_days = 9
# number of neurons in the network
w=20
# empty array to save every model from each simulation
NN_mod_lst = []
# define 10-fold cross validation test harness
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=123)
cvscores = []
for train, test in kfold.split(X, y):
    # create model
    model = Sequential()
    model.add(Dense(w, input_dim=X.iloc[train].shape[1], activation='relu', kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(w, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # compile
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    # Fit the model
    model.fit(X.iloc[train], y.iloc[train], epochs=300, batch_size=10, verbose=0)
    # evaluate the model
    y_pred = model.predict(X.iloc[test])
    mse_value = mean_squared_error(y.iloc[test], y_pred)
    r2_value = r2_score(y.iloc[test], y_pred)
    y_benchmark = np.mean(y.iloc[train])
    benchmark = r2_score(y.iloc[test], np.repeat(y_benchmark, len(y.iloc[test])))
    print("benchmark:", benchmark)
    print("mse score:", mse_value)
    print("r2 score:", r2_value)
    cvscores.append([mse_value, r2_value, benchmark])

mse_NK = np.median(np.array(cvscores)[:, 0])    
benchmark_lst_NK = np.array(cvscores)[:, 2]
r2_lst_NK = np.array(cvscores)[:, 1]
NK_ind = r2_lst_NK.argsort()    
r2_lst_NK_sorted = r2_lst_NK[NK_ind]
benchmark_sorted = benchmark_lst_NK[NK_ind]
r2_NK = np.mean(r2_lst_NK_sorted[24:26])
benchmark_NK = np.mean(benchmark_sorted[24:26])

    # # callback
    # cb = [EarlyStopping(monitor="val_loss", min_delta=10**(-4), patience=10000),
    #       ModelCheckpoint(filepath='cyto-vl_best_in_latest_run.h5', save_best_only=True, monitor='mse')]
    # # fit
    # history = model.fit(X_train, y_train, validation_split=0.1, batch_size=9, verbose=0, epochs=300, callbacks=cb)
    # latest_model = load_model('cyto-vl_best_in_latest_run.h5')
    # # save best model
    # NN_mod_lst.append(latest_model)

#%% prediction on training data
# y_pred_train_lst = []
# for mod in NN_mod_lst:
#     y_pred_train_lst.append(mod.predict(X_train))

# # calculate individual mse, mae, r2 score for each run
# mse_train = []
# mae_train = []
# r2_train = []
# for y_p in y_pred_train_lst:
#     mse_train.append(mean_squared_error(y_train, y_p.flatten()))
#     mae_train.append(mean_absolute_error(y_train, y_p.flatten()))
#     r2_train.append(r2_score(y_train, y_p.flatten()))

# # calculate average mse, mae and r2 score
# mse_mean = np.mean(mse_train)
# mae_mean = np.mean(mae_train)
# r2_mean = np.mean(r2_train)

# # calculate average prediction
# y_pred_train = np.mean(y_pred_train_lst, axis=0).flatten()

# # calculate mse, mae and r2 score from average prediction
# mse_print = mean_squared_error(y_train, y_pred_train)
# mae_print = mean_absolute_error(y_train, y_pred_train)
# r2_print = r2_score(y_train, y_pred_train)

# # print results to console
# print("Training:")
# print("==================================")
# print("MSE:", mse_print)
# print("MAE:", mae_print)
# print("R2 score:", r2_print)
# print("==================================")

# # save results to file
# if save_results:
#     file1 = open("../Results/Cyto-VL_Mapping_Results_"+figname+".txt","a")
#     file1.write("=== VL Mapping (NN, Training, "+figname+") ===")
#     file1.write("\n MSE:        "+str(mse_print))
#     file1.write("\n MAE:        "+str(mae_print))
#     file1.write("\n R2 score:   "+str(r2_print))
#     file1.write("\n")
#     file1.close()


# #%% plot training data

# # backtransform data
# y_pred_train_bt = np.exp(y_pred_train)
# y_train_bt = np.exp(y_train)
# # set control group to zero
# y_train_bt = y_train_bt.replace(1, 0)

# # extract days
# df_day = vl_data["Day"]

# # days for plotting
# day_exp = np.array([0,1,2,3,4,5,7,9,11])

# #  to plot all days instead, uncomment the line below
# # day_exp = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

# # plotting
# for i,d in enumerate(day_exp):        
#     # relevant part of days dataframe
#     day = df_day[df_day == d]

#     # set plotting options
#     plt.style.use('seaborn-ticks')
#     fig = plt.figure(1, figsize=(15, 10))  # Create a figure instance
#     ms = 12                                # markersize
#     col = "#5974B3"                        # color prediction data
#     col_test = "#99B8FF"                   # color training data
#     col_vlin = "#B3C9FF"                   # color vertical line
#     # plot
#     if len(day) == 1:
#         plt.plot(d, y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
#         plt.plot(d, y_train_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
#         plt.vlines(d, y_pred_train_bt[day.index][0], y_train_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
#     elif len(day) == 2:
#         pos = 0.3
#         plt.plot([d-pos, d+pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
#         plt.plot([d-pos, d+pos], y_train_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
#         plt.vlines(d-pos, y_pred_train_bt[day.index][0], y_train_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+pos, y_pred_train_bt[day.index][1], y_train_bt[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
#     elif len(day) == 3:
#         pos = 0.3
#         plt.plot([d-pos, d, d+pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
#         plt.plot([d-pos, d, d+pos], y_train_bt[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
#         plt.vlines(d-pos, y_pred_train_bt[day.index][0], y_train_bt[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d, y_pred_train_bt[day.index][1], y_train_bt[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+pos, y_pred_train_bt[day.index][2], y_train_bt[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
#     elif len(day) == 4:
#         pos = 0.1
#         plt.plot([d-2*pos, d-pos, d+pos, d+2*pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
#         plt.plot([d-2*pos, d-pos, d+pos, d+2*pos], y_train_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
#         plt.vlines(d-2*pos, y_pred_train_bt[day.index][0], y_train_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d-pos, y_pred_train_bt[day.index][1], y_train_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+pos, y_pred_train_bt[day.index][2], y_train_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+2*pos, y_pred_train_bt[day.index][3], y_train_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
#     elif len(day) == 5 and d == 0:
#         pos = 0.1
#         plt.plot([d-2*pos, d-pos, d, d+pos, d+2*pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
#         plt.axvspan(day_exp[0]-0.5, (d+day_exp[i+1])/2, facecolor="lightgrey", alpha=0.5, zorder=-10)  # use axvspan to set grey area
#         #plt.axvline((-1+(d+day_exp[i+1])/2)/2, color="lightgrey", alpha=0.5, linewidth=133, zorder=-10)
#     elif len(day) == 5:
#         pos = 0.1
#         plt.plot([d-2*pos, d-pos, d, d+pos, d+2*pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
#         plt.plot([d-2*pos, d-pos, d, d+pos, d+2*pos], y_train_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
#         plt.vlines(d-2*pos, y_pred_train_bt[day.index][0], y_train_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d-pos, y_pred_train_bt[day.index][1], y_train_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d, y_pred_train_bt[day.index][2], y_train_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+pos, y_pred_train_bt[day.index][3], y_train_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+2*pos, y_pred_train_bt[day.index][4], y_train_bt.iloc[day.index].iloc[4], colors=col_vlin, linestyles="dashed")
#     elif len(day) == 6:
#         pos = 0.1
#         plt.plot([d-3*pos, d-2*pos, d-pos, d+pos, d+2*pos, d+3*pos], y_pred_train_bt[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
#         plt.plot([d-3*pos, d-2*pos, d-pos, d+pos, d+2*pos, d+3*pos], y_train_bt.iloc[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
#         plt.vlines(d-3*pos, y_pred_train_bt[day.index][0], y_train_bt.iloc[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d-2*pos, y_pred_train_bt[day.index][1], y_train_bt.iloc[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d-pos, y_pred_train_bt[day.index][2], y_train_bt.iloc[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+pos, y_pred_train_bt[day.index][3], y_train_bt.iloc[day.index].iloc[3], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+2*pos, y_pred_train_bt[day.index][4], y_train_bt.iloc[day.index].iloc[4], colors=col_vlin, linestyles="dashed")
#         plt.vlines(d+3*pos, y_pred_train_bt[day.index][5], y_train_bt.iloc[day.index].iloc[5], colors=col_vlin, linestyles="dashed")
#     # separate data with vertical lines
#     if d != 11:
#         plt.axvline((d+day_exp[i+1])/2, color="grey", linestyle="dashed")

# # general plotting setting    
# plt.xlim(day_exp[0]-0.5, day_exp[-1]+0.5)
# # plt.ylim(2, 10**8)
# plt.plot([], 'D', alpha=1, markersize=ms, color=col, label='Model Prediction')
# plt.plot([], 'o', alpha=1, markersize=ms, color=col_test, label='Experimental Data')
# plt.yscale('log')
# fs = 30
# plt.xticks(day_exp, size=fs)
# plt.yticks(size=fs)
# plt.title("Training Performance", size=45)
# plt.xlabel("Time/ Days", size=fs+2)
# plt.ylabel("Viral Load/ (NP copies/50ng RNA)", size=fs+2)
# plt.legend(loc="upper right", fontsize=fs-7, frameon=True)

# # save plot
# if save_results:
#     plt.savefig("../Plots/Supplemental/Cyto-Viral_load_NN_Training_"+figname+".png", dpi=300, bbox_inches="tight")
#     plt.close()
# else:
#     plt.show()
    
    
# print runtime of program
end = time.time()
print("Time/min:", (end - start)/60)


# # blood data
# X_train = data.iloc[:, 3:]
# # viral load
# y_train = data.iloc[:, 2]

# # remove repetitive variables
# X_train.drop("Lymphocytes (%)", axis=1, inplace=True)
# X_train.drop("Monocytes (%)", axis=1, inplace=True)
# X_train.drop("Granulocytes (%)", axis=1, inplace=True)
# X_train.drop("RDWc (%)", axis=1, inplace=True)
# X_train.drop("PCT (%)", axis=1, inplace=True)
# X_train.drop("PDWc (%)", axis=1, inplace=True)
# X_train = X_train.astype(float)







































































