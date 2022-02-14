#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create plots for supplemental material. Data is fitted with selected algorithms and the results
of the mapping on testing data are plotted. Additionally, corresponding feature importance analysis
plots are created.

To create and edit specific plots comment/uncomment the code at the bottom of this file.

The plots are saved under "Plots/Supplemental"

@author: Suneet Singh Jhutty
@date:   02.08.22
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import re

figname = "09-02-22"

# %% help functions
def get_train(target_name, target, data="VL", nf=None):
    """function to load training data
    
       target_name: name you want go give variable to estimate
       target:      name of variable in training data set
       target_test: name of variable in testing data set. If None, use
                    target
       data:        "VL" for Viral Load, "LL" for lung leukocyte or "C" for cytokine data
       nf:          specify features to use. If None, all features are used
       return:      X_train, y_train
    """
    name_blood = ['Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
                  'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs']

    if data == "C":
        # blood data from V2 ---------------------------------------------------------------
        blood_V2 = pd.read_excel('Experiment2_Vetscan_Lung_Leukocyte_and_Cytokine_data.xlsx',
                                       sheet_name='VetScan Data')

        # delete unneccessary columns with no relevant information
        df_blood_V2 = blood_V2.iloc[:, 4:53]       
        df_blood_V2 = df_blood_V2[df_blood_V2.columns.drop(list(df_blood_V2.filter(regex='Hinweis')))]
        df_blood_V2 = df_blood_V2[df_blood_V2.columns.drop(list(df_blood_V2.filter(regex='EOS')))]
        df_blood_V2 = df_blood_V2[df_blood_V2.columns.drop(list(df_blood_V2.filter(regex='BAS')))]
        df_blood_V2 = df_blood_V2[df_blood_V2.columns.drop(list(df_blood_V2.filter(regex='% %')))]
        df_blood_V2.drop("PCT %", axis=1, inplace=True)
        df_blood_V2.drop("PDWc %", axis=1, inplace=True)
        df_blood_V2.drop("RDWc %", axis=1, inplace=True)
        # delete unneccessary rows with no relevant information
        df_blood_V2 = df_blood_V2.dropna(axis=0).reset_index(drop=True)
        # rename day column
        df_blood_V2.replace("PBS d1", 0, inplace=True)
        df_blood_V2.replace("IAV d1", 1, inplace=True)
        df_blood_V2.replace("IAV d2", 2, inplace=True)
        df_blood_V2.replace("IAV d3", 3, inplace=True)
        df_blood_V2.replace("IAV d4", 4, inplace=True)
        df_blood_V2.replace("IAV d5", 5, inplace=True)
        df_blood_V2.replace("IAV d7", 7, inplace=True)
        df_blood_V2.replace("IAV d9", 9, inplace=True)
        df_blood_V2.replace("IAV d11", 11, inplace=True)
        df_blood_V2.drop("Mouse#", axis=1, inplace=True)
        # remove units from names
        name_blood = ['Day', 'Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
                      'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs']
        df_blood_V2.columns =  name_blood
        df_blood_V2 = df_blood_V2.astype(float)

        # load cytokine data
        cytokines_V2 = pd.read_excel('Experiment2_Vetscan_Lung_Leukocyte_and_Cytokine_data.xlsx', sheet_name='Airway cytokines')
        # clean_data
        df_cyto = cytokines_V2.copy()
        df_cyto = df_cyto.iloc[:, 2:]
        df_cyto.drop(0, inplace=True)
        df_cyto.reset_index(inplace=True, drop=True)
        for col in df_cyto.columns:
                df_cyto[col] = df_cyto[col].astype(str)
                df_cyto[col] = df_cyto[col].str.replace('<' , '')
                df_cyto[col] = df_cyto[col].str.replace(',' , '.')
                df_cyto[col] = df_cyto[col].astype(float)

        # exchange rows to match blood data
        def change_row(data, idx1, idx2):
            """remove row with index idx1 and put it on row with idx2"""
            mouse46 = data.iloc[idx1]
            data.drop(index=idx1, inplace=True)
            first_half = data.iloc[:idx2]
            second_half = data.iloc[idx2:]
            return pd.concat([first_half.append(mouse46), second_half], ignore_index=True)

        df_cyto = change_row(df_cyto, 5, 9)
        # create dataframe with all relevant data from V2
        df = df_blood_V2.copy()
        df.drop("Day", axis=1, inplace=True)
        df[target_name] = df_cyto[target]
        
    elif data == "LL":
        # blood data from V2 ---------------------------------------------------------------
        blood_V2 = pd.read_excel('Experiment2_Vetscan_Lung_Leukocyte_and_Cytokine_data.xlsx',
                                       sheet_name='VetScan Data')

        # delete unneccessary columns with no relevant information
        df_blood_V2 = blood_V2.iloc[:, 4:53]       
        df_blood_V2 = df_blood_V2[df_blood_V2.columns.drop(list(df_blood_V2.filter(regex='Hinweis')))]
        df_blood_V2 = df_blood_V2[df_blood_V2.columns.drop(list(df_blood_V2.filter(regex='EOS')))]
        df_blood_V2 = df_blood_V2[df_blood_V2.columns.drop(list(df_blood_V2.filter(regex='BAS')))]
        df_blood_V2 = df_blood_V2[df_blood_V2.columns.drop(list(df_blood_V2.filter(regex='% %')))]
        df_blood_V2.drop("PCT %", axis=1, inplace=True)
        df_blood_V2.drop("PDWc %", axis=1, inplace=True)
        df_blood_V2.drop("RDWc %", axis=1, inplace=True)
        # delete unneccessary rows with no relevant information
        df_blood_V2 = df_blood_V2.dropna(axis=0).reset_index(drop=True)
        # rename day column
        df_blood_V2.replace("PBS d1", 0, inplace=True)
        df_blood_V2.replace("IAV d1", 1, inplace=True)
        df_blood_V2.replace("IAV d2", 2, inplace=True)
        df_blood_V2.replace("IAV d3", 3, inplace=True)
        df_blood_V2.replace("IAV d4", 4, inplace=True)
        df_blood_V2.replace("IAV d5", 5, inplace=True)
        df_blood_V2.replace("IAV d7", 7, inplace=True)
        df_blood_V2.replace("IAV d9", 9, inplace=True)
        df_blood_V2.replace("IAV d11", 11, inplace=True)
        df_blood_V2.drop("Mouse#", axis=1, inplace=True)
        # remove units from names
        name_blood = ['Day', 'Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
                      'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs']
        df_blood_V2.columns =  name_blood
        df_blood_V2 = df_blood_V2.astype(float)

        # lung leukocyte data from V2
        lung_leukocytes = pd.read_excel('Experiment2_Vetscan_Lung_Leukocyte_and_Cytokine_data.xlsx',
                                         sheet_name='Cleaned lung tissue leukocytes')

        lung_cells = lung_leukocytes[[target]]

        # create dataframe with all relevant data from V2
        df = df_blood_V2.copy()
        df.drop("Day", axis=1, inplace=True)
        df[target_name] = lung_cells
        
    else:
        # blood data from V1 ---------------------------------------------------------------
        blood_V1 = pd.read_excel('Experiment1_Vetscan_and_Viral_load_data.xlsx',
                                       sheet_name='Vetscan Data')
        # get only relevant data columns
        df_blood_V1 = blood_V1.iloc[:, 2:23]
        # delete unneccessary rows with no relevant information
        df_blood_V1 = df_blood_V1.drop(0)
        df_blood_V1 = df_blood_V1.drop(33)
        df_blood_V1 = df_blood_V1.drop(35).reset_index(drop=True)
        # delete excess variables
        df_blood_V1.drop("Day", axis=1, inplace=True)
        df_blood_V1.drop("Lymphocytes (%)", axis=1, inplace=True)
        df_blood_V1.drop("Monocytes (%)", axis=1, inplace=True)
        df_blood_V1.drop("Granulocytes (%)", axis=1, inplace=True)
        df_blood_V1.drop("PCT (%)", axis=1, inplace=True)
        df_blood_V1.drop("PDWc (%)", axis=1, inplace=True)
        df_blood_V1.drop("RDWc (%)", axis=1, inplace=True)
        df_blood_V1 = df_blood_V1.astype(float)
        # remove units from names
        df_blood_V1.columns =  name_blood

        # load viral load data from lungs
        raw_vl_data = pd.read_excel('Experiment1_Vetscan_and_Viral_load_data.xlsx', sheet_name='Lung viral load')
        df_vl_V1 = raw_vl_data.copy()
        # get only relevant data columns
        df_vl_V1 = raw_vl_data.iloc[:, 2:5]
        # set NAN entries to zero
        df_vl_V1["Day"] = pd.to_numeric(df_vl_V1.iloc[:, 0], errors='coerce').fillna(0)
        df_vl_V1[df_vl_V1.columns[1]] = pd.to_numeric(df_vl_V1.iloc[:, 1], errors='coerce').fillna(0)
        df_vl_V1["Viral load STD"] = pd.to_numeric(df_vl_V1.iloc[:, 2], errors='coerce').fillna(0)
        # delete data with no Vetscan equivalent
        df_vl_V1 = df_vl_V1.drop([9, 19, 32, 42, 45, 46, 47, 48, 49])
        # reset index
        vl = df_vl_V1.iloc[:, 1:2].reset_index(drop=True)
        # relable columns
        vl.columns = ["Viral Load"]
        # create dataframe with all relevant data
        df = df_blood_V1.copy()
        df[target_name] = vl["Viral Load"]
    
    # features to use
    if nf == None:
        name_features = ['Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
                         'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs']
    else:
        name_features = nf
        
    X_train = df[name_features]
    y_train = df[target_name]
    return X_train, y_train


def get_test(target_name, target, target_test=None, data="VL", nf=None):
    """function to load training data
   
       target_name: name you want go give variable to estimate
       target:      name of variable in training data set
       target_test: name of variable in testing data set. If None, use
                    target
       data:        "VL" for Viral Load, "LL" for lung leukocyte or "C" for cytokine data
       nf:          specify features to use. If None, all features are used
       return:      X_train, y_train
    """
    name_blood = ['Day', 'Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
                  'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs']
    if data == "C":
        name_blood = ['Day', 'Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
              'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs']
        df_V4 = pd.read_excel('Experiment4_Vetscan_and_Lung_Leukocyte_data.xlsx')
        # drop unnecessary columns
        df_blood_V4 = df_V4.iloc[:, 0:16]
        df_blood_V4.drop("Experiment", axis=1, inplace=True)
        # rename day column
        df_blood_V4.replace("PBS", 0, inplace=True)
        df_blood_V4.replace("IAV d2", 2, inplace=True)
        df_blood_V4.replace("IAV d4", 4, inplace=True)
        df_blood_V4.replace("IAV d6", 6, inplace=True)
        df_blood_V4.replace("IAV d9 ", 9, inplace=True)
        df_blood_V4.replace("IAV d11", 11, inplace=True)
        df_blood_V4.columns = name_blood

        # remove row because cytokine equivalent is missing
        df_blood_V4.drop(5, inplace=True)
        df_blood_V4.reset_index(inplace=True, drop=True)

        cytokines_V3_V4 = pd.read_excel('Experiment3_and_4_Cytokine_data.xlsx')
        df_cyto_eval = cytokines_V3_V4.iloc[:, 4:17]
        df_cyto_eval.drop(0, inplace=True)
        df_cyto_eval.reset_index(inplace=True, drop=True)

        for col in df_cyto_eval.columns:
                df_cyto_eval[col] = df_cyto_eval[col].astype(str)
                df_cyto_eval[col] = df_cyto_eval[col].str.replace('<' , '')
                df_cyto_eval[col] = df_cyto_eval[col].str.replace(',' , '.')
                df_cyto_eval[col] = df_cyto_eval[col].astype(float)

        # create dataframe with all relevant data from V2
        df_eval = df_blood_V4.copy()
        df_day = df_eval["Day"]
        df_eval.drop("Day", axis=1, inplace=True)
        df_eval[target_name] = df_cyto_eval[target]
    elif data == "LL":
        df_V4 = pd.read_excel('Experiment4_Vetscan_and_Lung_Leukocyte_data.xlsx')
        # drop unnecessary columns
        df_blood_V4 = df_V4.iloc[:, 0:16]
        df_blood_V4.drop("Experiment", axis=1, inplace=True)
        # rename day column
        df_blood_V4.replace("PBS", 0, inplace=True)
        df_blood_V4.replace("IAV d2", 2, inplace=True)
        df_blood_V4.replace("IAV d4", 4, inplace=True)
        df_blood_V4.replace("IAV d6", 6, inplace=True)
        df_blood_V4.replace("IAV d9 ", 9, inplace=True)
        df_blood_V4.replace("IAV d11", 11, inplace=True)
        df_blood_V4.columns = name_blood
        df_eval = df_blood_V4.copy()
        df_day = df_eval["Day"]
        df_eval.drop("Day", axis=1, inplace=True)
        df_eval[target_name] = df_V4[target_test]
    else:
        # days of measurement
        dayd = pd.read_excel('Experiment3_Vetscan_and_Viral_load_data.xlsx', sheet_name="Raw Data")
        df_day = dayd["Treatment"]
        df_day.replace("IAV d2", 2, inplace=True)
        df_day.replace("IAV d4", 4, inplace=True)
        df_day.replace("IAV d6", 6, inplace=True)
        df_day.replace("IAV d9 ", 9, inplace=True)
        df_day.replace("IAV d11", 11, inplace=True)
        
        # %% blood data from V3  -------------------------------------------------------------
        blood_V3 = pd.read_excel('Experiment3_Vetscan_and_Viral_load_data.xlsx', sheet_name="Cleaned Data")
        df_blood_V3 = blood_V3.copy()
        df_vl_V3 = df_blood_V3.pop('Viral load [NP copies/50ng RNA]')
        print("länge columns=", len(df_blood_V3.columns))
        name_blood = ['Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
                      'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs']
        # remove units from names
        df_blood_V3.columns =  name_blood
        # create dataframe with all relevant data
        df_eval = df_blood_V3.copy()
        df_eval["Viral Load"] = df_vl_V3
    
    # features to use
    if nf == None:
            name_features = ['Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
                             'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs']
    else:
        name_features = nf

    X_test = df_eval[name_features]
    y_test = df_eval[target_name]
    return X_test, y_test, df_day


def plot_fig(y_pred, y_test, df_day, name="Viral Load/ (NP copies/50ng RNA)"):
    """function to plot results of mapping on testing data
    
    y_pred: values predicted by model
    y_test: true values from measurement
    df_day: measurement days
    name:   name of mapped quantity
    """
    
    # go through days
    day_exp = np.array([0,2,4,6,9,11])
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
            plt.plot(d, y_pred[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
            plt.plot(d, y_test[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
            plt.vlines(d, y_pred[day.index][0], y_test[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
        elif len(day) == 2:
            pos = 0.3
            plt.plot([d-pos, d+pos], y_pred[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
            plt.plot([d-pos, d+pos], y_test[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
            plt.vlines(d-pos, y_pred[day.index][0], y_test[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
            plt.vlines(d+pos, y_pred[day.index][1], y_test[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
        elif len(day) == 3 and d == 0:
            pos = 0.3
            plt.plot([d-pos, d, d+pos], y_pred[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
            plt.plot([d-pos, d, d+pos], y_test[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
            plt.vlines(d-pos, y_pred[day.index][0], y_test[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
            plt.vlines(d, y_pred[day.index][1], y_test[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
            plt.vlines(d+pos, y_pred[day.index][2], y_test[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
            plt.axvspan(day_exp[0]-0.5, (d+day_exp[i+1])/2, facecolor="lightgrey", alpha=0.5, zorder=-10)  # use axvspan to set grey area
        elif len(day) == 3:
            pos = 0.3
            plt.plot([d-pos, d, d+pos], y_pred[day.index], 'D', alpha=1, markersize=ms, color=col, zorder=3)
            plt.plot([d-pos, d, d+pos], y_test[day.index], 'o', alpha=1, markersize=ms, color=col_test, zorder=3)
            plt.vlines(d-pos, y_pred[day.index][0], y_test[day.index].iloc[0], colors=col_vlin, linestyles="dashed")
            plt.vlines(d, y_pred[day.index][1], y_test[day.index].iloc[1], colors=col_vlin, linestyles="dashed")
            plt.vlines(d+pos, y_pred[day.index][2], y_test[day.index].iloc[2], colors=col_vlin, linestyles="dashed")
        
        # separate data with vertical lines
        if d != 11:
            plt.axvline((d+day_exp[i+1])/2, color="grey", linestyle="dashed")
        
    plt.xlim(day_exp[0]-0.5, day_exp[-1]+0.5)
    plt.plot([], 'D', alpha=1, markersize=ms, color=col, label='Model Estimation')
    plt.plot([], 'o', alpha=1, markersize=ms, color=col_test, label='Experimental Data')
    # general plotting setting    
    plt.yscale('log')
    fs = 30
    plt.title("Testing Performance", size=45)
    plt.xticks(day_exp, size=fs)
    plt.yticks(size=fs+10)
    
    # plotting settings unique to AMs
    if name == "Total AMs":
        plt.yscale("linear")
        plt.yticks([200000, 400000, 600000, 800000],
                   [r"2 x 10$^5$", r"4 x 10$^5$", r"6 x 10$^5$", r"8 x 10$^5$"])
    
    plt.xlabel("Time/ Days", size=fs+2)
    plt.ylabel(name, size=fs+2)
    plt.legend(loc="upper left", fontsize=fs-7, frameon=True)
    
    # edit name under which to save plot
    save_name = re.sub(r'[^\w]', '-', name)
    save_name = re.sub(r'-+', '-', save_name)
    save_name = re.sub(r' ', '_', save_name)
    plt.savefig("Plots/Supplemental/"+save_name+"_Mapping_"+figname+".png", dpi=300, bbox_inches="tight")
    plt.close()


def fit_and_plot(target, target_name, target_test=None, method="GBRT", data="C", para={}, nf=None):
    """fit data with selected algorithm and plot results of mapping on testing data 
       and feature importance if applicable
    
    target_name: name you want go give variable to estimate
    target:      name of variable in training data set
    target_test: name of variable in testing data set. If None, use
                 target
    method:      chose model for the fitting: 
                    GBRT     - Gradient Boostet Regression Trees
                    GBRTwPCA - Gradient Boostet Regression Trees with Principal Component Analysis
                    LR       - Linear Regression
                    LRwPCA   - Linear Regression with Principal Component Analysis
                    SVM      - Support Vector Machine
    data:        "VL" for Viral Load, "LL" for lung leukocyte or "C" for cytokine data
    para:        parameters for the algorithm
    nf:          specify features to use. If None, all features are used
    """
    name_blood = pd.Series(['Leukocytes', 'Lymphocytes', 'Monocytes', 'Granulocytes', 'Erythrocytes',
                            'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','RDWs', 'Platelets', 'MPV', 'PDWs'])
 
    if target_test == None:
        target_test = target
        
    # get training and testing data
    X_train, y_train = get_train(target_name, target, data=data, nf=None)
    X_test, y_test, df_day = get_test(target_name, target, target_test=target_test, data=data, nf=None)
    
    # scale data
    Scaler = preprocessing.MinMaxScaler()
    X_train = Scaler.fit_transform(X_train)
    X_test = Scaler.transform(X_test)
    
    # fit model
    n_pca = 6
    if method=="GBRT":
        est = GradientBoostingRegressor(**para)
        est.fit(X_train, np.array(y_train).flatten())
        y_pred = est.predict(X_test)
    elif method=="GBRTwPCA":
        pca = PCA(n_components=n_pca)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        est = GradientBoostingRegressor(**para)
        est.fit(X_train, np.array(y_train).flatten())
        y_pred = est.predict(X_test)
    elif method=="LRwPCA":
        pca = PCA(n_components=n_pca)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        est = LinearRegression()
        est.fit(X_train, np.array(y_train).flatten())
        y_pred = est.predict(X_test)
    elif method=="LR":
        est = LinearRegression()
        est.fit(X_train, np.array(y_train).flatten())
        y_pred = est.predict(X_test)
    elif method=="SVM":
        est = svm.SVR(**para)
        est.fit(X_train, np.array(y_train).flatten())
        y_pred = est.predict(X_test)

    # print r2 score in console
    print("r2", r2_score(y_test, y_pred))
    # plot mapping on testing data
    plot_fig(y_pred, y_test, df_day, target_name)
    
    # if applicable, plot feature importance analysis
    if target != "CD4+ T" and target != "CD8+ T":
        dfeval = X_test
        y_eval = y_test
        fs = 18
        col = "#5974B3"
        col2 = "#B3C9FF"
        # premutation importance
        result = permutation_importance(est, dfeval, y_eval, n_repeats=1000,
                                        random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()
    
        # create figure
        fig = plt.figure(1, figsize=(15,12))
        # create an axes instance
        ax = fig.add_subplot(122)
        # create boxplot
        bppi=ax.boxplot(result.importances[sorted_idx].T, patch_artist=True,
                    vert=False, labels=name_blood[sorted_idx])
    
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
            plt.xlabel('Relative Importance', fontsize=fs)
            t=plt.title("Permutation Importance (test set)", fontsize=fs+8)
            t.set_y(1.01)
            plt.subplots_adjust(top=0.86)
            fig.tight_layout()
    
            # %% plot impurity-based feature importance if GBRT is used
            if method == "GBRT":
                feature_importance = est.feature_importances_
                sorted2_idx = np.argsort(feature_importance)
                pos = np.arange(sorted2_idx.shape[0]) + .5
                ax2 = fig.add_subplot(121)
                ax2.barh(pos, feature_importance[sorted2_idx], color=col, align='center')
                plt.xticks(fontsize=fs)
                plt.yticks(pos, name_blood[sorted2_idx], fontsize=fs)
                plt.xlabel('Relative Importance', fontsize=fs)
                t = plt.title('Impurity-Based Feature Importance', fontsize=fs+8)
                t.set_y(1.01)
                plt.subplots_adjust(top=0.86)
                
        # edit name under which to save plot
        save_name = re.sub(r'[^\w]', '-', target_name)
        save_name = re.sub(r'-+', '-', save_name)
        save_name = re.sub(r' ', '_', save_name)  
        plt.savefig("Plots/Supplemental/Feature_Importance_"+save_name+"_"+figname+".png", dpi=300, bbox_inches="tight")
        plt.close()

#### Uncomment the lung leukocyte or cytokine for which to plot the mapping ####
#%% TNF-alpha
# params = {'kernel': 'rbf'}
# TNF_alpha = fit_and_plot("A7.TNF-a", r"TNF-$\alpha$/ (pg/ml)", method="SVM", para=params)

#%% NK cells
# params = {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.041, 'random_state': 0, 'loss': 'ls'}
# NK_cells = fit_and_plot("NK", r"Total NK cells", "NK cells ", method="GBRT", para=params, data="LL")

#%% IFN-beta
# params = {'kernel': 'rbf'}
# IFN_beta = fit_and_plot("B7.IFN-b", r"IFN-$\beta$/ (pg/ml)", method="SVM", para=params)

#%% CD4+ T cells
# CD4 = fit_and_plot("CD4+ T", "CD4+ T cells", "CD4+ T cells ", method="LRwPCA", para={}, data="LL")

#%% CD8+ T cells
# CD8 = fit_and_plot("CD8+ T", "CD8+ T cells", "CD8+ T cells ", method="LRwPCA", para={}, data="LL")

#%% AMs
# AMs = fit_and_plot("Ams", "Total AMs", "Alveolar macrophages ", method="LR", para={}, data="LL")

















