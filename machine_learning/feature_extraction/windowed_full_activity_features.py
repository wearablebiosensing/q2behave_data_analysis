# ---------------- Standard Library Imports ----------------
import os
import sys
import math
import time
from datetime import datetime

# ---------------- Third-Party Imports ----------------
import pandas as pd
import numpy as np

# SciPy signal / stats / FFT
from scipy import stats
from scipy.signal import (
    welch, find_peaks, butter, lfilter, peak_prominences, chirp, peak_widths
)
from scipy.fft import fftshift
from scipy.stats import skew, kurtosis
import scipy.fftpack

# ---------------- Visualization Imports ----------------
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- Local Project Imports ----------------
sys.path.append(
    '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/CODE/GitQ2Bheave/q2behave_data_analysis/machine_learning/feature_extraction'
)
from statistical_features import  * #extract_psd_features, get_concated_features  # explicitly list used functions
from power_spectral_features import *  # replace * with explicit imports if possible
from filtering_module import  * #process_filter_code, butter_bandpass_filter  # explicitly list used functions
from helpers import *  # replace * with explicit imports if possible

#################################################################################################################################
# Iterates over a full activity instead of blobs of segments to extract features.
#################################################################################################################################

def max_of_acc(row):
    return max(row[' Accel_X'], row[' Accel_Y'], row[' Accel_Z'])

def max_of_gry(row):
    return max(row[' Gyro_X'], row[' Gyro_Y'], row[' Gyro_Z'])

def average_sampling_rate(df):
    sr_dict = df["NewClock"].value_counts().to_dict()
    total_sum = sum(sr_dict.values())
    num_values = len(sr_dict)
    if num_values != 0:
        average_sr = total_sum / num_values
        return average_sr
    else:
        return 0

def apply_moving_avg(df, window, Fs):

    mov_average_filt_AccX = process_filter_code(df,' Accel_X',Fs,window)  
    mov_average_filt_AccY = process_filter_code(df,' Accel_Y',Fs,window)
    mov_average_filt_AccZ = process_filter_code(df,' Accel_Z',Fs,window)
                        
    mov_average_filt_GryX = process_filter_code(df,' Gyro_X',Fs,window)
    mov_average_filt_GryY = process_filter_code(df,' Gyro_Y',Fs,window)
    mov_average_filt_GryZ = process_filter_code(df,' Gyro_Z',Fs,window)
    mov_average_filt_Gry_max  = process_filter_code(df,'max_of_gry',Fs,window)
    mov_average_filt_Acc_max  = process_filter_code(df,'max_of_acc',Fs,window)

    mov_avg_yf,mov_avg_xf,mov_avg_AMP = my_fft(df[' Accel_X'].to_numpy(),Fs)
                        
    df["Filtered_Accel_X"] = mov_average_filt_AccX
    df["Filtered_Accel_Y"] = mov_average_filt_AccY
    df["Filtered_Accel_Z"] = mov_average_filt_AccZ
    df["Filtered_Gryo_X"] = mov_average_filt_GryX
    df["Filtered_Gryo_Y"] = mov_average_filt_GryY
    df["Filtered_Gryo_Z"] = mov_average_filt_GryZ
    df["Filtered_Gryo_Max"] = mov_average_filt_Gry_max
    df["Filtered_Acc_Max"] = mov_average_filt_Acc_max

    return df

############################################################
# Windowed Feature Extraction
############################################################

def windowed_features(df, window_size, overlap):

    num_samples = len(df)
    window_counter = 0
    start = 0

    dfs_psd = []
    dfs_stat = []
    dfs_combined = []

    Fs = average_sampling_rate(df)
    moving_average_filter_window_size = 5
    moving_avereaged_df = apply_moving_avg(df,moving_average_filter_window_size,Fs)

    while start < num_samples:

        end = min(start + window_size, num_samples)
        window_data = moving_avereaged_df.iloc[int(start):int(end)]

        print("windowed_features()===========:/  window_data",window_data.shape)
        print("Window Number: ",window_counter )

        behavior_counts = window_data['BehaviorCode'].value_counts()
        max_behavior = behavior_counts.idxmax() if not behavior_counts.empty else None

        ############################################
        # PSD FEATURES
        ############################################

        feature_df_psd = extract_psd_features(
            window_data[['Filtered_Accel_X','Filtered_Accel_Y','Filtered_Accel_Z',
                         'Filtered_Gryo_X','Filtered_Gryo_Y','Filtered_Gryo_Z',
                         'Filtered_Gryo_Max','Filtered_Acc_Max']],Fs)

        feature_df_psd["BehaviorCode"] = max_behavior
        feature_df_psd["WindowNumber"] = window_counter

        ############################################
        # STAT FEATURES
        ############################################

        accel_gry_data = window_data[[
            "Filtered_Accel_X","Filtered_Accel_Y","Filtered_Accel_Z",
            "Filtered_Gryo_X","Filtered_Gryo_Y","Filtered_Gryo_Z",
            'Filtered_Gryo_Max','Filtered_Acc_Max'
        ]]

        features_df = get_concated_features(accel_gry_data)

        features_df["BehaviorCode"] = max_behavior
        features_df["WindowNumber"] = window_counter

        ############################################
        # COMBINED
        ############################################

        df_concat = pd.concat([feature_df_psd, features_df], axis=1)

        dfs_psd.append(feature_df_psd)
        dfs_stat.append(features_df)
        dfs_combined.append(df_concat)

        window_counter += 1
        start += window_size - overlap

    combined_df_psd = pd.concat(dfs_psd, axis=0)
    combined_df_stat = pd.concat(dfs_stat, axis=0)
    combined_df_all = pd.concat(dfs_combined, axis=0)

    return combined_df_psd, combined_df_stat, combined_df_all


############################################################
# Behavior Mapping (UNCHANGED)
############################################################

def map_values(value):
    
    if pd.isna(value):
        return -1
    elif 'chair' in value:
        return 1
    elif 'Standing' in value:
        return -1
    elif "Manipulating" in value:
        return 1
    elif "Twirling" in value:
        return 1
    elif "Drumming" in value:
        return 1
    elif "Finger" in value:
        return 1
    elif "peers" in value:
        return -1
    elif "self" in value:
        return -1
    elif "NoHyp" in value:
        return 0     
    else:
        print("in windowing features == map_values():/ -1 values ====",value)
        return -1


############################################################
# MAIN LOOP (Metadata enforced uniformly)
############################################################

def run_activity_based_windowing(folder_path):

    feature_dfs_list = []
    stat_feature_df_list = []
    all_feature_df_list = []

    file_list = os.listdir(folder_path)
    files_without_ds_store = [file for file in file_list if not file.startswith('.DS_Store')]

    for filename in files_without_ds_store:

        if "Talking_to_peers" in filename or "Talking_to_self" in filename:
            print("Skipping Talking behavior:", filename)
            continue

        if not (filename.endswith('.csv') and "Right" in filename):
            continue

        file_path = os.path.join(folder_path, filename)

        print(filename, "=========================================================================================================")
        print("filename: ",filename)

        df = pd.read_csv(file_path)

        df = df[df["HyperactiveBehaviourType"]!='-100']
        df = df[df["HyperactiveBehaviourType"]!=-100]
        if len(df) < 30:
            print("Skipping file (less than 30 rows):", filename)
            continue
        behavior_str = filename.split('_')[3]

        df['BehaviorCode'] = map_values(behavior_str)
        df['max_of_acc'] = df.apply(max_of_acc, axis=1)
        df['max_of_gry'] = df.apply(max_of_gry, axis=1)

        Fs = average_sampling_rate(df)
        wind_size = int(Fs * 2)
        overlap = int(wind_size / 2)

        psd_df, stat_df, combined_df = windowed_features(df, wind_size, overlap)

        participant_id = filename.split('_')[0]
        activity_id = filename.split('_')[2]

        ################################################
        # Uniform Metadata Injection
        ################################################

        for meta_df in [psd_df, stat_df, combined_df]:
            meta_df["filename"] = filename
            meta_df["participantId"] = participant_id
            meta_df["activityID"] = activity_id

        feature_dfs_list.append(psd_df)
        stat_feature_df_list.append(stat_df)
        all_feature_df_list.append(combined_df)

    return feature_dfs_list, stat_feature_df_list, all_feature_df_list


############################################################
# RUN
############################################################

root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/DATASET/"
folder_path = root + "BlobsOFSegments"

feature_dfs_list, stat_feature_df_list, all_feature_df_list = run_activity_based_windowing(folder_path)

df_feature = pd.concat(feature_dfs_list)
all_features_df = pd.concat(all_feature_df_list)
df_stat_features = pd.concat(stat_feature_df_list)

df_stat_features.to_csv(root + "CodeOutputs/2026/Windowing_Features/BlobsOFSegments/stat_features_two_second.csv")
df_feature.to_csv(root + "CodeOutputs/2026/Windowing_Features/BlobsOFSegments/psd_features_two_second.csv")
all_features_df.to_csv(root + "CodeOutputs/2026/Windowing_Features/BlobsOFSegments/all_features_two_second.csv")