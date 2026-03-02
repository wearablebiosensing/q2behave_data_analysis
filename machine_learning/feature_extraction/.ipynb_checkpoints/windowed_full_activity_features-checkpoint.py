import pandas as pd 
import numpy as np 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
# from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from jupyter_dash import JupyterDash
from scipy import signal
from scipy.fft import fftshift
import plotly.graph_objects as go
import matplotlib.dates as mdates
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch
from scipy.signal import find_peaks
import scipy as sc
from scipy.stats import skew, kurtosis
import scipy.fftpack                 
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import chirp, peak_widths
import math
import sys
import time

#################################################################################################################################
# Iterates over a full activity instead of blobs of segments to extract features.
#################################################################################################################################

sys.path.append('/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/GitQ2Bheave/q2behave_data_analysis/jupyternotebooks/signal_processing_box')
from statistical_features import *
from power_spectral_features import *
from filtering_module import *
from helpers import *
from plotly.subplots import make_subplots
from sklearn import preprocessing 
# Function to find maximum value across three columns
def max_of_acc(row):
    return max(row[' Accel_X'], row[' Accel_Y'], row[' Accel_Z'])
def max_of_gry(row):
    return max(row[' Gyro_X'], row[' Gyro_Y'], row[' Gyro_Z'])


def average_sampling_rate(df):
    sr_dict = df["NewClock"].value_counts().to_dict()
    # print("average_sampling_rate():/ sr_dict: ",sr_dict)
    # Step 1: Calculate the sum of all values
    total_sum = sum(sr_dict.values())
    # Step 2: Count the number of values
    num_values = len(sr_dict)
    if num_values !=0:
        # print("num_values: ",num_values)
        # print("total_sum: ",total_sum)
        # Step 3: Calculate the average
        average_sr = total_sum / num_values
        return average_sr
    else:
        return 0
# Takes in 2 parameters df,window
# df raw pandas df with accelerometer and gryoscope data
# window : window size in number of samples for the moving average filter.
# Returns DF with raw and filtered data.
def apply_moving_avg(df,window,Fs):
    mov_average_filt_AccX = process_filter_code(df,' Accel_X',Fs,window)  
    mov_average_filt_AccY = process_filter_code(df,' Accel_Y',Fs,window)
    mov_average_filt_AccZ = process_filter_code(df,' Accel_Z',Fs,window)
                        
    mov_average_filt_GryX = process_filter_code(df,' Gyro_X',Fs,window)
    mov_average_filt_GryY = process_filter_code(df,' Gyro_Y',Fs,window)
    mov_average_filt_GryZ = process_filter_code(df,' Gyro_Z',Fs,window)
    mov_average_filt_Gry_max  = process_filter_code(df,'max_of_gry',Fs,window)
    mov_average_filt_Acc_max  = process_filter_code(df,'max_of_acc',Fs,window)

    #     df['max_of_acc'] = df.apply(max_of_acc, axis=1)
    # df['max_of_gry'] = df.apply(max_of_gry, axis=1)

    ############# FFT of filtered data 
    mov_avg_yf,mov_avg_xf,mov_avg_AMP =  my_fft(df[' Accel_X'].to_numpy(),Fs)
                        
    df["Filtered_Accel_X"] = mov_average_filt_AccX
    df["Filtered_Accel_Y"] = mov_average_filt_AccY
    df["Filtered_Accel_Z"] = mov_average_filt_AccZ
    df["Filtered_Gryo_X"] = mov_average_filt_GryX
    df["Filtered_Gryo_Y"] = mov_average_filt_GryY
    df["Filtered_Gryo_Z"] = mov_average_filt_GryZ
    df["Filtered_Gryo_Max"] = mov_average_filt_Gry_max
    df["Filtered_Acc_Max"] = mov_average_filt_Acc_max
    return df
"""
    Iterate over a pandas DataFrame with a specified window size and overlap.

    Parameters:
        df (pandas DataFrame): The DataFrame to iterate over.
        window_size (int): The size of each window (number of samples).
        overlap (int): The number of samples to overlap between windows.
"""
#full_activities_folder
def windowed_features(df, window_size, overlap):
    max_behavior_list = []
    num_samples = len(df)
    window_counter = 0
    start = 0
    dfs = []
    dfs_stat = []
    dfs_combined = []
    Fs = average_sampling_rate(df)
    ### Process the windowed data feature to get all the features #######.
    moving_average_filter_window_size = 5
    moving_avereaged_df = apply_moving_avg(df,moving_average_filter_window_size,Fs)
    while start < num_samples:
        end = min(start + window_size, num_samples)
        window_data = moving_avereaged_df.iloc[int(start):int(end)]
        #window_data = df.iloc[start:end]
        print("windowed_features()===========:/  window_data",window_data.shape)
        print("Window Number: ",window_counter )
        # Get The behaviour with maximum count.
        behavior_counts = window_data['BehaviorCode'].value_counts()
        max_behavior = behavior_counts.idxmax() if not behavior_counts.empty else None
        max_behavior_list.append(max_behavior)
        feature_df_psd = extract_psd_features(window_data[['Filtered_Accel_X','Filtered_Accel_Y','Filtered_Accel_Z','Filtered_Gryo_X','Filtered_Gryo_Y','Filtered_Gryo_Z','Filtered_Gryo_Max','Filtered_Acc_Max'
                                                                                 ]],Fs)
        feature_df_psd["BehaviorCode"] = max_behavior
        feature_df_psd["WindowNumber"] = window_counter
        print("feature_df_psd: ",type(feature_df_psd),feature_df_psd.columns, feature_df_psd.shape)
        dfs.append(feature_df_psd)
        ####################################
        #  Extracting Statistical Features            
        ####################################
        accel_gry_data = window_data[["Filtered_Accel_X","Filtered_Accel_Y","Filtered_Accel_Z","Filtered_Gryo_X","Filtered_Gryo_Y","Filtered_Gryo_Z",'Filtered_Gryo_Max','Filtered_Acc_Max']]
        features_df = get_concated_features(accel_gry_data)
        #features_df["BehaviorCode"] = max_behavior
        #features_df["WindowNumber"] = window_counter
        dfs_stat.append(features_df)
        df_concat = pd.concat([feature_df_psd, features_df], axis=1)
        dfs_combined.append(df_concat)
        window_counter+=1
        # yield start, end, window_data, max_behavior
        start += window_size - overlap
    print("iterate_over_windows(): max_behavior_list: ",len(max_behavior_list))
    print("len(dfs): ",len(dfs))
    # Concatenate the DataFrames vertically, but only keep the header of the first one
    combined_df = pd.concat(dfs, axis=0)
    combined_df_stat = pd.concat(dfs_stat, axis=0)
    dfs_combined_concat = pd.concat(dfs_combined, axis=0)

    
    # Reset index if needed
    # combined_df.reset_index(drop=True, inplace=True)

    return combined_df,combined_df_stat,dfs_combined_concat
########################################################################
# Codes for Behavior Types
# Moving chair =1
# Standing = 2
# Manipulating = 3
# Twirling = 4
# Drumming = 5
# Finger = 6
# NoHyp = 7 is also no hyperactive behavior
# pd.isna(value) =0 is also no hyperactive behavior
# None Type =  -1 
########################################################################
# Define a mapping function
def map_values(value):
    if pd.isna(value): ###### This is no hyperactivity
        return 0
    elif 'chair' in value:
        return 1
    elif 'Standing' in value:
        return 2
    elif "Manipulating" in value:
        return 3
    elif "Twirling" in value:
        return 4
    elif "Drumming" in value:
        return 5
    elif "Finger" in value:
        return 6
    elif "NoHyp" in value:
        return 0     
    else:# These are Talking to peers and Talking to Self behaviors we are not interested in.
        print("in windowing features == map_values():/ -1 values ====",value)
        return -1  # for values not matching any criteria
def run_sctivity_based_windowing(folder_path):
    feature_dfs_list = []
    stat_feature_df_list=[]
    all_feature_df_list =[]
    file_list = os.listdir(folder_path)
    files_without_ds_store = [file for file in file_list if not file.startswith('.DS_Store')]

    for filename in files_without_ds_store:
        if filename.endswith('.csv') and "Right" in filename and ".DS_Store" not in filename: #and behaviour_type_str in filename: #in filename and behaviour_type_str in filename: # and "P6" in filename, "NoHyp" not in filename
            substring1 = "Talking_to_peers"
            substring2 = "Talking_to_self"
            if substring1 in filename or substring2 in filename:
                print("Skipping Talking_to_peers")
                print("Skipping Talking_to_self")
                pass   
        else:
            file_path = os.path.join(folder_path, filename)
            print(filename, "=========================================================================================================")
            print("filename: ",filename)
            df = pd.read_csv(file_path)
       #      ['level_0', 'index', ' DataTS', ' EventTS', ' Accel_X', ' Accel_Y',
       # ' Accel_Z', ' Gyro_X', ' Gyro_Y', ' Gyro_Z', ' Magno_X', ' Magno_Y',
       # ' Magno_Z', ' Heartrate', ' AudioLevel', ' BatteryLevel', 'WatchID',
       # 'timestamp', 'timestamp_str', 'timestamp_str_seconds', 'NewClock',
       # 'Elapsed_Time', 'Off seat\n (0 -On seat ,1- yes off seat ) \n',
       # 'Hyperactive or Restless', 'Hyperactive Behaviour Type',
       # 'Annotator2_HyperactiveBehaviourType_Second_Coder',
       # 'Second Hyperactive Behaviour Type ', 'new_zeroTS'],
            #### Remove values when the activity was over.#### 
            df = df[df["Hyperactive Behaviour Type"]!='-100']
            df =  df[df["Hyperactive Behaviour Type"]!=-100]
            # Apply the mapping function to the column
            df['BehaviorCode'] = df['Annotator2_HyperactiveBehaviourType_Second_Coder'].apply(map_values)
            # Get the Max of Acc and Gry
            df['max_of_acc'] = df.apply(max_of_acc, axis=1)
            df['max_of_gry'] = df.apply(max_of_gry, axis=1)
            # ####################################
            # #  Extracting PSD Features            
            # ####################################
            Fs = average_sampling_rate(df)
            wind_size = int(Fs*2) # 2 seconds worth of data, and 1 second.
            overlap = int(wind_size/2)
            combined_df,stat_features_df,dfs_combined = windowed_features(df,wind_size , overlap)
            dfs_combined["filename"] = filename
            dfs_combined['participantId'] =  filename.split('_')[0]
            dfs_combined["activityID"] = filename.split('_')[2]
            feature_dfs_list.append(combined_df)
            # ####################################
            # #  Extracting Statistical Features            
            # ####################################
            stat_features_df["filename"] = filename
            stat_features_df['participantId'] =  filename.split('_')[0]
            stat_features_df["activityID"] = filename.split('_')[3][:2]
            stat_feature_df_list.append(stat_features_df)
            all_feature_df_list.append(dfs_combined)
    return feature_dfs_list,stat_feature_df_list,all_feature_df_list
folder_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/Synced_Files_All_New"
feature_dfs_list,stat_feature_df_list,all_feature_df_list = run_sctivity_based_windowing(folder_path)

df_feature = pd.concat(feature_dfs_list)
all_features_df = pd.concat(all_feature_df_list)
df_stat_features = pd.concat(stat_feature_df_list)

df_stat_features.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/CodeOutputs/Windowing_Features/stat_features_two_second_P5.csv")
# print('It took', time.time()-start, 'seconds to calcualte PSD features.')
df_feature.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/CodeOutputs/Windowing_Features/psd_features_two_second_P.csv")
all_features_df.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/CodeOutputs/Windowing_Features/all_features_two_second_P5.csv")