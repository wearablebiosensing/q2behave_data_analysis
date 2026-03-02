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


# features to analyse
#mean
def feature_mean(feature_name,accel_gyro_data):
    mean = accel_gyro_data.mean()
    mean_transposed_df = mean.to_frame().T
    mean_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" +feature_name,'Gryo_Max_'+feature_name,'Acc_Max_'+feature_name]
    return mean_transposed_df
    
#std
def feature_std(feature_name,accel_gyro_data):
    std = accel_gyro_data.std()
    std_transposed_df = std.to_frame().T
    std_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" +feature_name,'Gryo_Max_'+feature_name,'Acc_Max_'+feature_name]
    return std_transposed_df

#kurtosis
def feature_kurtosis(feature_name,accel_gyro_data):
    kurtosis = accel_gyro_data.kurtosis()
    kurtosis_transposed_df = kurtosis.to_frame().T
    kurtosis_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_"+feature_name,'Gryo_Max_'+feature_name,'Acc_Max_'+feature_name]
    return kurtosis_transposed_df

#median
def feature_median(feature_name,accel_gyro_data):
    median = accel_gyro_data.median()
    median_transposed_df = median.to_frame().T
    median_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" + feature_name,'Gryo_Max_'+feature_name,'Acc_Max_'+feature_name]
    return median_transposed_df

#skewness
def feature_skewness(feature_name,accel_gyro_data):
    skewness = accel_gyro_data.skew()
    skewness_transposed_df = skewness.to_frame().T
    skewness_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" + feature_name,'Gryo_Max_'+feature_name,'Acc_Max_'+feature_name]
    return skewness_transposed_df
# vector distance mean
def feature_sum(feature_name,accel_gyro_data):
    sum_ = accel_gyro_data.sum()
    sum_transposed_df = sum_.to_frame().T
    sum_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" +feature_name, 'Gryo_Max_'+feature_name,'Acc_Max_'+feature_name]
    return sum_transposed_df

def calculate_zero_crossing_rate(accel_data):
    accel_data = (accel_data - np.mean(accel_data)) / np.std(accel_data)
    # Calculate Zerocrossing rate.
    zero_crossings = np.nonzero(np.diff(np.signbit(accel_data)))[0]
    zcr = len(zero_crossings) / (2.0 * len(accel_data))
    return zcr

def feature_zero_cross(feature_name,accel_gyro_data):
    zero_crossing_rates = {}
    for column in accel_gyro_data.columns:
        zcr = calculate_zero_crossing_rate(accel_gyro_data[column])
        zero_crossing_rates[column] = zcr
    zcr_series = pd.Series(zero_crossing_rates)
    zcr_transposed_df = zcr_series.to_frame().T
    zcr_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" +feature_name,'Gryo_Max_'+feature_name,'Acc_Max_'+feature_name]
    return zcr_transposed_df


# vector distance mean
def diff_sum(feature_name,accel_gyro_data):
    sum_ = accel_gyro_data.diff().sum()
    sum_transposed_df = sum_.to_frame().T
    sum_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" +feature_name,'Gryo_Max_'+feature_name,'Acc_Max_'+feature_name]
    return sum_transposed_df

# create table 
def get_concated_features(accel_gyro_data):    
    df1 = feature_mean("mean",accel_gyro_data)
    df2 = feature_median("median",accel_gyro_data)
    df3 = feature_std("std",accel_gyro_data)
    df4 = feature_kurtosis("kurtosis",accel_gyro_data)
    df5 =feature_skewness("skewness",accel_gyro_data)
    df6 = feature_sum("sum",accel_gyro_data)
    df7 = feature_zero_cross("zero_cross_rate", accel_gyro_data)
    df8 = diff_sum("diff_sum", accel_gyro_data)
    df = [df1, df2, df3, df4, df5, df6,df7,df8]
    result_df = pd.concat(df, axis=1)
    return result_df
