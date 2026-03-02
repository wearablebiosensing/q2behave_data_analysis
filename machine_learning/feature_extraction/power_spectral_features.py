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
"""
Returns a set of power spectral density feaure from the PSD plot.
- feature_max
- feature_min
- feature_skewness
- feature_kurtosis
- feature_mean
- feature_sum_freq_diff
- feature_num_peaks
- feature_average_power
Returns a datafram for Ax, Ay, Az, Gx,Gy,Gz
"""

def calculate_features(axis_data, T, N, axis_name,Fs):
    print("calculate_features(): axis_name = ",axis_name)
    freqs, psd = welch(axis_data, fs=1/Fs) # nperseg=N/2
    psd = np.abs(psd)  # Take absolute value to ensure positive values

    # Find peaks using scipy's find_peaks
    peak_indices, _ = find_peaks(psd)
    peaks_freq = freqs[peak_indices]
    peaks_psd = psd[peak_indices]
    peak_diffs_freq = np.diff(peaks_freq)

    feature_max = max(psd)
    feature_min = min(psd)
    feature_skewness = skew(psd)
    feature_kurtosis = kurtosis(psd)
    try:
        feature_mean = sum(psd) / len(psd)
    except ZeroDivisionError:
        feature_mean = 0  # or any other value you want to assign when there are no elements in psd
    feature_sum_freq_diff = sum(peak_diffs_freq)
    try:
        feature_average_power = sum(peaks_psd) / len(peaks_psd)
    except ZeroDivisionError:
        feature_average_power = 0  # or any other value you want to assign when there are no elements in peaks_psd
    feature_num_peaks = len(peaks_psd)

    # feature_mean = sum(psd) / len(psd)
    # feature_sum_freq_diff = sum(peak_diffs_freq)
    # feature_average_power = sum(peaks_psd) / len(peaks_psd)
    return {
        f"{axis_name}_max_power_psd": feature_max,
        f"{axis_name}_min_power_psd": feature_min,
        f"{axis_name}_skewness_power_psd": feature_skewness,
        f"{axis_name}_kurtosis_power_psd": feature_kurtosis,
        f"{axis_name}_mean_power_psd": feature_mean,
        f"{axis_name}_sum_freq_diff_power_psd": feature_sum_freq_diff,
        f"{axis_name}_average_power_power_psd": feature_average_power,
        f"{axis_name}feature_num_peaks_power_psd": feature_num_peaks
    }

def extract_psd_features(df,Fs):
    selected_columns = df[['Filtered_Accel_X','Filtered_Accel_Y','Filtered_Accel_Z','Filtered_Gryo_X','Filtered_Gryo_Y','Filtered_Gryo_Z','Filtered_Gryo_Max','Filtered_Acc_Max']]
    N = len(selected_columns)
    T = 1 / Fs
    feature_set = {}  # Dictionary to store feature sets
    
    for axis in selected_columns.columns:
        axis_data = selected_columns[axis].values
        features = calculate_features(axis_data, T, N, axis,Fs)
        feature_set.update(features)
    
    # Convert the dictionary to a dataframe with a single row
    feature_df = pd.DataFrame([feature_set])
    return feature_df
"""
Helper function to plot the PSD while taking it out.
"""
def my_plot_psd(df,Fs):
    selected_columns = df.iloc[:, 4:10]
    N = len(selected_columns)
    print("nperseg: ",N,N/2)
    T = 1/Fs #@calculate_sampling_frequency(df)
    num_rows = 2
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.35)  # Increase the spacing between rows
    feature_sets = []  # List to store feature sets for each axis of each sensor

    for k in range(num_rows):
        for j in range(num_cols):
            column_index = k * num_cols + j
            if column_index >= selected_columns.shape[1]:
                # If there are no more columns to plot, break out of the loop
                break
            data_no_dc = selected_columns.iloc[:, column_index].values
            
            ax = axes[k, j]  # Get the current axis from the 'axes' array
            freqs, psd = welch(data_no_dc, fs=1/T) # nperseg=int(N/2)
            psd = np.abs(psd)  # Take absolute value to ensure positive values
            
            # Find peaks using scipy's find_peaks
            peak_indices, _ = find_peaks(psd)
            peaks_freq = freqs[peak_indices]
            peaks_psd = psd[peak_indices]
            peak_diffs_freq = np.diff(peaks_freq)
            ax.plot(freqs, psd, color='blue', label='PSD')
            ax.plot(peaks_freq, peaks_psd, 'ro', markersize=5, label='Peaks')  # Plot peaks
            ax.set_yscale('log')  # Set y-axis to logarithmic scale
            ax.grid()
            ax.set_title(f"{selected_columns.columns[column_index]} - PSD, no DC")
    plt.suptitle('')
    plt.show()
   