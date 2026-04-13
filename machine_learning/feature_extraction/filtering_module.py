import pandas as pd 
import numpy as np 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
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

################################################# FROM BSN CODE #################################################
"""
Takes in a pandas dataframe and sample rate of that dataframe calculated by another function.
Returns: yf,xf,AMP parameters used to plot the FFT of a signal.
"""
def my_fft(df,Fs):
    N = len(df)
    yf = scipy.fftpack.fft(df,n=N)
    # n, period of the signal.
    #print("my_fft() N//2: ",N//2)
    xf = scipy.fftpack.fftfreq(N,1/Fs)[:N//2] # get the frequency component Fs  = 40 Hz
    #print("my_fft() xf:",xf)
    #print("xf: ",xf)
    AMP= 2.0/N * np.abs(yf[0:N//2])
    #print("AMP: ",AMP)
    return yf,xf,AMP
"""
Takes List X i.e the signal to be filtered, window size i.e numnber of samples 
Returns: Filtered signal with moving average.
"""
def moving_average(X, window_size):
    X_new =[]
    # end_index = window_size
    for i, value in enumerate(X):
        #print("index, value",i, value)
        X_new.insert(i, sum(X[i:i+window_size])/len(X[i:i+window_size]))
    return X_new

"""
Takes: lowcut, highcut, fs, order=5 parameters of a band pass filter.
Returns: band pass filtered signal.
"""

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


"""
RUN THIS FUNCTION TO GET FILTERED DATA.
Note: Detrends the signal before passing in the fillteres.
Main function to run the filter code above. Arrange the function calling for moving average and band pass accourding to the requirement.
Takes: lowcut, highcut, fs, order=5 parameters of a band pass filter.
Returns:Filtered signal with a combination of filteres or a sigle filter.
"""
# low=1,high=5 is Frequency (Hz)of the bands  of interest  
# window number of samples in one window in the moving average filter.
def process_filter_code(df,column_name,Fs,window,low=1,high=5): # ' Accel_X'
    # 1) DETREND
    detrended_signal = signal.detrend(df[column_name])
    # 2) BAND PASS 1-5 Hz (applied to detrended_signal)
    band_pass_filtered = butter_bandpass_filter(detrended_signal, low, high, Fs, order=5)
    
    return band_pass_filtered
"""
Helpler function to visvulize the filtered signals.
Plots a 2x3 plot containing Ax Ay Az Gx Gy Gz aling with their filtered data.
"""
def filtering_plots(df):
    # Setting fonts and styles
    title_font = 12
    axis_font = 40
    font = {'family': 'Sans'}
    plt.rc('font', **font)
    plt.rc('xtick', labelsize=50)
    plt.rc('ytick', labelsize=50)
    # Create a 2x3 grid of subplots (2 rows, 3 columns)
    fig, axs = plt.subplots(2, 3, figsize=(30, 20))  # Adjust figsize as needed
    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the space between the plots

    # Plotting Accelerometer data
    axs[0, 0].plot(df[' Accel_X'], color="#171ad1", linestyle='--', label="Raw Data")
    axs[0, 0].plot(df["Filtered_Accel_X"], color="#478F96", linewidth=4, label="Moving Average")
    axs[0, 0].set_title('Acceleration X', fontsize=title_font)
    axs[0, 0].legend(loc='upper right')

    axs[0, 1].plot(df[' Accel_Y'], color="#171ad1", linestyle='--', label="Raw Data")
    axs[0, 1].plot(df["Filtered_Accel_Y"], color="#478F96", linewidth=4, label="Moving Average")
    axs[0, 1].set_title('Acceleration Y', fontsize=title_font)
    axs[0, 1].legend(loc='upper right')

    axs[0, 2].plot(df[' Accel_Z'], color="#171ad1", linestyle='--', label="Raw Data")
    axs[0, 2].plot(df["Filtered_Accel_Z"], color="#478F96", linewidth=4, label="Moving Average")
    axs[0, 2].set_title('Acceleration Z', fontsize=title_font)
    axs[0, 2].legend(loc='upper right')

    # Plotting Gyroscope data
    axs[1, 0].plot(df[' Gyro_X'], color="#171ad1", linestyle='--', label="Raw Data")
    axs[1, 0].plot(df["Filtered_Gryo_X"], color="#478F96", linewidth=4, label="Moving Average")
    axs[1, 0].set_title('Gyroscope X', fontsize=title_font)
    axs[1, 0].legend(loc='upper right')

    axs[1, 1].plot(df[' Gyro_Y'], color="#171ad1", linestyle='--', label="Raw Data")
    axs[1, 1].plot(df["Filtered_Gryo_Y"], color="#478F96", linewidth=4, label="Moving Average")
    axs[1, 1].set_title('Gyroscope Y', fontsize=title_font)
    axs[1, 1].legend(loc='upper right')

    axs[1, 2].plot(df[' Gyro_Z'], color="#171ad1", linestyle='--', label="Raw Data")
    axs[1, 2].plot(df["Filtered_Gryo_Z"], color="#478F96", linewidth=4, label="Moving Average")
    axs[1, 2].set_title('Gyroscope Z', fontsize=title_font)
    axs[1, 2].legend(loc='upper right')

    return plt